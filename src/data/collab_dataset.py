"""
collab_dataset.py -- Dataset and data pipeline for Phase 2 collaboration training.

CollabPairDataset:
  - Loads annotated (Person A, Person B) pair clips from pair_catalog.csv
  - Runs frozen SwinClipModel on both clips to extract 768-d temporal features
  - Computes interaction signals (4-d) from spatial info stored in catalog
  - Returns (feat_A, feat_B, signals, label) tuples for CollaborationHead training

Feature pre-extraction:
  Running SwinClipModel on all pairs during training loop is slow.
  Instead, we pre-extract all features once and cache them.
  Training then only touches CollaborationHead (2.4M params) on cached features.

Usage:
  # Build dataset splits
  python src/data/collab_dataset.py --build_splits

  # In training script:
  dataset = CollabPairDataset(split='train', cache_dir='data/collab_cache')
  loader  = DataLoader(dataset, batch_size=16, shuffle=True)
"""

import os
import sys
import csv
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from collections import defaultdict
import torchvision.transforms as T
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CATALOG_CSV  = "data/collab_raw/pair_catalog.csv"
SPLITS_DIR   = "data/collab_splits"
CACHE_DIR    = "data/collab_cache"
CLIP_LEN     = 8

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Train/val/test splits at VIDEO level (all pairs from same video → same split)
TRAIN_FRAC = 0.75
VAL_FRAC   = 0.15
TEST_FRAC  = 0.10


# ---------------------------------------------------------------------------
# Image transform
# ---------------------------------------------------------------------------

def build_val_transform(image_size: int = 224) -> T.Compose:
    return T.Compose([
        T.Resize(int(image_size * 1.143)),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Feature pre-extraction
# ---------------------------------------------------------------------------

def extract_and_cache_features(
    catalog_csv : str,
    model_path  : str,
    cache_dir   : str,
    device      : str = "cpu",
):
    """
    Pre-extract Swin temporal features for all annotated pairs.
    Features are saved as numpy arrays to avoid re-running the backbone at training time.

    This is a ONE-TIME operation. Takes ~10-30 min on CPU for 800 pairs.
    On GPU: ~2-3 min.

    Output:
      cache_dir/
        {pair_id}_A.npy   ← (768,) feat for Person A
        {pair_id}_B.npy   ← (768,) feat for Person B
        feature_index.csv ← maps pair_id → cache file paths
    """
    from src.models.swin_clip_model import build_clip_model
    from src.models.collaboration_head import build_feature_extractor

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Load engagement model and wrap as feature extractor
    print(f"Loading engagement model from {model_path} ...")
    engagement_model = build_clip_model(num_classes=2, pretrained=False)
    ckpt = torch.load(model_path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)
    engagement_model.load_state_dict(state_dict, strict=False)

    extractor = build_feature_extractor(engagement_model)
    extractor = extractor.to(device)

    transform = build_val_transform()

    # Load catalog
    pairs = []
    with open(catalog_csv, newline='') as f:
        for row in csv.DictReader(f):
            if row.get('label') in ('C', 'N'):   # Only labeled pairs
                pairs.append(row)

    print(f"Extracting features for {len(pairs)} labeled pairs ...")
    index_rows = []

    for i, pair in enumerate(pairs):
        pair_id = pair['pair_id']

        path_A_feat = cache_path / f"{pair_id}_A.npy"
        path_B_feat = cache_path / f"{pair_id}_B.npy"

        # Skip if already cached
        if path_A_feat.exists() and path_B_feat.exists():
            index_rows.append({
                'pair_id': pair_id, 'label': pair['label'],
                'feat_A': str(path_A_feat), 'feat_B': str(path_B_feat),
                'video_id': pair['video_id'],
                'frame_w': pair.get('frame_w', 848),
                'frame_h': pair.get('frame_h', 480),
                'track_id_A': pair.get('track_id_A', 0),
                'track_id_B': pair.get('track_id_B', 1),
            })
            continue

        # Load clip frames for person A
        feat_A = _extract_clip_feature(pair['clip_dir_A'], extractor, transform, device)
        feat_B = _extract_clip_feature(pair['clip_dir_B'], extractor, transform, device)

        if feat_A is None or feat_B is None:
            print(f"  WARNING: Skipping {pair_id} — cannot load clip frames")
            continue

        np.save(str(path_A_feat), feat_A)
        np.save(str(path_B_feat), feat_B)

        index_rows.append({
            'pair_id': pair_id, 'label': pair['label'],
            'feat_A': str(path_A_feat), 'feat_B': str(path_B_feat),
            'video_id': pair['video_id'],
            'frame_w': pair.get('frame_w', 848),
            'frame_h': pair.get('frame_h', 480),
            'track_id_A': pair.get('track_id_A', 0),
            'track_id_B': pair.get('track_id_B', 1),
        })

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(pairs)} done")

    # Save index
    index_csv = cache_path / "feature_index.csv"
    if index_rows:
        with open(index_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(index_rows[0].keys()))
            writer.writeheader()
            writer.writerows(index_rows)
    print(f"Feature cache complete. Index: {index_csv} ({len(index_rows)} pairs)")
    return str(index_csv)


def _extract_clip_feature(
    clip_dir : str,
    extractor,
    transform,
    device   : str,
) -> Optional[np.ndarray]:
    """Load 8-frame clip and extract 768-d temporal feature."""
    clip_path = Path(clip_dir)
    frames = []

    for i in range(CLIP_LEN):
        fp = clip_path / f"frame_{i:04d}.jpg"
        if not fp.exists():
            fp = clip_path / f"frame_{i:04d}.png"
        if fp.exists():
            try:
                img = Image.open(fp).convert("RGB")
                frames.append(transform(img))
            except Exception:
                frames.append(torch.zeros(3, 224, 224))
        else:
            frames.append(torch.zeros(3, 224, 224))

    if not frames:
        return None

    # Pad if needed
    while len(frames) < CLIP_LEN:
        frames.append(frames[-1].clone())

    clip_tensor = torch.stack(frames[:CLIP_LEN]).unsqueeze(0).to(device)  # (1, 8, 3, 224, 224)

    with torch.no_grad():
        _, clip_feat = extractor(clip_tensor)   # (1, 768)

    return clip_feat.squeeze(0).cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Build video-level splits
# ---------------------------------------------------------------------------

def build_splits(
    feature_index_csv : str,
    splits_dir        : str = SPLITS_DIR,
    seed              : int = 42,
):
    """
    Split feature_index.csv at VIDEO level (all pairs from same video → same split).
    This prevents scene/appearance leakage between train and val.

    Also adds symmetric pairs: (A, B) → (B, A) in train split only.
    (Adding symmetric pairs to val/test would leak, since swapped pair is essentially same data)
    """
    np.random.seed(seed)
    Path(splits_dir).mkdir(parents=True, exist_ok=True)

    rows = []
    with open(feature_index_csv, newline='') as f:
        rows = list(csv.DictReader(f))

    # Group by video_id
    by_video: Dict[str, list] = defaultdict(list)
    for row in rows:
        by_video[row['video_id']].append(row)

    videos = sorted(by_video.keys())
    np.random.shuffle(videos)

    n = len(videos)
    n_train = int(n * TRAIN_FRAC)
    n_val   = int(n * VAL_FRAC)

    train_videos = set(videos[:n_train])
    val_videos   = set(videos[n_train : n_train + n_val])
    test_videos  = set(videos[n_train + n_val:])

    train_rows = [r for r in rows if r['video_id'] in train_videos]
    val_rows   = [r for r in rows if r['video_id'] in val_videos]
    test_rows  = [r for r in rows if r['video_id'] in test_videos]

    # Add symmetric pairs to train (A↔B swap)
    sym_rows = []
    for row in train_rows:
        sym = row.copy()
        sym['pair_id']    = row['pair_id'] + "_sym"
        sym['feat_A']     = row['feat_B']
        sym['feat_B']     = row['feat_A']
        sym['track_id_A'] = row['track_id_B']
        sym['track_id_B'] = row['track_id_A']
        sym_rows.append(sym)
    train_rows = train_rows + sym_rows

    def save_split(split_rows, name):
        if not split_rows:
            print(f"  WARNING: {name} split is empty!")
            return
        out_path = Path(splits_dir) / f"{name}.csv"
        with open(out_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(split_rows[0].keys()))
            writer.writeheader()
            writer.writerows(split_rows)
        n_C = sum(1 for r in split_rows if r['label'] == 'C')
        n_N = sum(1 for r in split_rows if r['label'] == 'N')
        print(f"  {name:5s}: {len(split_rows):4d} pairs  (C={n_C}, N={n_N})  "
              f"→ {out_path}")

    print(f"\nSplit breakdown ({len(videos)} videos total):")
    print(f"  Train videos: {len(train_videos)}")
    print(f"  Val   videos: {len(val_videos)}")
    print(f"  Test  videos: {len(test_videos)}")

    save_split(train_rows, 'train')
    save_split(val_rows,   'val')
    save_split(test_rows,  'test')

    print(f"\nSplits saved to {splits_dir}/")


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class CollabPairDataset(Dataset):
    """
    PyTorch Dataset for collaboration pair training.

    Returns (feat_A, feat_B, signals, label) tuples from pre-extracted features.

    Interaction signals are computed from stored spatial metadata (bbox positions).
    Because we don't have temporal history at load time, we use STATIC approximations:
      - proximity: computed from track bounding box positions (stored in catalog)
      - facing: estimated from relative position
      - correlation, turn_taking: set to 0.5 (neutral) — only valid during live inference
        where we have a rolling temporal history

    During training, the model must learn to collaborate from features + proximity/facing.
    Correlation and turn-taking improve it at inference time with real temporal data.

    Parameters:
        split:    'train', 'val', or 'test'
        splits_dir: directory containing {split}.csv
    """

    LABEL_MAP = {'C': 1.0, 'N': 0.0}

    def __init__(
        self,
        split      : str = 'train',
        splits_dir : str = SPLITS_DIR,
    ):
        split_csv = Path(splits_dir) / f"{split}.csv"
        if not split_csv.exists():
            raise FileNotFoundError(
                f"Split file not found: {split_csv}\n"
                f"Run first:\n"
                f"  python src/data/collab_dataset.py --build_splits"
            )

        self.rows = []
        with open(split_csv, newline='') as f:
            for row in csv.DictReader(f):
                if row['label'] in self.LABEL_MAP:
                    self.rows.append(row)

        self.split = split
        print(f"CollabPairDataset [{split}]: {len(self.rows)} pairs  "
              f"(C={sum(1 for r in self.rows if r['label']=='C')}, "
              f"N={sum(1 for r in self.rows if r['label']=='N')})")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            feat_A:  (768,) float32 tensor
            feat_B:  (768,) float32 tensor
            signals: (4,)   float32 tensor
            label:   () scalar float tensor (1.0=collab, 0.0=not collab)
        """
        row = self.rows[idx]

        # Load pre-extracted features
        feat_A = torch.from_numpy(np.load(row['feat_A'])).float()
        feat_B = torch.from_numpy(np.load(row['feat_B'])).float()

        # Compute static interaction signals
        signals = self._compute_static_signals(row)

        label = torch.tensor(self.LABEL_MAP[row['label']], dtype=torch.float32)

        return feat_A, feat_B, signals, label

    def _compute_static_signals(self, row: dict) -> torch.Tensor:
        """
        Compute 4-d interaction signals from stored spatial metadata.
        Temporal signals (correlation, turn_taking) set to 0.5 (neutral) at training time.
        These will be computed from live data during inference.
        """
        try:
            frame_w = float(row.get('frame_w', 848))
            frame_h = float(row.get('frame_h', 480))
            frame_diag = (frame_w**2 + frame_h**2)**0.5

            # For training, we don't have actual bboxes in the feature_index
            # We use the pair_id to get approximate positions via track_id ordering
            # (Track IDs lower = appeared earlier = likely left side in crowded scenes)
            tid_A = int(row.get('track_id_A', 0))
            tid_B = int(row.get('track_id_B', 1))

            # Approximate horizontal separation from track_id ordering
            # (rough heuristic — actual signal computed during live inference)
            h_sep = min(abs(tid_A - tid_B) * 0.1, 0.5)   # mild separation prior

            # Proximity: unknown at train time (we don't have bbox coords in feature cache)
            # Use 0.6 as "near" prior (collab pairs tend to be physically close)
            # The model will learn to use proximity from actual inference signals
            proximity     = 0.60
            facing        = min(h_sep * 1.5, 1.0)
            correlation   = 0.50   # neutral
            turn_taking   = 0.50   # neutral

        except Exception:
            proximity = 0.5
            facing    = 0.5
            correlation = 0.5
            turn_taking = 0.5

        return torch.tensor(
            [proximity, facing, correlation, turn_taking],
            dtype=torch.float32
        )

    def get_class_weights(self) -> float:
        """
        Returns pos_weight for BCEWithLogitsLoss (neg_count / pos_count).
        Used by training script to handle class imbalance.
        """
        n_pos = sum(1 for r in self.rows if r['label'] == 'C')
        n_neg = sum(1 for r in self.rows if r['label'] == 'N')
        if n_pos == 0:
            return 1.0
        return n_neg / n_pos


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build_splits",  action="store_true",
                        help="Build train/val/test splits from feature_index.csv")
    parser.add_argument("--extract_features", action="store_true",
                        help="Pre-extract Swin features from annotated pairs")
    parser.add_argument("--model_path",    default="weights/best_clip_model.pth")
    parser.add_argument("--catalog_csv",   default=CATALOG_CSV)
    parser.add_argument("--cache_dir",     default=CACHE_DIR)
    parser.add_argument("--splits_dir",    default=SPLITS_DIR)
    parser.add_argument("--device",        default="cpu")
    args = parser.parse_args()

    if args.extract_features:
        index_csv = extract_and_cache_features(
            catalog_csv = args.catalog_csv,
            model_path  = args.model_path,
            cache_dir   = args.cache_dir,
            device      = args.device,
        )
        print(f"Feature index: {index_csv}")

    if args.build_splits:
        index_csv = Path(args.cache_dir) / "feature_index.csv"
        if not index_csv.exists():
            print(f"ERROR: Feature index not found at {index_csv}")
            print(f"Run first: python src/data/collab_dataset.py --extract_features")
            return
        build_splits(str(index_csv), splits_dir=args.splits_dir)

    if not args.extract_features and not args.build_splits:
        parser.print_help()
        print("\nExample workflow:")
        print("  1. python src/data/collab_dataset.py --extract_features --model_path weights/best_clip_model.pth")
        print("  2. python src/data/collab_dataset.py --build_splits")
        print("  3. python src/training/train_collab.py")


if __name__ == "__main__":
    main()
