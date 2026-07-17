"""
build_custom_splits.py
Convert annotated custom clips into frame-level train/val/test CSVs
that are directly compatible with ClipDataset.

Split strategy: VIDEO-LEVEL (not person-level)
  Why: All clips from a given classroom video share the same room,
  lighting, and camera angle. A person-level split would allow the model
  to overfit to room appearance rather than engagement behaviour.
  28 videos → ~20 train / 4 val / 4 test (rough 70/15/15 split).

Output CSV columns (same schema as DAiSEE splits):
  image_path  str   -- path to one frame JPG
  label       int   -- 0 = Not Engaged, 1 = Engaged
  clip_id     str   -- groups 8 frames of one clip
  person_id   str   -- groups all clips of one person track

After creating custom splits you can EITHER:
  (a) train only on custom data:
        set CONFIG['train_csv'] = 'data/splits/custom_train.csv'
  (b) mix with DAiSEE (recommended when custom data is small):
        python build_custom_splits.py --merge-daisee
        then use 'data/splits/merged_train.csv'

Usage:
  python src/data/build_custom_splits.py
  python src/data/build_custom_splits.py --merge-daisee
  python src/data/build_custom_splits.py --val-frac 0.15 --test-frac 0.15 --seed 42
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import CONFIG


# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
CUSTOM_CLIPS_CSV = CONFIG.get('custom_clips_csv', 'custom_dataset/clips_catalog.csv')
CUSTOM_ANNOT_CSV = CONFIG.get('custom_annot_csv', 'custom_dataset/annotations.csv')
CUSTOM_TRAIN_CSV = CONFIG.get('custom_train_csv', 'data/splits/custom_train.csv')
CUSTOM_VAL_CSV   = CONFIG.get('custom_val_csv',   'data/splits/custom_val.csv')
CUSTOM_TEST_CSV  = CONFIG.get('custom_test_csv',  'data/splits/custom_test.csv')
MERGED_TRAIN_CSV = CONFIG.get('merged_train_csv', 'data/splits/merged_train.csv')
MERGED_VAL_CSV   = CONFIG.get('merged_val_csv',   'data/splits/merged_val.csv')
MERGED_TEST_CSV  = CONFIG.get('merged_test_csv',  'data/splits/merged_test.csv')


# ──────────────────────────────────────────────────────────────────────────────
# Clip → frame-level expansion
# ──────────────────────────────────────────────────────────────────────────────

def expand_clips_to_frames(clip_df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand clip-level rows (one row per clip) to frame-level rows
    (one row per frame) that ClipDataset expects.

    ClipDataset groups rows by clip_id, so each frame in the same clip
    must share the same clip_id.  Label is majority-voted per clip inside
    ClipDataset, but since we label whole clips here, all frames in a clip
    carry the same label.
    """
    rows = []
    for _, row in clip_df.iterrows():
        paths = str(row['frame_paths']).split(';')
        label = int(row['label_int'])
        for p in paths:
            # Normalise Windows backslashes → forward slashes so the CSV
            # works on Linux/Colab even when generated on Windows.
            p = p.strip().replace('\\', '/')
            if p:
                rows.append({
                    'image_path' : p,
                    'label'      : label,
                    'clip_id'    : str(row['clip_id']),
                    'person_id'  : str(row['person_id']),
                })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Build splits
# ──────────────────────────────────────────────────────────────────────────────

def build_splits(
    val_frac    : float = 0.15,
    test_frac   : float = 0.15,
    seed        : int   = 42,
    merge_daisee: bool  = False,
):
    # ── Load and validate inputs ───────────────────────────────────────────────
    for path in [CUSTOM_CLIPS_CSV, CUSTOM_ANNOT_CSV]:
        if not Path(path).exists():
            print(f"\n❌  File not found: {path}")
            if 'clips_catalog' in path:
                print("    Run  python src/data/custom_video_processor.py  first.")
            else:
                print("    Run  python src/data/annotate_tracks.py  first.")
            return

    clips_df = pd.read_csv(CUSTOM_CLIPS_CSV)
    annot_df = pd.read_csv(CUSTOM_ANNOT_CSV)

    # Join on clip_id; only keep clips that were annotated.
    # clips_catalog.csv may contain an empty placeholder 'label' column —
    # use suffixes=('_old', '') so the annotation label wins as 'label'.
    df = clips_df.merge(
        annot_df[['clip_id', 'label']],
        on='clip_id',
        how='inner',
        suffixes=('_old', ''),
    )

    # Keep only Engaged (E) and Not Engaged (N); drop Skipped (S)
    df = df[df['label'].isin(['E', 'N'])].copy()
    df['label_int'] = df['label'].map({'N': 0, 'E': 1})

    print(f"\n{'='*65}")
    print(f"  BUILD CUSTOM SPLITS")
    print(f"  Labeled clips   : {len(df)}")
    print(f"  Engaged (E)     : {(df['label']=='E').sum()}")
    print(f"  Not Engaged (N) : {(df['label']=='N').sum()}")
    print(f"  Skipped (removed): {(annot_df['label']=='S').sum()}")
    print(f"  Unique videos   : {df['video_id'].nunique()}")
    print(f"{'='*65}")

    if len(df) == 0:
        print("\n❌  No labeled clips found. Run annotate_tracks.py first.")
        return

    # ── Video-level stratified split ──────────────────────────────────────────
    # Aggregate: majority engagement label per video for stratification
    video_labels = (
        df.groupby('video_id')['label_int']
          .agg(lambda x: int(x.mode()[0]))
          .reset_index()
          .rename(columns={'label_int': 'vid_label'})
    )

    videos    = video_labels['video_id'].values
    vid_strat = video_labels['vid_label'].values

    # With very few videos, stratified split may fail (e.g., only 1 class in a split).
    # Fall back to random split gracefully.
    try:
        train_val_vids, test_vids = train_test_split(
            videos, test_size=test_frac,
            stratify=vid_strat, random_state=seed
        )
        # Build labels array for train+val subset
        tv_mask        = np.isin(videos, train_val_vids)
        tv_strat       = vid_strat[tv_mask]
        adjusted_val   = val_frac / (1.0 - test_frac)
        train_vids, val_vids = train_test_split(
            train_val_vids, test_size=adjusted_val,
            stratify=tv_strat, random_state=seed
        )
    except ValueError:
        print("  ⚠️   Too few videos for stratified split — using random split.")
        train_val_vids, test_vids = train_test_split(
            videos, test_size=test_frac, random_state=seed
        )
        adjusted_val = val_frac / (1.0 - test_frac)
        train_vids, val_vids = train_test_split(
            train_val_vids, test_size=adjusted_val, random_state=seed
        )

    train_set = set(train_vids)
    val_set   = set(val_vids)
    test_set  = set(test_vids)

    print(f"\n  Video-level split:")
    print(f"    Train : {len(train_set):>3} videos")
    print(f"    Val   : {len(val_set):>3} videos")
    print(f"    Test  : {len(test_set):>3} videos")

    # ── Expand to frame-level rows ─────────────────────────────────────────────
    train_df = expand_clips_to_frames(df[df['video_id'].isin(train_set)])
    val_df   = expand_clips_to_frames(df[df['video_id'].isin(val_set)])
    test_df  = expand_clips_to_frames(df[df['video_id'].isin(test_set)])

    # Shuffle train
    train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # ── Leakage check ─────────────────────────────────────────────────────────
    tc = set(train_df['clip_id'])
    vc = set(val_df['clip_id'])
    ec = set(test_df['clip_id'])
    assert len(tc & vc) == 0, "LEAKAGE: train/val share clip_ids!"
    assert len(tc & ec) == 0, "LEAKAGE: train/test share clip_ids!"
    assert len(vc & ec) == 0, "LEAKAGE: val/test share clip_ids!"

    # ── Save CSVs ──────────────────────────────────────────────────────────────
    os.makedirs('data/splits', exist_ok=True)
    train_df.to_csv(CUSTOM_TRAIN_CSV, index=False)
    val_df.to_csv(CUSTOM_VAL_CSV,     index=False)
    test_df.to_csv(CUSTOM_TEST_CSV,   index=False)

    def _report(name, split_df):
        n    = len(split_df)
        eng  = int(split_df['label'].sum())
        ne   = n - eng
        clips = split_df['clip_id'].nunique()
        print(f"\n  {name}:")
        print(f"    Clips   : {clips:>4}  |  Frames: {n:>5}")
        print(f"    Engaged : {eng:>4}  ({100*eng/max(n,1):.1f}%)")
        print(f"    NE      : {ne:>4}  ({100*ne/max(n,1):.1f}%)")

    print("\n  Frame-level statistics:")
    _report("Train", train_df)
    _report("Val",   val_df)
    _report("Test",  test_df)

    print(f"\n  ✅  Custom splits saved (ZERO leakage verified):")
    print(f"      {CUSTOM_TRAIN_CSV}")
    print(f"      {CUSTOM_VAL_CSV}")
    print(f"      {CUSTOM_TEST_CSV}")

    # ── Optional DAiSEE merge ──────────────────────────────────────────────────
    if merge_daisee:
        _merge_with_daisee(train_df, val_df, test_df)
    else:
        print(f"\n  TIP: Use  --merge-daisee  to combine with your DAiSEE splits.")
        print(f"  Training command:")
        print(f"    In train_clip.py or config.py, set:")
        print(f"      train_csv = '{CUSTOM_TRAIN_CSV}'")
        print(f"      val_csv   = '{CUSTOM_VAL_CSV}'")

    print(f"\n{'='*65}")
    return train_df, val_df, test_df


# ──────────────────────────────────────────────────────────────────────────────
# DAiSEE merge
# ──────────────────────────────────────────────────────────────────────────────

def _merge_with_daisee(
    custom_train : pd.DataFrame,
    custom_val   : pd.DataFrame,
    custom_test  : pd.DataFrame,
):
    """
    Concatenate custom splits with pre-existing DAiSEE splits.

    The merged CSVs can be passed directly to train_clip.py.
    clip_ids are unique by construction (DAiSEE uses numeric stems;
    custom uses f'{video_id}_{track_id}_C{n}').
    """
    print(f"\n  Merging with DAiSEE splits...")

    for name, custom_df, daisee_path, merged_path in [
        ('train', custom_train, CONFIG['train_csv'], MERGED_TRAIN_CSV),
        ('val',   custom_val,   CONFIG['val_csv'],   MERGED_VAL_CSV),
        ('test',  custom_test,  CONFIG['test_csv'],  MERGED_TEST_CSV),
    ]:
        if Path(daisee_path).exists():
            daisee_df  = pd.read_csv(daisee_path)
            # Ensure same schema
            for col in ['image_path', 'label', 'clip_id', 'person_id']:
                if col not in daisee_df.columns:
                    daisee_df[col] = ''
            merged = pd.concat(
                [daisee_df[['image_path','label','clip_id','person_id']],
                 custom_df[['image_path','label','clip_id','person_id']]],
                ignore_index=True,
            )
            if name == 'train':
                merged = merged.sample(frac=1, random_state=42).reset_index(drop=True)
            d_n = len(daisee_df); c_n = len(custom_df)
            print(f"    {name.capitalize():5} : DAiSEE {d_n:>7,} + Custom {c_n:>5} = "
                  f"{len(merged):>8,} frames")
        else:
            print(f"    ⚠️  DAiSEE {name} CSV not found at {daisee_path}")
            print(f"        Using custom data only for {name}.")
            merged = custom_df

        merged.to_csv(merged_path, index=False)

    print(f"\n  ✅  Merged splits saved:")
    print(f"      {MERGED_TRAIN_CSV}")
    print(f"      {MERGED_VAL_CSV}")
    print(f"      {MERGED_TEST_CSV}")
    print(f"\n  To train on merged data, set in config.py or train_clip.py:")
    print(f"      train_csv = '{MERGED_TRAIN_CSV}'")
    print(f"      val_csv   = '{MERGED_VAL_CSV}'")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Build train/val/test CSVs from annotated custom dataset'
    )
    parser.add_argument(
        '--merge-daisee', action='store_true',
        help='Also create merged CSVs combining DAiSEE + custom data'
    )
    parser.add_argument('--val-frac',  type=float, default=0.15,
                        help='Fraction of videos for validation (default: 0.15)')
    parser.add_argument('--test-frac', type=float, default=0.15,
                        help='Fraction of videos for test (default: 0.15)')
    parser.add_argument('--seed',      type=int,   default=42,
                        help='Random seed (default: 42)')
    args = parser.parse_args()

    build_splits(
        val_frac     = args.val_frac,
        test_frac    = args.test_frac,
        seed         = args.seed,
        merge_daisee = args.merge_daisee,
    )
