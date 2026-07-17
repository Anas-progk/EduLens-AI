"""
eduaction_processor.py
Extract frames from EduAction short video clips and produce a
frame-level CSV that ClipDataset can read directly.

Why this is needed
──────────────────
The original merge approach stored raw .mp4 video paths in the
image_path column.  ClipDataset groups rows by clip_id and then
opens each image_path with PIL.Image.open() — a path to a video
file produces an error and falls back to a black frame.  Since each
EduAction clip had only one row in the CSV, min_frames=4 dropped it
entirely.  Zero EduAction clips were actually used during training.

This script fixes that by:
  1. Uniformly sampling N_FRAMES=8 frames from every clip.
  2. Saving each frame as a JPEG in custom_dataset/processed_edu/.
  3. Writing a frame-level CSV (image_path, label, clip_id, person_id)
     that can be concatenated with custom_train.csv.

Label mapping (folder-based — no manual annotation needed)
──────────────────────────────────────────────────────────
  EduAction_E/   →  label=1 (Engaged):   writing_*.mp4, lecture_*.mp4, work_*.mp4
  EduAction_NE/  →  label=0 (Not Engaged): sleep_*.mp4, talk_*.mp4, phone_*.mp4

Clip naming convention (no collision with custom clips)
───────────────────────────────────────────────────────
  clip_id   = "EDU_E_{stem}"  or  "EDU_NE_{stem}"   e.g. "EDU_E_writing_1"
  person_id = "EDUACTION_E"   or  "EDUACTION_NE"

Usage
─────
  # Basic: process and save CSV only (no merge)
  python src/data/eduaction_processor.py

  # Process and immediately merge into train split
  python src/data/eduaction_processor.py --merge-train

  # Custom paths / frame count
  python src/data/eduaction_processor.py --n-frames 8 --quality 90 --merge-train

Output
──────
  custom_dataset/eduaction_frames.csv   -- standalone frame-level CSV
  custom_dataset/processed_edu/         -- extracted JPEG frames
  data/splits/merged_train.csv          -- (only with --merge-train)
"""

import os
import sys
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import CONFIG


# ─────────────────────────────────────────────────────────────────
# Paths  (resolved from CONFIG so they stay consistent with the rest
#         of the project; fallback strings match the agreed layout)
# ─────────────────────────────────────────────────────────────────

EDU_E_DIR       = Path(CONFIG.get('edu_e_dir',   'custom_dataset/EduAction_E'))
EDU_NE_DIR      = Path(CONFIG.get('edu_ne_dir',  'custom_dataset/EduAction_NE'))
EDU_PROC_DIR    = Path(CONFIG.get('edu_proc_dir','custom_dataset/processed_edu'))
EDU_FRAMES_CSV  = Path(CONFIG.get('edu_frames_csv',
                                  'custom_dataset/eduaction_frames.csv'))
CUSTOM_TRAIN_CSV = Path(CONFIG.get('custom_train_csv', 'data/splits/custom_train.csv'))
MERGED_TRAIN_CSV = Path(CONFIG.get('merged_train_csv', 'data/splits/merged_train.csv'))

VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.MP4', '.AVI', '.MOV'}


# ─────────────────────────────────────────────────────────────────
# Core frame extractor
# ─────────────────────────────────────────────────────────────────

def extract_frames(
    video_path : Path,
    out_dir    : Path,
    n_frames   : int = 8,
    quality    : int = 90,
) -> list:
    """
    Uniformly sample `n_frames` frames from a video and save as JPEGs.

    Returns a list of posix-format absolute path strings (forward slashes
    so the CSV works identically on Windows and Linux/Colab).

    If the video has fewer frames than n_frames, frames are repeated
    (same padding strategy as ClipDataset._sample_frames).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"    ⚠  Cannot open: {video_path.name}")
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        print(f"    ⚠  Zero frames: {video_path.name}")
        return []

    # Uniform indices (with repetition if clip is shorter than n_frames)
    if total >= n_frames:
        indices = np.linspace(0, total - 1, n_frames, dtype=int)
    else:
        indices = list(range(total))
        while len(indices) < n_frames:
            indices.append(indices[-1])
        indices = np.array(indices)

    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    for seq_idx, frame_idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ret, frame = cap.read()
        if not ret:
            # Fall back to previous saved frame on read failure
            if saved:
                saved.append(saved[-1])
            continue

        fname = out_dir / f"frame_{seq_idx:04d}.jpg"
        cv2.imwrite(str(fname), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        # Store as posix path for cross-platform CSV compatibility
        saved.append(fname.as_posix())

    cap.release()

    # Pad to exactly n_frames if any reads failed mid-video
    while len(saved) < n_frames and saved:
        saved.append(saved[-1])

    return saved


# ─────────────────────────────────────────────────────────────────
# Process one folder (E or NE)
# ─────────────────────────────────────────────────────────────────

def process_folder(
    folder    : Path,
    label     : int,
    label_str : str,   # "E" or "NE"
    proc_root : Path,
    n_frames  : int,
    quality   : int,
) -> list:
    """
    Process every video in `folder`.

    Returns a list of row-dicts with keys:
        image_path, label, clip_id, person_id
    (one row per extracted frame — ClipDataset reads frame-level rows)
    """
    if not folder.exists():
        print(f"  ⚠  Folder not found: {folder}  (skipping)")
        return []

    videos = sorted(p for p in folder.iterdir() if p.suffix in VIDEO_EXTS)
    if not videos:
        print(f"  ⚠  No videos in: {folder}")
        return []

    rows = []
    ok = 0
    for vid in videos:
        stem     = vid.stem                                  # e.g. "writing_1"
        clip_id  = f"EDU_{label_str}_{stem}"               # e.g. "EDU_E_writing_1"
        person_id = f"EDUACTION_{label_str}"               # e.g. "EDUACTION_E"
        out_dir  = proc_root / f"EDU_{label_str}" / stem   # processed_edu/EDU_E/writing_1/

        frame_paths = extract_frames(vid, out_dir, n_frames=n_frames, quality=quality)
        if not frame_paths:
            continue

        for fpath in frame_paths:
            rows.append({
                'image_path' : fpath,
                'label'      : label,
                'clip_id'    : clip_id,
                'person_id'  : person_id,
            })
        ok += 1

    print(f"  {'Engaged' if label==1 else 'Not Engaged':12s} : {ok:>3}/{len(videos)} clips processed"
          f"  →  {ok * n_frames:>4} frame rows  (source: {folder.name})")
    return rows


# ─────────────────────────────────────────────────────────────────
# Leakage guard
# ─────────────────────────────────────────────────────────────────

def _check_no_leakage(edu_df: pd.DataFrame, train_csv: Path):
    """
    Confirm no EDU clip_ids overlap with val/test CSVs.
    (EDU clips should go to train only — they're supplementary diversity,
    not the target evaluation domain.)
    """
    edu_clips = set(edu_df['clip_id'])
    val_csv   = Path(str(train_csv).replace('train', 'val'))
    test_csv  = Path(str(train_csv).replace('train', 'test'))

    for split_name, split_csv in [('val', val_csv), ('test', test_csv)]:
        if split_csv.exists():
            split_clips = set(pd.read_csv(split_csv)['clip_id'])
            overlap = edu_clips & split_clips
            assert len(overlap) == 0, (
                f"LEAKAGE: {len(overlap)} EduAction clip_ids appear in {split_name}!"
            )


# ─────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────

def process_eduaction(
    n_frames    : int  = 8,
    quality     : int  = 90,
    merge_train : bool = False,
):
    print(f"\n{'='*65}")
    print(f"  EDUACTION FRAME EXTRACTOR")
    print(f"  E  folder : {EDU_E_DIR}")
    print(f"  NE folder : {EDU_NE_DIR}")
    print(f"  Output    : {EDU_PROC_DIR}")
    print(f"  Frames/clip: {n_frames}   JPEG quality: {quality}")
    print(f"{'='*65}\n")

    all_rows = []

    # ── Engaged clips ──────────────────────────────────────────────
    all_rows += process_folder(
        folder=EDU_E_DIR, label=1, label_str='E',
        proc_root=EDU_PROC_DIR, n_frames=n_frames, quality=quality,
    )

    # ── Not Engaged clips ──────────────────────────────────────────
    all_rows += process_folder(
        folder=EDU_NE_DIR, label=0, label_str='NE',
        proc_root=EDU_PROC_DIR, n_frames=n_frames, quality=quality,
    )

    if not all_rows:
        print("\n❌  No frames extracted.  Check that the EduAction folders exist")
        print(f"    and contain .mp4 files:\n    {EDU_E_DIR}\n    {EDU_NE_DIR}")
        return None

    edu_df = pd.DataFrame(all_rows)

    n_clips = edu_df['clip_id'].nunique()
    n_e     = edu_df[edu_df['label'] == 1]['clip_id'].nunique()
    n_ne    = edu_df[edu_df['label'] == 0]['clip_id'].nunique()

    print(f"\n  Summary:")
    print(f"    Total clips   : {n_clips}  ({n_e} Engaged  +  {n_ne} Not Engaged)")
    print(f"    Total frames  : {len(edu_df)}")
    print(f"    Frames/clip   : {n_frames}")

    # ── Save standalone CSV ───────────────────────────────────────
    EDU_FRAMES_CSV.parent.mkdir(parents=True, exist_ok=True)
    edu_df.to_csv(EDU_FRAMES_CSV, index=False)
    print(f"\n  ✅  EduAction frame CSV saved:\n      {EDU_FRAMES_CSV}")

    # ── Optional merge into train split ───────────────────────────
    if merge_train:
        _merge_into_train(edu_df)

    print(f"\n{'='*65}\n")
    return edu_df


def _merge_into_train(edu_df: pd.DataFrame):
    """
    Concatenate EduAction frame rows with the existing custom train CSV.

    Design decisions:
      • EduAction clips go to TRAIN only — val/test stay pure classroom.
        This ensures evaluation reflects real deployment conditions.
      • A leakage guard confirms clip_ids are disjoint from val/test.
      • The merged file is saved to merged_train_csv (not overwriting
        the custom_train.csv baseline so you can always revert).
    """
    if not CUSTOM_TRAIN_CSV.exists():
        print(f"\n  ⚠  custom_train.csv not found at {CUSTOM_TRAIN_CSV}")
        print(f"      Run  python src/data/build_custom_splits.py  first.")
        return

    # Leakage guard
    try:
        _check_no_leakage(edu_df, CUSTOM_TRAIN_CSV)
    except AssertionError as e:
        print(f"\n  ❌  {e}")
        return

    custom_df = pd.read_csv(CUSTOM_TRAIN_CSV)

    # Ensure schema match
    for col in ['image_path', 'label', 'clip_id', 'person_id']:
        if col not in custom_df.columns:
            custom_df[col] = ''

    merged = pd.concat(
        [custom_df[['image_path', 'label', 'clip_id', 'person_id']],
         edu_df[['image_path', 'label', 'clip_id', 'person_id']]],
        ignore_index=True,
    ).sample(frac=1, random_state=42).reset_index(drop=True)

    c_clips  = custom_df['clip_id'].nunique()
    e_clips  = edu_df[edu_df['label'] == 1]['clip_id'].nunique()
    ne_clips = edu_df[edu_df['label'] == 0]['clip_id'].nunique()

    MERGED_TRAIN_CSV.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(MERGED_TRAIN_CSV, index=False)

    print(f"\n  Merged train split:")
    print(f"    Custom clips   : {c_clips:>4}  ({len(custom_df):>6} frame rows)")
    print(f"    EduAction E    : {e_clips:>4}  ({e_clips * 8:>6} frame rows)")
    print(f"    EduAction NE   : {ne_clips:>4}  ({ne_clips * 8:>6} frame rows)")
    print(f"    ─────────────────────────────────────────────────")
    print(f"    TOTAL clips    : {c_clips + e_clips + ne_clips:>4}  ({len(merged):>6} frame rows)")
    merged_e  = int(merged['label'].sum())
    merged_ne = len(merged) - merged_e
    print(f"    Engaged frames : {merged_e:>6}  ({100*merged_e/len(merged):.1f}%)")
    print(f"    NE frames      : {merged_ne:>6}  ({100*merged_ne/len(merged):.1f}%)")
    print(f"\n  ✅  Merged train CSV saved  (ZERO leakage verified):")
    print(f"      {MERGED_TRAIN_CSV}")
    print(f"\n  To train on merged data, ensure config.py has:")
    print(f"      'train_csv': 'data/splits/merged_train.csv'")


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract EduAction frames → frame-level CSV for ClipDataset'
    )
    parser.add_argument(
        '--merge-train', action='store_true',
        help='After extracting, merge EduAction frames into merged_train.csv'
    )
    parser.add_argument(
        '--n-frames', type=int, default=8,
        help='Frames to sample per clip (default: 8, matches ClipDataset n_frames)'
    )
    parser.add_argument(
        '--quality', type=int, default=90,
        help='JPEG quality 1-100 (default: 90)'
    )
    args = parser.parse_args()

    process_eduaction(
        n_frames    = args.n_frames,
        quality     = args.quality,
        merge_train = args.merge_train,
    )
