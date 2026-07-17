"""
Maps DAiSEE label CSVs to extracted frame paths.

Output CSV columns:
  image_path  — absolute/relative path to the .jpg frame
  label       — 0 (Not Engaged) or 1 (Engaged)
  clip_id     — e.g. "1100010100"  (video clip identifier)
  person_id   — e.g. "110001"      (first 6 chars of clip_id = student ID)
  split       — "Train" | "Validation" | "Test"

Why person_id matters:
  The first 6 characters of every DAiSEE ClipID uniquely identify a student.
  If person_id is not tracked, build_splits.py cannot do person-level splitting,
  which is the ROOT CAUSE of data leakage (same student in train AND val).

Engagement mapping (DAiSEE):
  Score 0 (very low) → Not Engaged
  Score 1 (low)      → Not Engaged
  Score 2 (nominal)  → Engaged
  Score 3 (high)     → Engaged
  Threshold = 2  (configurable via CONFIG['engagement_threshold'])
"""

import os
import pandas as pd
from tqdm import tqdm
import sys

sys.path.append('.')
from src.config import CONFIG

LABELS_FOLDER = os.path.join(CONFIG['raw_daisee_dir'], 'Labels')
FRAMES_DIR    = CONFIG['frames_dir']
OUTPUT_CSV    = CONFIG['labels_csv']
THRESHOLD     = CONFIG.get('engagement_threshold', 2)

SPLIT_MAP = {
    'TrainLabels.csv'      : 'Train',
    'ValidationLabels.csv' : 'Validation',
    'TestLabels.csv'       : 'Test',
}


def build_labels():
    all_data = []

    for label_file, split in SPLIT_MAP.items():
        label_path = os.path.join(LABELS_FOLDER, label_file)
        if not os.path.isfile(label_path):
            print(f"WARNING: {label_path} not found — skipping {split}")
            continue

        df = pd.read_csv(label_path)
        df['binary_label'] = (df['Engagement'] >= THRESHOLD).astype(int)

        split_frames_dir = os.path.join(FRAMES_DIR, split)
        print(f"\nProcessing {split} ({len(df):,} clips)...")

        found_clips   = 0
        missing_clips = 0

        for _, row in tqdm(df.iterrows(), total=len(df), desc=split):
            raw_clip  = row['ClipID'].strip()
            # clip_id   = raw_clip.replace('.avi', '')
            clip_id = os.path.splitext(raw_clip)[0]
            label     = int(row['binary_label'])
            person_id = clip_id[:6]   # First 6 chars = student ID in DAiSEE

            clip_dir = os.path.join(split_frames_dir, clip_id, clip_id)

            if not os.path.isdir(clip_dir):
                missing_clips += 1
                continue

            frames = sorted(
                f for f in os.listdir(clip_dir) if f.endswith('.jpg')
            )

            if not frames:
                missing_clips += 1
                continue

            found_clips += 1
            for frame in frames:
                all_data.append({
                    'image_path' : os.path.join(clip_dir, frame),
                    'label'      : label,
                    'clip_id'    : clip_id,
                    'person_id'  : person_id,
                    'split'      : split,
                })

        print(f"  Clips found   : {found_clips:,}")
        print(f"  Clips missing : {missing_clips:,} "
              f"(run frame_extractor.py first if large)")

    if not all_data:
        raise RuntimeError(
            "No data found. Make sure frame_extractor.py has been run "
            "and frames exist under data/processed/daisee/frames/."
        )

    df_out = pd.DataFrame(all_data)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df_out.to_csv(OUTPUT_CSV, index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"✅  labels.csv saved  →  {OUTPUT_CSV}")
    print(f"{'='*55}")
    print(f"  Total frames   : {len(df_out):>10,}")
    print(f"  Unique clips   : {df_out['clip_id'].nunique():>10,}")
    print(f"  Unique persons : {df_out['person_id'].nunique():>10,}")
    print(f"\n  Label distribution:")
    vc = df_out['label'].value_counts().sort_index()
    for lbl, cnt in vc.items():
        name = 'Engaged' if lbl == 1 else 'Not Engaged'
        print(f"    {name:12s}: {cnt:>8,}  ({100*cnt/len(df_out):.1f}%)")
    print(f"{'='*55}")


if __name__ == '__main__':
    build_labels()
