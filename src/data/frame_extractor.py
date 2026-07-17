"""
Frame extractor for DAiSEE videos.

Input structure:
    data/raw/daisee/{Split}/{PersonID}/{ClipID}.avi

Output structure:
    data/processed/daisee/frames/{Split}/{PersonID}/{ClipID}/frame_{n:04d}.jpg

Key design decisions:
  - FRAME_STRIDE: extract every Nth frame instead of all frames.
      • Reduces near-duplicate consecutive frames (same face, nearly identical).
      • Prevents a single 30-sec clip from dominating the dataset.
      • Keeps total dataset size manageable for Colab.
  - MAX_FRAMES: hard cap per clip — gives every clip equal representation.
  - Skips clips that are already fully extracted (safe to re-run).
"""

import os
import cv2
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.append('.')
from src.config import CONFIG

RAW_DIR      = CONFIG['raw_daisee_dir']
OUT_DIR      = CONFIG['frames_dir']
FRAME_STRIDE = CONFIG.get('frame_stride', 15)
MAX_FRAMES   = CONFIG.get('max_frames', 8)


def extract_clip(video_path: str, out_dir: str) -> int:
    """
    Extract up to MAX_FRAMES frames from a single video clip,
    sampled every FRAME_STRIDE frames.
    Returns the number of frames actually saved.
    """
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  WARNING: Cannot open {video_path}")
        return 0

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    saved = 0

    for step in range(MAX_FRAMES):
        target_frame = step * FRAME_STRIDE
        if target_frame >= total_video_frames:
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        if not ret:
            break

        out_path = os.path.join(out_dir, f"frame_{step:04d}.jpg")
        cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved += 1

    cap.release()
    return saved


def extract_all():
    splits = ['Train', 'Validation', 'Test']

    for split in splits:
        split_in_dir  = os.path.join(RAW_DIR, split)
        split_out_dir = os.path.join(OUT_DIR,  split)

        if not os.path.isdir(split_in_dir):
            print(f"Skipping {split} — not found at: {split_in_dir}")
            continue

        # video_paths = sorted(Path(split_in_dir).glob('**/*.avi'))
        video_paths = sorted(
            list(Path(split_in_dir).glob('**/*.avi')) +
            list(Path(split_in_dir).glob('**/*.mp4'))
        )
        print(f"\n{split}: {len(video_paths):,} videos  "
              f"(stride={FRAME_STRIDE}, max_frames={MAX_FRAMES})")

        total_saved  = 0
        total_skip   = 0

        for vp in tqdm(video_paths, desc=split):
            person_id = vp.parent.name   # e.g. "110001"
            clip_id   = vp.stem          # e.g. "1100010100"

            out_dir = os.path.join(split_out_dir, person_id, clip_id)

            # Skip if already fully extracted
            if (os.path.isdir(out_dir) and
                    len([f for f in os.listdir(out_dir) if f.endswith('.jpg')])
                    >= MAX_FRAMES):
                total_skip += 1
                continue

            n = extract_clip(str(vp), out_dir)
            total_saved += n

        print(f"  New frames saved : {total_saved:,}")
        print(f"  Clips skipped    : {total_skip:,} (already done)")

    print("\n✅ Frame extraction complete")


if __name__ == '__main__':
    extract_all()