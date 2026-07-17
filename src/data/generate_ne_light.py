"""
generate_ne_light.py -- Safe, minimal synthetic NE clip generation.

PURPOSE
───────
Adds a small number (≤ ~30-60) of lightly-augmented NE clip copies to the
training set. This supplements the hardest NE behaviour categories (phone use
and sleeping) without teaching the model that image distortions = Not Engaged.

WHY ONLY PHONE + SLEEP?
───────────────────────
- Phone: very distinct NE pattern (downward gaze, hand near face). Hard to spot
  from 1 frame but very clear over 8 frames. High-value hard negative.
- Sleep: slumped posture + closed eyes over time. Also a clear temporal pattern.
- Talk: excluded because "talking to a neighbour" looks very similar to engaged
  discussion. Augmenting this could confuse the boundary.

WHY ONLY 1 AUG PER CLIP?
─────────────────────────
Aggressive augmentation (run 5, 934 synthetic rows, 3 augs per clip) HURT
performance: NE_F1 dropped from 0.778 → 0.628. The model learned distortion
artefacts instead of behaviour. This script limits to 1 aug per original clip
with very mild transforms, so the model sees slightly more variation in NE
poses/lighting without hallucinating new NE "signatures".

AUGMENTATIONS USED (by design):
  - RandomBrightnessContrast ±8%: simulates natural lighting variation
  - MotionBlur (kernel=3, p=0.15): very mild camera shake
  - RandomResizedCrop (scale 0.93-1.0): tiny zoom, nothing cropped out

AUGMENTATIONS INTENTIONALLY AVOIDED:
  - CoarseDropout / GaussNoise: teach model artefacts = NE
  - Rotation: classroom cameras are fixed, rotation is unphysical
  - Strong crop: can remove the face/posture that defines NE behaviour
  - Perspective distortion: unphysical for seated classroom setting

USAGE (Colab):
  Run BEFORE train_clip.py. Output CSV path printed at end.

  python src/data/generate_ne_light.py [--dry-run]

  --dry-run: print counts only, no files written.

OUTPUT:
  custom_dataset/generated_ne_light/  (augmented frame folders)
  data/splits/generated_ne_light.csv  (frame-level rows for ClipDataset)
  data/splits/merged_train_light.csv  (merged_train.csv + light synthetic)
"""

import os
import sys
import argparse
import random
import numpy as np
import pandas as pd
import cv2

# Guard albumentations import (may not be installed locally, only Colab)
try:
    import albumentations as A
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False
    print("[WARNING] albumentations not found. Install with: pip install albumentations -q")

# ─── Paths ────────────────────────────────────────────────────────────────────

EDU_NE_DIR   = "custom_dataset/processed_edu/EDU_NE"     # processed EduAction NE frames
OUTPUT_DIR   = "custom_dataset/generated_ne_light/GEN_NE_LIGHT"
GEN_CSV_OUT  = "data/splits/generated_ne_light.csv"
MERGED_TRAIN = "data/splits/merged_train.csv"             # base training set
MERGED_OUT   = "data/splits/merged_train_light.csv"       # final output


# ─── Category filter ──────────────────────────────────────────────────────────

# Only augment these EduAction NE categories (by folder prefix)
ALLOWED_PREFIXES = ("phone", "sleep")


# ─── Light augmentation pipeline ──────────────────────────────────────────────

def build_light_transform():
    """
    Very mild augmentation that models realistic variation:
      - Small brightness/contrast shifts (±8%): lighting condition changes
      - Rare MotionBlur (15% prob): occasional slight camera movement
      - Very tiny crop (93-100% of original): negligible zoom only

    Returns an albumentations Compose object, or None if albumentations missing.
    """
    if not HAS_ALBUMENTATIONS:
        return None

    return A.Compose([
        A.RandomBrightnessContrast(
            brightness_limit=0.08,
            contrast_limit  =0.08,
            p=0.5,
        ),
        A.MotionBlur(
            blur_limit=3,
            p=0.15,
        ),
        A.RandomResizedCrop(
            size  =(224, 224),
            scale =(0.93, 1.0),
            ratio =(0.99, 1.01),
            p=0.3,
        ),
    ])


# ─── Core generation logic ────────────────────────────────────────────────────

def generate_ne_light(dry_run: bool = False) -> int:
    """
    Generate 1 lightly-augmented copy per eligible EduAction NE clip.

    Args:
        dry_run: if True, count only — no files written.

    Returns:
        n_generated: number of synthetic clips created.
    """
    if not os.path.isdir(EDU_NE_DIR):
        print(f"[ERROR] EDU_NE_DIR not found: {EDU_NE_DIR}")
        print("  Make sure you ran: python src/data/eduaction_processor.py")
        return 0

    transform = build_light_transform()
    if transform is None and not dry_run:
        print("[ERROR] albumentations required. pip install albumentations -q")
        return 0

    # Collect eligible clip folders
    all_clips = sorted(os.listdir(EDU_NE_DIR))
    eligible  = [
        c for c in all_clips
        if os.path.isdir(os.path.join(EDU_NE_DIR, c))
        and any(c.startswith(pfx) for pfx in ALLOWED_PREFIXES)
    ]

    print(f"\n{'='*60}")
    print(f"  Light NE Generation")
    print(f"  Source dir    : {EDU_NE_DIR}")
    print(f"  Allowed types : {ALLOWED_PREFIXES}")
    print(f"  Eligible clips: {len(eligible)}")
    print(f"  Augs per clip : 1")
    print(f"  Dry run       : {dry_run}")
    print(f"{'='*60}")

    if len(eligible) == 0:
        print("[WARNING] No eligible clips found. Check folder names in EDU_NE_DIR.")
        return 0

    if dry_run:
        print(f"  Would generate: {len(eligible)} synthetic clips")
        print(f"  Output frames : {len(eligible) * 8}")
        return len(eligible)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    rows       = []
    n_generated = 0

    for clip_name in eligible:
        clip_path = os.path.join(EDU_NE_DIR, clip_name)
        frame_files = sorted([
            f for f in os.listdir(clip_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        if len(frame_files) == 0:
            print(f"  [SKIP] {clip_name} — no frames found")
            continue

        # One augmented copy: seed from clip name for reproducibility
        aug_clip_name = f"{clip_name}_light_0"
        aug_clip_path = os.path.join(OUTPUT_DIR, aug_clip_name)
        os.makedirs(aug_clip_path, exist_ok=True)

        # Temporal consistency: all frames in this clip share the same seed
        seed = hash(clip_name) % 1_000_000

        for frame_file in frame_files:
            src_path = os.path.join(clip_path, frame_file)
            img      = cv2.imread(src_path)

            if img is None:
                # Write a black frame so clip length stays consistent
                img = np.zeros((224, 224, 3), dtype=np.uint8)

            # Seed-based random state so spatial transforms are clip-consistent
            random.seed(seed + frame_files.index(frame_file))
            np.random.seed(seed + frame_files.index(frame_file))

            augmented = transform(image=img)
            aug_img   = augmented["image"]

            dst_path = os.path.join(aug_clip_path, frame_file)
            cv2.imwrite(dst_path, aug_img, [cv2.IMWRITE_JPEG_QUALITY, 92])

            rows.append({
                "image_path": dst_path.replace("\\", "/"),
                "label"     : 0,                          # NE
                "clip_id"   : f"GEN_LIGHT_{aug_clip_name}",
                "person_id" : f"GEN_LIGHT_{aug_clip_name}",
            })

        n_generated += 1
        if n_generated % 5 == 0:
            print(f"  Generated {n_generated}/{len(eligible)} clips...")

    print(f"\n  Done: {n_generated} synthetic NE clips generated.")

    # Save generated CSV
    gen_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(GEN_CSV_OUT), exist_ok=True)
    gen_df.to_csv(GEN_CSV_OUT, index=False)
    print(f"  Saved: {GEN_CSV_OUT}  ({len(gen_df)} frame rows)")

    # Merge with base training CSV
    if os.path.isfile(MERGED_TRAIN):
        base_df    = pd.read_csv(MERGED_TRAIN)
        merged_df  = pd.concat([base_df, gen_df], ignore_index=True)
        merged_df.to_csv(MERGED_OUT, index=False)

        base_ne  = (base_df["label"] == 0).sum()
        base_e   = (base_df["label"] == 1).sum()
        total_ne = (merged_df["label"] == 0).sum()
        total_e  = (merged_df["label"] == 1).sum()

        print(f"\n  Merge summary:")
        print(f"    Base  — NE frames: {base_ne:>5}  E frames: {base_e:>5}")
        print(f"    Added — NE frames: {len(gen_df):>5}")
        print(f"    Final — NE frames: {total_ne:>5}  E frames: {total_e:>5}")
        print(f"  Saved: {MERGED_OUT}")
        print(f"\n  To use this in training, update config.py:")
        print(f"    'train_csv': '{MERGED_OUT}',")
    else:
        print(f"  [INFO] Base CSV not found at {MERGED_TRAIN}; skipping merge.")

    return n_generated


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate light synthetic NE clips from EduAction phone+sleep data."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count eligible clips without writing any files.",
    )
    args = parser.parse_args()

    n = generate_ne_light(dry_run=args.dry_run)

    if not args.dry_run and n > 0:
        print(f"\n{'='*60}")
        print(f"  NEXT STEPS:")
        print(f"  1. Check the generated clips in: {OUTPUT_DIR}")
        print(f"  2. To train WITH light synthetic NE, update src/config.py:")
        print(f"       'train_csv': '{MERGED_OUT}',")
        print(f"  3. Run: python -m src.training.train_clip")
        print(f"\n  RECOMMENDATION: First train on original merged_train.csv")
        print(f"  to get your baseline. Then try merged_train_light.csv if you")
        print(f"  want to see if the extra clips help.")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
