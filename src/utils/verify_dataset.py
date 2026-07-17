"""
Dataset split verification — run this AFTER build_splits.py, BEFORE training.

Checks performed:
  [1] Frame path overlap between splits (direct leakage)
  [2] Person ID overlap between splits  (identity leakage — most critical)
  [3] Clip ID overlap between splits
  [4] Class distribution in each split
  [5] Split size sanity (percentages)
  [6] Missing image files (spot-check)

Exit code: 0 if all checks pass, 1 if any leakage is detected.
"""

import os
import sys
import random
import pandas as pd

sys.path.append('.')
from src.config import CONFIG

SEP = '=' * 60


def _overlap_report(name_a, set_a, name_b, set_b, noun):
    overlap = set_a & set_b
    ok      = len(overlap) == 0
    status  = "✅" if ok else "❌  LEAKAGE!"
    print(f"  {name_a} ∩ {name_b} : {len(overlap):>5} {noun}  {status}")
    return ok


def verify_splits(
    train_csv : str = CONFIG['train_csv'],
    val_csv   : str = CONFIG['val_csv'],
    test_csv  : str = CONFIG['test_csv'],
    spot_check: int = 20,          # number of random frames to check for existence
) -> bool:

    print(SEP)
    print("  DATASET SPLIT VERIFICATION")
    print(SEP)

    # ── Load ─────────────────────────────────────────────────────────────────
    for path in (train_csv, val_csv, test_csv):
        if not os.path.isfile(path):
            print(f"❌  Split file not found: {path}")
            print("    Run build_splits.py first.")
            sys.exit(1)

    train = pd.read_csv(train_csv)
    val   = pd.read_csv(val_csv)
    test  = pd.read_csv(test_csv)

    all_passed = True

    # ── [1] Frame path overlap ────────────────────────────────────────────────
    print("\n[1] Frame path overlap:")
    t_paths = set(train['image_path'])
    v_paths = set(val['image_path'])
    e_paths = set(test['image_path'])

    all_passed &= _overlap_report("Train", t_paths, "Val",  v_paths, "frames")
    all_passed &= _overlap_report("Train", t_paths, "Test", e_paths, "frames")
    all_passed &= _overlap_report("Val",   v_paths, "Test", e_paths, "frames")

    # ── [2] Person ID overlap ─────────────────────────────────────────────────
    print("\n[2] Person ID overlap (most critical):")
    if 'person_id' in train.columns:
        t_persons = set(train['person_id'])
        v_persons = set(val['person_id'])
        e_persons = set(test['person_id'])

        all_passed &= _overlap_report("Train", t_persons, "Val",  v_persons, "persons")
        all_passed &= _overlap_report("Train", t_persons, "Test", e_persons, "persons")
        all_passed &= _overlap_report("Val",   v_persons, "Test", e_persons, "persons")
    else:
        print("  SKIPPED — no 'person_id' column found.")
        print("  Re-run label_mapper.py and build_splits.py to add it.")
        all_passed = False

    # ── [3] Clip ID overlap ───────────────────────────────────────────────────
    print("\n[3] Clip ID overlap:")
    if 'clip_id' in train.columns:
        t_clips = set(train['clip_id'])
        v_clips = set(val['clip_id'])
        e_clips = set(test['clip_id'])

        all_passed &= _overlap_report("Train", t_clips, "Val",  v_clips, "clips")
        all_passed &= _overlap_report("Train", t_clips, "Test", e_clips, "clips")
        all_passed &= _overlap_report("Val",   v_clips, "Test", e_clips, "clips")
    else:
        print("  SKIPPED — no 'clip_id' column found.")

    # ── [4] Class distribution ────────────────────────────────────────────────
    print("\n[4] Class distribution:")
    header = f"  {'Split':6s}  {'Frames':>9}  {'Engaged':>8}  {'%':>6}  {'Not Eng':>8}  {'%':>6}"
    print(header)
    print("  " + "─" * (len(header) - 2))

    for name, df in [("Train", train), ("Val", val), ("Test", test)]:
        n    = len(df)
        eng  = int(df['label'].sum())
        neng = n - eng
        print(f"  {name:6s}  {n:>9,}  {eng:>8,}  {100*eng/n:>5.1f}%  "
              f"{neng:>8,}  {100*neng/n:>5.1f}%")

    # ── [5] Split size sanity ─────────────────────────────────────────────────
    print("\n[5] Split sizes:")
    total = len(train) + len(val) + len(test)
    for name, df in [("Train", train), ("Val", val), ("Test", test)]:
        pct = 100 * len(df) / total
        print(f"  {name:6s}: {len(df):>9,} frames  ({pct:.1f}%)")
    print(f"  {'TOTAL':6s}: {total:>9,} frames")

    # ── [6] Spot-check file existence ─────────────────────────────────────────
    print(f"\n[6] Spot-checking {spot_check} random frame paths ...")
    all_paths   = list(t_paths | v_paths | e_paths)
    sample      = random.sample(all_paths, min(spot_check, len(all_paths)))
    missing     = [p for p in sample if not os.path.isfile(p)]

    if missing:
        print(f"  ⚠️  {len(missing)}/{spot_check} sampled files are MISSING:")
        for m in missing[:5]:
            print(f"     {m}")
        if len(missing) > 5:
            print(f"     ... and {len(missing)-5} more")
    else:
        print(f"  ✅  All {spot_check} sampled files exist")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    if all_passed:
        print("  ✅  ALL CHECKS PASSED — No data leakage detected")
        print("      Safe to start training.")
    else:
        print("  ❌  LEAKAGE DETECTED — Fix before training!")
        print("      Re-run label_mapper.py → build_splits.py.")
    print(SEP)

    return all_passed


if __name__ == '__main__':
    ok = verify_splits()
    sys.exit(0 if ok else 1)
