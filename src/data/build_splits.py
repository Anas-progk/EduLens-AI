"""
Build train / val / test splits at PERSON level.

WHY PERSON-LEVEL (not clip-level, not frame-level)?
────────────────────────────────────────────────────
In DAiSEE the first 6 characters of every ClipID identify the student.
Frames from the same student share:
  • the same face
  • the same background / room lighting
  • the same camera angle
If the same student appears in BOTH train and val, the model can simply
learn to recognise them rather than learning engagement patterns →
validation F1 is inflated (the 0.92+ result you saw).

WHAT THIS SCRIPT DOES
──────────────────────
1. Reads labels.csv (must have person_id and clip_id columns).
2. Assigns one label per person (majority-vote over their clips).
3. Stratified split: 70 % train / 15 % val / 15 % test (person-level).
4. Maps persons back to their frames.
5. Saves train/val/test CSVs WITHOUT any resampling.
   → Class imbalance is handled in training via WeightedRandomSampler
     and class-weighted CrossEntropyLoss (no duplicate frames in CSV).
6. Asserts zero person overlap between every pair of splits.

IMPORTANT: Do NOT resample val/test. Their real-world class distribution
must be preserved so evaluation reflects true deployment performance.
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.append('.')
from src.config import CONFIG


def build_splits(
    labels_csv : str   = CONFIG['labels_csv'],
    train_out  : str   = CONFIG['train_csv'],
    val_out    : str   = CONFIG['val_csv'],
    test_out   : str   = CONFIG['test_csv'],
    val_frac   : float = 0.15,
    test_frac  : float = 0.15,
    seed       : int   = 42,
):
    # ── Load ─────────────────────────────────────────────────────────────────
    df = pd.read_csv(labels_csv)

    required = ['image_path', 'label', 'person_id', 'clip_id']
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns in labels.csv: {missing}\n"
            "Re-run label_mapper.py — the new version writes person_id and clip_id."
        )

    print(f"Loaded labels.csv")
    print(f"  Total frames   : {len(df):>10,}")
    print(f"  Unique clips   : {df['clip_id'].nunique():>10,}")
    print(f"  Unique persons : {df['person_id'].nunique():>10,}")

    # ── Person-level label (majority vote across all of their clips) ──────────
    person_df = (
        df.groupby('person_id')['label']
          .agg(lambda x: int(x.mode()[0]))
          .reset_index()
          .rename(columns={'label': 'person_label'})
    )
    n_persons = len(person_df)

    # ── Stratified person-level split ─────────────────────────────────────────
    # Step 1: carve out test
    train_val_p, test_p = train_test_split(
        person_df,
        test_size  = test_frac,
        stratify   = person_df['person_label'],
        random_state = seed,
    )

    # Step 2: carve out val from what's left
    adjusted_val = val_frac / (1.0 - test_frac)
    train_p, val_p = train_test_split(
        train_val_p,
        test_size  = adjusted_val,
        stratify   = train_val_p['person_label'],
        random_state = seed,
    )

    print(f"\nPerson-level split (N={n_persons}):")
    print(f"  Train : {len(train_p):>5} persons")
    print(f"  Val   : {len(val_p):>5} persons")
    print(f"  Test  : {len(test_p):>5} persons")

    # ── Map persons → frames ──────────────────────────────────────────────────
    train_ids = set(train_p['person_id'])
    val_ids   = set(val_p['person_id'])
    test_ids  = set(test_p['person_id'])

    train_df = df[df['person_id'].isin(train_ids)].copy()
    val_df   = df[df['person_id'].isin(val_ids)].copy()
    test_df  = df[df['person_id'].isin(test_ids)].copy()

    # Shuffle train so DataLoader sees mixed persons/clips from the start
    train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)

    # ── Verify no leakage BEFORE saving ──────────────────────────────────────
    assert len(train_ids & val_ids)  == 0, "LEAKAGE: train/val share persons!"
    assert len(train_ids & test_ids) == 0, "LEAKAGE: train/test share persons!"
    assert len(val_ids   & test_ids) == 0, "LEAKAGE: val/test share persons!"

    t_clips = set(train_df['clip_id'])
    v_clips = set(val_df['clip_id'])
    e_clips = set(test_df['clip_id'])
    assert len(t_clips & v_clips) == 0, "LEAKAGE: train/val share clips!"
    assert len(t_clips & e_clips) == 0, "LEAKAGE: train/test share clips!"

    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(train_out), exist_ok=True)
    train_df.to_csv(train_out, index=False)
    val_df.to_csv(val_out,     index=False)
    test_df.to_csv(test_out,   index=False)

    # ── Report ────────────────────────────────────────────────────────────────
    def _report(name, split_df):
        n       = len(split_df)
        eng     = split_df['label'].sum()
        not_eng = n - eng
        print(f"\n  {name}:")
        print(f"    Persons : {split_df['person_id'].nunique():>5}")
        print(f"    Clips   : {split_df['clip_id'].nunique():>5}")
        print(f"    Frames  : {n:>8,}")
        print(f"    Engaged : {eng:>8,}  ({100*eng/n:.1f}%)")
        print(f"    Not Eng : {not_eng:>8,}  ({100*not_eng/n:.1f}%)")

    print("\nFinal split statistics:")
    _report("Train", train_df)
    _report("Val",   val_df)
    _report("Test",  test_df)

    print(f"\n{'='*55}")
    print("✅  Splits saved (PERSON-level, ZERO leakage verified)")
    print(f"    {train_out}")
    print(f"    {val_out}")
    print(f"    {test_out}")
    print(f"{'='*55}")


if __name__ == '__main__':
    build_splits()
