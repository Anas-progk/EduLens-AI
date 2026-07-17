"""
build_unique_pairs.py -- Materialize the 883 unique-undirected-pair dataset as ONE
compact, inspectable artifact, so Phase-2 experiments (Stage 1/2) never have to
re-read ~7k .npy from the cache on every run.

WHY THIS FILE EXISTS
--------------------
data/collab_cache/ stores per-CLIP pooled features (one .npy per person per clip,
~13k files). The honest training unit is the 883 UNIQUE undirected pairs, each
aggregated over its clip time-series with 6 relational signals (see
data.collab_pairs.load_pairs). Reading 13k small files over a slow mount on every
experiment is wasteful; this script does that pass ONCE and saves:

  data/collab_pairs_unique/pairs_features.npz   <- the "800+ features" copy
  data/collab_pairs_unique/pairs_index.csv      <- human-readable (labels + signals)

npz keys:
  pooled_A  (P, 768) float32  mean engagement feature of the lower-id person
  pooled_B  (P, 768) float32  mean engagement feature of the higher-id person
  signals   (P, 6)   float32  [state_cos,state_close,traj_cos,dyn_corr,turn_taking,joint_active]
  labels    (P,)     int64    1=Collaborative 0=Not
  n_clips   (P,)     int64    #time-steps (clips) backing the pair
  video_ids (P,)     str      session id (video)
  a_ids,b_ids (P,)   str      canonical lower/higher track ids
  signal_names (6,)  str

This is a numpy-only, torch-free artifact. Stage-1/2 trainers load it directly.
"""

import os
import sys
import csv
import argparse
import numpy as np

# make data.collab_pairs importable whether run from repo root or src/
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE), os.path.dirname(os.path.dirname(_HERE))):
    if _p not in sys.path:
        sys.path.insert(0, _p)
try:
    from data.collab_pairs import load_pairs, SIGNAL_NAMES
except ImportError:
    from collab_pairs import load_pairs, SIGNAL_NAMES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/collab_cache/feature_index.csv")
    ap.add_argument("--cache", default="data/collab_cache")
    ap.add_argument("--out_dir", default="data/collab_pairs_unique")
    args = ap.parse_args()

    pairs = load_pairs(args.index, args.cache, drop_label_conflicts=True, verbose=True)
    P = len(pairs)
    if P == 0:
        print("[build] ERROR: 0 pairs loaded -- check --index/--cache paths")
        sys.exit(1)

    pooled_A = np.stack([p["pooled_A"] for p in pairs]).astype(np.float32)
    pooled_B = np.stack([p["pooled_B"] for p in pairs]).astype(np.float32)
    signals  = np.stack([p["signals"]  for p in pairs]).astype(np.float32)
    labels   = np.array([p["label"]    for p in pairs], dtype=np.int64)
    n_clips  = np.array([p["n_clips"]  for p in pairs], dtype=np.int64)
    video_ids = np.array([p["video"] for p in pairs])
    a_ids     = np.array([p["a"]     for p in pairs])
    b_ids     = np.array([p["b"]     for p in pairs])

    os.makedirs(args.out_dir, exist_ok=True)
    npz = os.path.join(args.out_dir, "pairs_features.npz")
    np.savez_compressed(
        npz,
        pooled_A=pooled_A, pooled_B=pooled_B, signals=signals,
        labels=labels, n_clips=n_clips, video_ids=video_ids,
        a_ids=a_ids, b_ids=b_ids, signal_names=np.array(SIGNAL_NAMES),
    )

    idx = os.path.join(args.out_dir, "pairs_index.csv")
    with open(idx, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "a", "b", "label", "n_clips"] + list(SIGNAL_NAMES))
        for p in pairs:
            w.writerow([p["video"], p["a"], p["b"],
                        "C" if p["label"] == 1 else "N", p["n_clips"]]
                       + [f"{x:.4f}" for x in p["signals"]])

    C = int(labels.sum())
    print(f"[build] P={P}  C={C}  N={P - C}  videos={len(set(video_ids.tolist()))}")
    print(f"[build] pooled_A{pooled_A.shape} pooled_B{pooled_B.shape} signals{signals.shape}")
    print(f"[build] saved -> {npz}")
    print(f"[build] saved -> {idx}")
    print("[build] DONE")


if __name__ == "__main__":
    main()
