"""
ceiling_probe.py -- Measure the INFORMATION CEILING of the current (engagement)
features for PAIR-level collaboration, using WITHIN-balanced-video cross-validation.

WHY (read before trusting any number here)
------------------------------------------
The honest question is: "can these features separate collaboration AT ALL, if we
remove the cross-scene generalization problem?" To answer it we hold the scene
CONSTANT (one internally balanced video) and run honest k-fold CV INSIDE it.

This is "leakage-allowed" ONLY in the sense that train and test share the room. It is
NOT train==test -- training and testing on the same rows would let a 768-d model
memorize to ~100%, which is a meaningless number. With real k-fold separation, the
BEST within-scene macro-F1 is the ceiling: no head / brain / signal-injection on THESE
features can beat it on a balanced scene. If it sits ~0.56, the bottleneck is the
FEATURES (Phase-1 was trained to treat talking/discussion as NOT engaged, i.e. to
suppress the very cue collaboration needs) -- which is the evidence that justifies
Stage-2 fresh-feature re-extraction, instead of more trainer tuning.

Modes compared (all under the same CV):
  scalars      4 engagement-derived scalars [cos, dist, |A|, |B|]   (what the LR head saw)
  full768      raw [pooled_A | pooled_B] = 1536-d                   (everything the backbone gives)
  full768+sig  full768 + 6 relational signals
  signals      6 relational signals only

For each mode we sweep a small L2 grid and report the BEST (an optimistic upper bound,
so if even this is ~0.56 the ceiling claim is strong). A label-shuffled control gives
the chance floor for this small set, and the majority baseline is printed too.

Numpy only. Prefers data/collab_pairs_unique/pairs_features.npz (fast); falls back to
re-reading the cache via data.collab_pairs.load_pairs if the npz is absent.
"""

import os
import sys
import argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_SRC)
for _p in (_SRC, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from training.train_collab_video_level import fit_lr, predict_proba, metrics, _standardize_fit
except ImportError:
    from src.training.train_collab_video_level import fit_lr, predict_proba, metrics, _standardize_fit


# ---------------------------------------------------------------------------
# data loading (npz preferred)
# ---------------------------------------------------------------------------

def load_arrays(npz_path, index_csv, cache_dir):
    if os.path.exists(npz_path):
        d = np.load(npz_path, allow_pickle=True)
        return (d["pooled_A"].astype(np.float64), d["pooled_B"].astype(np.float64),
                d["signals"].astype(np.float64), d["labels"].astype(np.int64),
                d["video_ids"].astype(str))
    # fallback: slow path
    try:
        from data.collab_pairs import load_pairs
    except ImportError:
        from src.data.collab_pairs import load_pairs
    pairs = load_pairs(index_csv, cache_dir, drop_label_conflicts=True, verbose=True)
    A = np.stack([p["pooled_A"] for p in pairs]).astype(np.float64)
    B = np.stack([p["pooled_B"] for p in pairs]).astype(np.float64)
    S = np.stack([p["signals"] for p in pairs]).astype(np.float64)
    y = np.array([p["label"] for p in pairs], dtype=np.int64)
    v = np.array([p["video"] for p in pairs])
    return A, B, S, y, v


def pick_balanced_video(y, v, min_total=20):
    vids = {}
    for vi in np.unique(v):
        m = v == vi
        c = int(y[m].sum()); n = int(m.sum() - c); t = c + n
        if t >= min_total:
            vids[vi] = (min(c, n) / t, t, c, n)
    if not vids:
        return None
    best = max(vids.items(), key=lambda kv: kv[1][0])
    return best[0], vids[best[0]]


# ---------------------------------------------------------------------------
# feature construction
# ---------------------------------------------------------------------------

def scalars_of(A, B):
    na = np.linalg.norm(A, axis=1); nb = np.linalg.norm(B, axis=1)
    cos = (A * B).sum(1) / (na * nb + 1e-8)
    dist = np.linalg.norm(A - B, axis=1)
    return np.stack([cos, dist, na, nb], axis=1)


def build_X(A, B, S, mode, swap=False):
    """swap=True applies the A<->B symmetry augmentation (collaboration is symmetric)."""
    if swap:
        A, B = B, A
    if mode == "scalars":
        return scalars_of(A, B)
    if mode == "signals":
        return S.copy()
    if mode == "full768":
        return np.concatenate([A, B], axis=1)
    if mode == "full768+sig":
        return np.concatenate([A, B, S], axis=1)
    raise ValueError(mode)


# ---------------------------------------------------------------------------
# stratified k-fold within one scene
# ---------------------------------------------------------------------------

def stratified_folds(y, k, seed=0):
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    for cls in (0, 1):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        for i, j in enumerate(idx):
            folds[i % k].append(j)
    return [np.array(sorted(f)) for f in folds]


def cv_macro_f1(A, B, S, y, mode, l2, k=5, seed=0, augment=True):
    folds = stratified_folds(y, k, seed=seed)
    yt_all, yp_all = [], []
    do_aug = augment and mode in ("full768", "full768+sig")
    for i in range(k):
        te = folds[i]
        tr = np.concatenate([folds[j] for j in range(k) if j != i])
        Xtr = build_X(A[tr], B[tr], S[tr], mode)
        ytr = y[tr]
        if do_aug:
            Xtr = np.concatenate([Xtr, build_X(A[tr], B[tr], S[tr], mode, swap=True)], axis=0)
            ytr = np.concatenate([ytr, y[tr]])
        Xte = build_X(A[te], B[te], S[te], mode)
        mu, sd = _standardize_fit(Xtr)
        w, b = fit_lr((Xtr - mu) / sd, ytr, l2=l2)
        pr = predict_proba((Xte - mu) / sd, w, b)
        yt_all.extend(y[te].tolist())
        yp_all.extend((pr >= 0.5).astype(int).tolist())
    return metrics(yt_all, yp_all)["macro_f1"]


def best_over_l2(A, B, S, y, mode, l2_grid, k=5, seed=0):
    best_f1, best_l2 = -1.0, None
    for l2 in l2_grid:
        f1 = cv_macro_f1(A, B, S, y, mode, l2, k=k, seed=seed)
        if f1 > best_f1:
            best_f1, best_l2 = f1, l2
    return best_f1, best_l2


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="data/collab_pairs_unique/pairs_features.npz")
    ap.add_argument("--index", default="data/collab_cache/feature_index.csv")
    ap.add_argument("--cache", default="data/collab_cache")
    ap.add_argument("--video", default="", help="force a video id; default = auto most-balanced")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    A, B, S, y, v = load_arrays(args.npz, args.index, args.cache)
    print(f"[ceiling] loaded {len(y)} pairs over {len(np.unique(v))} videos")

    if args.video:
        vid = args.video
        m = v == vid
        info = (None,)
    else:
        pick = pick_balanced_video(y, v)
        if pick is None:
            print("[ceiling] no video with enough pairs"); sys.exit(1)
        vid, info = pick
    m = v == vid
    c = int(y[m].sum()); n = int(m.sum() - c)
    print(f"[ceiling] balanced scene = {vid!r}  pairs={int(m.sum())}  C={c}  N={n}  "
          f"(C-rate {c/(c+n)*100:.0f}%)")

    Av, Bv, Sv, yv = A[m], B[m], S[m], y[m]
    l2_grid = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]

    print(f"\nWITHIN-SCENE {args.k}-fold CV macro-F1 (best over L2 {l2_grid}):")
    print(f"  {'mode':14}  macro-F1   bestL2")
    results = {}
    for mode in ("scalars", "signals", "full768", "full768+sig"):
        f1, l2 = best_over_l2(Av, Bv, Sv, yv, mode, l2_grid, k=args.k, seed=args.seed)
        results[mode] = f1
        print(f"  {mode:14}  {f1:.3f}      {l2}")

    # label-shuffle control on the strongest mode
    strong = max(results, key=results.get)
    rng = np.random.default_rng(args.seed)
    shuf = []
    for s in range(5):
        ys = rng.permutation(yv)
        f1, _ = best_over_l2(Av, Bv, Sv, ys, strong, l2_grid, k=args.k, seed=args.seed)
        shuf.append(f1)
    shuf_mean = float(np.mean(shuf))

    n1 = int(yv.sum()); n0 = len(yv) - n1
    maj = max(n0, n1) / len(yv)
    base_f1 = metrics(yv, np.full_like(yv, 1 if n1 >= n0 else 0))["macro_f1"]

    print("\n" + "=" * 64)
    print("CEILING VERDICT (within balanced scene, leakage-allowed upper bound)")
    print("=" * 64)
    print(f"  best mode           : {strong}  -> macro-F1 {results[strong]:.3f}")
    print(f"  majority baseline   : acc {maj*100:.0f}%   macro-F1 {base_f1:.3f}")
    print(f"  label-shuffle floor : macro-F1 {shuf_mean:.3f} (+/- {np.std(shuf):.3f})")
    real = results[strong] > shuf_mean + 0.03
    print(f"  -> features carry {'SOME' if real else 'NO clear'} within-scene pair signal "
          f"({'above' if real else 'not above'} shuffle floor).")
    print(f"  -> CEILING ~ {results[strong]:.2f}.  If ~0.56, Stage-2 fresh features is JUSTIFIED.")
    print("=" * 64)


if __name__ == "__main__":
    main()
