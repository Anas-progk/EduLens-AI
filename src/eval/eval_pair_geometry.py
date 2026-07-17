"""
eval_pair_geometry.py -- the DECISION for the geometry track. Self-contained (numpy only),
reads pairs_features_geom.npz, and judges geometry on the WITHIN-SCENE metric (the thing
appearance failed), not the pooled number.

For each mode it reports, under honest Leave-One-Video-Out:
    pooled macro-F1     (session-leaky reference -- high here just means "guessed the room")
    VID_(4) LOVO        (the only balanced scene -> the cleanest within-scene number)
    per-video median    (broad within-scene signal? want > 0.50)
    shuffle floor       (chance control)

Modes: signals(6) | geom(10) | signals+geom(16) | full768+sig | full768+sig+geom
Go/no-go: geometry is real iff it lifts VID_(4) LOVO and the per-video median clearly above
the appearance baseline and above ~0.50. Otherwise the frozen session-level 0.667 stands.
"""

import os
import argparse
import numpy as np
from collections import defaultdict


def macro_f1(yt, yp):
    yt, yp = np.asarray(yt), np.asarray(yp)
    fs = []
    for c in (0, 1):
        tp = int(((yp == c) & (yt == c)).sum()); fp = int(((yp == c) & (yt != c)).sum())
        fn = int(((yp != c) & (yt == c)).sum())
        pr = tp / (tp + fp) if (tp + fp) else 0.0
        rc = tp / (tp + fn) if (tp + fn) else 0.0
        fs.append(2 * pr * rc / (pr + rc) if (pr + rc) else 0.0)
    return sum(fs) / 2.0


def standardize_fit(X):
    mu = X.mean(0); sd = X.std(0); sd[sd < 1e-8] = 1.0
    return mu, sd


def fit_lr(X, y, l2=1.0, iters=1500, lr=0.3):
    n, d = X.shape; w = np.zeros(d); b = 0.0
    pos = max(y.sum(), 1e-8); neg = max(n - y.sum(), 1e-8)
    sw = np.where(y == 1, n / (2 * pos), n / (2 * neg))
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ w + b, -30, 30)))
        g = (p - y) * sw
        w -= lr * (X.T @ g / n + l2 * w / n); b -= lr * (g.sum() / n)
    return w, b


def scalars_of(A, B):
    na = np.linalg.norm(A, axis=1); nb = np.linalg.norm(B, axis=1)
    return np.stack([(A * B).sum(1) / (na * nb + 1e-8), np.linalg.norm(A - B, axis=1), na, nb], 1)


def build_X(d, mode, swap=False):
    A, B = d["A"], d["B"]
    if swap:
        A, B = B, A
    S, G = d["S"], d["G"]
    if mode == "signals":
        return S
    if mode == "geom":
        return G
    if mode == "signals+geom":
        return np.concatenate([S, G], 1)
    if mode == "full768+sig":
        return np.concatenate([A, B, S], 1)
    if mode == "full768+sig+geom":
        return np.concatenate([A, B, S, G], 1)
    raise ValueError(mode)


def lovo(d, y, v, mode, l2, augment=False):
    """Return (pooled_macro_f1, {video: per_video_macro_f1})."""
    vids = sorted(set(v))
    yt_all, yp_all, per = [], [], {}
    for vid in vids:
        te = v == vid; tr = ~te
        Xtr = build_X({k: d[k][tr] for k in d}, mode); ytr = y[tr]
        if augment and mode.startswith("full768"):
            Xs = build_X({k: d[k][tr] for k in d}, mode, swap=True)
            Xtr = np.concatenate([Xtr, Xs], 0); ytr = np.concatenate([ytr, ytr])
        Xte = build_X({k: d[k][te] for k in d}, mode); yte = y[te]
        mu, sd = standardize_fit(Xtr); w, b = fit_lr((Xtr - mu) / sd, ytr, l2=l2)
        pr = (1.0 / (1.0 + np.exp(-np.clip(((Xte - mu) / sd) @ w + b, -30, 30))) >= 0.5).astype(int)
        yt_all += yte.tolist(); yp_all += pr.tolist()
        per[vid] = macro_f1(yte, pr) if len(set(yte.tolist())) > 0 else 0.0
    return macro_f1(yt_all, yp_all), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="data/collab_pairs_unique_fresh/pairs_features_geom.npz")
    ap.add_argument("--balanced_video", default="VID_ (4)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--modes", nargs="+",
                    default=["signals", "geom", "signals+geom", "full768+sig", "full768+sig+geom"],
                    help="subset to run (the full768 modes are slow; drop them for a quick check)")
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    d = {"A": z["pooled_A"].astype(np.float64), "B": z["pooled_B"].astype(np.float64),
         "S": z["signals"].astype(np.float64), "G": z["geom"].astype(np.float64)}
    y = z["labels"].astype(np.int64); v = z["video_ids"].astype(str)
    mask = z["geom_mask"].astype(np.int64) if "geom_mask" in z.files else np.ones(len(y), int)
    print(f"[eval] {len(y)} pairs / {len(set(v))} videos | geometry coverage {int(mask.sum())}/{len(y)} "
          f"({mask.mean()*100:.0f}%) | geom dims {d['G'].shape[1]} ({list(z['geom_names'])})")
    bv = args.balanced_video
    if bv in set(v):
        m = v == bv
        print(f"[eval] balanced scene {bv!r}: n={int(m.sum())} C={int(y[m].sum())} "
              f"N={int(m.sum()-y[m].sum())} | geom-covered {int(mask[m].sum())}/{int(m.sum())}")

    modes = list(args.modes)
    l2_for = {"signals": 1.0, "geom": 1.0, "signals+geom": 1.0,
              "full768+sig": 10.0, "full768+sig+geom": 10.0}
    print(f"\n{'mode':18} {'pooled':>7} {'VID4-LOVO':>10} {'perVid-med':>11} {'>0.5':>6}")
    results = {}
    for mode in modes:
        pooled, per = lovo(d, y, v, mode, l2_for[mode], augment=mode.startswith("full768"))
        vid4 = per.get(bv, float("nan"))
        med = float(np.median(list(per.values())))
        nabove = sum(1 for f in per.values() if f > 0.5)
        results[mode] = (pooled, vid4, med, nabove)
        print(f"{mode:18} {pooled:7.3f} {vid4:10.3f} {med:11.3f} {nabove:3d}/{len(per)}")

    # shuffle floor on the strongest within-scene mode (by per-video median)
    strong = max(results, key=lambda k: results[k][2])
    rng = np.random.default_rng(args.seed)
    sh = []
    for _ in range(5):
        ys = rng.permutation(y)
        _, per = lovo(d, ys, v, strong, l2_for[strong], augment=strong.startswith("full768"))
        sh.append(float(np.median(list(per.values()))))
    floor = float(np.mean(sh))

    print("\n" + "=" * 70)
    print("GEOMETRY VERDICT (within-scene is decisive; pooled is session-leaky)")
    print("=" * 70)
    base = results.get("signals")
    geo_name = "signals+geom" if "signals+geom" in results else ("geom" if "geom" in results else None)
    geo = results.get(geo_name) if geo_name else None
    if base is None or geo is None:
        print("  (need 'signals' and ('signals+geom' or 'geom') in --modes for the verdict)")
        print("=" * 70)
        return
    print(f"  appearance/signals within-scene: VID4-LOVO {base[1]:.3f}  per-video median {base[2]:.3f}")
    print(f"  + geometry ({geo_name})      : VID4-LOVO {geo[1]:.3f}  per-video median {geo[2]:.3f}")
    print(f"  per-video-median shuffle floor : {floor:.3f}")
    lift = geo[2] - base[2]
    real = (geo[2] > floor + 0.03) and (geo[2] > 0.50) and (lift > 0.03)
    print(f"  -> geometry {'LIFTS within-scene pair signal (build a compact pair head)' if real else 'does NOT lift within-scene (session-0.667 stands; documented negative)'}")
    print(f"     (median lift {lift:+.3f}; within-scene must clear 0.50 AND the shuffle floor)")
    print("=" * 70)


if __name__ == "__main__":
    main()
