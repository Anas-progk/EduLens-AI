"""
day1_session_bakeoff.py -- Phase-2 Day-1 honest session-level bake-off (existing data only).

Goal: the cheap, decisive check from PHASE2_FRESH_PLAN.md -- can ANY principled, no-new-data change
honestly beat the frozen session-level macro-F1 0.667? Every config is judged under leak-free LOVO
with a label-shuffle floor and a noise-aware gate. Nothing here learns per-video weights (that would
memorize the room on 30 sessions -- the scene-confound trap).

Levers tested:
  - subgroup-aware pooling: max / high-quantile (q90) instead of plain mean+std
    (a session is collaborative if SOME pairs interact strongly, not if the average does)
  - gaze and geometry session-aggregates added to the 0.667 recipe
  - ensembles of per-view OOF probabilities:
      * E_avg  : equal average
      * E_conf : CONFIDENCE-WEIGHTED -- weight each view by its own |p-0.5| decisiveness.
                 Label-free, FIXED rule: a view that says ~0.5 (uninformative) on a session is
                 auto-down-weighted. This is the honest version of "use the strong channel,
                 suppress the weak one" -- it never looks at the label, so it does not overfit.

DECISION GATE: adopt a new config ONLY if LOVO macro-F1 >= ~0.71 reproducibly (>= +1.5 of 30 videos),
clearly above the shuffle floor, AND not driven solely by co-tracking-duration geometry
(g_approach/g_covis/g_logn are recording artefacts, not collaboration). Else FREEZE at 0.667.

Run (Colab / wherever all three npzs exist):
    python src/eval/day1_session_bakeoff.py
    # custom dir: --dir data/collab_pairs_unique_fresh
"""

import os
import argparse
import numpy as np
from collections import defaultdict


def _scalars(A, B):
    na = np.linalg.norm(A, axis=1); nb = np.linalg.norm(B, axis=1)
    return np.stack([(A * B).sum(1) / (na * nb + 1e-8), np.linalg.norm(A - B, axis=1), na, nb], 1)


def _sig(z): return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _fit(X, y, l2=1.0, it=800, lr=0.3):
    n, d = X.shape; w = np.zeros(d); b = 0.0
    pos = max(y.sum(), 1e-8); neg = max(n - y.sum(), 1e-8)
    sw = np.where(y == 1, n / (2 * pos), n / (2 * neg))
    for _ in range(it):
        g = (_sig(X @ w + b) - y) * sw
        w -= lr * (X.T @ g / n + l2 * w / n); b -= lr * g.sum() / n
    return w, b


def _macro(yt, yp):
    fs = []
    for c in (0, 1):
        tp = ((yp == c) & (yt == c)).sum(); fp = ((yp == c) & (yt != c)).sum(); fn = ((yp != c) & (yt == c)).sum()
        pr = tp / (tp + fp) if tp + fp else 0.0; rc = tp / (tp + fn) if tp + fn else 0.0
        fs.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return sum(fs) / 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/collab_pairs_unique_fresh")
    ap.add_argument("--min_pairs", type=int, default=3)
    a = ap.parse_args()

    d = np.load(os.path.join(a.dir, "pairs_features.npz"), allow_pickle=True)
    S = d["signals"].astype(float); A = d["pooled_A"].astype(float); B = d["pooled_B"].astype(float)
    Y = d["labels"].astype(int); V = d["video_ids"].astype(str)
    SC = _scalars(A, B)

    def _load(name, key, mkey):
        p = os.path.join(a.dir, name)
        if os.path.exists(p):
            z = np.load(p, allow_pickle=True); return z[key].astype(float), z[mkey].astype(int)
        return None, None
    G, GM = _load("pairs_features_geom.npz", "geom", "geom_mask")
    Z, ZM = _load("pairs_features_gaze.npz", "gaze", "gaze_mask")
    print(f"channels present: signals+scalars=yes  geom={G is not None}  gaze={Z is not None}")
    if Z is None:
        print("  !! gaze npz missing -> gaze configs skipped. Build it first (build_gaze_features.py).")

    by = defaultdict(list)
    for i in range(len(Y)):
        by[V[i]].append(i)
    vids = sorted([v for v in by if len(by[v]) >= a.min_pairs])
    yv = np.array([1 if sum(int(Y[i]) for i in by[v]) >= len(by[v]) - sum(int(Y[i]) for i in by[v]) else 0
                   for v in vids])
    print(f"sessions={len(vids)}  C={int(yv.sum())}  N={int((1 - yv).sum())}\n")

    def pool(M, mode):
        parts = [M.mean(0), M.std(0)]
        if "max" in mode: parts.append(M.max(0))
        if "q90" in mode: parts.append(np.quantile(M, 0.9, axis=0))
        return np.concatenate(parts)

    def blockpool(v, M, mask, mode):
        ii = [i for i in by[v] if (mask is None or mask[i] == 1)]
        if not ii: ii = by[v]
        return pool(M[ii], mode)

    def buildX(blocks, mode):
        return np.array([np.concatenate([blockpool(v, M, mk, mode) for (M, mk) in blocks]) for v in vids])

    def lovo(X, y):
        oof = np.zeros(len(y))
        for i in range(len(y)):
            tr = [j for j in range(len(y)) if j != i]
            mu = X[tr].mean(0); sd = X[tr].std(0); sd[sd < 1e-8] = 1.0
            w, b = _fit((X[tr] - mu) / sd, y[tr]); oof[i] = _sig(((X[i] - mu) / sd) @ w + b)
        return _macro(y, (oof >= 0.5).astype(int)), oof

    sigb = [(S, None), (SC, None)]
    configs = [("B0 signals+scalars [mean,std]  (=0.667)", sigb, "meanstd"),
               ("P1 +max pooling", sigb, "meanstdmax"),
               ("P2 +q90 pooling", sigb, "meanstdq90")]
    if Z is not None:
        configs += [("Z1 sig+sc+gaze [mean,std]", sigb + [(Z, ZM)], "meanstd"),
                    ("Z2 sig+sc+gaze +max", sigb + [(Z, ZM)], "meanstdmax")]
    if G is not None:
        configs += [("G1 sig+sc+geom [mean,std]", sigb + [(G, GM)], "meanstd")]
    if Z is not None and G is not None:
        configs += [("ALL sig+sc+geom+gaze", sigb + [(G, GM), (Z, ZM)], "meanstd")]

    res = {}
    print(f"{'config':40} {'LOVO-F1':>8}")
    print("-" * 50)
    for name, bl, mode in configs:
        f1, oof = lovo(buildX(bl, mode), yv); res[name] = (f1, oof, bl, mode)
        print(f"{name:40} {f1:8.3f}")

    # ---- ensembles over per-view OOF probabilities ----
    views = {"sig": res["B0 signals+scalars [mean,std]  (=0.667)"][1]}
    if Z is not None: views["gaze"] = lovo(buildX([(Z, ZM)], "meanstd"), yv)[1]
    if G is not None: views["geom"] = lovo(buildX([(G, GM)], "meanstd"), yv)[1]
    print(f"\nensembles over views {list(views)}:")
    names = list(views); Pm = np.stack([views[k] for k in names], 1)
    ens = {}
    pe = Pm.mean(1); ens["E_avg (equal)"] = _macro(yv, (pe >= 0.5).astype(int))
    w = np.abs(Pm - 0.5) + 1e-6; pc = (w * Pm).sum(1) / w.sum(1)
    ens["E_conf (|p-0.5| weighted)"] = _macro(yv, (pc >= 0.5).astype(int))
    if "gaze" in views:
        sg = np.stack([views["sig"], views["gaze"]], 1)
        ens["E_sig+gaze avg"] = _macro(yv, (sg.mean(1) >= 0.5).astype(int))
        wsg = np.abs(sg - 0.5) + 1e-6
        ens["E_sig+gaze conf"] = _macro(yv, ((wsg * sg).sum(1) / wsg.sum(1) >= 0.5).astype(int))
    for k, vF in ens.items():
        print(f"  {k:40} {vF:8.3f}")

    # ---- shuffle floor on the best single config + gate verdict ----
    best = max(res, key=lambda k: res[k][0]); bf1, _, bl, mode = res[best]
    all_best = max([bf1] + list(ens.values()))
    rng = np.random.default_rng(0)
    fl = [lovo(buildX(bl, mode), rng.permutation(yv))[0] for _ in range(20)]
    floor = float(np.mean(fl))

    print("\n" + "=" * 64)
    print("DAY-1 VERDICT  (baseline session-level = 0.667; noise ~+-0.03 = 1 video)")
    print("=" * 64)
    print(f"  best single config : {best.split('  ')[0]:30} {bf1:.3f}")
    print(f"  best overall (incl ensembles)                     : {all_best:.3f}")
    print(f"  label-shuffle floor (20x)                         : {floor:.3f}")
    clears = all_best >= 0.71 and all_best > floor + 0.05
    if clears:
        print("  -> A config clears the gate. INSPECT it: confirm the gain is NOT from geometry")
        print("     co-tracking-duration features (g_approach/g_covis/g_logn = artefacts) before adopting.")
    else:
        print("  -> Nothing clears >=0.71 above the floor. The 0.667 ceiling holds: FREEZE the")
        print("     classifier at 0.667 and win via the interaction-analysis layer + review readiness.")
    print("=" * 64)


if __name__ == "__main__":
    main()
