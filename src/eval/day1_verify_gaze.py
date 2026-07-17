"""
day1_verify_gaze.py -- stress-test the session-level 0.764 (signals+scalars+gaze) BEFORE adopting it.

The label-shuffle floor only proves the model isn't fitting random labels; it does NOT rule out a
CONFOUND (gaze stats that secretly encode which-video-it-is -- e.g. collaborative rooms filmed more
frontally -> faces more detectable -> higher gaze coverage). At session level that is the dominant
risk. This script runs the checks the shuffle floor misses and prints ADOPT / CAUTION / REJECT.

Checks (all honest LOVO, 30 sessions):
  1. CONFOUND discriminator -- baseline + coverage/count WITHOUT any gaze content. If that alone ~=
     the gaze score, the "win" is coverage/which-video, not collaboration.
  2. Does gaze add BEYOND coverage -- baseline + coverage + gaze vs baseline + coverage.
  3. ROBUSTNESS -- Z1 across L2 in {0.3..30} and with fewer features (gaze-mean-only, mutual+turntake
     only, gaze-only). A real effect survives strong regularization; an overfit one collapses.
  4. Per-gaze-feature session AUC -- is the driver collaboration-meaningful (gz_mutual / gz_turntake /
     gz_oneway) or a proxy (coverage / n_pairs)?
  5. Per-video flips -- broad gain or a couple of lucky sessions?

Run on Colab (where pairs_features_gaze.npz exists):
    python src/eval/day1_verify_gaze.py
"""

import argparse
import numpy as np
from collections import defaultdict


def scalars(A, B):
    na = np.linalg.norm(A, axis=1); nb = np.linalg.norm(B, axis=1)
    return np.stack([(A * B).sum(1) / (na * nb + 1e-8), np.linalg.norm(A - B, axis=1), na, nb], 1)


def sigf(z): return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def fit(X, y, l2, it=800, lr=0.3):
    n, d = X.shape; w = np.zeros(d); b = 0.0
    pos = max(y.sum(), 1e-8); neg = max(n - y.sum(), 1e-8)
    sw = np.where(y == 1, n / (2 * pos), n / (2 * neg))
    for _ in range(it):
        g = (sigf(X @ w + b) - y) * sw
        w -= lr * (X.T @ g / n + l2 * w / n); b -= lr * g.sum() / n
    return w, b


def macro(yt, yp):
    fs = []
    for c in (0, 1):
        tp = ((yp == c) & (yt == c)).sum(); fp = ((yp == c) & (yt != c)).sum(); fn = ((yp != c) & (yt == c)).sum()
        pr = tp / (tp + fp) if tp + fp else 0.0; rc = tp / (tp + fn) if tp + fn else 0.0
        fs.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return sum(fs) / 2.0


def auc(s, y):
    s = np.asarray(s, float); y = np.asarray(y)
    pos = s[y == 1]; neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0: return 0.5
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    a = (r[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return max(a, 1 - a)


def lovo(X, y, l2=1.0):
    oof = np.zeros(len(y))
    for i in range(len(y)):
        tr = [j for j in range(len(y)) if j != i]
        mu = X[tr].mean(0); sd = X[tr].std(0); sd[sd < 1e-8] = 1.0
        w, b = fit((X[tr] - mu) / sd, y[tr], l2); oof[i] = sigf(((X[i] - mu) / sd) @ w + b)
    return macro(y, (oof >= 0.5).astype(int)), oof


def meanstd(M): return np.concatenate([M.mean(0), M.std(0)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="data/collab_pairs_unique_fresh/pairs_features_gaze.npz")
    ap.add_argument("--min_pairs", type=int, default=3)
    a = ap.parse_args()
    d = np.load(a.npz, allow_pickle=True)
    S = d["signals"].astype(float); A = d["pooled_A"].astype(float); B = d["pooled_B"].astype(float)
    Y = d["labels"].astype(int); V = d["video_ids"].astype(str)
    Z = d["gaze"].astype(float); ZM = d["gaze_mask"].astype(int)
    gn = [str(x) for x in d["gaze_names"]]
    SC = scalars(A, B)

    by = defaultdict(list)
    for i in range(len(Y)):
        by[V[i]].append(i)
    vids = sorted([v for v in by if len(by[v]) >= a.min_pairs])
    yv = np.array([1 if sum(int(Y[i]) for i in by[v]) >= len(by[v]) - sum(int(Y[i]) for i in by[v]) else 0
                   for v in vids])
    print(f"sessions={len(vids)} C={int(yv.sum())} N={int((1-yv).sum())} | gaze={gn}")

    def covpairs(v): return [i for i in by[v] if ZM[i] == 1] or by[v]
    Xbase = np.array([np.concatenate([meanstd(S[by[v]]), meanstd(SC[by[v]])]) for v in vids])           # 20-d
    cov = np.array([[sum(ZM[i] for i in by[v]) / len(by[v]), len(by[v]), sum(ZM[i] for i in by[v])]
                    for v in vids], float)                                                              # 3-d

    def Xg(agg):
        rows = []
        for v in vids:
            cp = covpairs(v)
            sig = np.concatenate([meanstd(S[by[v]]), meanstd(SC[by[v]])])
            if agg == "meanstd": g = np.concatenate([Z[cp].mean(0), Z[cp].std(0)])
            elif agg == "mean": g = Z[cp].mean(0)
            elif agg == "mutturn": g = Z[cp][:, [gn.index("gz_mutual"), gn.index("gz_turntake")]].mean(0)
            rows.append(np.concatenate([sig, g]))
        return np.array(rows)
    Xgaze_only = np.array([meanstd(Z[covpairs(v)]) for v in vids])

    R = {}
    R["base (=0.667)"] = lovo(Xbase, yv)[0]
    R["base + coverage/count (NO gaze content)"] = lovo(np.hstack([Xbase, cov]), yv)[0]
    R["base + gaze[mean,std]  (Z1)"] = lovo(Xg("meanstd"), yv)[0]
    R["base + gaze[mean] only"] = lovo(Xg("mean"), yv)[0]
    R["base + gaze(mutual,turntake) only"] = lovo(Xg("mutturn"), yv)[0]
    R["base + coverage + gaze[mean,std]"] = lovo(np.hstack([Xg("meanstd"), cov]), yv)[0]
    R["gaze[mean,std] only"] = lovo(Xgaze_only, yv)[0]
    print("\n--- configs (LOVO macro-F1, L2=1) ---")
    for k, vF in R.items():
        print(f"  {k:44} {vF:.3f}")

    print("\n--- Z1 robustness across L2 (overfit check; 32-d on 30 samples) ---")
    Xz = Xg("meanstd")
    l2tab = {l2: lovo(Xz, yv, l2)[0] for l2 in [0.3, 1, 3, 10, 30]}
    for l2, f in l2tab.items():
        print(f"  L2={l2:>4}  F1={f:.3f}")

    print("\n--- per-session gaze-mean AUC vs label (meaningful driver?) ---")
    gm = np.array([Z[covpairs(v)].mean(0) for v in vids])
    for j, nm in enumerate(gn):
        print(f"  {nm:14} AUC={auc(gm[:, j], yv):.3f}")
    print(f"  {'coverage':14} AUC={auc(cov[:, 0], yv):.3f}")
    print(f"  {'n_pairs':14} AUC={auc(cov[:, 1], yv):.3f}")

    _, ob = lovo(Xbase, yv); _, oz = lovo(Xz, yv)
    pb = (ob >= 0.5).astype(int); pz = (oz >= 0.5).astype(int)
    fixed = [vids[i] for i in range(len(vids)) if pb[i] != yv[i] and pz[i] == yv[i]]
    broke = [vids[i] for i in range(len(vids)) if pb[i] == yv[i] and pz[i] != yv[i]]
    print(f"\n--- per-video flips (base -> +gaze) ---\n  fixed ({len(fixed)}): {fixed}\n  broke ({len(broke)}): {broke}")
    rng = np.random.default_rng(0); fl = [lovo(Xz, rng.permutation(yv))[0] for _ in range(20)]
    print(f"  Z1 shuffle floor (20x) = {np.mean(fl):.3f} ± {np.std(fl):.3f}")

    # ---- verdict ----
    z1 = R["base + gaze[mean,std]  (Z1)"]; covonly = R["base + coverage/count (NO gaze content)"]
    addbeyond = R["base + coverage + gaze[mean,std]"]
    l2hi = l2tab[10]
    meaningful = max(auc(gm[:, gn.index("gz_mutual")], yv), auc(gm[:, gn.index("gz_turntake")], yv),
                     auc(gm[:, gn.index("gz_oneway")], yv))
    not_conf = covonly < z1 - 0.05
    robust = l2hi >= 0.71
    adds = addbeyond >= covonly + 0.03
    mean_ok = meaningful >= 0.60
    print("\n" + "=" * 66)
    print("GAZE-0.764 VERDICT")
    print("=" * 66)
    print(f"  Z1={z1:.3f} | coverage-only={covonly:.3f} | +gaze-beyond-cov={addbeyond:.3f} | "
          f"Z1@L2=10={l2hi:.3f} | best meaningful gaze AUC={meaningful:.3f}")
    print(f"  checks: not-coverage-confound={not_conf}  robust-to-L2={robust}  "
          f"adds-beyond-coverage={adds}  meaningful-driver={mean_ok}")
    if not_conf and robust and adds and mean_ok:
        print(f"  -> ADOPT (cautiously): real session signal beyond coverage, robust, meaningful.")
        print(f"     New honest deliverable ~{z1:.2f} (signals+scalars+gaze). Update demo/docs/deck.")
    elif (not not_conf) or (not adds):
        print("  -> REJECT / CONFOUND: the lift is largely coverage / which-video, not gaze content.")
        print("     Keep 0.667 as the honest headline.")
    else:
        print("  -> CAUTION: real but fragile (overfit-sensitive on 30 sessions). Report 0.667 as the")
        print("     headline; present signals+gaze (~0.76) as a promising secondary result, not the claim.")
    print("=" * 66)


if __name__ == "__main__":
    main()
