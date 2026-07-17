"""
demo_group_collab.py -- Phase-2 GROUP-LEVEL collaboration: the review-safe MVP demo.

This is the deterministic demo to run in the review. It uses ONLY numpy + two saved
files (the group head + the cached per-pair features), so there is no camera, no GPU,
no torch, and nothing that can fail live. It reproduces the honest headline number and
shows the deployed group head making a Collaborative / Not-Collaborative verdict for a
whole session, with the relational signals that drove the decision.

WHAT TO CLAIM IN THE REVIEW  (honest, leak-free)
------------------------------------------------
  * GROUP / SESSION level -- "is this whole session collaborating?":
        macro-F1 = 0.667 (66.7% accuracy) under Leave-One-Video-Out (LOVO) on 30
        sessions of the full 33-video set. Majority baseline F1 0.348; label-shuffled
        floor 0.396  ->  the result is real and well above chance. Built on the FROZEN
        Swin-Tiny engagement backbone + 6 relational signals + 4 feature scalars.
  * PAIR level -- "which two specific people collaborate?":  deliberately NOT claimed.
        It is ~chance with appearance / relational / geometry features (diagnosed, with
        the experiment). Claiming it would not survive scrutiny.

The session-vector math here is byte-identical to
    src/inference/group_collab.py        : session_vector(..., mode="both")
    src/training/train_collab_video_level.py : video_vector(..., "both")
so DEMO == DEPLOY == TRAIN.

Usage:
    python demo_group_collab.py                      # 1 collaborative + 1 not + headline
    python demo_group_collab.py --all                # every session + LOVO headline
    python demo_group_collab.py --video "VID_ (3)"   # one session, full breakdown
"""

import os
import sys
import argparse
import numpy as np
from collections import defaultdict

DEF_HEADS = ["weights/best_collab_group_fresh.npz", "weights/best_collab_group.npz"]
DEF_NPZS = ["data/collab_pairs_unique_fresh/pairs_features.npz",
            "data/collab_pairs_unique/pairs_features.npz"]

VERDICT_C, VERDICT_N = "COLLABORATIVE", "NOT COLLABORATIVE"

# documented honest floors (see PHASE2_HONEST_RESULT.md) -- printed for context
MAJ_BASELINE_F1 = 0.348
SHUFFLE_FLOOR = 0.396


def _first(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return paths[0]


# --------------------------------------------------------------------------
# feature math -- identical to group_collab.session_vector(..., "both")
# --------------------------------------------------------------------------
def _scalars(a, b):
    """[cos, dist, |a|, |b|] of two pooled 768-d vectors."""
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return np.array([float(a @ b / (na * nb + 1e-8)), float(np.linalg.norm(a - b)),
                     float(na), float(nb)])


def _agg(M):
    return np.concatenate([M.mean(0), M.std(0)])


def session_vector(S, A, B):
    """[mean,std] of 6 signals ++ [mean,std] of 4 feature scalars = 20-d 'both'."""
    F = np.stack([_scalars(A[i], B[i]) for i in range(len(A))])
    return np.concatenate([_agg(S), _agg(F)])


# --------------------------------------------------------------------------
# tiny logistic read-out -- identical math to group_collab._fit_lr
# --------------------------------------------------------------------------
def _sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _fit_lr(X, y, l2=1.0, iters=2000, lr=0.3):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    pos = max(y.sum(), 1e-8)
    neg = max(n - y.sum(), 1e-8)
    sw = np.where(y == 1, n / (2 * pos), n / (2 * neg))
    for _ in range(iters):
        g = (_sig(X @ w + b) - y) * sw
        w -= lr * (X.T @ g / n + l2 * w / n)
        b -= lr * g.sum() / n
    return w, b


def _macro_f1(yt, yp):
    yt, yp = np.asarray(yt), np.asarray(yp)
    fs = []
    for c in (0, 1):
        tp = ((yp == c) & (yt == c)).sum()
        fp = ((yp == c) & (yt != c)).sum()
        fn = ((yp != c) & (yt == c)).sum()
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        fs.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return sum(fs) / 2.0


# --------------------------------------------------------------------------
# data + head
# --------------------------------------------------------------------------
def load_sessions(npz_path, min_pairs=3):
    d = np.load(npz_path, allow_pickle=True)
    A, B, S = d["pooled_A"].astype(float), d["pooled_B"].astype(float), d["signals"].astype(float)
    Y, V = d["labels"].astype(int), d["video_ids"].astype(str)
    names = ([str(x) for x in d["signal_names"]] if "signal_names" in d.files
             else ["sig%d" % i for i in range(S.shape[1])])
    by = defaultdict(list)
    for i in range(len(Y)):
        by[str(V[i])].append(i)
    sess = {}
    for v, ix in by.items():
        if len(ix) < min_pairs:
            continue
        c = sum(int(Y[i]) for i in ix)
        sess[v] = dict(S=S[ix], A=A[ix], B=B[ix],
                       vec=session_vector(S[ix], A[ix], B[ix]),
                       label=1 if c >= len(ix) - c else 0, n=len(ix),
                       sig_mean=S[ix].mean(0))
    return sess, names


def load_head(path):
    d = np.load(path, allow_pickle=True)
    return dict(mu=d["mu"], sd=d["sd"], w=d["w"], b=float(d["b"]),
                mode=str(d["mode"]) if "mode" in d.files else "both")


def fit_head_from(sess):
    vids = sorted(sess)
    X = np.stack([sess[v]["vec"] for v in vids])
    y = np.array([sess[v]["label"] for v in vids])
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-8] = 1.0
    w, b = _fit_lr((X - mu) / sd, y)
    return dict(mu=mu, sd=sd, w=w, b=b, mode="both")


def prob(head, vec):
    return float(_sig(((np.asarray(vec, float) - head["mu"]) / head["sd"]) @ head["w"] + head["b"]))


def lovo(sess):
    """Live, honest Leave-One-Video-Out over the sessions (the number to defend)."""
    vids = sorted(sess)
    X = np.stack([sess[v]["vec"] for v in vids])
    y = np.array([sess[v]["label"] for v in vids])
    pred = np.zeros(len(y), int)
    for i in range(len(y)):
        tr = [j for j in range(len(y)) if j != i]
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd[sd < 1e-8] = 1.0
        w, b = _fit_lr((X[tr] - mu) / sd, y[tr])
        pred[i] = int(_sig(((X[i] - mu) / sd) @ w + b) >= 0.5)
    return _macro_f1(y, pred), float((pred == y).mean()), y, pred


def _bar(v, w=20):
    v = max(0.0, min(1.0, float(v)))
    n = int(round(v * w))
    return "#" * n + "." * (w - n)


# --------------------------------------------------------------------------
# views
# --------------------------------------------------------------------------
def show_one(head, sess, names, v):
    if v not in sess:
        print(f"[demo] no session {v!r} with >= 3 pairs. Available:\n   " +
              ", ".join(sorted(sess)))
        return
    s = sess[v]
    p = prob(head, s["vec"])
    verdict = VERDICT_C if p >= 0.5 else VERDICT_N
    truth = VERDICT_C if s["label"] else VERDICT_N
    print("=" * 64)
    print(f"  SESSION: {v}")
    print("=" * 64)
    print(f"  P(collaborative) = {p:.3f}  |{_bar(p, 30)}|")
    print(f"  VERDICT          = {verdict}   (confidence {abs(p - 0.5) * 2:.2f})")
    print(f"  group size       = {s['n']} person-pairs analysed")
    print(f"  ground truth     = {truth}")
    print(f"  -> {'MATCH' if verdict == truth else 'miss (honest, in-sample example)'}")
    print("\n  Relational signals that drove the decision (session mean):")
    for nm, val in sorted(zip(names, s["sig_mean"]), key=lambda kv: -kv[1]):
        print(f"     {nm:14} {val:5.2f}  {_bar(val)}")
    print()


def show_all(head, sess, names):
    vids = sorted(sess)
    ok = 0
    print("\n" + "=" * 74)
    print("  GROUP-LEVEL COLLABORATION  --  all sessions (deployed fresh head)")
    print("=" * 74)
    print(f"  {'session':22} {'pairs':>5} {'P(C)':>6}  {'predicted':18} truth")
    print("  " + "-" * 70)
    for v in vids:
        s = sess[v]
        p = prob(head, s["vec"])
        pred = int(p >= 0.5)
        verdict = VERDICT_C if pred else VERDICT_N
        truth = VERDICT_C if s["label"] else VERDICT_N
        ok += (pred == s["label"])
        print(f"  {v:22} {s['n']:5d} {p:6.3f}  {verdict:18} {truth} "
              f"{'ok' if pred == s['label'] else 'XX'}")
    print("  " + "-" * 70)
    print(f"  deployed-head agreement (in-sample): {ok}/{len(vids)} = {100 * ok / len(vids):.0f}%")
    f1, acc, y, _ = lovo(sess)
    base = max(int(y.sum()), len(y) - int(y.sum())) / len(y)
    print("\n  >>> HONEST generalization (Leave-One-Video-Out, leak-free) -- the number to defend:")
    print(f"      macro-F1 = {f1:.3f}   accuracy = {acc * 100:.1f}%   sessions = {len(y)} "
          f"(C={int(y.sum())} / N={len(y) - int(y.sum())})")
    print(f"      majority-baseline acc = {base * 100:.1f}%   |  documented: "
          f"majority-F1 {MAJ_BASELINE_F1:.3f}, label-shuffle floor {SHUFFLE_FLOOR:.3f}")
    print("      Pair level ('which two people') is NOT claimed -- it is chance (documented).")
    print("=" * 74)


def main():
    ap = argparse.ArgumentParser(description="Phase-2 group-level collaboration -- review-safe demo")
    ap.add_argument("--video", default="", help='one session id, e.g. "VID_ (3)"')
    ap.add_argument("--all", action="store_true", help="score every session + LOVO headline")
    ap.add_argument("--head", default="", help="group head .npz (default: fresh head)")
    ap.add_argument("--npz", default="", help="pairs_features.npz (default: fresh features)")
    ap.add_argument("--min_pairs", type=int, default=3)
    a = ap.parse_args()

    npz = a.npz or _first(DEF_NPZS)
    if not os.path.exists(npz):
        print(f"[demo] features npz not found: {npz}")
        sys.exit(1)
    sess, names = load_sessions(npz, min_pairs=a.min_pairs)

    head_path = a.head or _first(DEF_HEADS)
    if os.path.exists(head_path):
        head, src = load_head(head_path), head_path
    else:
        head, src = fit_head_from(sess), "(fitted on the fly from the npz)"

    print(f"[demo] features  = {npz}")
    print(f"[demo] group head = {src}  (mode={head['mode']}, {len(head['w'])}-d)")

    if a.all:
        show_all(head, sess, names)
    elif a.video:
        show_one(head, sess, names, a.video)
    else:
        for v in ("VID_ (3)", "VID_ (10)"):
            if v in sess:
                show_one(head, sess, names, v)
        print('Tip: run  --all  for every session + the LOVO headline, '
              'or  --video "VID_ (4)"  for one.')


if __name__ == "__main__":
    main()
