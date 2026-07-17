"""
train_collab_video_level.py -- Honest, leak-free evaluation of Phase-2 collaboration
using Leave-One-Video-Out (LOVO) cross-validation.

WHY THIS FILE EXISTS (read before changing anything)
----------------------------------------------------
The single-held-out-balanced-video run (train_collab_honest.py on VID_ (4)) COLLAPSED
to predicting all-N. That is not a bug in the trainer -- it is the data telling us
something true:

  * The pair-level collaboration label is ~83-90% determined by WHICH VIDEO the pair
    is in (the scene confound). Collaboration in this dataset is essentially a
    SESSION-level property: in a given recording the whole group is either doing a
    discussion/joint task (Collaborative) or individual work (Not-Collaborative).
  * Only ONE video (VID_ (4)) is internally balanced. If we hold it out as the only
    test, TRAIN contains almost no within-scene C-vs-N contrast, so a frozen-feature
    head can only learn "scene -> label", which cannot transfer to an unseen scene.
    -> all-N collapse, macro-F1 == scene baseline.
  * The frozen engagement features are collab-BLIND on a held-out balanced scene
    (Phase 1 was trained to treat talking as NOT engaged, i.e. to SUPPRESS exactly
    the discussion cue that defines collaboration).

So a single split is the wrong instrument. The honest instrument for a confounded,
session-structured dataset is GROUPED cross-validation, holding out a WHOLE video each
fold so scene identity can never leak. This file answers two honest questions:

  (Q1) SESSION level  -- can aggregate interaction statistics classify a whole UNSEEN
        session as Collaborative vs Not? (LOVO over all videos; N = #videos)
  (Q2) PAIR level / deployment -- train on all-but-one video, predict the held-out
        video's pairs, pool predictions over all folds. (the real deployment question:
        "given a brand-new classroom video, label its pairs")

Both run an ABLATION that is the scientific crux of the project:
    signals-only   vs   features-only   vs   combined
If the confound-RESISTANT relational signals (trajectory correlation, turn-taking,
joint activity, ... computed from the per-pair feature time-series) beat the
scene-correlated engagement features under LOVO, that is direct evidence the signals
carry collaboration information that GENERALIZES -- which is the honest contribution.

A label-shuffled control is also run: if the real model beats its own shuffled-label
version, the result is not luck.

Pure numpy. No torch, no sklearn -> runs locally for verification AND on Colab
unchanged. Consumes data/collab_pairs.py (the de-duplicated pair builder).

Usage:
    python src/training/train_collab_video_level.py \
        --index data/collab_cache/feature_index_33.csv \
        --cache data/collab_cache
    # optional: also save a deployable head fit on ALL pairs
    python src/training/train_collab_video_level.py ... \
        --save_final weights/collab_head_lr.npz
"""

import os
import sys
import argparse
import numpy as np
from collections import defaultdict

# make `data.collab_pairs` importable whether run from repo root or src/
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)             # .../src
for _p in (_SRC, os.path.dirname(_SRC)):  # src and repo root
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from data.collab_pairs import load_pairs, SIGNAL_NAMES
except ImportError:
    from src.data.collab_pairs import load_pairs, SIGNAL_NAMES


def load_pairs_from_npz(npz_path):
    """Build the same list-of-dicts load_pairs returns, but from a prebuilt
    pairs_features.npz (pooled_A/B, signals, labels, video_ids, a_ids, b_ids). Lets the
    honest LOVO run on the EXACT fresh arrays the ceiling probe scored, with no risk of
    an --index/--cache mismatch. The npz is already de-conflicted by build_unique_pairs."""
    d = np.load(npz_path, allow_pickle=True)
    A, B, S = d["pooled_A"], d["pooled_B"], d["signals"]
    y = d["labels"].astype(int)
    v = d["video_ids"].astype(str)
    a_ids = d["a_ids"].astype(str) if "a_ids" in d.files else [""] * len(y)
    b_ids = d["b_ids"].astype(str) if "b_ids" in d.files else [""] * len(y)
    n_clips = d["n_clips"] if "n_clips" in d.files else np.zeros(len(y), dtype=int)
    pairs = []
    for i in range(len(y)):
        pairs.append({
            "pooled_A": A[i].astype(np.float64), "pooled_B": B[i].astype(np.float64),
            "signals": S[i].astype(np.float64), "label": int(y[i]), "video": str(v[i]),
            "a": str(a_ids[i]), "b": str(b_ids[i]), "n_clips": int(n_clips[i]),
        })
    return pairs


# ===========================================================================
# feature construction
# ===========================================================================

def _scalars_from(a, b):
    """cos, dist, |a|, |b| for two pooled vectors (order: a then b)."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    cos = float(np.dot(a, b) / (na * nb + 1e-8))
    dist = float(np.linalg.norm(a - b))
    return np.array([cos, dist, float(na), float(nb)], dtype=np.float64)


def _pair_feature_scalars(p):
    """Engagement-feature-derived scalars for ONE pair (proxy for 'what the frozen
    backbone sees'): cosine, distance, and the two magnitudes of the pooled features.
    These are scene-correlated, so they are the 'features-only' arm of the ablation."""
    return _scalars_from(p["pooled_A"].astype(np.float64), p["pooled_B"].astype(np.float64))


def pair_vector(p, mode, swap=False):
    """Per-pair vector for the PAIR-level model (Q2). No aggregation.

    swap=True applies the A<->B symmetry augmentation (collaboration is undirected),
    mirroring ceiling_probe.build_X so the honest-LOVO full768 test is apples-to-apples
    with the within-scene ceiling.
    """
    sig = p["signals"].astype(np.float64)          # 6 relational signals
    A = p["pooled_A"].astype(np.float64)
    B = p["pooled_B"].astype(np.float64)
    if swap:
        A, B = B, A
    if mode == "signals":
        return sig
    if mode == "features":
        return _scalars_from(A, B)                 # 4 engagement-derived scalars
    if mode == "both":
        return np.concatenate([sig, _scalars_from(A, B)])
    if mode == "full768":                          # raw [A | B] = 1536-d
        return np.concatenate([A, B])
    if mode == "full768+sig":                      # raw [A | B | 6 signals]
        return np.concatenate([A, B, sig])
    raise ValueError(mode)


def _agg(M):
    """Aggregate a (P, d) matrix of per-pair values into mean+std (-> 2d) summary."""
    return np.concatenate([M.mean(0), M.std(0)])


def video_vector(pairs_in_video, mode):
    """Session-level vector for the VIDEO-level model (Q1): aggregate the per-pair
    values across all pairs in the video into mean+std."""
    S = np.stack([p["signals"].astype(np.float64) for p in pairs_in_video])      # P x 6
    F = np.stack([_pair_feature_scalars(p) for p in pairs_in_video])             # P x 4
    if mode == "signals":
        return _agg(S)
    if mode == "features":
        return _agg(F)
    return np.concatenate([_agg(S), _agg(F)])


def video_label(pairs_in_video):
    """Session label = majority label of its pairs (1 = Collaborative)."""
    c = sum(p["label"] for p in pairs_in_video)
    return 1 if c >= (len(pairs_in_video) - c) else 0


# ===========================================================================
# tiny, dependency-free logistic regression (L2, class-weighted, standardized)
# ===========================================================================

def _standardize_fit(X):
    mu = X.mean(0)
    sd = X.std(0)
    sd[sd < 1e-8] = 1.0
    return mu, sd


def fit_lr(X, y, l2=1.0, iters=1500, lr=0.3, class_weight=True):
    """Full-batch gradient-descent logistic regression. Deterministic (zero init)."""
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    if class_weight:
        pos = max(y.sum(), 1e-8)
        neg = max(n - y.sum(), 1e-8)
        wpos, wneg = n / (2 * pos), n / (2 * neg)
        sw = np.where(y == 1, wpos, wneg)
    else:
        sw = np.ones(n)
    for _ in range(iters):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        g = (p - y) * sw
        gw = X.T @ g / n + l2 * w / n
        gb = g.sum() / n
        w -= lr * gw
        b -= lr * gb
    return w, b


def predict_proba(X, w, b):
    return 1.0 / (1.0 + np.exp(-np.clip(X @ w + b, -30, 30)))


def save_final_model(pairs, mode, path, l2=1.0):
    """Fit the honest collaboration head on ALL pairs and save it for deployment.
    The head is a regularized logistic model over [6 relational signals (+4 engagement
    scalars)] sitting on top of the FROZEN engagement backbone -- i.e. exactly the
    Phase-2 'collaboration head', but small/robust enough not to collapse on this small,
    confounded dataset. Saves a self-contained .npz (mu, sd, w, b, mode, feature_names)."""
    X = np.stack([pair_vector(p, mode) for p in pairs])
    y = np.array([p["label"] for p in pairs])
    mu, sd = _standardize_fit(X)
    w, b = fit_lr((X - mu) / sd, y, l2=l2)
    names = list(SIGNAL_NAMES) if mode != "features" else []
    if mode != "signals":
        names = names + ["feat_cos", "feat_dist", "feat_magA", "feat_magB"]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, mu=mu, sd=sd, w=w, b=float(b), mode=mode,
             feature_names=np.array(names))
    print(f"\n[deploy] saved collaboration head -> {path}")
    print(f"[deploy] predict a new pair p:  x = pair_vector(p, '{mode}'); "
          f"prob_C = sigmoid(((x - mu) / sd) @ w + b)")


def save_session_model(by_vid_session, mode, path, l2=1.0):
    """Fit the SESSION / group head on ALL videos (video_vector aggregation = the honest
    claim, LOVO macro-F1 ~0.667 in 'both' mode) and save a self-contained .npz. This is
    the deployable group verdict head: session vector = [mean, std] of per-pair `mode`."""
    vids = sorted(by_vid_session.keys())
    X = np.stack([video_vector(by_vid_session[v], mode) for v in vids])
    y = np.array([video_label(by_vid_session[v]) for v in vids])
    mu, sd = _standardize_fit(X)
    w, b = fit_lr((X - mu) / sd, y, l2=l2)
    base = list(SIGNAL_NAMES) if mode != "features" else []
    if mode != "signals":
        base = base + ["feat_cos", "feat_dist", "feat_magA", "feat_magB"]
    names = [f"mean_{n}" for n in base] + [f"std_{n}" for n in base]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, mu=mu, sd=sd, w=w, b=float(b), mode=mode, level="session",
             feature_names=np.array(names))
    print(f"\n[deploy] saved SESSION/group head ({mode}, {len(vids)} videos) -> {path}")
    print(f"[deploy] session vector = [mean,std] of per-pair {mode} ({len(names)}-d); "
          f"predict: prob_C = sigmoid(((v - mu)/sd) @ w + b)")


# ===========================================================================
# metrics
# ===========================================================================

def metrics_majority(y_true, maj):
    y_pred = np.full_like(np.asarray(y_true), maj)
    f1s = []
    for cls in (0, 1):
        tp = int(((y_pred == cls) & (y_true == cls)).sum())
        fp = int(((y_pred == cls) & (y_true != cls)).sum())
        fn = int(((y_pred != cls) & (y_true == cls)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return sum(f1s) / 2.0


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out = {}
    out["acc"] = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    f1s = {}
    for cls in (0, 1):
        tp = int(((y_pred == cls) & (y_true == cls)).sum())
        fp = int(((y_pred == cls) & (y_true != cls)).sum())
        fn = int(((y_pred != cls) & (y_true == cls)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s[cls] = f1
    out["f1_N"], out["f1_C"] = f1s[0], f1s[1]
    out["macro_f1"] = (f1s[0] + f1s[1]) / 2.0
    # confusion [[TN,FP],[FN,TP]] with rows=true(N,C), cols=pred(N,C)
    out["cm"] = [
        [int(((y_true == 0) & (y_pred == 0)).sum()), int(((y_true == 0) & (y_pred == 1)).sum())],
        [int(((y_true == 1) & (y_pred == 0)).sum()), int(((y_true == 1) & (y_pred == 1)).sum())],
    ]
    n1 = int(y_true.sum())
    n0 = len(y_true) - n1
    maj = 1 if n1 >= n0 else 0
    out["baseline_acc"] = max(n0, n1) / len(y_true) if len(y_true) else 0.0
    out["baseline_macro_f1"] = metrics_majority(y_true, maj)
    return out


# ===========================================================================
# Q1: session-level Leave-One-Video-Out
# ===========================================================================

def lovo_video(by_vid, mode, l2=1.0, verbose=False):
    videos = sorted(by_vid.keys())
    X = np.stack([video_vector(by_vid[v], mode) for v in videos])
    y = np.array([video_label(by_vid[v]) for v in videos])
    preds = np.zeros(len(videos), dtype=int)
    for i, v in enumerate(videos):
        tr = [j for j in range(len(videos)) if j != i]
        Xtr, ytr = X[tr], y[tr]
        mu, sd = _standardize_fit(Xtr)
        w, b = fit_lr((Xtr - mu) / sd, ytr, l2=l2)
        p = predict_proba((X[i:i + 1] - mu) / sd, w, b)[0]
        preds[i] = int(p >= 0.5)
    m = metrics(y, preds)
    if verbose:
        for i, v in enumerate(videos):
            mark = "ok " if preds[i] == y[i] else "XX "
            print(f"    {mark}{v:22} true={'C' if y[i] else 'N'} pred={'C' if preds[i] else 'N'}")
    return m, y, preds, videos


# ===========================================================================
# Q2: pair-level Leave-One-Video-Out (deployment)
# ===========================================================================

def lovo_pair(by_vid, mode, l2=1.0, augment=False, per_video=False):
    """Train on all-but-one video, predict the held-out video's pairs, pool over folds.
    augment=True adds the A<->B swap to the TRAIN set only (undirected-pair symmetry),
    matching ceiling_probe's treatment of the full768 modes.
    per_video=True also prints the held-out macro-F1 per video (signal-breadth check)."""
    videos = sorted(by_vid.keys())
    all_true, all_pred = [], []
    per = []
    for v in videos:
        test = by_vid[v]
        train = [p for vv in videos if vv != v for p in by_vid[vv]]
        Xtr = np.stack([pair_vector(p, mode) for p in train])
        ytr = np.array([p["label"] for p in train])
        if augment:
            Xtr = np.concatenate(
                [Xtr, np.stack([pair_vector(p, mode, swap=True) for p in train])], axis=0)
            ytr = np.concatenate([ytr, ytr])
        Xte = np.stack([pair_vector(p, mode) for p in test])
        yte = np.array([p["label"] for p in test])
        mu, sd = _standardize_fit(Xtr)
        w, b = fit_lr((Xtr - mu) / sd, ytr, l2=l2)
        pr = predict_proba((Xte - mu) / sd, w, b)
        ypv = (pr >= 0.5).astype(int)
        all_true.extend(yte.tolist())
        all_pred.extend(ypv.tolist())
        if per_video:
            per.append((v, len(test), metrics(yte, ypv)["macro_f1"]))
    if per_video:
        print("    per-held-out-video macro-F1 (broad signal vs a few lucky scenes?):")
        above = 0
        for v, n, f1 in per:
            if f1 > 0.5:
                above += 1
            print(f"      {'ok ' if f1 > 0.5 else '   '}{v:24} n={n:3d}  macro-F1 {f1:.3f}")
        print(f"    -> {above}/{len(per)} held-out videos above 0.50 "
              f"(median {np.median([f for _, _, f in per]):.3f})")
    return metrics(all_true, all_pred), np.array(all_true), np.array(all_pred)


def _shuffle_labels(by_vid, rng):
    """Copy of by_vid with pair labels globally permuted (breaks the pair<->label link,
    preserves class balance) -> empirical chance floor for the pooled-pair LOVO metric."""
    flat = [(v, p) for v in by_vid for p in by_vid[v]]
    perm = rng.permutation([p["label"] for _, p in flat])
    out = defaultdict(list)
    for (v, p), y in zip(flat, perm):
        q = dict(p); q["label"] = int(y); out[v].append(q)
    return out


def validate_pair(by_vid, mode="full768+sig", l2=10.0, augment=True, n_shuffle=5, seed=0):
    """The leakage/significance gate the pooled-pair LOVO was missing: empirical shuffle
    floor, L2-stability sweep, per-video breadth. Run BEFORE committing to Stage 3."""
    print("\n" + "-" * 78)
    print(f"VALIDATION GATE  (pair LOVO, mode={mode}, L2={l2:g}, A<->B aug)")
    print("-" * 78)
    m, _, _ = lovo_pair(by_vid, mode, l2=l2, augment=augment, per_video=True)
    print(f"  real macro-F1        = {m['macro_f1']:.3f}  (acc {m['acc'] * 100:.1f}%)")
    rng = np.random.default_rng(seed)
    sh = [lovo_pair(_shuffle_labels(by_vid, rng), mode, l2=l2, augment=augment)[0]["macro_f1"]
          for _ in range(n_shuffle)]
    floor = float(np.mean(sh))
    verdict = "ABOVE chance" if m["macro_f1"] > floor + 0.03 else "NOT clearly above chance"
    print(f"  label-shuffled floor = {floor:.3f} +/- {np.std(sh):.3f}  -> real is {verdict}")
    print("  L2 stability sweep (real labels):")
    for l2v in (1.0, 3.0, 10.0, 30.0):
        mm, _, _ = lovo_pair(by_vid, mode, l2=l2v, augment=augment)
        print(f"    L2={l2v:>4g}: macro-F1 {mm['macro_f1']:.3f}")
    print("-" * 78)


# ===========================================================================
# reporting
# ===========================================================================

def _print_block(title, m):
    print(f"\n  {title}")
    print(f"    accuracy   {m['acc'] * 100:5.1f}%   (majority baseline {m['baseline_acc'] * 100:5.1f}%)")
    print(f"    macro-F1   {m['macro_f1']:.3f}    (majority baseline {m['baseline_macro_f1']:.3f})")
    print(f"    F1_N {m['f1_N']:.3f}   F1_C {m['f1_C']:.3f}")
    print(f"    confusion (rows=true N,C / cols=pred N,C): {m['cm']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/collab_cache/feature_index_33.csv")
    ap.add_argument("--cache", default="data/collab_cache")
    ap.add_argument("--npz", default="",
                    help="OPT-IN: load pooled features straight from a pairs_features.npz "
                         "(e.g. data/collab_pairs_unique_fresh/pairs_features.npz). Empty = read "
                         "--index/--cache via load_pairs. Default is empty ON PURPOSE so this "
                         "never silently reads a stale npz -- the exact bug that fooled "
                         "ceiling_probe's first 'fresh' run.")
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--min_pairs", type=int, default=3,
                    help="drop videos with fewer than this many pairs (too noisy to aggregate)")
    ap.add_argument("--save_final", default="",
                    help="if set, fit the head on ALL pairs and save to this .npz path "
                         "(e.g. weights/collab_head_lr.npz)")
    ap.add_argument("--final_mode", default="both", choices=["signals", "features", "both"],
                    help="feature set for the saved deployment head")
    ap.add_argument("--validate", action="store_true",
                    help="run the pair-level significance/stability gate (shuffle floor + "
                         "L2 sweep + per-video breadth) on full768+sig before any Stage-3 build")
    ap.add_argument("--save_session", default="",
                    help="if set, fit the SESSION/group head on ALL videos and save to this "
                         ".npz (the honest deliverable; e.g. weights/best_collab_group_fresh.npz)")
    ap.add_argument("--session_mode", default="both", choices=["signals", "features", "both"],
                    help="feature set for the saved SESSION head (default both = LOVO ~0.667)")
    args = ap.parse_args()

    print("=" * 78)
    print("HONEST LEAVE-ONE-VIDEO-OUT EVALUATION  (no scene can leak: whole video held out)")
    print("=" * 78)

    if args.npz:
        if not os.path.exists(args.npz):
            print(f"[error] --npz {args.npz} not found"); sys.exit(1)
        print(f"[load] reading pooled features directly from npz: {args.npz}")
        pairs = load_pairs_from_npz(args.npz)
    else:
        pairs = load_pairs(args.index, args.cache, drop_label_conflicts=True, verbose=True)
    by_vid = defaultdict(list)
    for p in pairs:
        by_vid[p["video"]].append(p)
    by_vid_session = {v: ps for v, ps in by_vid.items() if len(ps) >= args.min_pairs}

    nC = sum(p["label"] for p in pairs)
    print(f"\nDataset: {len(pairs)} unique pairs over {len(by_vid)} videos "
          f"(C={nC} N={len(pairs) - nC}).")
    nvidC = sum(video_label(ps) for ps in by_vid_session.values())
    print(f"Session-level: {len(by_vid_session)} videos with >= {args.min_pairs} pairs "
          f"(C-sessions={nvidC} N-sessions={len(by_vid_session) - nvidC}).")

    # ---- Q1: session level ----
    print("\n" + "-" * 78)
    print("Q1  SESSION-LEVEL  -- classify a whole UNSEEN video (LOVO).  Ablation:")
    print("-" * 78)
    results_session = {}
    for mode in ("features", "signals", "both"):
        m, y, preds, vids = lovo_video(by_vid_session, mode, l2=args.l2,
                                       verbose=(mode == "signals"))
        results_session[mode] = m
        _print_block(f"[{mode:8}]", m)

    # label-shuffled control on the combined arm
    rng = np.random.default_rng(0)
    vids = sorted(by_vid_session.keys())
    Xb = np.stack([video_vector(by_vid_session[v], "both") for v in vids])
    yb = np.array([video_label(by_vid_session[v]) for v in vids])
    shuf_scores = []
    for s in range(5):
        ys = rng.permutation(yb)
        preds = np.zeros(len(vids), dtype=int)
        for i in range(len(vids)):
            tr = [j for j in range(len(vids)) if j != i]
            mu, sd = _standardize_fit(Xb[tr])
            w, b = fit_lr((Xb[tr] - mu) / sd, ys[tr], l2=args.l2)
            preds[i] = int(predict_proba((Xb[i:i + 1] - mu) / sd, w, b)[0] >= 0.5)
        shuf_scores.append(metrics(ys, preds)["macro_f1"])
    shuf_mean = float(np.mean(shuf_scores))
    print(f"\n  [control] label-shuffled macro-F1 = {shuf_mean:.3f} "
          f"+/- {np.std(shuf_scores):.3f}  (real 'both' = {results_session['both']['macro_f1']:.3f})")
    print("            -> real must beat shuffled for the result to be meaningful.")

    # ---- Q2: pair level (deployment) ----
    print("\n" + "-" * 78)
    print("Q2  PAIR-LEVEL / DEPLOYMENT -- predict pairs of an UNSEEN video (LOVO, pooled).")
    print("-" * 78)
    results_pair = {}
    for mode in ("features", "signals", "both"):
        m, _, _ = lovo_pair(by_vid, mode, l2=args.l2)
        results_pair[mode] = m
        _print_block(f"[{mode:8}]", m)

    # Honest-LOVO counterpart of the within-scene ceiling: do the RAW fresh features
    # [A|B] generalize to UNSEEN scenes/people, or was the 0.79 within-scene ceiling just
    # identity memorization (same people in train+test folds)?  A<->B augmented, L2=10
    # (the value the ceiling probe picked for these modes).
    print("\n  -- raw fresh features [A|B] under the SAME honest LOVO (the real test) --")
    for mode in ("full768", "full768+sig"):
        m, _, _ = lovo_pair(by_vid, mode, l2=10.0, augment=True)
        results_pair[mode] = m
        _print_block(f"[{mode:11}]", m)

    if args.validate:
        validate_pair(by_vid, mode="full768+sig", l2=10.0, augment=True, n_shuffle=5, seed=0)

    # ---- verdict ----
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    sb = results_session["both"]
    sf = results_session["features"]
    ss = results_session["signals"]
    lift = ss["macro_f1"] - sf["macro_f1"]
    pb = max(results_pair.values(), key=lambda m: m["macro_f1"])
    print(f"  Session-level best macro-F1 = {max(sb['macro_f1'], ss['macro_f1'], sf['macro_f1']):.3f} "
          f"vs baseline {sb['baseline_macro_f1']:.3f}.")
    print(f"  Pair-level (deployment) best macro-F1 = {pb['macro_f1']:.3f} "
          f"(acc {pb['acc'] * 100:.1f}%) vs baseline {pb['baseline_macro_f1']:.3f} "
          f"({pb['baseline_acc'] * 100:.1f}%).")
    f768 = results_pair.get("full768+sig") or results_pair.get("full768")
    if f768 is not None:
        gen = f768["macro_f1"] > f768["baseline_macro_f1"] + 0.03
        print(f"  full768 (raw feats) LOVO macro-F1 = {f768['macro_f1']:.3f} "
              f"vs baseline {f768['baseline_macro_f1']:.3f}  "
              f"-> raw features {'GENERALIZE across scenes' if gen else 'do NOT generalize'} "
              f"(within-scene ceiling was ~0.79; any gap = scene/identity leakage).")
    print(f"  Signals vs features lift (session) = {lift:+.3f}  "
          f"-> {'signals carry GENERALIZABLE collab info' if lift > 0.0 else 'signals tie/loss vs features'}.")
    print(f"  Label-shuffled control = {shuf_mean:.3f} "
          f"-> real is {'ABOVE' if sb['macro_f1'] > shuf_mean + 0.03 else 'NOT clearly above'} chance.")
    print("=" * 78)

    if args.save_final:
        save_final_model(pairs, args.final_mode, args.save_final, l2=args.l2)

    if args.save_session:
        save_session_model(by_vid_session, args.session_mode, args.save_session, l2=args.l2)


if __name__ == "__main__":
    main()
