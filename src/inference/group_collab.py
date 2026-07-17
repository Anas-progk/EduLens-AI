"""
group_collab.py -- Phase-2 GROUP-LEVEL collaboration head (honest, deployable).

WHY GROUP-LEVEL  (read this before changing anything)
-----------------------------------------------------
Leak-free Leave-One-Video-Out (LOVO) on the FULL 33-video cache, re-run on the FRESH
per-person features, settled what is honestly learnable here:

  * WITHIN-SCENE PAIR level ("which two people in this room collaborate?") -> CHANCE.
      On the only balanced scene (VID_(4)) honest-LOVO is 0.544; per-video median is
      0.442 (6/33 videos > 0.50). The pooled pair score (0.608) only *looks* above
      chance because it is the SESSION signal in disguise -- each scene is near
      single-class, so "guess the video's majority and stamp all its pairs" wins when
      pooled. Per-person *appearance* features cannot separate pairs that coexist in
      one frame (that information is geometric, not appearance).

  * GROUP / SESSION level ("is this group collaborating?") -> REAL.
      Aggregating per-pair statistics over a whole video and asking one yes/no question
      per session gives macro-F1 0.667 (66.7% acc) under LOVO on fresh features, vs
      majority baseline 0.348 and a label-shuffled floor 0.396. Generalizes to unseen
      rooms. This is the honest, deployable Phase-2 product.

ARCHITECTURE (still Swin-transformer-centered)
----------------------------------------------
    frames --> [frozen Swin-Tiny + TemporalTransformer = Phase-1 backbone]
            --> 768-d feature per person per frame
            --> per pair: 6 relational signals (data.collab_pairs._compute_signals)
                          + 4 feature scalars [cos, dist, |A|, |B|] of the pooled feats
            --> session aggregate = [mean,std] over the group's pairs
                 'both' mode -> [meanSig(6), stdSig(6), meanFeat(4), stdFeat(4)] = 20-d
            --> THIS small regularized logistic read-out --> Collaborative? yes/no

The deep model is the Swin backbone; the group read-out is deliberately small (only ~30
sessions -- a heavier head would memorize sessions, not generalize). Aggregation here is
byte-identical to train_collab_video_level.video_vector, so DEPLOY == TRAIN.

Self-contained at inference (numpy only). 768-d features come from the engagement model.
"""

import os
import sys
import argparse
import numpy as np
from collections import deque, defaultdict

# --- make data.collab_pairs importable whether run from repo root, src/, or src/inference/
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)              # .../src
_ROOT = os.path.dirname(_SRC)              # repo root
for _p in (_SRC, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from data.collab_pairs import _compute_signals, SIGNAL_NAMES, load_pairs
except ImportError:  # running from repo root with src. prefix
    from src.data.collab_pairs import _compute_signals, SIGNAL_NAMES, load_pairs

VERDICT_C = "COLLABORATIVE"
VERDICT_N = "NOT COLLABORATIVE"
VERDICT_UNKNOWN = "UNKNOWN"

FEAT_SCALAR_NAMES = ["feat_cos", "feat_dist", "feat_magA", "feat_magB"]


# ===========================================================================
# per-pair feature scalars + session aggregation
# (MUST match train_collab_video_level._pair_feature_scalars / video_vector)
# ===========================================================================

def _scalars_from(a, b):
    """[cos, dist, |a|, |b|] for two pooled 768-d vectors -- the 'features' arm."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    return np.array([float(a @ b / (na * nb + 1e-8)), float(np.linalg.norm(a - b)),
                     float(na), float(nb)], dtype=np.float64)


def session_vector(pair_records, mode):
    """Aggregate a list of per-pair records into the session vector.

    pair_records: list of dicts with 'signals' (6-d) and, for feature modes,
    'pooled_A'/'pooled_B' (768-d pooled feature of each person).
      signals  -> [mean(6), std(6)]                                   = 12-d
      features -> [mean(4), std(4)]                                   =  8-d
      both     -> [meanSig(6), stdSig(6), meanFeat(4), stdFeat(4)]    = 20-d
    Byte-identical to train_collab_video_level.video_vector.
    """
    S = np.stack([np.asarray(r["signals"], dtype=np.float64).ravel() for r in pair_records])
    if mode in ("signals", "signals_session"):
        return np.concatenate([S.mean(0), S.std(0)])
    F = np.stack([_scalars_from(r["pooled_A"], r["pooled_B"]) for r in pair_records])
    if mode == "features":
        return np.concatenate([F.mean(0), F.std(0)])
    return np.concatenate([S.mean(0), S.std(0), F.mean(0), F.std(0)])  # both


def session_vector_from_signals(pair_signals):
    """Legacy signals-only aggregation (12-d). Kept for back-compat with old heads."""
    return session_vector([{"signals": s} for s in pair_signals], "signals")


def vector_names(mode):
    sm = [f"mean_{n}" for n in SIGNAL_NAMES] + [f"std_{n}" for n in SIGNAL_NAMES]
    fm = [f"mean_{n}" for n in FEAT_SCALAR_NAMES] + [f"std_{n}" for n in FEAT_SCALAR_NAMES]
    if mode in ("signals", "signals_session"):
        return sm
    if mode == "features":
        return fm
    return sm + fm


# ===========================================================================
# tiny, dependency-free logistic regression (identical math to the trainer)
# ===========================================================================

def _standardize_fit(X):
    mu = X.mean(0)
    sd = X.std(0)
    sd[sd < 1e-8] = 1.0
    return mu, sd


def _fit_lr(X, y, l2=1.0, iters=2000, lr=0.3, class_weight=True):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    if class_weight:
        pos = max(y.sum(), 1e-8)
        neg = max(n - y.sum(), 1e-8)
        sw = np.where(y == 1, n / (2 * pos), n / (2 * neg))
    else:
        sw = np.ones(n)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ w + b, -30, 30)))
        g = (p - y) * sw
        w -= lr * (X.T @ g / n + l2 * w / n)
        b -= lr * (g.sum() / n)
    return w, b


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _macro_f1(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    f1s = []
    for cls in (0, 1):
        tp = int(((y_pred == cls) & (y_true == cls)).sum())
        fp = int(((y_pred == cls) & (y_true != cls)).sum())
        fn = int(((y_pred != cls) & (y_true == cls)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return sum(f1s) / 2.0


# ===========================================================================
# the deployable head
# ===========================================================================

class GroupCollabHead:
    """A saved, self-contained group-level collaboration classifier.

    prob_C = sigmoid( ((session_vector - mu) / sd) @ w + b )
    where session_vector is built by `session_vector(pairs, mode)`.
    """

    _VEC_MODES = ("signals", "signals_session", "features", "both")

    def __init__(self, mu, sd, w, b, mode="both", feature_names=None):
        self.mu = np.asarray(mu, dtype=np.float64)
        self.sd = np.asarray(sd, dtype=np.float64)
        self.w = np.asarray(w, dtype=np.float64)
        self.b = float(b)
        self.mode = str(mode)
        self.feature_names = list(feature_names) if feature_names is not None else vector_names(self.mode)

    def _vec_mode(self):
        return self.mode if self.mode in self._VEC_MODES else "signals"

    # ---- io ----
    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=True)
        mode = str(d["mode"]) if "mode" in d.files else "both"
        names = None
        for k in ("feature_names", "signal_names"):
            if k in d.files:
                names = list(d[k])
                break
        return cls(d["mu"], d["sd"], d["w"], float(d["b"]), mode=mode, feature_names=names)

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez(path, mu=self.mu, sd=self.sd, w=self.w, b=self.b,
                 mode=self.mode, level="session",
                 feature_names=np.array(self.feature_names))

    # ---- scoring ----
    def prob_from_session_vector(self, x):
        x = np.asarray(x, dtype=np.float64).ravel()
        return float(_sigmoid(((x - self.mu) / self.sd) @ self.w + self.b))

    def prob_from_pairs(self, pair_records):
        """pair_records: list of dicts with 'signals' and (for feature/both) pooled_A/B."""
        if not pair_records:
            return None
        return self.prob_from_session_vector(session_vector(pair_records, self._vec_mode()))

    def prob_from_pair_signals(self, pair_signals):
        """Legacy entry: list of 6-d signal vectors (signals-only heads)."""
        if not pair_signals:
            return None
        return self.prob_from_pairs([{"signals": s} for s in pair_signals])

    def predict(self, pair_records, thresh=0.5):
        """Return a verdict dict from a list of per-pair records."""
        p = self.prob_from_pairs(pair_records)
        if p is None:
            return {"prob": None, "verdict": VERDICT_UNKNOWN, "confidence": 0.0, "n_pairs": 0}
        return {"prob": p,
                "verdict": VERDICT_C if p >= thresh else VERDICT_N,
                "confidence": float(abs(p - 0.5) * 2.0),
                "n_pairs": len(pair_records)}


# ===========================================================================
# live aggregator -- feed it per-person 768-d Swin features, get a group verdict
# ===========================================================================

class LiveGroupCollab:
    """Rolling per-person feature buffers -> live group collaboration verdict.

    Usage in the inference loop (per frame):
        for track_id, feat768 in current_people.items():
            live.update(track_id, feat768)        # feat768 from the engagement model
        result = live.verdict()                   # {'verdict','prob','confidence','n_pairs'}
    Call live.drop(track_id) when a person leaves so stale buffers don't pollute pairs.

    Pair signals use data.collab_pairs._compute_signals and the pooled-feature scalars --
    the EXACT features used in training -- so the live verdict is train-identical.
    """

    def __init__(self, head, window=30, min_frames=3, thresh=0.5):
        self.head = head
        self.window = int(window)
        self.min_frames = int(min_frames)
        self.thresh = float(thresh)
        self._buf = defaultdict(lambda: deque(maxlen=self.window))

    def update(self, track_id, feat_768):
        self._buf[track_id].append(np.asarray(feat_768, dtype=np.float64).ravel())

    def drop(self, track_id):
        self._buf.pop(track_id, None)

    def reset(self):
        self._buf.clear()

    def active_tracks(self):
        return [t for t, b in self._buf.items() if len(b) >= self.min_frames]

    def pair_records(self):
        ids = sorted(self.active_tracks(), key=lambda x: (len(str(x)), str(x)))
        recs = []
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                A = np.stack(list(self._buf[ids[i]]))
                B = np.stack(list(self._buf[ids[j]]))
                T = min(len(A), len(B))
                if T < self.min_frames:
                    continue
                recs.append({
                    "signals": np.asarray(_compute_signals(A[:T], B[:T]), dtype=np.float64),
                    "pooled_A": A[:T].mean(0),
                    "pooled_B": B[:T].mean(0),
                })
        return recs

    def verdict(self):
        return self.head.predict(self.pair_records(), thresh=self.thresh)


# ===========================================================================
# data loading (npz preferred; cache fallback) + session dataset
# ===========================================================================

def load_pairs_from_npz(path):
    """List-of-dicts pairs from a pairs_features.npz (pooled_A/B, signals, labels, video_ids)."""
    d = np.load(path, allow_pickle=True)
    A, B, S = d["pooled_A"], d["pooled_B"], d["signals"]
    y = d["labels"].astype(int)
    v = d["video_ids"].astype(str)
    return [{"signals": S[i].astype(np.float64),
             "pooled_A": A[i].astype(np.float64), "pooled_B": B[i].astype(np.float64),
             "label": int(y[i]), "video": str(v[i])} for i in range(len(y))]


def _load_pairs(args, verbose=False):
    if getattr(args, "npz", "") and os.path.exists(args.npz):
        return load_pairs_from_npz(args.npz)
    return load_pairs(args.index, args.cache, drop_label_conflicts=True, verbose=verbose)


def _group_by_video(pairs, min_pairs=1):
    by = defaultdict(list)
    for p in pairs:
        by[p["video"]].append(p)
    return {v: ps for v, ps in by.items() if len(ps) >= min_pairs}


def _build_session_dataset(pairs, mode, min_pairs=3):
    by = _group_by_video(pairs, min_pairs=min_pairs)
    vids = sorted(by.keys())
    X = np.stack([session_vector(by[v], mode) for v in vids])

    def _lbl(ps):
        c = sum(p["label"] for p in ps)
        return 1 if c >= (len(ps) - c) else 0

    y = np.array([_lbl(by[v]) for v in vids])
    return vids, X, y, by


def lovo_report(X, y, l2=1.0):
    """Leave-One-Video-Out sanity check on the session dataset."""
    preds = np.zeros(len(y), dtype=int)
    for i in range(len(y)):
        tr = [j for j in range(len(y)) if j != i]
        mu, sd = _standardize_fit(X[tr])
        w, b = _fit_lr((X[tr] - mu) / sd, y[tr], l2=l2)
        preds[i] = int(_sigmoid(((X[i] - mu) / sd) @ w + b) >= 0.5)
    acc = float((preds == y).mean())
    return acc, _macro_f1(y, preds), preds


def fit_group_head(pairs, out_path, mode="both", l2=1.0, min_pairs=3, report=True):
    """Fit the deployable group head on ALL sessions; print the honest LOVO number."""
    vids, X, y, by = _build_session_dataset(pairs, mode, min_pairs=min_pairs)
    nC = int(y.sum())
    if report:
        acc, f1, _ = lovo_report(X, y, l2=l2)
        base = max(len(y) - nC, nC) / len(y)
        print(f"\n[group head] mode={mode}  sessions={len(y)} (C={nC} N={len(y)-nC})  min_pairs={min_pairs}")
        print(f"[group head] HONEST LOVO  acc={acc*100:.1f}%  macro-F1={f1:.3f}  "
              f"(majority baseline acc={base*100:.1f}%)")
    mu, sd = _standardize_fit(X)
    w, b = _fit_lr((X - mu) / sd, y, l2=l2)
    head = GroupCollabHead(mu, sd, w, b, mode=mode, feature_names=vector_names(mode))
    head.save(out_path)
    if report:
        print(f"[group head] fit on all {len(y)} sessions -> saved {out_path}")
    return head


def score_video(head, by, video):
    """Score ONE video from a pre-grouped {video: [pair_records]} dict."""
    if video not in by:
        return None
    recs = by[video]
    res = head.predict(recs, thresh=0.5)
    nC = sum(p["label"] for p in recs)
    res["true_label"] = VERDICT_C if nC >= (len(recs) - nC) else VERDICT_N
    return res


# ===========================================================================
# CLI
# ===========================================================================

def _default_npz():
    for p in ("data/collab_pairs_unique_fresh/pairs_features.npz",
              "data/collab_pairs_unique/pairs_features.npz"):
        if os.path.exists(p):
            return p
    return ""


def _default_paths():
    cands = [
        ("data/collab_cache_fresh/feature_index.csv", "data/collab_cache_fresh"),
        ("data/collab_cache/feature_index_33.csv", "data/collab_cache"),
    ]
    for idx, cache in cands:
        if os.path.exists(idx):
            return idx, cache
    return cands[0]


def main():
    di, dc = _default_paths()
    ap = argparse.ArgumentParser(description="Phase-2 group-level collaboration head (fresh, both-recipe)")
    ap.add_argument("--npz", default=_default_npz(),
                    help="pairs_features.npz to load (preferred). Default = fresh npz if present.")
    ap.add_argument("--index", default=di)
    ap.add_argument("--cache", default=dc)
    ap.add_argument("--fit", action="store_true", help="fit + save the group head")
    ap.add_argument("--mode", default="both", choices=["signals", "features", "both"])
    ap.add_argument("--out", default="weights/best_collab_group_fresh.npz")
    ap.add_argument("--head", default="weights/best_collab_group_fresh.npz")
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--min_pairs", type=int, default=3)
    ap.add_argument("--score_video", default="", help="score ONE video id")
    ap.add_argument("--all", action="store_true", help="score every session + print the LOVO headline")
    args = ap.parse_args()

    if args.fit:
        pairs = _load_pairs(args, verbose=True)
        fit_group_head(pairs, args.out, mode=args.mode, l2=args.l2, min_pairs=args.min_pairs, report=True)
        return

    if args.all:
        pairs = _load_pairs(args)
        head = GroupCollabHead.load(args.head)
        by1 = _group_by_video(pairs, min_pairs=1)
        print(f"\nDeployed head: {args.head}  (mode={head.mode}, {len(head.w)}-d)")
        print(f"{'video':16} {'pairs':>5}  {'P(C)':>5}  predicted            true")
        for v in sorted(by1.keys()):
            res = score_video(head, by1, v)
            print(f"{v:16} {res['n_pairs']:5d}  {res['prob']:.3f}  "
                  f"{res['verdict']:20} {res['true_label']}")
        _, X, y, _ = _build_session_dataset(pairs, head._vec_mode(), min_pairs=args.min_pairs)
        acc, f1, _ = lovo_report(X, y, l2=args.l2)
        print(f"\n[HONEST LOVO headline]  macro-F1 = {f1:.3f}   acc = {acc*100:.1f}%   "
              f"sessions = {len(y)}  (this is the number to defend)")
        return

    if args.score_video:
        pairs = _load_pairs(args)
        head = GroupCollabHead.load(args.head)
        by1 = _group_by_video(pairs, min_pairs=1)
        res = score_video(head, by1, args.score_video)
        if res is None:
            print(f"no pairs for video {args.score_video!r}")
            return
        print(f"video={args.score_video!r}  pairs={res['n_pairs']}")
        print(f"  predicted : {res['verdict']}  (P(collab)={res['prob']:.3f}, "
              f"confidence={res['confidence']:.2f})")
        print(f"  true label: {res['true_label']}")
        return

    ap.print_help()


if __name__ == "__main__":
    main()
