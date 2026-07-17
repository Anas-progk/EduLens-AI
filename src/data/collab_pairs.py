"""
collab_pairs.py -- Honest, de-duplicated, pair-level dataset builder for Phase 2.

WHY THIS FILE EXISTS (read before changing anything)
----------------------------------------------------
The previous pipeline trained on CLIP-level rows with two hidden problems:

  1. DUPLICATION (your annotation instinct was right):
     - Each person-pair appears as ~5-6 near-duplicate clips (different frames).
     - 325 of the relationships are stored as BOTH (A,B) and (B,A) -> double counted.
     - Net: 6,869 "clips" -> 1,209 directed pairs -> only 884 UNIQUE relationships.
     This file collapses everything to the 884 unique undirected pairs. One pair =
     one training sample (not 5-6). That is the honest unit.

  2. DEAD INTERACTION SIGNALS:
     - The 6-d interaction signals were constant 0.5 at train time, so the only
       thing that varied per sample was the two engagement features -- which are
       correlated with WHICH VIDEO the pair is from. With a label that is ~83%
       determined by the video (the scene confound), the optimal thing for the
       model to do was memorize the scene. That is exactly what happened
       (val macro-F1 == scene-only baseline, to the decimal).

THE FIX HERE:
  - Collapse to unique undirected pairs (kills duplication + double counting).
  - Compute REAL relational signals from the per-frame feature TIME-SERIES that
    already exists in the cache (each pair has multiple clips at f0000, f0004,...).
    These signals describe the RELATIONSHIP between the two people over time
    (do their states move together? do they take turns? are they jointly active?)
    and are NOT a function of video identity -> they resist the scene confound.
  - Spatial signals (proximity, facing) need bbox trajectories that were NOT
    saved during processing, so they are left neutral (0.5) and flagged. They are
    the documented next upgrade (requires re-running the detector to log bboxes).

This module has NO torch dependency so it can be unit-tested without a GPU.
train_collab_honest.py consumes it.
"""

import os
import csv
import numpy as np
from collections import defaultdict

NEUTRAL = 0.5
SIGNAL_NAMES = [
    "state_cos",      # 0: cosine(pooled_A, pooled_B)            -- similar overall state
    "state_close",    # 1: exp(-||pA-pB||/scale)                 -- closeness in feature space
    "traj_cos",       # 2: mean_t cosine(A_t, B_t)               -- moment-to-moment co-state
    "dyn_corr",       # 3: corr_t(activityA_t, activityB_t)      -- synchronized dynamics
    "turn_taking",    # 4: anti-corr of activity CHANGES         -- one acts then the other
    "joint_active",   # 5: mean_t min(actA, actB)                -- both engaged at once
]
SPATIAL_UNAVAILABLE = {"proximity", "facing"}  # need bbox; not in cache (documented gap)


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def _norm_path(p):
    return os.path.basename(p.replace("\\", "/"))


def _activity(seq):
    """Per-frame 'activity' = L2 distance of each frame-feature from the clip mean.
    Captures how much the person's state moves over the clip (a behaviour proxy)."""
    if len(seq) == 0:
        return np.zeros(0)
    m = seq.mean(0, keepdims=True)
    return np.linalg.norm(seq - m, axis=1)


def _safe_corr(a, b):
    if len(a) < 3 or len(b) < 3:
        return None
    sa, sb = a.std(), b.std()
    if sa < 1e-8 or sb < 1e-8:
        return None
    return float(np.clip(np.corrcoef(a, b)[0, 1], -1, 1))


def _compute_signals(A_seq, B_seq):
    """Compute the 6-d relational signal vector from two aligned feature time-series.
    A_seq, B_seq: (T, 768) ordered by frame. Returns float list of length 6 in [0,1]."""
    pA, pB = A_seq.mean(0), B_seq.mean(0)

    # 0: state cosine
    cos = float(np.dot(pA, pB) / (np.linalg.norm(pA) * np.linalg.norm(pB) + 1e-8))
    state_cos = (cos + 1.0) / 2.0  # -> [0,1]

    # 1: closeness (scale by typical 768-d distance ~ sqrt(dim))
    d = float(np.linalg.norm(pA - pB))
    state_close = float(np.exp(-d / 20.0))

    T = min(len(A_seq), len(B_seq))
    if T >= 3:
        At, Bt = A_seq[:T], B_seq[:T]
        # 2: trajectory cosine (mean per-frame cosine)
        num = (At * Bt).sum(1)
        den = np.linalg.norm(At, axis=1) * np.linalg.norm(Bt, axis=1) + 1e-8
        traj_cos = float(((num / den).mean() + 1.0) / 2.0)
        # activity series + dynamics
        actA, actB = _activity(At), _activity(Bt)
        c = _safe_corr(actA, actB)
        dyn_corr = NEUTRAL if c is None else (c + 1.0) / 2.0
        # 4: turn-taking = anti-correlation of activity CHANGES
        dA, dB = np.diff(actA), np.diff(actB)
        ct = _safe_corr(dA, dB)
        turn_taking = NEUTRAL if ct is None else (1.0 - (ct + 1.0) / 2.0)
        # 5: joint activity (normalize activities to [0,1] by their own max, then min)
        na = actA / (actA.max() + 1e-8)
        nb = actB / (actB.max() + 1e-8)
        joint_active = float(np.minimum(na, nb).mean())
    else:
        traj_cos = dyn_corr = turn_taking = joint_active = NEUTRAL

    return [
        float(np.clip(state_cos, 0, 1)),
        float(np.clip(state_close, 0, 1)),
        float(np.clip(traj_cos, 0, 1)),
        float(np.clip(dyn_corr, 0, 1)),
        float(np.clip(turn_taking, 0, 1)),
        float(np.clip(joint_active, 0, 1)),
    ]


def load_pairs(feature_index_csv, cache_dir, drop_label_conflicts=True,
               videos=None, verbose=True):
    """
    Build UNIQUE UNDIRECTED person-pairs from a clip-level feature_index.

    Returns list of dict:
      { 'video', 'a', 'b', 'label' (0/1), 'pooled_A' (768,), 'pooled_B' (768,),
        'signals' (6,), 'n_clips' }

    Orientation: A is always the smaller track_id (canonical) so (A,B) and (B,A)
    collapse to one sample. Label = majority vote across all clips of both directions.

    videos: optional iterable of video_ids to restrict to (speeds up testing).
    """
    rows = list(csv.DictReader(open(feature_index_csv, newline="", encoding="utf-8")))
    present = set(f for f in os.listdir(cache_dir) if f.endswith(".npy"))
    vid_filter = set(videos) if videos else None

    # gather, per undirected pair: time-ordered (frame -> featLow, featHigh) where
    # "low" = smaller-id person's feature, "high" = larger-id person's feature.
    store = defaultdict(lambda: {"low": {}, "high": {}, "labels": [], "video": None})
    for r in rows:
        vid = r["video_id"]
        if vid_filter is not None and vid not in vid_filter:
            continue
        lab = (r.get("label") or "").strip()
        a, b = r["track_id_A"], r["track_id_B"]
        key = (vid,) + tuple(sorted([a, b], key=lambda x: (len(x), x)))
        d = store[key]
        d["video"] = vid
        if lab in ("C", "N"):
            d["labels"].append(lab)
        # frame index from pair_id tail (..._fXXXX) or fallback to row order
        pid = r.get("pair_id", "")
        frame = pid.split("_")[-1] if "_f" in pid else f"f{len(d['low'])+len(d['high']):04d}"
        bn_a, bn_b = _norm_path(r["feat_A"]), _norm_path(r["feat_B"])
        if bn_a not in present or bn_b not in present:
            continue
        fa = os.path.join(cache_dir, bn_a)
        fb = os.path.join(cache_dir, bn_b)
        # map A/B onto low/high by canonical order
        if sorted([a, b], key=lambda x: (len(x), x))[0] == a:
            d["low"][frame] = fa
            d["high"][frame] = fb
        else:
            d["low"][frame] = fb
            d["high"][frame] = fa

    pairs, conflicts, missing = [], 0, 0
    for key, d in store.items():
        labs = d["labels"]
        if not labs:
            continue
        nC, nN = labs.count("C"), labs.count("N")
        if nC == nN and drop_label_conflicts and nC > 0:
            conflicts += 1
            continue
        label = 1 if nC >= nN else 0
        frames = sorted(set(d["low"].keys()) & set(d["high"].keys()))
        if not frames:
            missing += 1
            continue
        A_seq = np.stack([np.load(d["low"][f]) for f in frames])
        B_seq = np.stack([np.load(d["high"][f]) for f in frames])
        pairs.append({
            "video": d["video"], "a": key[1], "b": key[2], "label": label,
            "pooled_A": A_seq.mean(0).astype(np.float32),
            "pooled_B": B_seq.mean(0).astype(np.float32),
            "signals": np.array(_compute_signals(A_seq, B_seq), dtype=np.float32),
            "n_clips": len(frames),
        })

    if verbose:
        nC = sum(p["label"] for p in pairs)
        print(f"[load_pairs] {os.path.basename(feature_index_csv)}: "
              f"{len(rows)} clip-rows -> {len(pairs)} UNIQUE undirected pairs "
              f"(C={nC} N={len(pairs)-nC}); dropped {conflicts} label-conflict, "
              f"{missing} feature-missing")
    return pairs


# ---------------------------------------------------------------------------
# honest split
# ---------------------------------------------------------------------------

def _balance(c, n):
    t = c + n
    return 0.0 if t == 0 else min(c, n) / t


def _scene_baseline(pairs):
    vid = defaultdict(lambda: [0, 0])
    for p in pairs:
        vid[p["video"]][0 if p["label"] == 1 else 1] += 1
    tot = sum(c + n for c, n in vid.values())
    if tot == 0:
        return 0.0
    return sum(max(c, n) for c, n in vid.values()) / tot


def honest_split(pairs, test_videos=("VID_ (4)",), val_min=50, seed=0, verbose=True):
    """
    Video-level split that puts the BALANCED video(s) in TEST so scene-memorization
    scores ~chance there. VAL = next-most-balanced videos (for early stopping).
    TRAIN = everything else (the pure/skewed videos still teach the easy structure).

    No pair can appear in two splits because the split is by video.
    Returns dict of lists: {'train','val','test'} of pair dicts.
    """
    by_vid = defaultdict(list)
    for p in pairs:
        by_vid[p["video"]].append(p)

    test_set = set(test_videos)
    test = [p for v in test_videos for p in by_vid.get(v, [])]

    # rank remaining videos by balance (most balanced first) for val
    rem = [v for v in by_vid if v not in test_set]
    def vbal(v):
        c = sum(x["label"] for x in by_vid[v]); n = len(by_vid[v]) - c
        return _balance(c, n)
    rem.sort(key=lambda v: (-vbal(v), -len(by_vid[v])))

    val, val_vids = [], []
    for v in rem:
        if len(val) >= val_min and any(x["label"] == 1 for x in val) and any(x["label"] == 0 for x in val):
            break
        val += by_vid[v]; val_vids.append(v)
    val_set = set(val_vids)
    train = [p for v in rem if v not in val_set for p in by_vid[v]]

    if verbose:
        for name, split in [("TRAIN", train), ("VAL", val), ("TEST", test)]:
            c = sum(p["label"] for p in split); n = len(split) - c
            sb = _scene_baseline(split)
            flag = "  <-- honest (scene~chance)" if sb < 0.62 else "  <-- still confounded"
            print(f"[split] {name:5} pairs={len(split):4} C={c:3} N={n:3} "
                  f"scene_baseline={sb*100:4.0f}%{flag if name=='TEST' else ''}")
        print(f"[split] TEST videos={list(test_videos)}  VAL videos={val_vids}")
    return {"train": train, "val": val, "test": test}


def add_symmetric(train_pairs):
    """Augment TRAIN ONLY with A<->B swapped copies (collaboration is symmetric).
    Safe because we collapsed duplicates first, so this exactly doubles train."""
    out = list(train_pairs)
    for p in train_pairs:
        q = dict(p)
        q["pooled_A"], q["pooled_B"] = p["pooled_B"], p["pooled_A"]
        out.append(q)
    return out


# ---------------------------------------------------------------------------
# self-test (no torch needed)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/collab_cache/feature_index_33.csv")
    ap.add_argument("--cache", default="data/collab_cache")
    args = ap.parse_args()
    ps = load_pairs(args.index, args.cache)
    sig = np.stack([p["signals"] for p in ps])
    print("\nsignal means (should NOT all be 0.5 -> proves signals carry info):")
    for i, nm in enumerate(SIGNAL_NAMES):
        print(f"  {nm:14} mean={sig[:,i].mean():.3f}  std={sig[:,i].std():.3f}")
    honest_split(ps)
