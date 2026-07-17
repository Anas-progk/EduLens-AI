"""
build_gaze_features.py -- headpose.csv + bboxes_geom.csv + pair_catalog_33.csv
                          -> per-pair GAZE features -> merged into pairs_features_gaze.npz

Reuses the geometry join verbatim: unique pair key = (video, min(track), max(track));
per pair-clip window [start_frame, start_frame+CLIP_LEN); features averaged over the pair's
face-valid co-visible frames, then aligned 1:1 to the fresh pairs npz order.

Six transparent gaze features per pair (all in [0,1] except the two correlations in [-1,1]):
    gz_mutual      frac co-frames A looks->B AND B looks->A           (turn-taking face-to-face)
    gz_oneway      frac co-frames exactly one looks at the other      (address / listen)
    gz_converge    frac co-frames both turned, gaze x-targets agree    (shared external focus)
    gz_jointdown   frac co-frames both heads pitched DOWN together     (shared desk/workspace)
    gz_yawsync     corr(yaw_A series, yaw_B series)                    (head-turn synchrony)
    gz_turntake    lagged |Δyaw| cross-corr (A turns -> B turns)       (question->response)

"A looks toward B" uses yaw sign vs the image-x bearing to B. The global yaw-sign convention
is fixed by the estimator, so even if it is inverted it is CONSISTENT across all pairs and a
linear head handles the sign -- only consistency matters for the within-scene gate.

RUN (Colab, after extract_gaze.py and the geometry bbox re-detection):
    # 1) positions (same re-detection the geometry track used; regenerates bboxes_geom.csv)
    python src/data/extract_pair_geometry.py --videos videos --out data/collab_raw/bboxes_geom.csv
    # 2) head pose
    python src/data/extract_gaze.py --crops data/collab_raw/crops --out data/gaze/headpose.csv
    # 3) THIS builder
    python src/data/build_gaze_features.py \
        --headpose data/gaze/headpose.csv --bboxes data/collab_raw/bboxes_geom.csv \
        --catalog data/collab_raw/pair_catalog_33.csv \
        --pairs_npz data/collab_pairs_unique_fresh/pairs_features.npz \
        --out data/collab_pairs_unique_fresh/pairs_features_gaze.npz
    # 4) the honest gate
    python src/eval/eval_gaze.py --npz data/collab_pairs_unique_fresh/pairs_features_gaze.npz
"""

import os
import csv
import math
import argparse
import numpy as np
from collections import defaultdict

GAZE_NAMES = ["gz_mutual", "gz_oneway", "gz_converge", "gz_jointdown", "gz_yawsync", "gz_turntake"]
CLIP_LEN = 8
YAW_MIN = 12.0      # deg: a meaningful head turn
PITCH_DOWN = 10.0   # deg: head pitched down (looking at desk)
MIN_FRAMES = 3      # need this many valid co-frames or the pair is left uncovered


def _clip_num(clip_dir):
    """'...\\clip_0007' -> 7."""
    base = os.path.basename(str(clip_dir).replace("\\", "/").rstrip("/"))
    digits = "".join(ch for ch in base if ch.isdigit())
    return int(digits) if digits else -1


def load_bboxes(path):
    """-> D[(video,track)] = {frame_rel:(cx,cy,h)} ; FW[video]=frame_w."""
    D = defaultdict(dict); FW = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                v = r["video_id"]; t = int(float(r["track_id"])); fr = int(float(r["frame_rel"]))
                x, y, w, h = (float(r["x"]), float(r["y"]), float(r["w"]), float(r["h"]))
                FW[v] = int(float(r.get("frame_w", 1) or 1))
            except (KeyError, ValueError):
                continue
            D[(v, t)][fr] = (x + w / 2.0, y + h / 2.0, h)
    return D, FW


def load_headpose(path):
    """-> H[(video,track,clip,frame_in_clip)] = (yaw,pitch) for face-found rows only."""
    H = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if str(r.get("face_found", "0")).strip() not in ("1", "1.0", "True", "true"):
                continue
            try:
                key = (r["video_id"], int(float(r["track"])), int(float(r["clip"])), int(float(r["frame"])))
                yaw = float(r["yaw"]); pitch = float(r["pitch"])
            except (KeyError, ValueError):
                continue
            H[key] = (yaw, pitch)
    return H


def _safe_corr(u, v):
    if len(u) < 3:
        return 0.0
    u = np.asarray(u, float); v = np.asarray(v, float)
    if u.std() < 1e-6 or v.std() < 1e-6:
        return 0.0
    return float(np.clip(np.corrcoef(u, v)[0, 1], -1, 1))


def _turn_take(dyA, dyB):
    """Max lagged cross-corr of |Δyaw| at lag 1..2, both directions (A->B and B->A)."""
    a = np.abs(np.asarray(dyA, float)); b = np.abs(np.asarray(dyB, float))
    best = 0.0
    for lag in (1, 2):
        if len(a) > lag + 2:
            best = max(best, _safe_corr(a[:-lag], b[lag:]), _safe_corr(b[:-lag], a[lag:]))
    return best


def pair_gaze(seqA, seqB, frameW):
    """seqX = list of (yaw, pitch, cx, cy, h) aligned by co-frame. Returns 6-d vec or None."""
    if len(seqA) < MIN_FRAMES:
        return None
    yawA = [s[0] for s in seqA]; yawB = [s[0] for s in seqB]
    mutual = oneway = converge = jointdown = 0
    n = len(seqA)
    for (ya, pa, ax, ay, ah), (yb, pb, bx, by, bh) in zip(seqA, seqB):
        dirAB = 1.0 if bx >= ax else -1.0          # B is to image-right(+)/left(-) of A
        a_to_b = abs(ya) >= YAW_MIN and math.copysign(1, ya) == dirAB
        b_to_a = abs(yb) >= YAW_MIN and math.copysign(1, yb) == -dirAB
        if a_to_b and b_to_a:
            mutual += 1
        elif a_to_b or b_to_a:
            oneway += 1
        # shared external focus: both turned, gaze x-targets land near the same spot
        scale = 0.5 * (ah + bh)
        tax = ax + scale * math.sin(math.radians(ya))
        tbx = bx + scale * math.sin(math.radians(yb))
        if abs(ya) >= YAW_MIN and abs(yb) >= YAW_MIN and not (a_to_b and b_to_a):
            if abs(tax - tbx) < 0.12 * max(frameW, 1):
                converge += 1
        if pa <= -PITCH_DOWN and pb <= -PITCH_DOWN:
            jointdown += 1
    dyA = np.diff(yawA); dyB = np.diff(yawB)
    return np.array([mutual / n, oneway / n, converge / n, jointdown / n,
                     _safe_corr(yawA, yawB), _turn_take(dyA, dyB)], dtype=np.float64)


def main():
    ap = argparse.ArgumentParser(description="Per-pair gaze features -> merged npz")
    ap.add_argument("--headpose", default="data/gaze/headpose.csv")
    ap.add_argument("--bboxes", default="data/collab_raw/bboxes_geom.csv")
    ap.add_argument("--catalog", default="data/collab_raw/pair_catalog_33.csv")
    ap.add_argument("--pairs_npz", default="data/collab_pairs_unique_fresh/pairs_features.npz")
    ap.add_argument("--out", default="data/collab_pairs_unique_fresh/pairs_features_gaze.npz")
    args = ap.parse_args()

    D, FW = load_bboxes(args.bboxes)
    H = load_headpose(args.headpose)
    print(f"[gaze] bbox series: {len(D)} (video,track) | headpose face-valid rows: {len(H)}")

    # accumulate per-frame samples for each UNIQUE pair key over all its clips
    samplesA = defaultdict(list); samplesB = defaultdict(list); fw_of = {}
    n_rows = n_used = 0
    with open(args.catalog, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            n_rows += 1
            try:
                v = r["video_id"]; tA = int(float(r["track_id_A"])); tB = int(float(r["track_id_B"]))
                start = int(float(r["start_frame"]))
                cA = _clip_num(r["clip_dir_A"]); cB = _clip_num(r["clip_dir_B"])
            except (KeyError, ValueError):
                continue
            lo, hi = sorted((tA, tB))
            clo, chi = (cA, cB) if tA <= tB else (cB, cA)   # clip num that belongs to lo / hi track
            fw_of[(v, lo, hi)] = FW.get(v, 1)
            bxLo = D.get((v, lo)); bxHi = D.get((v, hi))
            if not bxLo or not bxHi:
                continue
            used_any = False
            for fi in range(CLIP_LEN):
                fr = start + fi
                hpL = H.get((v, lo, clo, fi)); hpH = H.get((v, hi, chi, fi))
                boL = bxLo.get(fr); boH = bxHi.get(fr)
                if hpL is None or hpH is None or boL is None or boH is None:
                    continue
                samplesA[(v, lo, hi)].append((hpL[0], hpL[1], boL[0], boL[1], boL[2]))
                samplesB[(v, lo, hi)].append((hpH[0], hpH[1], boH[0], boH[1], boH[2]))
                used_any = True
            n_used += used_any

    gaze_by_key = {}
    for key in samplesA:
        vec = pair_gaze(samplesA[key], samplesB[key], fw_of.get(key, 1))
        if vec is not None:
            gaze_by_key[key] = vec
    print(f"[gaze] catalog rows={n_rows}  pair-clips with gaze frames={n_used}  unique pairs w/ gaze={len(gaze_by_key)}")

    # align to the fresh pairs npz order
    d = np.load(args.pairs_npz, allow_pickle=True)
    vids = d["video_ids"].astype(str); a_ids = d["a_ids"].astype(str); b_ids = d["b_ids"].astype(str)
    P = len(vids); K = len(GAZE_NAMES)
    gaze = np.zeros((P, K), dtype=np.float32); mask = np.zeros(P, dtype=np.int64)
    for i in range(P):
        try:
            lo, hi = sorted((int(float(a_ids[i])), int(float(b_ids[i]))))
        except ValueError:
            continue
        gv = gaze_by_key.get((vids[i], lo, hi))
        if gv is not None:
            gaze[i] = gv.astype(np.float32); mask[i] = 1
    cov = int(mask.sum())
    print(f"[gaze] coverage: {cov}/{P} pairs got gaze features ({cov / P * 100:.1f}%)")

    out = {k: d[k] for k in d.files}
    out["gaze"] = gaze; out["gaze_mask"] = mask; out["gaze_names"] = np.array(GAZE_NAMES)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, **out)
    print(f"[gaze] wrote {args.out}  (added 'gaze' {gaze.shape}, 'gaze_mask', 'gaze_names')")
    print("[gaze] next: python src/eval/eval_gaze.py --npz", args.out)


if __name__ == "__main__":
    main()
