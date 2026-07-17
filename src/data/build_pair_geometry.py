"""
build_pair_geometry.py -- bboxes_geom.csv + pair_catalog_33.csv -> per-pair GEOMETRIC signals,
aligned to the existing 883 unique pairs. Numpy/CSV only (no torch).

Per pair clip window [start_frame, start_frame+CLIP_LEN), over the co-visible frames of the two
tracks (centers c=(x+w/2, y+h/2), scale = mean person height = 0.5*(hA+hB)):
    g_prox      mean 1/(1 + d/scale)         (closeness; high = close)
    g_dist      mean d/scale                  (normalized centre distance)
    g_dx, g_dy  mean |dx|/scale, |dy|/scale   (side-by-side vs stacked)
    g_iou       mean bbox IoU                 (shared space / leaning in)
    g_sizeratio min(h)/max(h)                 (similar depth = same table)
    g_comove    corr of the two centre paths  (move together)
    g_approach  -slope of distance over time  (positive = leaning in)
    g_covis     fraction of frames both seen
    g_logn      log(1 + #co-frames)
Clip values are averaged per UNIQUE pair (key = video, min(track), max(track)) so they match the
883-pair grouping, then joined to the fresh pairs npz by (video_id, a_id, b_id).

Outputs (existing npz untouched):
    data/collab_pairs_unique_fresh/pairs_geometry.npz     (geom aligned to the npz pair order)
    data/collab_pairs_unique_fresh/pairs_features_geom.npz (fresh npz + geom + geom_mask)
"""

import os
import csv
import argparse
from collections import defaultdict
import numpy as np

GEOM_NAMES = ["g_prox", "g_dist", "g_dx", "g_dy", "g_iou",
              "g_sizeratio", "g_comove", "g_approach", "g_covis", "g_logn"]


def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2, bx2, by2 = ax + aw, ay + ah, bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _corr(u, v):
    u, v = np.asarray(u, float), np.asarray(v, float)
    if len(u) < 3 or u.std() < 1e-8 or v.std() < 1e-8:
        return 0.0
    c = np.corrcoef(u, v)[0, 1]
    return float(c) if np.isfinite(c) else 0.0


def load_bboxes(path):
    """-> D[(video, track)] = dict{frame_rel: (x,y,w,h)} ;  FW[video]=frame_w, FH[video]=frame_h"""
    D = defaultdict(dict)
    FW, FH = {}, {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            v = r["video_id"]; t = int(float(r["track_id"])); fr = int(float(r["frame_rel"]))
            D[(v, t)][fr] = (float(r["x"]), float(r["y"]), float(r["w"]), float(r["h"]))
            FW[v] = int(float(r["frame_w"])); FH[v] = int(float(r["frame_h"]))
    return D, FW, FH


def clip_geometry(boxesA, boxesB, start, clip_len):
    """boxesA/boxesB: dict{frame_rel:(x,y,w,h)}. Returns (vec8, n_co) over the window co-frames."""
    co = [fr for fr in range(start, start + clip_len) if fr in boxesA and fr in boxesB]
    if not co:
        return None, 0
    prox = dist = dxn = dyn = iou = sr = 0.0
    cAx, cAy, cBx, cBy, dser = [], [], [], [], []
    for fr in co:
        ax, ay, aw, ah = boxesA[fr]
        bx, by, bw, bh = boxesB[fr]
        caX, caY = ax + aw / 2.0, ay + ah / 2.0
        cbX, cbY = bx + bw / 2.0, by + bh / 2.0
        scale = 0.5 * (ah + bh) + 1e-6
        d = ((caX - cbX) ** 2 + (caY - cbY) ** 2) ** 0.5
        dn = d / scale
        prox += 1.0 / (1.0 + dn); dist += dn
        dxn += abs(caX - cbX) / scale; dyn += abs(caY - cbY) / scale
        iou += _iou(boxesA[fr], boxesB[fr])
        sr += min(ah, bh) / (max(ah, bh) + 1e-6)
        cAx.append(caX); cAy.append(caY); cBx.append(cbX); cBy.append(cbY); dser.append(dn)
    n = len(co)
    comove = 0.5 * (_corr(cAx, cBx) + _corr(cAy, cBy))
    if len(dser) >= 3:
        slope = float(np.polyfit(np.arange(len(dser)), dser, 1)[0])
        approach = -slope
    else:
        approach = 0.0
    vec = np.array([prox / n, dist / n, dxn / n, dyn / n, iou / n,
                    sr / n, comove, approach], dtype=np.float64)
    return vec, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bboxes", default="data/collab_raw/bboxes_geom.csv")
    ap.add_argument("--catalog", default="data/collab_raw/pair_catalog_33.csv")
    ap.add_argument("--pairs_npz", default="data/collab_pairs_unique_fresh/pairs_features.npz")
    ap.add_argument("--out_dir", default="data/collab_pairs_unique_fresh")
    ap.add_argument("--clip_len", type=int, default=8)
    args = ap.parse_args()

    D, FW, FH = load_bboxes(args.bboxes)
    print(f"[geom] bboxes: {len(D)} (video,track) series over {len(FW)} videos")

    # accumulate clip-level geometry per UNIQUE pair key (video, lowTrack, highTrack)
    acc = defaultdict(lambda: {"vsum": np.zeros(8), "wsum": 0.0, "co": 0, "clips": 0})
    n_rows = n_used = 0
    with open(args.catalog, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            n_rows += 1
            v = r["video_id"]
            try:
                tA = int(float(r["track_id_A"])); tB = int(float(r["track_id_B"]))
                start = int(float(r["start_frame"]))
            except (KeyError, ValueError):
                continue
            lo, hi = min(tA, tB), max(tA, tB)
            bxLo = D.get((v, lo)); bxHi = D.get((v, hi))
            if not bxLo or not bxHi:
                continue
            vec, n_co = clip_geometry(bxLo, bxHi, start, args.clip_len)
            if vec is None:
                continue
            a = acc[(v, lo, hi)]
            a["vsum"] += vec * n_co; a["wsum"] += n_co; a["co"] += n_co; a["clips"] += 1
            n_used += 1
    print(f"[geom] catalog rows={n_rows}  clips with geometry={n_used}  unique pairs={len(acc)}")

    def pair_vec(a):
        base = a["vsum"] / max(a["wsum"], 1e-6)                 # weighted-mean 8-d
        covis = a["co"] / max(a["clips"] * args.clip_len, 1e-6)
        logn = float(np.log1p(a["co"]))
        return np.concatenate([base, [covis, logn]]).astype(np.float64)  # 10-d

    geom_by_key = {k: pair_vec(a) for k, a in acc.items()}

    # ---- align to the fresh pairs npz order ----
    d = np.load(args.pairs_npz, allow_pickle=True)
    vids = d["video_ids"].astype(str)
    a_ids = d["a_ids"].astype(str); b_ids = d["b_ids"].astype(str)
    P = len(vids)
    G = len(GEOM_NAMES)
    geom = np.zeros((P, G), dtype=np.float32)
    mask = np.zeros(P, dtype=np.int64)
    for i in range(P):
        try:
            lo, hi = sorted((int(float(a_ids[i])), int(float(b_ids[i]))))
        except ValueError:
            continue
        gv = geom_by_key.get((vids[i], lo, hi))
        if gv is not None:
            geom[i] = gv.astype(np.float32); mask[i] = 1
    cov = int(mask.sum())
    print(f"[geom] coverage: {cov}/{P} pairs got geometry ({cov/P*100:.1f}%)")
    if cov < P:
        miss = P - cov
        print(f"[geom] {miss} pairs missing geometry (video not re-detected, or no co-frames) "
              f"-> geom=0, geom_mask=0 for those; keep this in mind in eval.")

    os.makedirs(args.out_dir, exist_ok=True)
    geo_npz = os.path.join(args.out_dir, "pairs_geometry.npz")
    np.savez_compressed(geo_npz, video_ids=vids, a_ids=a_ids, b_ids=b_ids,
                        geom=geom, geom_mask=mask, geom_names=np.array(GEOM_NAMES))
    # merged: fresh npz + geometry (existing npz untouched)
    merged = {k: d[k] for k in d.files}
    merged["geom"] = geom; merged["geom_mask"] = mask; merged["geom_names"] = np.array(GEOM_NAMES)
    mer_npz = os.path.join(args.out_dir, "pairs_features_geom.npz")
    np.savez_compressed(mer_npz, **merged)
    print(f"[geom] saved -> {geo_npz}")
    print(f"[geom] saved -> {mer_npz}")
    print(f"[geom] next: python src/eval/eval_pair_geometry.py --npz {mer_npz}")


if __name__ == "__main__":
    main()
