# Phase-2 Geometry Track — plan (isolated branch, baseline frozen)

## Why
Appearance features (fresh or not) cannot separate collaborative from non-collaborative pairs
*within one scene*: balanced VID_(4) honest-LOVO = 0.544, per-video median 0.442. That information
is **geometric** — two pairs in the same frame look identical in per-person appearance but differ in
how the people are arranged (close + facing + leaning in vs apart + parallel). The tracker already
computes per-frame bounding boxes and **throws them away** (only crops are saved). This track recovers
them and tests whether geometry lifts the *within-scene* signal that appearance can't.

## Hard rule: the frozen baseline is untouched
The locked deliverable — `group_collab.py`, `weights/best_collab_group_fresh.npz`, the session 0.667
LOVO path — is **not modified**. Geometry lives entirely in new files:
- `src/data/extract_pair_geometry.py` — recover bboxes (re-detection)
- `src/data/build_pair_geometry.py` — bboxes → per-pair geometric signals → merged npz
- `src/eval/eval_pair_geometry.py` — within-scene gate (the decision)

## Step 1 — recover bboxes (Colab GPU; the only step that needs torch/YOLO)
`extract_pair_geometry.py` re-runs the **exact** detector + `SimpleIoUTracker` from
`collab_video_processor` (imported, not modified) with the same constants (FPS_EXTRACT=3, conf=0.40,
IoU=0.30, the `extract_upper_body_crop` validity gate). The tracker is deterministic, so it reproduces
the **same track IDs and frame_rel** numbering as the original run → the bboxes join 1:1 to the existing
883 pairs. It writes `data/collab_raw/bboxes_geom.csv` (`video_id, track_id, frame_rel, x, y, w, h,
frame_w, frame_h`). No crops are written, so it is detection-only and fast.

```
python src/data/extract_pair_geometry.py --videos videos --out data/collab_raw/bboxes_geom.csv
# (custom lectures, if used as collab videos: add --videos custom_dataset/EduAction_E)
```

## Step 2 — geometric pair signals (local, numpy only, no torch)
`build_pair_geometry.py` joins `bboxes_geom.csv` to `pair_catalog_33.csv` by `(video, track, frame_rel)`
and, per pair's clip window `[start_frame, start_frame+8)`, computes over the co-visible frames
(centers `c=(x+w/2,y+h/2)`, scale = mean person height):

- `g_prox`     — closeness, mean `1/(1+ d/scale)`  (high = close)
- `g_dist`     — mean normalized center distance `d/scale`
- `g_dx,g_dy`  — normalized horizontal / vertical gap (side-by-side vs stacked)
- `g_iou`      — mean bbox IoU (shared space / leaning into the same materials)
- `g_sizeratio`— `min(h)/max(h)` (similar depth = same table)
- `g_comove`   — Pearson corr of the two center trajectories over the clip (move together)
- `g_approach` — −slope of distance over the clip (positive = leaning in)
- `g_covis`    — fraction of frames both are detected; `g_logn` — log #co-frames

Per **unique** pair these clip values are averaged (matching the 883-pair grouping), giving a ~10-d
geometry vector. It writes `data/collab_pairs_unique_fresh/pairs_geometry.npz` and a merged
`pairs_features_geom.npz` (the fresh npz **plus** an aligned `geom` array + `geom_mask`), so nothing
in the existing npz changes.

```
python src/data/build_pair_geometry.py \
  --bboxes data/collab_raw/bboxes_geom.csv \
  --catalog data/collab_raw/pair_catalog_33.csv \
  --pairs_npz data/collab_pairs_unique_fresh/pairs_features.npz \
  --out_dir data/collab_pairs_unique_fresh
```

## Step 3 — the decision (within-scene gate, not the pooled number)
`eval_pair_geometry.py` runs the honest gate on the merged npz for modes
`signals` / `geom` / `signals+geom` / `full768+sig+geom`, reporting **VID_(4) honest-LOVO** and the
**per-video median** (the metrics that matter) alongside the pooled number and a shuffle floor.

```
python src/eval/eval_pair_geometry.py --npz data/collab_pairs_unique_fresh/pairs_features_geom.npz
```

**Go / no-go:** if geometry raises the within-scene metric materially above appearance — VID_(4) LOVO
and per-video median clearing ~0.50–0.55 — pair-level becomes real and a compact pair head is justified
(then it can feed a group aggregator). If geometry does **not** lift the within-scene metric, that is a
clean, documented negative: within-scene pair detection is not achievable on this data, and the frozen
session-level 0.667 stands as the result. Either outcome is reported honestly; the baseline is safe.
