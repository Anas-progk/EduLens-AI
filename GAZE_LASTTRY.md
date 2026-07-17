# Phase-2.5 — Gaze last-try (the one untested channel), gated honestly

## Why this is a real attempt and not a repeat
Within-scene pair collaboration was a diagnosed **negative** across three families: appearance
(Swin features), relational signals, and **geometry** (bbox proximity / IoU / dx-dy / co-movement —
every spatial cue ≈ 0.5 AUC on the balanced scene VID_(4); only co-tracking-*duration* correlated and
it didn't survive LOVO). None of those ever encoded **head direction**. Gaze adds exactly one new
measurement — per-person **yaw/pitch** — and reuses everything else (the 20,440 crops, the bbox
positions from the geometry re-detection, the pair catalog, and the *same* within-scene gate).
This is the "Plan 3" scope: lightweight gaze features appended to the existing pipeline. **No graph
transformer, no Swin retrain** — 33 videos / 883 pairs would overfit either instantly.

## The six gaze features (per unique pair, over face-valid co-visible frames)
| feature | meaning |
|---|---|
| `gz_mutual` | frac frames A looks→B **and** B looks→A (face-to-face turn-taking) |
| `gz_oneway` | frac frames exactly one looks at the other (address / listen) |
| `gz_converge` | frac frames both turned, gaze x-targets agree (shared external focus) |
| `gz_jointdown` | frac frames both heads pitched down together (shared desk / workspace) |
| `gz_yawsync` | corr(yaw_A series, yaw_B series) (head-turn synchrony) |
| `gz_turntake` | lagged \|Δyaw\| cross-corr, A turns → B turns (question → response) |

"A looks toward B" = yaw past a turn threshold whose sign matches the image-x bearing to B. The
estimator's yaw-sign convention is fixed, so even if globally inverted it is *consistent* across all
pairs and the linear head handles the sign — only consistency matters for the gate.

## Run order (Colab — only the first step needs GPU/torch-ish deps)
```bash
# 0) deps  -- MediaPipe is DROPPED (its FaceMesh graph init fails on Colab Python 3.12).
#           InsightFace is onnxruntime-based: no torch / no protobuf / no graph issues.
pip install insightface onnxruntime-gpu opencv-python-headless numpy
#   torch fallback backend (you already run torch):  pip install sixdrepnet

# 0b) SMOKE-TEST the backend on ONE crop BEFORE the 20k run (sanity, ~10s):
python -c "from src.data.extract_gaze import make_predictor as M; print(M('insightface').pose_from_path('data/collab_raw/crops/VID_ (4)/1/clip_0000/frame_0000.jpg'))"
#   -> expect a tuple like (yaw, pitch, roll, 1).  If it prints (...,0) the face wasn't found on that crop; try another.

# 1) positions — the SAME re-detection the geometry track used (regenerates bboxes_geom.csv)
python src/data/extract_pair_geometry.py --videos videos --out data/collab_raw/bboxes_geom.csv

# 2) head pose on the existing crops  ->  prints FACE-FOUND COVERAGE per video (the first signal)
python src/data/extract_gaze.py --crops data/collab_raw/crops --out data/gaze/headpose.csv
#    switch backend if InsightFace misbehaves:  --backend sixdrepnet

# 3) build per-pair gaze features, merged into a new npz
python src/data/build_gaze_features.py \
    --headpose data/gaze/headpose.csv --bboxes data/collab_raw/bboxes_geom.csv \
    --catalog data/collab_raw/pair_catalog_33.csv \
    --pairs_npz data/collab_pairs_unique_fresh/pairs_features.npz \
    --out data/collab_pairs_unique_fresh/pairs_features_gaze.npz

# 4) THE HONEST GATE — within-VID_(4) LOVO + per-video median + shuffle floor + per-feature AUC
python src/eval/eval_gaze.py --npz data/collab_pairs_unique_fresh/pairs_features_gaze.npz
```

## The decision rule (identical to the geometry gate — no moved goalposts)
Gaze is real **iff** `signals+gaze` per-video median macro-F1 is **> 0.50**, **> the shuffle floor + 0.03**,
**and** beats the `signals` baseline median by **> 0.03**. VID_(4)-LOVO (the only balanced scene) and the
per-feature within-VID_(4) AUC table tell you *which* cue, if any, carries it.

- **PASS** → a compact pair-level head over `signals + gaze` is justified; we extend
  `group_collab.py` to derive the group verdict from per-pair P(collab). You'd have a defensible
  *pair-level* result for the first time.
- **FAIL** → documented negative. The frozen **session-level 0.667** is the final Phase-2 deliverable,
  and gaze is written up as "future work: who-looks-at-whom needs higher-res / frame-annotated data."

## Honest expectation (stated up front)
Roughly **1-in-3**. The idea is the strongest remaining hypothesis, but the likely failure mode is not
the idea — it's **measurement**: 2fps, low-res, multi-person classroom crops where faces are often
small or non-frontal. That's why step 2 prints **face-found coverage per video**: if most videos are
**< ~40%**, gaze features are too sparse to separate pairs and that *is* the answer — do not force a
pair head. This gate has correctly rejected appearance, signals, and geometry; it will tell the truth
about gaze too. Either outcome is a clean result for the review.

## Self-tested before handoff (in this environment, numpy parts)
- `eval_gaze.py` runs end-to-end and, on the real geometry features, reproduces the documented
  geometry rejection (per-feature AUC ≈ 0.5 except co-tracking duration; verdict: does not lift).
- `build_gaze_features.py` joins catalog↔bboxes↔headpose, computes features (verified `gz_mutual=1.0`
  on a constructed mutual-gaze pair), aligns 1:1 to the fresh npz, and writes the merged file.
- `extract_gaze.py` scaffolding (crop walk / parse / CSV / per-video coverage) verified with a mock
  backend; the head-pose backends (InsightFace default, SixDRepNet fallback) run on Colab. MediaPipe
  was removed after it proved unrunnable on Colab Python 3.12 (FaceMesh graph init explodes).
