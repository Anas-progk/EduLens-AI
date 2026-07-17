# Phase 2 — Collaboration: Honest Result (fresh per-person features, full 33 videos)

## TL;DR — what to claim in the review
- **Headline finding (verified positive):** *mutual gaze is a real, mechanistically-sound group-collaboration signal.* Collaborative sessions show mean mutual-gaze ratio **0.34 vs 0.22** for non-collaborative (single-feature AUC **0.705**, model-free). Adding head-pose / gaze features to the group model lifts honest LOVO macro-F1 from **0.667 → 0.764** (≈0.70–0.73 regularized). This is Phase-2's first **verified improvement** — it turns the earlier honest *negative* into an honest *gain*.
- **Claim this (session / group level):** *"is this whole session collaborative?"* — under honest Leave-One-Video-Out (LOVO) on all 33 videos: signals+scalars baseline **0.667**, **signals + gaze 0.764** (verified — *not* a coverage confound; mutual-gaze-driven; small-sample sensitive, 95% bootstrap CI **[0.60, 0.90]**). Majority baseline 0.348; label-shuffle floor 0.396–0.503 → real and well above chance. Built on the frozen **Swin-Tiny** backbone + relational signals + feature scalars (+ gaze).
- **Do NOT claim within-scene pair level** ("which two specific people in this room collaborate"). It is **chance**. A naive pooled pair score (0.608) *looks* above chance but is the **session signal in disguise** — proven below with a per-video breakdown. **Four** signal families (appearance, relational, geometry, gaze/head-pose) were each tested against an honest within-scene gate; none clears it. Claiming pair-level would not survive scrutiny.
- The model uses the **Swin-Tiny engagement backbone (frozen)** + 6 relational signals + 4 per-pair feature scalars, with a small deployable head. Architecture stays Swin-centered.
- A deterministic **MVP demo** runs end-to-end on cached Swin features (`demo_group_collab.py`) and prints a Collaborative/Not verdict per session with the signals that drove it.

## What changed since the first pass — the feature fix
The first cache stored **whole-frame** features: within any frame every tracked person collapsed to *one identical vector* (`feat_A == feat_B` for all 883 pairs). Every number was secretly built on person-blind features. We re-extracted **genuinely per-person** features into `collab_cache_fresh/` (self-test: two different tracks in the same frame now differ, `|A−B| = 11.6`; 0 / 200 pairs have `feat_A == feat_B`). **Every result below is on the corrected features.** This both (a) confirmed the session result is robust, and (b) let us run the *real* pair-level test for the first time.

## The trap we avoid: the scene confound
Collaboration in this dataset is effectively a **scene/session-level property**: in each recording the *whole group* is either collaborating or not, so a pair's label is **~83–90% determined by which video it came from**. Any split that lets pairs from one video appear in both train and test (random split, stratified k-fold) lets the model **recognize the room, not the collaboration** — inflated and false.

This is the same failure mode as the paper's **97.58%** Swin number (`StudentAttention-2508.15782v2`): stratified 5-fold CV **plus minority oversampling before the split**, on individual frames — duplicated frames and the same scene land on both sides. Not comparable to an honest split.

**The honest instrument is LOVO:** hold out one *entire* video each fold, train on the rest, predict the held-out video. The scene can never leak.

## Honest results — fresh features, full 33 videos (LOVO)
883 unique undirected pairs (C=452 / N=431) over 33 videos; 30 sessions have ≥3 pairs (14 collaborative, 16 not).

**Q1 — Session / group level** (the claim): *classify a whole unseen video.*

| feature set | accuracy | macro-F1 | confusion [N,C] |
|---|---|---|---|
| **signals + scalars + gaze** (verified; ≈0.70–0.73 regularized; 95% CI [0.60, 0.90]) | **76.4%** | **0.764** | — |
| both (signals + feature scalars) — prior baseline | 66.7% | 0.667 | [[10, 6], [4, 10]] |
| relational signals only | 53.3% | 0.531 | [[9, 7], [7, 7]] |
| engagement features only | 50.0% | 0.499 | [[7, 9], [6, 8]] |
| majority baseline | 53.3% | 0.348 | — |
| label-shuffled control | — | 0.396 ± 0.076 | — |

**Q2 — Pair level** (NOT claimed): *label the pairs of an unseen video, pooled over folds.*

| feature set | accuracy | macro-F1 |
|---|---|---|
| engagement feature scalars | 49.3% | 0.484 |
| relational signals | 57.2% | 0.572 |
| signals + scalars | 57.3% | 0.573 |
| full 1536-d [A\|B] | 55.7% | 0.551 |
| full 1536-d + signals | 60.9% | **0.608** |
| majority baseline | 51.2% | 0.339 |

**Reading Q1:** the session signal is real and generalizes across unseen scenes. The signals+scalars baseline is **0.667**; **adding head-pose / gaze features raises it to 0.764** (verified — see *Session-level: gaze improves the group model* below). The mechanism is simple and sound: collaborative sessions exhibit more **mutual gaze** (mean ratio 0.34 vs 0.22), a real group-collaboration cue the engagement features alone cannot see. (On the old whole-frame cache, signals-only looked dominant at 0.697 because those "signals" were scene-level dynamics; on correct per-person features that inflation disappears and the signals baseline is 0.667.)

## Why the pooled 0.608 is NOT within-scene pair detection
The pooled pair score (0.608, full768+sig) looks like it beats chance. It does not, *in the way that matters*. The validation panel (`--validate`) breaks it down per held-out video:

- **Per-video median macro-F1 = 0.442**; only **6 / 33** held-out videos score above 0.50.
- **VID_(4) — the only internally balanced scene — honest-LOVO = 0.544 ≈ chance.**
- The within-VID_(4) *leakage-allowed* ceiling rose to **0.79**, but that is **identity memorization**: inside one video the folds split *pairs*, so the same people appear in train and test, and only the high-capacity 1536-d mode spikes (scalars 0.38 / signals 0.43 / full768 0.78). The 0.79 → 0.544 collapse on the same scene *is* the leakage, quantified.

Why does pooling reach 0.608 while every individual scene is ≈ chance? Because each scene is near single-class, the model wins by learning *"what kind of video is this?"* and stamping that label on all the video's pairs. Pooled over the balanced 452C/431N set that scores 0.608 — a **between-video (session)** effect, not within-scene pair discrimination. The pooled shuffle floor (0.503) and L2-stability (0.60–0.64) confirm it is *stable*, but it is stably measuring the session signal.

**Consequence for the goal:** a single frame can hold both a collaborative and a non-collaborative pair. These features cannot tell them apart (median within-scene 0.442). Within-scene pair detection remains an **honest negative** on appearance features — fresh or not.

## Retired / superseded numbers
- **0.795 pair-level** (15-video local subset) — **retired**: a non-random subset; collapses to chance on the full 33.
- **0.697 signals-only** (old whole-frame cache) — **superseded** by the fresh **0.667 (both)**. Statistically the same (~1 video of 30), but the fresh number is built on correct per-person features and is the one to defend. Do not cite the ~90% random-split number.

## Deployable model + MVP demo
- **Session head** — standardizer (mu, sd) + logistic weights over the session vector `[mean(signals+scalars), std(signals+scalars)]`, fit on all videos in the `both` recipe; reproduces LOVO 0.667. Regenerate on the fresh cache with one command:
  ```bash
  python src/training/train_collab_video_level.py \
    --npz data/collab_pairs_unique_fresh/pairs_features.npz \
    --save_session weights/best_collab_group_fresh.npz --session_mode both
  ```
- `src/inference/group_collab.py` — numpy inference: `GroupCollabHead` (session verdict from pair signals) and `LiveGroupCollab` (rolling per-track 768-d buffers → pair signals → live verdict). Reuses the training signal function so **deploy == train**.
- `demo_group_collab.py` — deterministic review demo on cached features (no camera/GPU/torch): `--video "VID_ (3)"` (one session) or `--all` (every session + the LOVO headline).

## Reproduce / verification
1. **Fresh per-person cache** is built by `src/data/reextract_per_person.py` (self-verifying; aborts if the backbone fails to load; post-checks `feat_A ≠ feat_B`).
2. **Rebuild the pair dataset** from the fresh cache:
   ```bash
   python src/data/build_unique_pairs.py \
     --index data/collab_cache_fresh/feature_index.csv \
     --cache data/collab_cache_fresh --out_dir data/collab_pairs_unique_fresh
   ```
3. **Honest LOVO + the significance/stability gate**:
   ```bash
   python src/training/train_collab_video_level.py \
     --npz data/collab_pairs_unique_fresh/pairs_features.npz --validate
   ```
   Reads: Q1 session (0.667, shuffle 0.396), Q2 pair (pooled 0.608) **and** the validation gate (per-video median 0.442, VID_4 LOVO 0.544, pooled shuffle floor 0.503, L2 sweep) that exposes 0.608 as session-in-disguise.
4. **Leak-free?** Yes — LOVO holds out a whole video per fold; the session shuffle control (0.396) confirms 0.667 is signal, not luck.

## Geometry branch — tested and rejected (the final pair-level hypothesis)
Appearance features can't encode the geometry *between* people, so we recovered the bounding boxes the tracker computes but discards (`extract_pair_geometry.py` — deterministic YOLO re-detection, 95% pair coverage) and built 10 per-pair geometric signals (proximity, normalized offset, IoU, size-ratio, co-movement, approach, co-visibility). Judged on the within-scene gate (`eval_pair_geometry.py`):

| mode | pooled | VID4-LOVO | per-video median |
|---|---|---|---|
| signals | 0.572 | 0.560 | 0.406 |
| geom | 0.512 | 0.457 | 0.375 |
| signals + geom | 0.529 | 0.542 | 0.345 |
| full768 + sig | 0.609 | 0.412 | 0.412 |
| full768 + sig + geom | 0.531 | 0.563 | 0.400 |

Per-video-median shuffle floor = **0.413**. Every mode's within-scene median sits *at* that floor (0.34–0.41); adding geometry does not lift it (it slightly lowers it — noise, not signal).

**The mechanistic why** — inside the balanced scene VID_(4) (C=47 / N=50), per-feature AUC for separating collaborative vs non-collaborative pairs *within the room*: g_prox 0.52, g_dist 0.48, g_dx 0.47, g_dy 0.52, g_iou 0.50, g_sizeratio 0.48, g_comove 0.48 — every **spatial** cue is at chance. Two students at a shared desk are geometrically identical whether they collaborate or work independently. The only features above chance (g_approach 0.67, g_covis 0.61, g_logn 0.58) are co-tracking duration/stability correlates of the session label, not spatial collaboration, and they do not survive cross-scene LOVO. **Geometry is rejected.**

## Gaze branch — the strongest-but-still-insufficient pair hypothesis (tested)
Geometry encodes where bodies *are*, never where heads *point*. The one untested channel was
**who-looks-at-whom**, which the reference paper flags as the primary collaboration cue. We added a single
new measurement — per-person head **yaw/pitch** over the existing crops (`extract_gaze.py`) — and built 6
per-pair gaze features (`build_gaze_features.py`): mutual gaze, one-directional gaze, gaze convergence
(shared external focus), joint downward focus (shared desk), head-turn synchrony, and turn-taking latency.
(MediaPipe FaceMesh was abandoned — its graph init fails on Colab Python 3.12 — and replaced with
**InsightFace** head pose. This is *not* a measurement excuse: **12,809 face-valid frames / 69% pair
coverage** means gaze got a genuine test.) Judged on the same within-scene gate (`eval_gaze.py`):

| mode | pooled | VID4-LOVO | per-video median |
|---|---|---|---|
| signals | 0.572 | 0.560 | 0.406 |
| gaze | 0.459 | 0.542 | 0.371 |
| signals + gaze | 0.551 | **0.602** | 0.418 |
| full768 + sig | 0.644 | 0.322 | 0.442 |
| full768 + sig + gaze | 0.615 | 0.465 | 0.455 |

Per-video-median shuffle floor = **0.414**. Within-VID_(4) per-feature AUC: gz_turntake **0.597**,
gz_oneway **0.589**, gz_jointdown 0.568, gz_mutual 0.559, gz_yawsync 0.549, gz_converge 0.523.

**An honest negative — but a more interesting one than geometry.** Gaze is the *only* channel that lifts
the cleanest within-scene number: VID_(4)-LOVO rises **0.560 → 0.602**, clearing 0.50, and *every* gaze
feature sits above chance (geometry had only co-tracking-duration). The most informative cues are
turn-taking and one-directional gaze — exactly the theory-predicted collaboration signal. **But it does
not clear the gate:** the per-video median is **0.418** — a lift of only +0.011 over signals and barely
above the 0.414 shuffle floor, far short of the required >0.50 and >floor+0.03. The signal is real on the
one balanced scene but **does not generalize across scenes**, because the dataset has exactly one
internally balanced video and ~83% scene-confound — not enough balanced data to convert a faint
within-scene lift into a deployable pair model. Building a pair head on a +0.04 single-scene lift would be
overfitting to VID_(4), the trap avoided throughout. **Gaze is rejected on the gate — and in doing so it
localizes the residual signal to interaction/gaze (not appearance or geometry) and pins the bottleneck to DATA.**

## Session-level: gaze improves the group model (the verified win)
The within-scene gate asks *"which two people in this room collaborate?"* — gaze cannot answer that here
(above). But the deployable question is *"is this whole session collaborative?"*, and there gaze helps.
Added to the 0.667 recipe, `[mean,std]` of the 6 gaze features (over each session's gaze-covered pairs)
lifts honest LOVO macro-F1 from **0.667 → 0.764** (`day1_session_bakeoff.py`). We then tried hard to break
it (`day1_verify_gaze.py`):

- **Not a coverage confound.** Baseline + coverage/count *without any gaze content* scores **0.598**
  (*below* baseline) — the model is not reading "faces more detectable = collaborative."
- **Driven by a meaningful, model-free cue.** Mean **mutual-gaze** separates collaborative from
  non-collaborative sessions at **AUC 0.705** (collaborative mean **0.34** vs **0.22**) — no classifier
  involved. (`gz_yawsync` AUC 0.737 is higher but may reflect joint attention to the teacher rather than
  pairwise collaboration, so we lead with mutual gaze.)
- **Broad, not lucky.** 6 of 30 held-out videos flip correct (incl. the balanced VID_(4)), 3 break,
  net +3 — consistent with 0.667 → 0.764. Label-shuffle floor 0.503.
- **Honest caveat.** With only 30 sessions the gain is small-sample sensitive: regularized it is **0.732
  (L2=3) / 0.700 (L2=10)** — direction robust, magnitude softens — and the 95% bootstrap CI is **[0.60,
  0.90]** (its lower edge touches the baseline). So we report **a verified, promising improvement**, not a
  solved production number.

Feature-fusing gaze *before* the classifier is what worked: pooling tricks, geometry, and prediction-level
ensembles (including the confidence-weighted "dynamic combination" idea) all stayed ≤ 0.667. New model =
**signals + gaze** (`weights/best_collab_group_gaze.npz`, `adopt_gaze_model.py`).

## Conclusion — pair-level a diagnosed negative; session-level a verified gaze improvement
Two honest results, on two different questions. **Within-scene pair** ("which two people collaborate") is a
**clean, diagnosed negative**: four signal families — appearance, relational, geometry, gaze — were each
tested under one honest within-scene gate, and none clears it, because the label is ~83–90% scene-determined
and only one of 33 videos is internally balanced. **Session / group level** ("is this room collaborating")
is where the win is: the signals baseline (0.667) improves to **0.764 with head-pose / gaze** — verified not
to be a coverage confound, driven by the model-free **mutual-gaze** cue (AUC 0.705; 0.34 vs 0.22), broad
across 6 held-out videos, reported honestly as small-sample sensitive (≈0.70–0.73 regularized, 95% CI
[0.60, 0.90]). The deployable product is **`group_collab` + the signals+gaze head**. **Future work is
precise:** more internally **balanced, frame-annotated** scenes would (a) let the within-scene pair question
be tested at real statistical power and (b) tighten the session-level gaze estimate. **Phase-1 engagement
(held-out TEST macro-F1 ~0.73 / 76.7% acc on 5 unseen classrooms; validation 0.90 / 92.4% was the
epoch+threshold selection set, audited via `eval_engagement_honest.py`) + Phase-2 group collaboration
(signals + gaze, ≈0.76 / 0.73 regularized, mutual-gaze-driven) is the complete, honestly-evaluated system.**
