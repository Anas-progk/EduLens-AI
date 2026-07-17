# Phase-2 Blueprint — Pair-level interaction → Group collaboration

Status: PLAN ONLY (no training run yet). Written 2026-05-31 for execution in the next session.
Scope: give the pair-level idea (unique pairs + signal injection + a "brain") its best **honest** shot,
then derive the **group** verdict from pair reasoning — without collapsing straight to a session classifier.

---

## 0. My verdict up front (read this first)

Your instinct and ChatGPT's three ideas are **worth testing**, and two of them are genuinely *untested* in your
repo. But I have to be straight about the ceiling, because the numbers are already measured:

1. **The single most important fact:** on the *only* internally balanced video, `VID_ (4)`, even when I let the
   model **train and test on the same video (full leakage allowed)**, the frozen **engagement** features reach only
   **~56%** at the pair level. That is not a training bug or an optimizer problem — it is an *information ceiling*.
   Phase-1 was trained to treat talking/discussion as **Not Engaged**, i.e. it was explicitly taught to **suppress the
   exact cue collaboration depends on**. No head, no signal-injection, and no "brain" stacked on top can recover
   information the features do not contain.

2. **So "92.4% for collab" is not a realistic target, and here's the precise reason it worked for engagement but
   won't transfer:** the Phase-1 winner was `BOOTSTRAP Ep01` = *ImageNet features + minimal temporal training*. It
   won because **pretrained features are already informative for engagement**. The half of your intuition that is
   *right* — "keep the pretrained backbone, adapt lightly, save the best model early" — points at **fresh ImageNet/
   DINOv2 features (Stage 2)**, NOT at reusing the engagement-tuned backbone (which is collab-blind and is what
   creates the 56% ceiling). Your idea is half-right, and the right half is Stage 2.

3. **What is real and defensible today:** group/session-level collaboration, LOVO macro-F1 **0.697** (70% acc),
   majority baseline 0.348, label-shuffle floor 0.464. The signals **beat** the engagement features at session level
   (0.697 vs 0.253) — i.e. the relational signals carry collaboration information that *generalizes*. That is the
   honest contribution already in hand (`weights/best_collab_group.npz`, `src/inference/group_collab.py`).

**Decision:** we do NOT abandon pair-level. We run a short, ranked sequence of experiments that (a) *answers
ChatGPT's exact open question* cheaply and decisively, then (b) attacks the real lever (fresh features), then
(c) derives the group output from per-pair reasoning so a single frame can hold both collaborative and
non-collaborative students. Every number is measured under Leave-One-Video-Out with a label-shuffle control and
the within-`VID_ (4)` ceiling probe. We never re-cite the retired 0.795 / 90% pair numbers.

---

## 1. Ground truth we are building on (measured, not assumed)

| Fact | Value | Source in repo |
|---|---|---|
| Phase-1 engagement (locked, frozen) | acc 92.4%, macro-F1 0.9019 | `weights/best_clip_model.pth` |
| Clip rows in full cache | 6,869 | `data/collab_cache/feature_index.csv` |
| **Unique undirected pairs** | **884** raw; **883** after dropping label-conflicts (C=452 / N=431) | `data.collab_pairs.load_pairs` |
| Sessions with ≥3 pairs | ~30 (14 C / 16 N) | `train_collab_video_level.py` |
| Session LOVO macro-F1 (signals) | **0.697** (70% acc) | `group_collab.py` / honest result |
| Pair LOVO macro-F1 | ~chance (feat 0.499, sig 0.485) | `train_collab_video_level.py` Q2 |
| Within-`VID_ (4)` ceiling, leakage allowed | **~56%** | the killer constraint |
| Scene confound (label ≈ which video) | ~83% | `collab_confound_report.py` |

**Data layout that constrains what's possible:**
- `feature_index.csv` columns: `pair_id, label, feat_A, feat_B, video_id, frame_w, frame_h, track_id_A, track_id_B`.
  There are frame **dimensions** but **no bbox coordinates** anywhere → real spatial proximity/facing signals are
  *not* available without re-running the detector to log bboxes. (This is why `proximity`/`facing` are left neutral.)
- Each `.npy` = **one pooled 768-d vector per person per clip** (within-clip frames already averaged).
- The per-pair **time series** = the sequence of *clips* for that pair (median 4, up to 28). The 6 relational signals
  are computed across this clip sequence (`data.collab_pairs._compute_signals`).

**Existing assets to reuse (do NOT rebuild):**
- `src/data/collab_pairs.py` — de-dup to 884 unique pairs, 6 signals, `honest_split`, `add_symmetric`. numpy-only.
- `src/models/collaboration_head.py` — `CollaborationHead` (PairwiseCrossAttention over feat_A 768 + feat_B 768 +
  signals 6) — **built but never honestly trained.** This is exactly ChatGPT's "inject signals into a transformer."
- `src/training/train_collab_video_level.py` — canonical honest LOVO evaluator (Q1 session, Q2 pair, ablation,
  shuffle control). Currently its pair model is **logistic regression over 4 feature *scalars* + 6 signals**, not the
  full 768-d — so a neural model over the full vectors is genuinely untested.
- `src/inference/group_collab.py` — `GroupCollabHead`, `LiveGroupCollab`, session aggregation. Deploy == train.

The 6 signals (`SIGNAL_NAMES`): `state_cos, state_close, traj_cos, dyn_corr, turn_taking, joint_active`.

---

## 2. The four ideas, mapped honestly

| Idea (yours / ChatGPT) | Already done? | Untested part = the experiment |
|---|---|---|
| **1. Use only the 884 unique pairs** | ✅ done (`load_pairs` collapses 6,869→884) | nothing — this is already the unit |
| **2. Inject 6 signals into a transformer** | ⚠️ partial: signals are *concatenated to LR*; the **neural** `CollaborationHead` over full 768-d **was never trained honestly** | **Stage 1** |
| **3. Fresh pretrained-Swin features (not engagement-tuned)** | ❌ not done | **Stage 2** (the real lever) |
| **4. A "brain" (meta / stacking head)** | ❌ not done | **Stage 4** (polish; cannot create missing info) |
| **Derive group from pair reasoning** | ⚠️ group head exists but aggregates *signals*, not *pair predictions* | **Stage 3** (honors your constraint) |

---

## 3. The staged plan (ranked by risk/reward; each stage has a decision gate)

### Stage 0 — Reproduce baseline + nail the ceiling number (~30 min, do first)
**Why:** every later claim is "better than X." We need X measured on the machine we'll train on, plus an explicit
ceiling probe so we can tell "the model is bad" from "the information isn't there."

**Do:**
- Run the canonical evaluator on the full cache:
  ```
  python src/training/train_collab_video_level.py --index data/collab_cache/feature_index.csv --cache data/collab_cache --l2 1.0 --min_pairs 3
  ```
  Confirm: session-level ~0.697, pair-level ~chance, shuffle control below real.
- **New tiny probe** `src/training/ceiling_probe.py`: train+test on `VID_ (4)` only (leakage allowed), report the best
  achievable pair macro-F1 with (a) feature scalars, (b) full 768-d, (c) +signals. This number is the honest ceiling
  of the current features. Expectation: ~0.56.

**Gate:** record Stage-0 baseline + ceiling. Proceed.

---

### Stage 1 — Neural signal-injection pair model on EXISTING features (cheap, decisive) (~2-4 hrs)
**This answers ChatGPT's exact open question.** It is the cheapest untested experiment and it *settles* whether the
56% ceiling is the features or just the simple LR.

**Why:** the current pair model throws away the 768-d (keeps only cos/dist/mag). A small transformer over the full
vectors + injected signals is the strongest model these features allow. If it still can't beat the ~0.56 ceiling on
`VID_ (4)`-with-leakage, the ceiling is *confirmed* to be in the features → Stage 2 is justified, not optional.

**Files:**
- *Reuse* `src/models/collaboration_head.py::CollaborationHead` (already takes feat_A 768 + feat_B 768 + signals 6).
  Token layout: `[proj(feat_A), proj(feat_B), proj(signals)]` → PairwiseCrossAttention → MLP → logit. (Signals as a
  **third token** is the cleaner injection; concat is the fallback.)
  - **⚠ Signal-mismatch to fix:** the head's docstring names an OLD 6-signal set
    `[proximity, facing, cosine_sim, l2_sim, feat_corr, mad_sim]`, but the de-dup pipeline now emits
    `[state_cos, state_close, traj_cos, dyn_corr, turn_taking, joint_active]` (`data.collab_pairs.SIGNAL_NAMES`).
    Same dim (6), different meaning — the trainer **must feed the new `collab_pairs` signals**, and we update the
    docstring. (`proximity`/`facing` are still neutral 0.5 — no bbox in cache.)
- *New* `src/training/train_collab_pair_nn.py`: consumes `data.collab_pairs.load_pairs` (884 pairs) +
  `honest_split` + `add_symmetric` (train only). **Honest protocol = LOVO**, not a single split (reuse the LOVO loop
  shape from `train_collab_video_level.lovo_pair`). BCE + pos_weight + label smoothing 0.05, AdamW lr 3e-4, cosine,
  grad-clip 1.0, **collapse guard** (the focal-collapse lesson — stop if val pC≈pN). Report pair LOVO macro-F1,
  within-`VID_ (4)` ceiling, label-shuffle control.

**Honest expected range:** 0.50 → **0.55–0.60** pair-level. Bounded by the ceiling. Treat **>0.60 honest** as a
genuine (mild) win; **≤0.56** = ceiling confirmed.

**Gate:**
- If pair LOVO clearly beats signals-LR *and* the `VID_ (4)` ceiling rises meaningfully above 0.56 → the features had
  more than LR could extract; push Stage 1 harder before Stage 2.
- Else (most likely) → ceiling confirmed in the features → **go to Stage 2.** This is a *result*, not a failure: it's
  the evidence that justifies re-extraction.

---

### Stage 2 — Fresh, non-collab-blind features (the real lever for pair-level) (~half to one day)
**Why:** this is the right half of your "keep the pretrained backbone" intuition. Engagement features were trained to
suppress talking; fresh features were not. Replacing them is the only thing that can raise the *information* ceiling.

**Precondition to verify first (shell was flaky — check at session start):** do `data/collab_raw/` crops or the
`videos/` mp4s exist for re-extraction? (Memory says both should.) If crops exist, re-extract is cheap; if only
videos, re-run the detector at the same frames first.

**Two options, cheapest first:**
- **2a — ImageNet Swin-Tiny features (closest analog to your `BOOTSTRAP Ep01`).** Same architecture, *no* engagement
  fine-tuning. Directly tests "did engagement training destroy collab info?" Cheapest re-extraction.
- **2b — DINOv2 / CLIP features (higher reward).** Self-supervised, richer for interaction/pose. More compute.

**Files:**
- *New* `src/data/extract_fresh_features.py`: read the same crop list → fresh backbone → write a **parallel cache**
  `data/collab_cache_fresh/` with the **identical `feature_index.csv` schema** (so everything downstream is unchanged).
- *Re-run, unchanged:* `train_collab_video_level.py` and `train_collab_pair_nn.py` pointed at the fresh cache.

**Decisive probe:** does the within-`VID_ (4)` ceiling rise above ~0.56?
- **Yes** → pair-level collaboration is *unlocked*; carry the better backbone into Stage 3.
- **No** → the ceiling is the labels/scene, not the backbone → group-level is the honest product and we say so.

**Honest expected range (if it works):** pair LOVO **0.60–0.70**; not 0.90. Plan around 0.65.

---

### Stage 3 — Derive the GROUP verdict from PAIR reasoning (honors your hard constraint) (~2 hrs)
**Why:** you explicitly do not want a pure session classifier. A frame can hold both collaborative and
non-collaborative students. So the group verdict must be an **aggregation of per-pair predictions**, with the per-pair
verdicts kept visible.

**Files:**
- *Extend* `src/inference/group_collab.py`: add `PairCollabHead` (the Stage-1/2 pair model, saved) and
  `predict_group_from_pairs(pair_probs)` = aggregate per-pair P(collab) → group verdict via **mean + high-quantile**
  (e.g. "group is collaborating if a meaningful sub-group is"), and return the **per-pair breakdown** for
  explainability. This replaces "aggregate signals" with "aggregate pair *decisions*" — the bridge you asked for.
- Keep the existing signal-aggregation `GroupCollabHead` as the robust fallback/ensemble member.

**Evaluate:** session LOVO macro-F1 from pair-derived group verdicts vs the current 0.697. Target: ≥ 0.697 *and*
now backed by interpretable per-pair reasoning.

---

### Stage 4 — The "brain" (meta / stacking head) (~1-2 hrs, polish only)
**Why & honest limit:** stacking `[pair_nn_logit, 6 signals, engagement scalars]` → small meta-LR can calibrate and
add a few points. It **cannot manufacture missing information** — on engagement features it stays under the ~0.56
ceiling; on fresh features it polishes whatever Stage 2 unlocked.

**Files:** *new* `src/training/train_meta_head.py` — fit meta-LR on **out-of-fold** Stage-1/2 predictions (must use
OOF to avoid leakage), evaluate under the same LOVO. Adopt only if it beats the base model under LOVO.

---

## 4. Honest expectations (what each stage can realistically deliver)

| Stage | Pair-level macro-F1 | Group/session macro-F1 | Confidence |
|---|---|---|---|
| 0 baseline | ~0.46–0.50 | 0.697 | measured |
| 1 NN + signals (engagement feats) | 0.55–0.60 | ≥0.697 | bounded by ceiling |
| 2 fresh features (if it works) | 0.60–0.70 | 0.70–0.78 | uncertain, real upside |
| 3 pair→group derivation | — | ≥0.697, interpretable | high |
| 4 meta brain | +0.00–0.03 | +0.00–0.03 | low, polish |

**90% honestly at pair level is not on the table with this dataset.** The path to genuinely higher numbers is
*more balanced videos* (re-annotate the suspected autopilot pure videos) and *real spatial/gaze signals* (needs bbox
logging) — both are data work, not model work, and are the honest long-term ceiling-raisers.

---

## 5. Success vs. mirage (so we don't fool ourselves)

**A real result:** beats its own label-shuffled control by >0.03 under LOVO, AND the within-`VID_ (4)` ceiling moves.
**The mirage to reject:** any pair/random-split number near 0.80–0.90 that *collapses on a held-out video* — that is
scene memorization (this is exactly how the retired 0.795 happened on the 15-video subset). If a number looks too
good, hold out a whole video and watch it fall.

---

## 6. Guardrails (from prior hard-won lessons)

- **Engagement model is frozen.** Never touch Phase-1 weights or architecture.
- **LOVO + shuffle control + `VID_ (4)` ceiling on every claim.** A single split is the wrong instrument here.
- **Collapse guard** in any neural trainer (focal loss already burned us once → BCE + guard).
- **Full files, not snippets. Reasoning before code. Incremental "do this now."**
- **Never cite the retired 0.795 / ~90% pair numbers.** The number to defend is session-level **0.697**.
- New code connects via the **same `feature_index.csv` schema and `collab_pairs` API** — no pipeline redesign.

---

## 7. First actions when the new session starts (ordered)

1. Verify shell + paths; confirm `data/collab_cache/feature_index.csv` (6,869 rows) and `.npy` cache load.
2. **Stage 0:** run `train_collab_video_level.py` (reproduce 0.697 / chance) and add `ceiling_probe.py` (get the ~0.56).
3. **Stage 1:** build `train_collab_pair_nn.py` around the existing `CollaborationHead`; LOVO + controls.
4. Read the Stage-1 gate. Most likely → **Stage 2** precondition check (crops/videos), then `extract_fresh_features.py`
   (ImageNet Swin first), re-run the evaluator on the fresh cache, read the ceiling.
5. **Stage 3:** wire pair predictions → group verdict in `group_collab.py`; re-measure session LOVO.
6. Optional **Stage 4** meta head only if it beats base under LOVO.

Tell me "new session — start Stage 0" (or wherever you want to begin) and I'll build the files one by one.
