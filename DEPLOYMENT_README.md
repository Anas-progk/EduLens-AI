# Deployment — RTRP review run guide

The deployable system is two frozen models on one Swin-Tiny backbone.

| Phase | Question | Honest metric | Weights |
|---|---|---|---|
| 1 — Engagement | Is this person engaged? | macro-F1 **0.90** (acc 92.4%) | `weights/best_clip_model.pth` |
| 2 — Group collaboration | Is this whole session collaborating? | **LOVO macro-F1 0.667** (acc 66.7%) | `weights/best_collab_group_fresh.npz` |

## 1. The review-safe demo (run this live — numpy only, no GPU/torch/camera)

```bash
python demo_group_collab.py --all                 # every session + the honest LOVO headline
python demo_group_collab.py --video "VID_ (3)"    # one session, full signal breakdown
python demo_group_collab.py                        # 1 collaborative + 1 not, then a tip
```

It loads `weights/best_collab_group_fresh.npz` + `data/collab_pairs_unique_fresh/pairs_features.npz`
and is byte-identical in its feature math to training, so **demo == deploy == train**. It cannot
fail live. Expected headline: `macro-F1 = 0.667  accuracy = 66.7%  sessions = 30 (C=14/N=16)`,
deployed-head in-sample agreement 23/30.

## 2. What to claim (and not claim) to reviewers

- **Claim — group/session level:** "Is this whole group collaborating?" → macro-F1 **0.667** under
  Leave-One-Video-Out on all 33 videos. Majority baseline F1 0.348; label-shuffle floor 0.396 → real,
  well above chance. Frozen Swin-Tiny engagement backbone + 6 relational signals + 4 feature scalars,
  aggregated `[mean,std]` over the group's pairs (20-d) → small regularized logistic head.
- **Do NOT claim — pair level:** "Which two specific people collaborate?" is **chance** within a scene.
  A pooled pair score (0.608) only *looks* above chance — it is the session signal in disguise (proven
  per-video in `PHASE2_HONEST_RESULT.md`). Appearance, relational, and geometry features were all tested
  and rejected on the within-scene gate (VID_(4) per-feature AUC ≈ 0.5).

## 3. Reproduce the number (numpy only)

```bash
python src/training/train_collab_video_level.py --npz data/collab_pairs_unique_fresh/pairs_features.npz --l2 1.0 --min_pairs 3
# or, from the deploy module:
python src/inference/group_collab.py --all
```

## 4. Live video demo (optional, GPU/Colab) — one wiring TODO

The deterministic demo above is the recommended review artifact. For a live-on-video group verdict,
the building block is `LiveGroupCollab` in `src/inference/group_collab.py`: feed it each person's 768-d
Swin feature per frame (from the engagement model) and call `.verdict()`.

**TODO (only if you want the live video demo):** `run_collab_inference.py` / `multi_person_collab_inference.py`
still default to the retired pair model `weights/best_collab_model.pth` (the dead pair-level path).
Point the live path at `LiveGroupCollab(GroupCollabHead.load("weights/best_collab_group_fresh.npz"))`
instead. Left unchanged here because it needs torch + video and cannot be tested in this environment —
flagged rather than edited blind.

## 5. Files that matter

```
weights/best_clip_model.pth                       Phase-1 engagement (Swin-Tiny + TemporalTransformer)
weights/best_collab_group_fresh.npz               Phase-2 group head (20-d, mode=both)  <-- the 0.667 model
data/collab_pairs_unique_fresh/pairs_features.npz fresh per-person features (883 pairs / 33 videos)
demo_group_collab.py                              review-safe deterministic demo
src/inference/group_collab.py                     GroupCollabHead + LiveGroupCollab (deploy module)
src/training/train_collab_video_level.py          canonical honest evaluator (LOVO)
PHASE2_HONEST_RESULT.md                           full write-up incl. the pair-level negative result
Engagement_Collaboration_RTRP.pptx                17-slide review deck (KMIT template)
```
