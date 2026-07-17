# Phase-2 — Honest 4-Day Plan (existing 33-video data, current codebase)

## Bottom line (read this first)
1. **No 4-day modeling change on the existing data has a *strong* chance of honestly beating the
   session-level 0.667.** Four signal families already failed within-scene; the limit is the dataset
   (scene confound 83–90%, one balanced video, coarse labels), not the model. So **do not bet the
   4 days on beating 0.667.**
2. **Run ONE cheap, decisive check on Day 1** (≤ half a day): a unified all-channel session model +
   ensemble, judged by the existing honest LOVO harness against a noise band. It will most likely
   *confirm the ceiling*; small chance of a real, small gain. Either way you then know, with evidence.
3. **Spend the rest delivering the "richer collaboration analysis" you chose as success** — a
   descriptive temporal + interaction-graph layer built on the signals we already have. It is
   honest, achievable in the time, needs no new data, and is far more compelling for a review than a
   0.667→0.69 number-chase. Plus review readiness.

This matches your answers: data **as-is**, success = **richer analysis (staged)**, runway = **4 days,
freeze if nothing has a strong honest chance**.

---

## Why the limit is data, not the Swin model (what four negatives proved, now backed by the field)
- The professional way to detect "who interacts/collaborates with whom" is **Actor Relation Graphs**
  (Wu, CVPR 2019), **Actor-Transformers** (Gavrilyuk, CVPR 2020), and **Dual-AI** (CVPR 2022). They
  reach state-of-the-art on **Volleyball (~4,830 labelled clips)** and **Collective Activity** —
  datasets with *many balanced groups* and actor/frame-level annotations.
- We have **33 videos, 1 internally balanced scene, pair-level (not frame-level) labels.** That is
  orders of magnitude below what those relational models need; on this data they overfit — exactly
  the "session-in-disguise" effect we measured. A bigger/graph/transformer head will not fix a data
  deficit.
- Pairwise collaboration-from-video is **data-scarce**: classroom video research works at
  behaviour/engagement/group level (EduNet, SCB-Dataset, classroom *group* engagement). Our
  **session-level 0.667 is a defensible result in that class** — not an underachievement.

**Conclusion:** "make Swin learn within-scene collaboration" is blocked by the dataset, not solvable
by a 4-day modelling trick. The honest, winnable move is to deliver richer *analysis* on top of the
representation Swin already learned.

---

## Day 1 (morning) — the one cheap experiment, with a hard go/no-go
**What:** the 0.667 head used only `signals(6) + feat-scalars(4)`. We now also have **geometry**
(`pairs_features_geom.npz`) and **gaze** (`pairs_features_gaze.npz`) per pair — never combined at the
session level. Build a session model over the **union**, aggregated `[mean, std]` per video:
- relational signals (6) + feature scalars (4)  ← current 0.667 recipe
- gaze aggregates: mean mutual-gaze, mean turn-taking, mean one-way gaze (plausible collaboration mechanism)
- geometry aggregates: proximity / co-visibility / approach (session-correlated)

**Three principled levers — judged honestly; pick the gate-winner, do not stack blindly:**
- **(most promising) subgroup-aware pooling** — replace plain `[mean, std]` over pairs with
  **top-k / high-quantile** pooling of the interaction + gaze signals. Mechanism: a session is
  collaborative if *some* pairs interact strongly, not if the *average* does — `mean` washes that
  out. Cheapest change, best-motivated, the single best shot at a real honest gain.
- **gaze + interaction session-aggregates** — add mean/peak mutual-gaze, turn-taking, joint-activity.
  Gaze separated sessions only weakly within-scene, but *aggregated across a session* it may add real
  signal. Exclude pure co-tracking-duration features (recording artefact, not collaboration).
- **ensemble** — average the LOVO probabilities of the separate honest per-view models
  (signals, gaze, geometry); ensembling is the safest small-N generalisation squeeze.

Keep every config tiny (30 sessions ⇒ ≤ ~8 features, strong L2) and let the honest gate choose.

**Evaluate with the existing harness** (`train_collab_video_level.py --validate` style): LOVO
macro-F1 + label-shuffle floor + per-video breakdown.

**DECISION GATE (noise-aware — 30 sessions ⇒ 1 flip ≈ 0.03 F1):**
- Adopt a new config **only if** it reaches **≥ ~0.71 reproducibly** (≥ +1.5 videos over 0.667),
  **clearly above the shuffle floor**, **stable across L2**, and **not driven solely by
  co-tracking-duration features** (those are recording artefacts, not collaboration).
- Otherwise the gain is within noise → **do not claim it; freeze at 0.667.**

**Honest odds:** ~30–45 % chance of a small real gain, ~55–70 % it's flat/within-noise. It costs a
few hours and removes all doubt. *This is the highest-probability honest improvement on this data —
and it is deliberately modest, because small-N honest modelling cannot safely attempt more.*

---

## Days 1 (afternoon)–3 — the real deliverable: a *richer collaboration analysis* (your "success")
Stop framing success as a binary score and deliver what the data genuinely supports: a **descriptive,
honest interaction layer**, built entirely from existing artefacts (per-clip Swin features, relational
signals, `headpose.csv`, `bboxes_geom.csv`). No new claim of within-scene classification — it is an
*analysis/exploration* tool, with the binary collaboration verdict kept honestly at session level.

Three components (all achievable, numpy + matplotlib/networkx or one HTML view):
1. **Per-session collaboration timeline** — plot the interaction signals over time for each video:
   mutual-gaze rate, joint-activity, turn-taking, proximity. Shows *when* a session looks
   collaborative. Turns the frozen 0.667 verdict into an interpretable story.
2. **Who-interacts-with-whom graph** — per session, nodes = students, edges weighted by
   mutual-gaze / co-activity. A clean, intuitive visual of group structure (this is the
   actor-relation idea, used *descriptively* at a scale the data supports).
3. **Driver breakdown per verdict** — for each session's Collaborative/Not call, show the signals
   that drove it (you already have this in `demo_group_collab.py --video`; extend it visually).

**Framing for the review (honest):** "Collaboration is *classified* at the session level (0.667,
leak-free). On top of that we provide an *interaction-analysis* layer — timelines and interaction
graphs — that makes the group dynamics interpretable. Pair-level *classification* is a documented
negative on this data; the analysis layer is descriptive, not a validated pair classifier." This is
exactly the **richer, staged** success you picked, and it cannot be honestly attacked.

---

## Days 3–4 — review readiness (the guaranteed value)
- **Deck**: done (4-family story, slide 13 four cards). Add 1 slide showing the new interaction
  timeline/graph. Rehearse the honest narrative.
- **Reviewer-Q&A prep**: ready answers for "why 0.667 not 0.90?", "what is LOVO / why?", "why is
  pair-level a negative?", "could gaze/audio fix it?" (you now have evidence-backed answers).
- **Demo**: `demo_group_collab.py --all` is deterministic; add the timeline/graph view; verify on a
  fresh clone / Colab.
- **Repo hygiene**: `DEPLOYMENT_README.md` + `PHASE2_HONEST_RESULT.md` are the basis; ensure scripts
  run end-to-end; tag a release.

---

## Explicitly NOT attempted in 4 days (considered and deferred — with reasons)
- **Within-scene pair head (any architecture incl. actor-transformer/graph):** data-blocked; would
  overfit 33 videos. Four families already failed honestly.
- **Audio / active-speaker turn-taking (TalkNet / LoCoNet on AVA-ActiveSpeaker style):** the
  **single highest-upside *new* signal** — who-speaks-when and A↔B alternation is the strongest
  collaboration cue we have never used. But ASD on **noisy, far-field, overlapping WhatsApp classroom
  audio** is a multi-day engineering build with real failure risk (the literature flags overlap +
  noise + the ~6 s context needed). **This is the #1 *future* lever, not a 4-day safe bet.** First
  cheap step (out of scope now): check whether the video files even carry usable audio tracks.
- **Bigger backbone / more epochs / deeper transformer / DINOv2 re-extraction:** proven not the
  bottleneck (Swin features already generalize; engagement = 0.90).
- **Data redesign (new balanced recordings + frame-level interaction labels):** the *real* unlock and
  the correct semester-scale project — but you ruled out new data, and it cannot be done in 4 days.

---

## Recommendation
Run the Day-1 cheap check; **expect to freeze the classifier at 0.667.** Put your real energy into the
**interaction-analysis layer + review readiness** — that is where 4 days converts into a visibly
stronger, still-honest Phase-2. If you later want to genuinely push pair-level, the highest-probability
direction is **audio active-speaker turn-taking + a small amount of balanced, frame-annotated data** —
documented here as future work.

## Sources (professional grounding)
- Actor Relation Graphs — Wu et al., CVPR 2019: https://arxiv.org/abs/1904.10117
- Actor-Transformers — Gavrilyuk et al., CVPR 2020: https://arxiv.org/pdf/2003.12737
- Dual-AI — CVPR 2022: https://arxiv.org/pdf/2204.02148
- AVA-ActiveSpeaker: https://arxiv.org/pdf/1901.01342 · LoCoNet: https://arxiv.org/pdf/2301.08237
- EduNet classroom video dataset: https://www.semanticscholar.org/paper/18e8d02d7219029b5009915674f204e492acf097 · Classroom group engagement dataset: https://pmc.ncbi.nlm.nih.gov/articles/PMC12003871/
