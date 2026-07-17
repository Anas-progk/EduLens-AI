# Collaboration Annotation Guide
## Phase 2 — Pair Labeling Rulebook

> **Your labels are the most important input to this model.**
> A perfectly designed architecture trained on noisy labels will fail.
> A simple architecture trained on clean, consistent labels will succeed.
> Read this guide fully before annotating a single pair.

---

## 1. The Two Labels

| Label | Key | Meaning |
|-------|-----|---------|
| **C** — Collaborative | `C` | These two people are actively working together, exchanging ideas, or jointly attending to a shared task |
| **N** — Not Collaborative | `N` | These two people are physically near each other but NOT engaged in a joint activity |
| **S** — Skip | `S` | Video quality, occlusion, or ambiguity makes it impossible to label confidently |

**There is no "maybe" label.** If you cannot decide in 5 seconds, press `S`. Ambiguous labels corrupt the model worse than fewer labels.

---

## 2. The Collaboration Definition

Two people are **Collaborative (C)** if they satisfy **at least 2 of the following 4 criteria** simultaneously during the clip:

| # | Criterion | What to look for |
|---|-----------|-----------------|
| 1 | **Verbal exchange visible** | Mouth movement from one person directed toward the other; the other person reacts (nods, replies, turns head) |
| 2 | **Body oriented toward each other** | Torsos or faces angled toward each other — NOT both facing the teacher/board |
| 3 | **Shared focus on a common object** | Both looking at the same laptop, notebook, phone, paper, or screen |
| 4 | **Active listening signals** | One person speaking, the other showing clear attention: nodding, leaning forward, eye contact direction, responsive posture |

**If only 1 criterion is satisfied: label N.**
**If 2 or more are satisfied: label C.**

---

## 3. Step-by-Step Annotation Workflow

### Before you start a session

**Do a scene-level scan first.** Watch the full video clip (8 frames) for both persons together before pressing any key. Ask:

1. What are they doing individually?
2. Are they interacting with each other or ignoring each other?
3. What is the context — group task, individual work, listening to teacher?

This 5-second scan prevents the most common mistake: snap-judging from a single frame.

### For each pair

1. Watch Person A's 8-frame grid
2. Watch Person B's 8-frame grid
3. Ask: "Is there evidence of a joint activity between them?"
4. Check the 2-of-4 rule
5. Press `C`, `N`, or `S`

---

## 4. Role Identification — The Most Important Concept

Real classroom collaboration has **roles**. Misunderstanding these roles is the #1 source of annotation errors.

### Active Speaker
- Mouth moving, gesturing, leaning forward
- Engagement probability typically HIGH
- **Always label C if the other person is Active Listener**

### Active Listener
- Facing the speaker, nodding, still but attentive
- Engagement probability may be MODERATE (not moving = lower HOG activity)
- **CRITICAL: Active Listener is COLLABORATIVE, not "not engaged"**
- Do NOT label N just because one person is quieter or more still

### Passive Observer
- Present near collaborating pair but NOT part of the exchange
- Looking at their own work, phone, or at the teacher
- Body NOT oriented toward the pair
- **Label N for this person when paired with either of the collaborators**

### Independent Worker
- Working alone on their laptop/notebook
- Physical proximity to others means nothing
- **Label N when paired with anyone they are not directly exchanging with**

### Distracted Person
- Phone, sleeping, looking around randomly
- **Label N for any pair involving this person**

---

## 5. Group Annotation Strategy

When you see a group of 3+ people, think in pairs:

**Example: A, B, C where A and B are collaborating, C is working independently**

| Pair | Label | Reason |
|------|-------|--------|
| (A, B) | C | Directly exchanging, facing each other |
| (A, C) | N | A ignoring C, C working alone |
| (B, C) | N | B ignoring C, C working alone |

**Example: A, B, C all collaborating as a group**

| Pair | Label | Reason |
|------|-------|--------|
| (A, B) | C | Both part of group discussion |
| (A, C) | C | Both part of group discussion |
| (B, C) | C | Both part of group discussion |

**Key rule:** Collaboration is transitive in a group. If A, B, C are all in the same conversation, all three pairs are C — even if B and C happen to not be facing each other directly at this moment.

---

## 6. Hard Cases — Explicit Rules

These are the cases where annotators most commonly disagree. Follow these rules exactly for consistency.

### Hard Case 1: Silent Laptop Collaboration
**Scenario:** Two students working on laptops side by side, occasionally glancing at each other's screens but not talking.

**Rule:** If you can observe **both looking at the same screen** OR **one pointing at the other's screen**, label C. If they are both staring at their own screens with no visible exchange: label N.

### Hard Case 2: Back-of-Head Only
**Scenario:** Person B is facing away from camera; you can only see the back of their head.

**Rule:** Use Person A's behavior as your primary evidence. If Person A is clearly speaking to, listening to, or orienting toward B — and B's posture suggests engagement with A (leaning toward A, not toward the board) — label C. If Person A is ignoring B, label N. Press S only if you truly cannot tell either person's behavior.

### Hard Case 3: Teacher-Student Interaction
**Scenario:** Teacher is walking around, briefly interacting with one student while another student is nearby.

**Rule:** The student directly talking to the teacher — label that pair as N (teacher is not a "peer collaborator"). The nearby student who is just watching: also N. Teacher-student pairs should generally be avoided or skipped.

### Hard Case 4: Quiet Listener During Group Discussion
**Scenario:** A, B, C are at the same table. A and B are talking. C is silently listening, nodding occasionally, clearly part of the discussion.

**Rule:** C is an Active Listener. Label (A,C)=C and (B,C)=C. A person does not need to speak to be a collaborator.

### Hard Case 5: One Person Sleeping / Distracted
**Scenario:** A and B are sitting together. A is working; B is on their phone or asleep.

**Rule:** Always N. There is no collaboration if one person is checked out, regardless of proximity or what the other person is doing.

### Hard Case 6: Delayed Response / Looking Away Momentarily
**Scenario:** A says something to B. B looks at their notebook for 2 seconds then responds.

**Rule:** The 8-frame window (covering ~2.5 seconds) should capture the full exchange. If you can see the initiation AND the response within the clip: label C. If only one side is visible: use the 2-of-4 rule on what you CAN see.

### Hard Case 7: Students Facing Teacher But Discussing with Neighbor
**Scenario:** Both A and B face the board, but you can see A whispering to B and B reacting.

**Rule:** Visible verbal exchange (criterion 1) + active listening signals (criterion 4) = 2 criteria satisfied = C. Facing the teacher does not override visible peer exchange.

### Hard Case 8: Very Large Group (5+ people)
**Scenario:** 6 students around a table, all part of one discussion.

**Rule:** Annotate every visible pair systematically. Two people at opposite ends of a large table who are NOT directly engaged with each other despite both being "in the group" should be labeled N. Focus on direct dyadic exchange, not group membership.

---

## 7. Common Mistakes to Avoid

| Mistake | Why it happens | Correct approach |
|---------|---------------|-----------------|
| Labeling Active Listener as N | They are quiet/still so seem "not engaged" | Stillness + attention = collaboration |
| Labeling proximity as C | Two students sit next to each other | Proximity alone is NEVER enough for C |
| Labeling N because you can't see faces | Back-of-head clips | Use body orientation and Person A's behavior |
| Labeling C for teacher-student | Teacher is helping a student | Not peer collaboration; label N or skip |
| Labeling C for a group member who is clearly distracted | They're "at the same table" | Check each person's actual attention state |
| Over-using S | Caution | Skip only when genuinely impossible, not just uncertain |
| Under-using S | Wanting to annotate everything | Forced guesses corrupt the model more than fewer samples |

---

## 8. Consistency Rules (Follow These Exactly)

These rules exist so that your annotations are consistent session to session.

**Rule 1: Symmetric labeling**
If you label (A, B) = C, you should also label (B, A) = C if the annotator sees that pair. Collaboration is symmetric.

**Rule 2: The 2-second rule**
If you have been staring at a pair for more than 5 seconds without deciding, press S. Prolonged indecision = genuine ambiguity = noisy label.

**Rule 3: Do not over-correct for class balance**
Do not think "I've been pressing N too much, let me press C more." Label what you see. Class imbalance is handled by the training code with pos_weight, not by forcing balance in labels.

**Rule 4: Re-watch before S**
Before pressing S, watch the clip one more time. Many "skip" decisions disappear on the second watch.

**Rule 5: Context window**
The annotator shows 8 frames. These cover roughly 2-3 seconds of video. Collaboration does not have to be visible in ALL 8 frames — a clear exchange in 3-4 frames is enough for C.

---

## 9. Pilot-First Workflow (Do This Before Annotating All 40 Videos)

**This is the most important section. Do not skip it.**

### Phase A: Micro Pilot (2-3 videos, ~80-120 pairs)

1. Run the processor on 5 videos only (pick diverse ones: different room sizes, lighting, group sizes)
2. Annotate 80-120 pairs
3. **Stop and review your annotations:**
   - Are your C labels obviously collaborative when you re-watch?
   - Are your N labels obviously non-collaborative?
   - Are you pressing S more than 15% of the time? (If yes, your criteria may be too strict)
   - Look at the C/N ratio. Is it roughly 30-60% C? (If C < 20%, you may be under-labeling; if C > 70%, you may be over-labeling)

4. Only proceed to full 40-video annotation after this self-check passes.

### Phase B: Full annotation (all 40 videos)

- Target: 400+ labeled pairs (C or N), S does not count
- Aim for roughly 35-50% C across all labels
- Take breaks every 30-40 minutes; annotation fatigue causes drift

### Phase C: Quick model pilot

After 150+ labels (even before finishing all 40 videos):

```bash
python src/data/collab_dataset.py --extract_features --model_path weights/best_clip_model.pth
python src/data/collab_dataset.py --build_splits
python src/training/train_collab.py
```

Check the confusion matrix. If the model predicts everything as N: your labels are too imbalanced or the model needs more C pairs. If accuracy is 50%: something is wrong with features or labels. If F1 > 0.55 on 150 pairs: you are on the right track.

---

## 10. Signal Meanings for Annotation Context

Understanding what the model sees helps you annotate better:

| Signal | What the model uses it for | Annotation implication |
|--------|--------------------------|----------------------|
| Proximity | Gate — far apart = definitely not collab | Make sure collab pairs are visibly close |
| Facing score | Directional orientation heuristic | Note if people are side-by-side vs. face-to-face |
| Activity correlation | Shared engagement patterns | If both always high or both always low, that's a clue |
| Turn-taking | Verbal exchange detection | Alternating engagement = conversation |
| Engagement sync | Joint focus moments | Both suddenly attentive = shared stimulus |
| Bbox movement | Physical response sync | One gestures, other reacts physically |

None of these signals alone determines the label. The model combines all 6. Your annotation should reflect what a human observer sees, not what you think the signals will detect.

---

## 11. Quick Reference Card

```
COLLABORATIVE (C) — needs 2+ of:
  [ ] Visible verbal exchange (mouth movement + reaction)
  [ ] Body oriented toward each other
  [ ] Shared focus on common object/screen
  [ ] Active listening signals (nod, lean, eye direction)

NOT COLLABORATIVE (N):
  [ ] Physically close but doing own thing
  [ ] Both facing teacher / board
  [ ] One person sleeping / on phone / distracted
  [ ] Near a group but not part of exchange

SKIP (S):
  [ ] Cannot see either person's face or body orientation
  [ ] Clip is too blurry / dark
  [ ] Genuinely 50/50 after two watches

HARD CASES:
  Active Listener = C (not N)
  Laptop collaboration without talking = check shared screen gaze
  Back-of-head = use Person A's behavior
  Teacher-student = N
  Quiet group member clearly participating = C
  One person distracted = always N
```

---

## 12. Annotation Session Log

Keep a brief log as you annotate (no need to be formal — even mental notes help):

- What proportion of pairs are you labeling C today?
- Are there specific video types that are consistently hard to label?
- Are there any labels you are unsure about that you should revisit?
- Any patterns in what makes C vs. N clear or ambiguous in your videos?

This reflection improves consistency across multiple annotation sessions.
