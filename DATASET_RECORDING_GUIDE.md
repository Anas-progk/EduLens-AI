# Custom Engagement Dataset Recording Protocol

## Why you need custom data

DAiSEE has a fundamental imbalance: 95% Engaged, 5% Not Engaged.
With only ~333 NE clips in the training set, no model can learn stable NE features.
A custom dataset with a **controlled 60:40 or 50:50 ratio** will directly fix this.

---

## What the model needs to learn

Your model should learn engagement as **sustained behavioral patterns**, not appearance.

| Engaged behaviors                   | Not Engaged behaviors              |
|--------------------------------------|------------------------------------|
| Looking at screen / notebook / board | Persistent gaze away from workspace|
| Typing / writing actively            | Phone usage (scrolling, texting)   |
| Reading, studying, problem-solving   | Sleeping / resting head on desk    |
| Active discussion with partner       | Off-topic conversation             |
| Focused posture, leaning in          | Slouching, leaning far back/away   |
| Occasional natural head movement     | Sustained looking out of window    |
| Brief stretches, then back to work   | Staring blankly for extended time  |

**Key rule**: A 2-second phone glance is Engaged. A 25-second phone scroll is Not Engaged.
Duration and persistence define the label — not isolated appearance.

---

## Equipment requirements

| Item                  | Specification                            |
|-----------------------|------------------------------------------|
| Camera                | 720p minimum (1080p recommended)         |
| Frame rate            | 30 FPS (do not go below 24 FPS)          |
| Lens angle            | Normal/wide (70-90°), not fisheye        |
| Lighting              | Consistent, diffuse — no harsh shadows  |
| Mounting height       | Eye-level to slightly above (not from floor) |
| Distance from person  | 1.5–2.5 meters (full upper body visible) |

---

## Scene setup

### Room layout
- Desk + laptop/notebook visible in frame
- Person seated (not standing)
- Background: neutral — wall, shelf, or neutral poster
- Lighting: 2 light sources minimum to avoid harsh one-sided shadows

### Multiple lighting conditions to record
1. Well-lit room (daytime, natural + artificial)
2. Dimmer room (evening, desk lamp only)
3. Side lighting (lamp to one side)

Record EACH PERSON in ALL THREE lighting conditions.
This prevents the model from learning "bright room = engaged".

### Multiple camera angles to record
1. Frontal (camera directly facing person)
2. Slight off-center left (15–20° angle)
3. Slight off-center right (15–20° angle)

Record EACH SESSION from the primary frontal angle.
Off-center angles are for augmentation variety — keep majority frontal.

---

## How many people

| Minimum (viable dataset)   | Recommended (research quality) |
|----------------------------|--------------------------------|
| 20 people                  | 40–60 people                   |
| 10 Engaged sessions each   | 10 sessions each               |
| 10 Not Engaged sessions    | 10 sessions each               |

**Important for generalization:**
- Include diverse: gender, age (18–35 best), glasses/no glasses, hair styles
- Include: people with laptops, people with notebooks, people with tablets
- Do NOT use the same room for all recordings if possible (2–3 rooms minimum)

---

## Session structure

### Each recording session = 5 minutes of continuous video

**Engaged session script** (person should be genuinely working):
- 0:00–1:30 → Actively reading / typing (steady gaze on screen/notebook)
- 1:30–2:00 → Brief natural pause, look around, then back to work (STILL ENGAGED)
- 2:00–3:30 → Continue working (mix of typing, reading, thinking with minor head moves)
- 3:30–4:00 → Stretch, adjust posture, glance away 2–3 sec (STILL ENGAGED)
- 4:00–5:00 → Resume focused work

**Not Engaged session script** (deliberate distraction behaviors):
- 0:00–0:45 → Stare blankly at nothing, no activity
- 0:45–1:30 → Scroll phone / look at phone for extended time
- 1:30–2:00 → Look away from workspace repeatedly (out window, at ceiling)
- 2:00–2:45 → Head resting on arms, drowsy appearance
- 2:45–3:30 → Talk to someone off-camera (no work being done)
- 3:30–4:30 → Distracted social media browsing on device
- 4:30–5:00 → Random gaze drift, fidgeting with objects, no task focus

---

## Clip labeling rules (CRITICAL)

DAiSEE labeled individual frames — that is why it fails.
Your dataset should be labeled at the **clip level (8-second windows)**.

### Labeling rules:
- Label is assigned per 8-second clip window
- If > 60% of those 8 seconds show NE behavior → label = **Not Engaged**
- If > 60% of those 8 seconds show Engaged behavior → label = **Engaged**
- Clips where the person transitions mid-clip → discard (ambiguous)
- Never mix labels within a single clip

### Clip extraction:
```
Clip 1:  frames 0–119    (0.0–3.9s @ 30fps, sample every 4th frame = 8 frames)
Clip 2:  frames 60–179   (2.0–5.9s, overlapping windows by 2s)
Clip 3:  frames 120–239  ...
```
Use overlapping windows (50% overlap) to get more clips from the same video.

---

## Folder structure

```
data/custom/
    raw/
        person_001/
            session_engaged_01.mp4
            session_engaged_02.mp4
            session_notengaged_01.mp4
            session_notengaged_02.mp4
        person_002/
            ...
    frames/
        person_001_session_e01_clip001_f0001.jpg
        person_001_session_e01_clip001_f0002.jpg
        ...
    labels_custom.csv
        columns: image_path, label, person_id, clip_id
```

---

## How to combine with DAiSEE

After recording, combine both datasets:

```python
# In build_splits.py, add:
custom_df = pd.read_csv("data/custom/labels_custom.csv")
daisee_df = pd.read_csv("data/processed/daisee/labels.csv")
combined  = pd.concat([daisee_df, custom_df], ignore_index=True)
combined.to_csv("data/processed/combined/labels.csv", index=False)
```

Person-level splitting will automatically prevent leakage across datasets.

---

## Expected dataset size and balance

| Split | Persons | Clips (est) | Engaged | Not Engaged |
|-------|---------|-------------|---------|-------------|
| Train | 28      | ~2,800      | ~1,400  | ~1,400      |
| Val   | 6       | ~600        | ~300    | ~300        |
| Test  | 6       | ~600        | ~300    | ~300        |

With 40 people and this ratio, your dataset will have ~3.5x more NE clips than DAiSEE alone.
The combined dataset (DAiSEE + custom) will be:
- Total NE clips: ~333 (DAiSEE) + ~1,400 (custom) = ~1,733
- Total E clips:  ~5,250 (DAiSEE) + ~1,400 (custom) = ~6,650
- Combined NE%: ~21% (vs 5.8% before)

This is a fundamentally different and much more learnable distribution.

---

## Recording checklist (per session)

Before recording:
- [ ] Camera stable (tripod or mounted, no hand-holding)
- [ ] Consistent lighting (no window behind person, no glare on screen)
- [ ] Full upper body visible in frame
- [ ] Background is neutral (not cluttered)
- [ ] Subject is NOT looking at the camera
- [ ] Record 5-minute session without interruption

After recording:
- [ ] Note: Person ID, session type (E/NE), lighting condition, angle
- [ ] Label the session (E or NE) immediately — memory is unreliable later
- [ ] Store in correct folder structure

---

## Quality control

Discard clips if:
- Person is fully outside frame for > 2 consecutive seconds
- Severe glare, blur, or over-exposure makes face unrecognizable
- Mixed behavior (person starts engaged then gets distracted mid-clip)
- Recording equipment moved during session

Accept with note:
- Brief occlusion < 1 second (natural behavior)
- Slight motion blur (real classroom condition, model should be robust)
- Glasses, hats, hoods (important for generalization — keep these)

---

## Single-person vs multi-person recording

**Phase 1 (start here): Single-person recordings**
- Easier to control and label
- Each person fills the frame
- Direct use with current SwinClipModel

**Phase 2 (after Phase 1 works): Multi-person recordings**
- 3–6 people at desks in same frame
- Use multi_person_inference.py (YOLO detection + tracking)
- Label each person independently from the same video
- Richer collaborative/non-collaborative signal

Do Phase 1 first. Add Phase 2 data once the single-person model is validated.
