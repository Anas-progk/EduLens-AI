# Phase 2 Architecture: Collaboration Detection
## Status: BUILDING — May 2025

---

## 1. What Phase 2 Is and Is NOT

**Phase 1 (DONE):** Per-person binary engagement: Engaged vs Not Engaged.  
**Phase 2 (NOW):** Per-person binary collaboration: Collaborative vs Not Collaborative.

Collaboration is NOT a visual label you can read from a single frame.  
It is an **interaction property** derived from how two or more people behave **relative to each other over time**.

**Key definitions (locked):**
- Engagement = individual focus on a relevant task (can happen alone or in a group)
- Collaboration = active participation in an interaction loop with another person

A person is labeled **Collaborative** if they are part of an active exchange:
- talking + getting a response
- explaining + being listened to (with active signals: nods, lean-in, eye contact direction)
- turn-taking behavior confirmed over multiple seconds

A person is labeled **Not Collaborative** if:
- they are studying alone (engaged but isolated)
- they are sitting in a group but not interacting (passive)
- they are distracted (phone, sleep) — also not collaborative
- they are a passive audience for a teacher (listening to lecture ≠ collaboration)

---

## 2. Full System Architecture (Phase 2)

```
Video Frame (any source)
        ↓
  Person Detection (HOG / YOLO fallback)
        ↓
  Identity Tracking — ByteTrack / SimpleIoU
        ↓
  ┌─────────────────────────────────────────────┐
  │  PermanentReID Database  (SQLite)           │
  │  Track_ID → GlobalPersonID (persists)       │
  └─────────────────────────────────────────────┘
        ↓ GlobalPersonID assigned
  Per-Person 8-frame buffer (same as Phase 1)
        ↓
  ┌──────────────────────────────────────────────────────┐
  │  SwinClipModel  (FULLY FROZEN, Phase 1 weights)      │
  │  Input: (1, 8, 3, 224, 224)                          │
  │  Output A: clip_feat (768-d) ← temporal embedding    │
  │  Output B: engagement_logit (2-d) ← E / NE decision  │
  └──────────────────────────────────────────────────────┘
        ↓ clip_feat (768-d) per person
  ┌──────────────────────────────────────────────────────┐
  │  InteractionSignalComputer                           │
  │  For each PAIR (person_A, person_B):                 │
  │    - proximity_score (0-1)                           │
  │    - facing_score (0-1)                              │
  │    - activity_correlation (−1 to +1)                 │
  │    - turn_taking_score (0-1)                         │
  │  → interaction_signals (4-d vector)                  │
  └──────────────────────────────────────────────────────┘
        ↓ (feat_A, feat_B, interaction_signals)
  ┌──────────────────────────────────────────────────────┐
  │  CollaborationHead  (TRAINABLE, Phase 2)             │
  │  Projection: 768 → 128 (each person)                 │
  │  Cross-attention: attend A features onto B           │
  │  MLP: [128 + 128 + 4] → 64 → 1 (sigmoid)            │
  │  Output: collab_prob (0-1) for this PAIR             │
  └──────────────────────────────────────────────────────┘
        ↓ max collab_prob across all pairs for person X
  Per-Person Final Output:
    {
      "global_id":    12,           ← permanent across sessions
      "track_id":     3,            ← local to this session
      "engagement":   "Engaged",
      "collaboration":"Collaborative",
      "eng_prob":     0.87,
      "collab_prob":  0.74,
    }
```

---

## 3. CollaborationHead Architecture (Detail)

```python
class CollaborationHead(nn.Module):
    """
    Input: two temporal clip features (768-d each) + 4 interaction signals
    Output: pairwise collaboration probability (sigmoid)

    Design choices:
      - Project 768 → 128: reduce dimensionality before cross-attention
        (avoids overfitting on small collab dataset ~200-400 pairs)
      - Cross-attention: A attends over B's tokens (and vice versa)
        This is what distinguishes collaboration from individual behavior.
        Two engaged people who don't interact should NOT be labeled collaborative.
        Cross-attention forces the model to ask: "does A's behavior relate to B's?"
      - Concat with interaction_signals: spatial proximity and temporal patterns
        These 4 signals encode PHYSICAL interaction evidence that appearance alone misses
      - Symmetric loss: train with both (A,B) and (B,A) orderings
        Collaboration is symmetric: if A collab with B, then B collab with A
    """
    def __init__(self, feat_dim=768, proj_dim=128, n_heads=4, signal_dim=4):
        # proj_A, proj_B: Linear(768, 128) with LayerNorm
        # cross_attn: MultiheadAttention(128, 4 heads)  A queries B's keys/values
        # mlp: Linear(128+128+4, 64) → GELU → Dropout(0.3) → Linear(64, 1)
        ...
```

**Why NOT a GNN:** GNNs need graph construction per frame which is brittle with noisy tracking (we saw tracking fluctuate 5→0→5 persons). The cross-attention approach handles variable numbers of people naturally and degrades gracefully when tracking is noisy.

**Why NOT full TimeSformer for collaboration:** We already have temporal modeling in the engagement backbone. Adding a second full temporal model doubles compute and requires much more data. The collaboration signal is in the RELATIONSHIP between two people's engagement features, not in temporal dynamics of the collaboration itself.

---

## 4. Interaction Signals (The 4-d Vector)

These 4 signals are computed from geometry and temporal patterns — NOT from the model:

### Signal 1: Proximity Score
```
proximity = 1 - (center_dist / frame_diagonal)
```
- `center_dist`: Euclidean distance between bbox centers (pixels)
- `frame_diagonal`: diagonal of the full frame
- Result: 0 = far apart, 1 = touching/overlapping
- Collaboration threshold: typically proximity > 0.4 for seated students

### Signal 2: Facing Score
```
# Estimate whether person A "faces toward" person B
# Uses relative x-position and bbox shape as proxy for orientation
if A is to the LEFT of B:
    facing_score = sigmoid(-(center_A.x - center_B.x) / frame_width)
    # higher when A's center is clearly left of B (natural "facing right toward B")
```
- This is a HEURISTIC — we cannot do full 3D head pose estimation without depth
- But it's directionally correct: students facing each other have their centers arranged consistently
- Range: 0 to 1

### Signal 3: Activity Correlation
```
# Pearson correlation of engagement probability timeseries
# over last 16 inference decisions per person
corr = pearsonr(engagement_probs_A[-16:], engagement_probs_B[-16:])
```
- High positive correlation → both reacting similarly (e.g., both focusing when teacher speaks)
- Negative correlation → one active while other passive → possible turn-taking
- Both high correlation AND high proximity → strong collaboration signal

### Signal 4: Turn-Taking Score
```
# Compute frame-to-frame DELTA of engagement prob
delta_A = diff(engagement_probs_A[-16:])
delta_B = diff(engagement_probs_B[-16:])
# Anti-correlation of deltas = alternating activity = turn-taking
turn_taking = max(0, -pearsonr(delta_A, delta_B))
```
- High score → they alternate being "active" — classic conversation pattern
- Range: 0 to 1

**Why these 4 specifically:** They cover the 3 key collaboration signals researchers use:
1. Physical proximity (must be near each other)
2. Social orientation (must be facing each other)
3. Synchronized behavior (engaged together)
4. Turn-taking (they exchange turns — the strongest signal of actual conversation)

---

## 5. Permanent ID System (ReID Database)

**Problem:** ByteTrack/IoU trackers assign NEW IDs every session. Student who appears Monday gets Track_3; same student Tuesday gets Track_7. This breaks analytics across sessions.

**Solution:** Lightweight appearance-based re-identification using Swin features.

```
New person detected in session:
  1. Extract appearance embedding: mean of last 8 Swin backbone features (768-d)
  2. Query database: cosine_similarity(new_embedding, all stored embeddings)
  3a. sim > 0.75 → this is a known person → reuse their GlobalID
  3b. 0.5 < sim < 0.75 → ambiguous → flag as "possible match", assign tentative ID
  3c. sim < 0.5 → new person → assign new GlobalID, store embedding
  4. Update stored embedding with exponential moving average (alpha=0.1)
     to handle appearance drift (lighting, clothing changes)

Database schema (SQLite):
  persons(
    global_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    first_seen       TIMESTAMP,
    last_seen        TIMESTAMP,
    appearance_emb   BLOB,         -- 768-d float32 numpy array
    appearance_count INTEGER,      -- how many times embedding was updated
    notes            TEXT          -- optional: "seat A3", "student 22CS101"
  )

  sessions(
    session_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    start_time   TIMESTAMP,
    video_source TEXT
  )

  detections(
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   INTEGER,
    global_id    INTEGER,
    frame_num    INTEGER,
    timestamp    REAL,
    bbox_x       INTEGER,
    bbox_y       INTEGER,
    bbox_w       INTEGER,
    bbox_h       INTEGER,
    engagement   TEXT,       -- "Engaged" / "Not Engaged" / "Unknown"
    eng_prob     REAL,
    collaboration TEXT,      -- "Collaborative" / "Not Collaborative" / "Unknown"
    collab_prob  REAL
  )
```

**Privacy note:** No face images stored — only learned feature vectors (768 floats). These are abstract representations, not recoverable photographs. The system can be wiped (DELETE FROM persons) for privacy compliance.

---

## 6. Labeling Strategy for the 40 Videos in /videos/

These 40 WhatsApp-recorded classroom videos are your Phase 2 training data.

### Step 1: Process videos (automated)
```
python src/data/collab_video_processor.py --input_dir videos/ --output_dir data/collab_raw/
```
This will:
- Extract frames at 3fps (slightly higher than engagement pipeline's 2fps — we need more temporal context for interaction)
- Detect persons using HOG (same as working inference)
- Track with SimpleIoU
- Save per-person crops + track metadata
- Generate pair catalog: all (person_A, person_B) pairs visible in same frames

### Step 2: Annotate pairs (manual, ~3-4 hours for 40 videos)
```
python src/data/collab_annotator.py --catalog data/collab_raw/pair_catalog.csv
```
GUI shows:
- Left panel: 8 frames of Person A
- Right panel: 8 frames of Person B  
- Full scene thumbnail at bottom (so you see context)
- **Key C → Collaborative** (they are in active interaction)
- **Key N → Not Collaborative** (no interaction, even if close)
- **Key S → Skip** (too ambiguous, occlusion, back-head with no other signals)
- Q → Quit and save progress

### Step 3: Labeling rules (STRICT — follow exactly)

**Label C (Collaborative) ONLY IF at least 2 of the following are true:**
- [ ] Visible verbal exchange: mouth movement in at least one person, other person reacting
- [ ] Physical co-orientation: both bodies angled toward each other
- [ ] Shared focus: both looking at same object (laptop, paper, board)
- [ ] Response behavior: one person nods, leans in, or gestures in reaction to other

**Label N (Not Collaborative) if:**
- They are sitting near each other but working independently
- One is talking, other is not reacting (lecture/presentation context)
- Both are looking at teacher/board (shared attention to 3rd object ≠ collaboration)
- One or both are on phone or sleeping

**Label S (Skip) if:**
- You cannot see both people clearly for at least 5 of 8 frames
- Scene changes completely mid-clip

### Expected yield from 40 videos:
- ~3-6 persons per video × 40 videos = ~120-240 tracked persons
- Each person tracked for 10-25 seconds → 5-8 clips per person
- Pairs per frame: average 2-4 unique pairs visible simultaneously
- Expected pairs to annotate: **600-1200 pairs**
- After skipping ambiguous: **400-800 usable pairs**
- Class distribution target: aim for ~40% Collaborative (some videos are pure discussion)

---

## 7. Training Plan (Phase 2)

### Data
- Input: (feat_A [768], feat_B [768], signals [4]) → label (0/1)
- Split: video-level (ALL pairs from same video → same split, prevents scene leakage)
- Augmentation: (A,B) and (B,A) both added (symmetric pairs) → doubles dataset

### Training
```python
# Phase 2-A: Bootstrap CollaborationHead only (backbone FULLY frozen)
optimizer = AdamW(collab_head.parameters(), lr=3e-4)
epochs = 30, early_stop patience = 8
loss = BCEWithLogitsLoss(pos_weight=neg_count/pos_count)  # class balance

# Phase 2-B: If val F1 < 0.65 after Phase 2-A:
# Unfreeze ONLY engagement temporal transformer at LR=1e-7
# (last resort — only if collab signals are too weak)
```

### Metrics
- Primary: F1 per-pair (Collab vs Not Collab)
- Secondary: Precision (avoid false collaboration alerts)
- Target: F1 > 0.70 (collaboration is harder than engagement — this is realistic)

### Loss weighting
- Collaborative pairs tend to be LESS common in classroom recordings
- Use pos_weight = 1.5-2.0 to compensate if needed

---

## 8. What We Are NOT Doing (and Why)

| Idea | Why we reject it |
|------|-----------------|
| Direct collaboration label from video | Too noisy — can't reliably label "is this person collaborating" from isolated video |
| GNN for multi-person interaction | Too complex, brittle with fluctuating tracking, need more data |
| YOLO as hard dependency | Was failing in Phase 1, HOG works well enough for person-level crops |
| Face recognition for permanent ID | Privacy risk, legal issues, requires consent — Swin embedding ReID is safer |
| Training new backbone for collab | We have 400-800 pairs max — not enough to train from scratch. Freeze and reuse. |
| TimeSformer for collab | Already have temporal model in engagement backbone. Adding second = overfit. |
| 4-class label (C1/C2/C3/C4) | Too much label noise. Binary Collab/NotCollab is more reliable and trainable. |

---

## 9. File Map (Phase 2)

```
src/
  models/
    collaboration_head.py      ← NEW: PairwiseCollaborationHead
  tracking/
    reid_database.py           ← NEW: SQLite persistent person ID
  inference/
    interaction_signals.py     ← NEW: 4-signal computation
    multi_person_collab_inference.py  ← NEW: Phase 2 full inference
  data/
    collab_video_processor.py  ← NEW: Process /videos/ for Phase 2
    collab_annotator.py        ← NEW: GUI for pair collaboration labeling
    collab_dataset.py          ← NEW: CollabPairDataset
  training/
    train_collab.py            ← NEW: Phase 2 training

weights/
  best_clip_model.pth          ← Phase 1 engagement (FROZEN)
  best_collab_model.pth        ← Phase 2 collab head (to be trained)

data/
  collab_raw/
    pair_catalog.csv           ← All person pairs from /videos/
    annotations.csv            ← Your manual labels
  collab_splits/
    train.csv / val.csv / test.csv

database/
  persons.db                   ← Persistent ReID SQLite database
```

---

## 10. Session Checklist

**For Nikhil to do (in order):**

1. **Run video processor** on /videos/:
   ```
   python src/data/collab_video_processor.py
   ```
   Takes ~20-40 min for 40 videos.

2. **Run annotation GUI**:
   ```
   python src/data/collab_annotator.py
   ```
   Takes ~3-4 hours. Do in multiple sessions (auto-saves progress).
   TARGET: annotate at least 400 pairs before training.

3. **Build splits**:
   ```
   python src/data/collab_dataset.py --build_splits
   ```

4. **Train CollaborationHead**:
   ```
   python src/training/train_collab.py
   ```
   Can run on CPU (CollaborationHead is small, ~2M params).
   Recommended: Google Colab for speed.

5. **Save weights** → `weights/best_collab_model.pth`

6. **Run Phase 2 inference** on test video:
   ```
   python run_collab_inference.py
   ```
   Output: per-person engagement + collaboration labels with permanent IDs.
