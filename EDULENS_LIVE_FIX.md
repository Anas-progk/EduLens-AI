# EduLens — making it actually LIVE (not demo)

## Why it was fake/static (root cause)
The backend code is **real** — it loads `best_clip_model.pth`, runs YOLO→tracking→Swin, extracts real
bboxes + 768-d features, and scores the collab head. **But it silently falls back to demo data when
`import torch` fails** — and your backend env never had the ML libraries installed (the setup only
installed `fastapi uvicorn python-multipart pydantic fpdf2`; torch/opencv are *commented out* in
`backend/requirements_backend.txt`). So every number was synthetic and there were no real bboxes.

The frontend was already wired correctly (snake→camel mapper, real bbox canvas, dashboard reads the real
session from localStorage). I removed one leftover bug (a dead `useLiveSim` referencing un-imported demo
constants that broke `npm run build`).

## THE fix — install the ML deps in the backend env
```bash
cd Ai_based_engagement_system

# CPU build (works anywhere; slower). For GPU use the CUDA wheel from pytorch.org.
pip install torch torchvision --index-url https://download.pytorch.org/cpu
pip install timm ultralytics opencv-python pillow scikit-learn numpy fpdf2
pip install fastapi "uvicorn[standard]" python-multipart pydantic
```
Make sure `weights/best_clip_model.pth` exists (it does) and `yolov8n.pt` is in the project root
(ultralytics auto-downloads it on first run if online).

## Verify it's REAL (not demo) — look at the BACKEND terminal
Start the backend and watch the log:
```bash
python -m uvicorn backend.main:app --reload --port 8000
```
- On startup you want: `✓ best_clip_model.pth loaded on cpu` and `✓ GroupCollabHead loaded`.
- When you upload + analyze, you want: `✓ REAL analysis — running best_clip_model on <file>`.
- If you instead see the big `DEMO MODE — best_clip_model.pth NOT loaded` banner, a dep is still
  missing — read the line under it and install what it names.

The log is now the **source of truth** for real-vs-demo (I made it loud and added a `mode` field to the
result). Don't trust the UI badge alone until you've confirmed the log.

## Make a real run actually finish (CPU is slow)
Running a Swin temporal model per student on CPU is heavy. I added a wall-clock cap so analysis always
returns instead of hanging:
```bash
# default 120s; raise it if you want the whole video processed (it will be slow)
set EDULENS_MAX_ANALYSIS_SEC=180   # Windows
# export EDULENS_MAX_ANALYSIS_SEC=180   # macOS/Linux
```
**For the demo, use a SHORT clip (30–60 s).** A 10-minute video on CPU can take many minutes; a 45-second
clip finishes in ~1–2 minutes and gives you real bboxes + real engagement + a real collab verdict.

## Recommended for the review: pre-cache one real run (zero risk)
The safest way to demo live data without waiting (or crashing) in front of reviewers:
1. Pick ONE good short classroom clip.
2. Before the review, upload it once and let the real analysis finish (raise the time budget if needed).
3. The result is saved (SQLite + browser localStorage), so the **dashboard shows that real session
   instantly**, and the monitor replays the real bboxes synced to the video.
4. During the review, open that session — no live inference lag, but 100% real model output.

This is exactly the "graceful, real, smooth" demo flow. It's not faking — it's caching a real result.

## Test checklist (you run; I can't reach your localhost)
- [ ] Backend log shows `✓ best_clip_model.pth loaded` (NOT the DEMO banner)
- [ ] Upload a 45 s clip → backend log shows `✓ REAL analysis`
- [ ] Monitor draws colored bboxes that move with the video (green=Engaged, red=Not)
- [ ] Top cards show real %s that differ per video; before upload they show "—"
- [ ] "View Detailed Analytics" → dashboard shows the SAME session's numbers, "✓ Real Analysis" badge
- [ ] Different clip → different numbers (proves it's not static)

## What I changed in code (this session)
- `backend/services/engagement_service.py`: loud REAL/DEMO logging + `mode` field + wall-clock cap
  (`EDULENS_MAX_ANALYSIS_SEC`) on both real paths so CPU runs finish.
- `frontend/app/monitor/page.tsx`: removed the dead `useLiveSim` (un-imported demo refs that broke build).
- (Frontend api.ts / monitor / dashboard were already correctly wired by the prior session.)

## If anything still shows demo after installing deps
Send me the backend terminal output from startup + one upload. The log now says exactly which dep/model
failed to load — that tells us the remaining gap in one line.

---

## Round 2 — REAL mode works, now tuning the bboxes (you're here)
Your logs confirm real mode (`✓ best_clip_model.pth loaded`, `✓ REAL analysis`, `YOLOv8n + ByteTrack`).
Two symptoms remained: only ~2 boxes, appearing late and looking static. Causes + fixes:

**1. ByteTrack tracker needs `lap` (your log: `Ultralytics requirement ['lap>=0.5.12'] not found`).**
Without it, persons don't get stable track IDs, so few boxes survive. Install the maintained wheel:
```bash
pip install lapx        # provides the `lap` module ByteTrack needs (works on new Python)
```
Then restart the backend. Now every student gets a stable ID and a box.

**2. Boxes appeared late / only for 2 students (fixed in code).** The service used to draw a box only
*after* a track buffered 8 frames (≈4 s) and only stored boxes every 2 s. Fixed: it now stores a box for
**every** detected person on **every** 0.5 s sample. Tracks still warming up show a gray **"Analyzing"**
box that turns green (Engaged) / red (Not Engaged) once the model has enough frames. The frontend window
was tightened so boxes follow the video.

**3. Use a longer, clearer clip (30 s is too short).** The temporal model needs ~8 frames (~4 s) per
student before it can label them, so a 20 s video barely warms up. A **1–3 minute** clip where faces are
reasonably visible gives all students boxes + labels and smoother tracking. Raise the budget if needed:
`set EDULENS_MAX_ANALYSIS_SEC=240`.

**About the low engagement % (e.g., 19%):** that is the **real** model output, not a bug — it's the honest
0.73-F1 model, which is imperfect on uncurated classroom footage (students looking down at desks/phones
read as "not engaged"). It is *not* faked. For a strong demo, pick a clip where students are visibly
on-task (looking at board/notebook); the real numbers will be higher and sensible. Don't inflate them.

**Note on "static" boxes:** in a classroom students barely move, so boxes are *meant* to stay roughly in
place — that's correct, not broken. The fix makes them appear for everyone, immediately, and update live.

### After this: restart both, hard-refresh, re-upload
```bash
pip install lapx
# backend auto-reloads (uvicorn --reload); if not, restart it
# in the browser: Ctrl+Shift+R to clear the old cached session, then upload a 1–3 min clip
```
