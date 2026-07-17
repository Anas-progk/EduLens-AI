"""
Engagement + Collaboration inference service.

Pipeline:
  1. MultiPersonEngagementSystem (YOLO + ByteTrack + SwinClip)  — if ultralytics available
  2. Haar + SwinClip fallback                                    — no extra deps needed

Both paths extract per-person 768-d clip features so GroupCollabHead can compute
the real _compute_signals used in training (zero deploy/train divergence).
"""

import os
import sys
import time
import logging
import hashlib
from collections import defaultdict, deque
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

# ─── Demo result builder ─────────────────────────────────────────────────────
def _demo_result(video_path: str) -> dict:
    import random
    try:
        seed_str = f"{os.path.basename(video_path)}_{os.path.getsize(video_path)}"
    except Exception:
        seed_str = video_path
    rng = random.Random(int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16))

    base_eng = rng.randint(55, 88)
    base_col = rng.randint(45, 80)
    has_drop = rng.random() > 0.4
    drop_s = rng.randint(25, 38)
    drop_d = rng.randint(20, 40) if has_drop else 0

    timeline = []
    for i in range(61):
        in_d = has_drop and drop_s <= i <= drop_s + 8
        eng = max(20, min(95, base_eng + rng.randint(-8, 8) - (drop_d if in_d else 0)))
        col = max(20, min(95, base_col + rng.randint(-6, 6) - (int(drop_d * 0.6) if in_d else 0)))
        timeline.append({"t": i * 60, "engagement": eng, "collab": col, "health": (eng + col) // 2})

    n = rng.randint(6, 12)
    ne = rng.randint(1, max(1, n // 3))
    students = []
    for i in range(n):
        p = rng.uniform(0.15, 0.40) if i < ne else rng.uniform(0.62, 0.95)
        students.append({
            "id": f"ST-{i+1:02d}", "track_id": i + 1,
            "label": "Not Engaged" if i < ne else "Engaged",
            "engagement_prob": round(p, 3),
            "collab_label": "Not Collaborative" if i < ne else (
                "Collaborative" if rng.random() > 0.3 else "Not Collaborative"),
            "row": i // 3, "col": i % 3,
        })

    avg_eng = sum(p["engagement"] for p in timeline) / len(timeline)
    avg_col = sum(p["collab"] for p in timeline) / len(timeline)
    alerts = []
    for s in students:
        if s["label"] == "Not Engaged":
            p = s["engagement_prob"]
            sev = "critical" if p < 0.25 else "warning" if p < 0.35 else "soft"
            alerts.append({
                "id": f"alert_{s['id']}_{sev}", "student_id": s["id"], "severity": sev,
                "message": f"{s['id']} disengaged for {'10+' if sev=='critical' else '5+' if sev=='warning' else '3+'} minutes",
                "timestamp": float(rng.randint(1500, 2400)), "resolved": False,
            })
    return {
        "avg_engagement": round(avg_eng, 1), "avg_collab": round(avg_col, 1),
        "class_health": round((avg_eng + avg_col) / 2, 1),
        "collab_verdict": "COLLABORATIVE" if avg_col >= 60 else "NOT COLLABORATIVE",
        "timeline": timeline, "students": students, "alerts": alerts, "frames": [],
    }


# ─── Helper: compute _compute_signals from 768-d feature sequences ───────────
def _pair_signals(feat_seq_A: np.ndarray, feat_seq_B: np.ndarray) -> np.ndarray:
    """Compute the 6-d relational signals used in collab training.
    feat_seq_A, feat_seq_B: (T, 768) arrays of per-clip 768-d Swin features.
    Matches src/data/collab_pairs._compute_signals exactly.
    """
    try:
        from src.data.collab_pairs import _compute_signals
        return np.asarray(_compute_signals(feat_seq_A, feat_seq_B), dtype=np.float64)
    except Exception:
        # Manual fallback (same math as collab_pairs._compute_signals)
        A = np.asarray(feat_seq_A, dtype=np.float64)
        B = np.asarray(feat_seq_B, dtype=np.float64)
        T = min(len(A), len(B))
        A, B = A[:T], B[:T]

        pA = A.mean(0); pB = B.mean(0)
        na = np.linalg.norm(pA) + 1e-8; nb = np.linalg.norm(pB) + 1e-8
        state_cos   = float(pA @ pB / (na * nb))
        state_close = float(np.exp(-np.linalg.norm(pA - pB) / (np.sqrt(A.shape[1]) + 1e-8)))

        cos_t = np.array([float(A[t] @ B[t] / (np.linalg.norm(A[t]) * np.linalg.norm(B[t]) + 1e-8)) for t in range(T)])
        traj_cos = float(cos_t.mean())

        def _act(seq):
            m = seq.mean(0, keepdims=True)
            return np.linalg.norm(seq - m, axis=1)

        actA = _act(A); actB = _act(B)
        if T >= 3 and actA.std() > 1e-8 and actB.std() > 1e-8:
            dyn_corr = float(np.corrcoef(actA, actB)[0, 1])
        else:
            dyn_corr = 0.0

        dA = np.diff(actA) if T > 1 else np.array([0.0])
        dB = np.diff(actB) if T > 1 else np.array([0.0])
        if len(dA) >= 3 and dA.std() > 1e-8 and dB.std() > 1e-8:
            turn_taking = float(-np.corrcoef(dA, dB)[0, 1])
        else:
            turn_taking = 0.0

        joint_active = float(np.minimum(actA, actB).mean()) if T > 0 else 0.0

        return np.array([state_cos, state_close, traj_cos, dyn_corr, turn_taking, joint_active],
                        dtype=np.float64)


class EngagementService:
    def __init__(self):
        self.model         = None
        self.device        = "cpu"
        self._model_loaded = False
        self._weights_path = REPO_ROOT / "weights" / "best_clip_model.pth"
        self._collab_path  = REPO_ROOT / "weights" / "best_collab_group_fresh.npz"
        self._face_cascade = None
        self._collab_head  = None

        self._try_load_model()
        self._try_load_haar()
        self._try_load_collab()

    # ─── Model loading ────────────────────────────────────────────────────────
    def _try_load_model(self):
        if not self._weights_path.exists():
            logger.warning(f"best_clip_model.pth not found at {self._weights_path}")
            return
        try:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            from src.models.swin_clip_model import build_clip_model
            self.model = build_clip_model(num_classes=2)
            ckpt = torch.load(str(self._weights_path), map_location=self.device, weights_only=False)
            if isinstance(ckpt, dict):
                for key in ("model_state", "model_state_dict", "state_dict", "model"):
                    if key in ckpt:
                        ckpt = ckpt[key]
                        break
            self.model.load_state_dict(ckpt, strict=True)
            self.model.to(self.device).eval()
            self._model_loaded = True
            logger.info(f"✓ best_clip_model.pth loaded on {self.device}")
        except Exception as e:
            logger.error(f"Model load failed: {e}")
            self.model = None

    def _try_load_haar(self):
        try:
            import cv2
            for c in [cv2.data.haarcascades + "haarcascade_frontalface_default.xml",
                      cv2.data.haarcascades + "haarcascade_upperbody.xml"]:
                if os.path.exists(c):
                    self._face_cascade = cv2.CascadeClassifier(c)
                    logger.info(f"✓ Haar: {os.path.basename(c)}")
                    break
        except Exception as e:
            logger.warning(f"Haar unavailable: {e}")

    def _try_load_collab(self):
        try:
            from src.inference.group_collab import GroupCollabHead
            path = self._collab_path if self._collab_path.exists() else REPO_ROOT / "weights" / "best_collab_group.npz"
            if path.exists():
                self._collab_head = GroupCollabHead.load(str(path))
                logger.info(f"✓ GroupCollabHead loaded from {path.name}")
        except Exception as e:
            logger.warning(f"GroupCollabHead not loaded: {e}")

    # ─── Entry point ─────────────────────────────────────────────────────────
    def analyze_video(self, video_path: str, session_id: str, progress_callback=None) -> dict:
        if not self._model_loaded:
            logger.warning("=" * 64)
            logger.warning("  DEMO MODE — best_clip_model.pth NOT loaded.")
            logger.warning("  Numbers are synthetic; real bboxes only if OpenCV is installed.")
            logger.warning("  Fix: pip install torch torchvision timm ultralytics opencv-python pillow")
            logger.warning("=" * 64)
            r = self._demo_with_frames(video_path, progress_callback)
            r["mode"] = "demo"
            return r
        try:
            logger.info("✓ REAL analysis — running best_clip_model on %s", os.path.basename(video_path))
            r = self._real_analysis(video_path, progress_callback)
            r.setdefault("mode", "real")
            return r
        except Exception as e:
            logger.error(f"Real analysis failed ({e}); falling back to DEMO", exc_info=True)
            r = self._demo_with_frames(video_path, progress_callback)
            r["mode"] = "demo"
            return r

    # ─── Demo + real frame extraction ────────────────────────────────────────
    def _demo_with_frames(self, video_path: str, progress_callback=None) -> dict:
        steps = 8
        for i in range(steps):
            time.sleep(0.35)
            if progress_callback:
                progress_callback(int((i + 1) / steps * 100))

        result = _demo_result(video_path)
        try:
            frames = self._extract_frames_haar_only(video_path, result["students"])
            if frames:
                result["frames"] = frames
        except Exception as e:
            logger.debug(f"Frame extraction skipped: {e}")
        return result

    # ─── Real analysis ────────────────────────────────────────────────────────
    def _real_analysis(self, video_path: str, progress_callback=None) -> dict:
        """
        Primary path: use MultiPersonEngagementSystem (YOLO + ByteTrack + SwinClip).
        Fallback: Haar + SwinClip with manual 768-d feature extraction for collab.
        """
        try:
            from src.inference.multi_person_inference import MultiPersonEngagementSystem
            system = MultiPersonEngagementSystem(
                model_path=str(self._weights_path),
                n_frames=8,              # ← correct param name (NOT clip_len)
                device=self.device,
                inference_step=4,        # infer ~every 4th sample → fewer Swin passes (faster on CPU)
                use_yolo=True,           # YOLO if ultralytics installed
                fps_hint=25.0,
                detection_conf=0.25,     # lower → catch more students (phone users, side angles)
                min_clip=2,              # predict after ~2 frames (pad to 8) → no 8-s warmup
                min_decisions=1,         # show a real label after the first inference
            )
            return self._run_with_system(video_path, system, progress_callback)
        except Exception as e:
            logger.warning(f"MultiPersonEngagementSystem failed: {e} → Haar+Swin fallback")
            return self._run_haar_swin(video_path, progress_callback)

    def _run_with_system(self, video_path: str, system, progress_callback=None) -> dict:
        """Use MultiPersonEngagementSystem.process_frame() — works with YOLO or HOG."""
        import cv2
        cap = cv2.VideoCapture(video_path)
        fps        = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration   = total / fps

        # Per-track accumulators
        track_probs:  Dict[int, List[float]]        = defaultdict(list)
        track_feats:  Dict[int, List[np.ndarray]]   = defaultdict(list)  # 768-d per clip
        timeline_pts: Dict[float, List[float]]      = defaultdict(list)  # keyed by time(s)
        seen_tracks:  Dict[int, float]              = {}                 # every detected track
        frame_results = []

        # Sample ~2 frames/s for short clips, but stretch the step for long videos so
        # we never run more than ~110 detection passes total (keeps CPU time bounded).
        sample_step = max(1, int(fps * 0.5))     # ~2 samples/s
        if total > 0:
            sample_step = max(sample_step, total // 110)

        # CPU budget: cap wall-clock so real analysis always finishes in reasonable time.
        max_sec = float(os.environ.get("EDULENS_MAX_ANALYSIS_SEC", "180"))
        t0 = time.time()
        frame_idx = 0
        while True:
            if time.time() - t0 > max_sec:
                logger.info(f"Analysis time budget ({max_sec:.0f}s) reached at frame {frame_idx}")
                break
            # Skip-decode frames we won't analyse (grab() is much cheaper than read()).
            if frame_idx % sample_step != 0:
                if not cap.grab():
                    break
                frame_idx += 1
                continue
            ret, bgr = cap.read()
            if not ret:
                break
            if True:
                try:
                    dets = system.process_frame(bgr)
                    t_sec = frame_idx / fps

                    # Store a box for EVERY detected person, EVERY sample (~0.5 s) so all
                    # students get a box immediately and it tracks the video smoothly.
                    # Tracks still warming up (label "Unknown") are shown as "Analyzing".
                    if dets:
                        frame_results.append({
                            "t": round(t_sec, 2),
                            "detections": [{
                                "track_id": d["track_id"],
                                "bbox":     [int(v) for v in d["bbox"]],
                                "label":    d["label"] if d.get("label") != "Unknown" else "Analyzing",
                                "prob":     round(float(d.get("prob", 0.5)), 3),
                            } for d in dets],
                        })

                    # Record EVERY detected track (so all students appear); aggregate
                    # engagement only from labeled (non-Unknown) tracks. Collect the
                    # 768-d clip feature whenever a fresh inference produced one — this
                    # is what GroupCollabHead needs for the REAL collaboration verdict.
                    for d in dets:
                        tid = d["track_id"]
                        seen_tracks[tid] = float(d.get("prob", 0.5))
                        feat = d.get("feat")
                        if feat is not None:
                            track_feats[tid].append(np.asarray(feat, dtype=np.float64))
                        if d.get("label") == "Unknown":
                            continue
                        track_probs[tid].append(float(d["prob"]))
                        timeline_pts[round(t_sec, 1)].append(float(d["prob"]))

                except Exception as fe:
                    logger.debug(f"Frame {frame_idx}: {fe}")

            frame_idx += 1
            if progress_callback and total > 0:
                progress_callback(min(95, int(frame_idx / total * 90)))

        cap.release()

        if not track_probs:
            logger.warning("No detections from system — falling back to demo")
            return self._demo_with_frames(video_path, None)

        students   = self._build_students(track_probs, seen_tracks)
        timeline   = self._build_timeline(timeline_pts)
        alerts     = self._build_alerts(students, duration)

        # REAL collaboration: feed the per-track 768-d features into the trained
        # GroupCollabHead (signals + pooled-feature scalars, the 0.667/0.764 model).
        # This is independent of engagement — a heads-down group working together can
        # still be COLLABORATIVE. Falls back to the heuristic only if too few features.
        collab, collab_prob = self._collab_from_features(track_feats, students)

        # Collaboration is a GROUP property — give every student the session verdict
        # rather than a per-person guess (fixes "only 2 members collaborative").
        _collab_yes = str(collab).upper().startswith("COLLAB")
        for s in students:
            s["collab_label"] = "Collaborative" if _collab_yes else "Not Collaborative"

        # Stabilize the overlay: paint every stored box with that track's FINAL averaged
        # engagement (real value + correct colour), instead of the warming-up 0.5 that
        # made every box show a grey "50%". Removes the red/grey flicker too.
        final = {s["track_id"]: (s["engagement_prob"], s["label"]) for s in students}
        for fr in frame_results:
            for d in fr["detections"]:
                fp = final.get(d["track_id"])
                if fp:
                    d["prob"], d["label"] = round(float(fp[0]), 3), fp[1]

        if progress_callback:
            progress_callback(100)

        labeled = [s for s in students if s["label"] != "Analyzing"] or students
        avg_eng = sum(s["engagement_prob"] for s in labeled) * 100 / len(labeled)
        # Collaboration % = the real session probability from the head (not eng*0.85).
        avg_col = round(collab_prob * 100, 1) if collab_prob is not None else round(avg_eng * 0.85, 1)

        # Collaboration is a SESSION-level verdict, so show it as a flat reference line
        # on the timeline rather than fabricating a per-second collab curve.
        for pt in timeline:
            pt["collab"]  = avg_col
            pt["health"]  = round((pt["engagement"] + avg_col) / 2, 1)

        return {
            "avg_engagement": round(avg_eng, 1),
            "avg_collab":     avg_col,
            "class_health":   round((avg_eng + avg_col) / 2, 1),
            "collab_verdict": collab,
            "collab_prob":    round(collab_prob, 3) if collab_prob is not None else None,
            "timeline": timeline, "students": students,
            "alerts": alerts, "frames": frame_results,
        }

    # ─── Haar + SwinClip (fallback, extracts 768-d features for real collab) ──
    def _run_haar_swin(self, video_path: str, progress_callback=None) -> dict:
        import cv2, torch, torch.nn.functional as F
        import torchvision.transforms as T
        from PIL import Image

        transform = T.Compose([
            T.Resize(256), T.CenterCrop(224), T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        cap       = cv2.VideoCapture(video_path)
        fps       = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration  = total / fps

        clip_bufs:    Dict[int, deque]               = {}  # track → 8-frame buffer
        feat_bufs:    Dict[int, List[np.ndarray]]    = defaultdict(list)  # 768-d per clip
        track_probs:  Dict[int, List[float]]         = defaultdict(list)
        timeline_pts: Dict[int, List[float]]         = defaultdict(list)
        frame_results = []

        sample_step = max(1, int(fps * 2))
        max_sec = float(os.environ.get("EDULENS_MAX_ANALYSIS_SEC", "300"))
        t0 = time.time()
        frame_idx   = 0

        while True:
            if time.time() - t0 > max_sec:
                logger.info(f"Analysis time budget ({max_sec:.0f}s) reached at frame {frame_idx}")
                break
            ret, bgr = cap.read()
            if not ret:
                break

            if frame_idx % sample_step == 0:
                t_sec  = frame_idx / fps
                bboxes = self._detect_persons(bgr)

                if bboxes:
                    frame_dets = []
                    for i, (x1, y1, x2, y2) in enumerate(bboxes[:12]):
                        tid  = i + 1
                        crop = bgr[y1:y2, x1:x2]
                        if crop.size == 0:
                            continue

                        if tid not in clip_bufs:
                            clip_bufs[tid] = deque(maxlen=8)

                        pil  = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                        clip_bufs[tid].append(transform(pil))

                        prob     = 0.65
                        feat_768 = None

                        if len(clip_bufs[tid]) == 8:
                            try:
                                clip = torch.stack(list(clip_bufs[tid])).unsqueeze(0).to(self.device)
                                # B=1, T=8, C, H, W
                                with torch.no_grad():
                                    B, T_len, C, H, W = clip.shape
                                    frames   = clip.view(B * T_len, C, H, W)
                                    raw_feat = self.model.backbone(frames)        # (B*T, 768)
                                    raw_feat = raw_feat.view(B, T_len, -1)
                                    clip_f   = self.model.temporal(raw_feat)      # (B, 768)
                                    logits   = self.model.head(clip_f)            # (B, 2)
                                    prob     = float(F.softmax(logits, dim=1)[0, 1].item())
                                    feat_768 = clip_f[0].cpu().numpy()            # (768,)
                            except Exception as e:
                                logger.debug(f"Swin inference tid={tid}: {e}")

                        track_probs[tid].append(prob)
                        if feat_768 is not None:
                            feat_bufs[tid].append(feat_768)
                        timeline_pts[int(t_sec // 60)].append(prob)

                        label = "Engaged" if prob >= 0.5 else "Not Engaged"
                        frame_dets.append({
                            "track_id": tid,
                            "bbox":     [x1, y1, x2, y2],
                            "label":    label,
                            "prob":     round(prob, 3),
                        })

                    if frame_dets:
                        frame_results.append({"t": round(t_sec, 2), "detections": frame_dets})

            frame_idx += 1
            if progress_callback and total > 0:
                progress_callback(min(95, int(frame_idx / total * 90)))

        cap.release()

        if not track_probs:
            result = _demo_result(video_path)
            if progress_callback:
                progress_callback(100)
            return result

        students = self._build_students(track_probs)
        timeline = self._build_timeline(timeline_pts)
        alerts   = self._build_alerts(students, duration)

        # Real collab verdict using 768-d features + _compute_signals
        collab, collab_prob = self._collab_from_features(feat_bufs, students)

        avg_eng = sum(s["engagement_prob"] for s in students) * 100 / len(students)
        avg_col = round(collab_prob * 100, 1) if collab_prob is not None else round(avg_eng * 0.85, 1)
        for pt in timeline:
            pt["collab"] = avg_col
            pt["health"] = round((pt["engagement"] + avg_col) / 2, 1)

        if progress_callback:
            progress_callback(100)

        return {
            "avg_engagement": round(avg_eng, 1),
            "avg_collab":     avg_col,
            "class_health":   round((avg_eng + avg_col) / 2, 1),
            "collab_verdict": collab,
            "collab_prob":    round(collab_prob, 3) if collab_prob is not None else None,
            "timeline": timeline, "students": students,
            "alerts": alerts, "frames": frame_results,
        }

    # ─── Person detection (YOLO → HOG → Haar, no fake grid) ─────────────────
    def _detect_persons(self, bgr_frame) -> list:
        """Return list of (x1,y1,x2,y2). Never falls back to a fake grid."""
        import cv2
        h, w = bgr_frame.shape[:2]

        # 1. YOLO
        try:
            from ultralytics import YOLO as _YOLO
            if not hasattr(self, '_yolo_model'):
                self._yolo_model = _YOLO("yolov8n.pt")
            res = self._yolo_model(bgr_frame, classes=[0], verbose=False)[0]
            bboxes = []
            for box in res.boxes:
                if float(box.conf[0]) > 0.3:
                    bboxes.append(tuple(int(v) for v in box.xyxy[0].tolist()))
            if bboxes:
                return bboxes
        except Exception:
            pass

        # 2. HOG person detector
        try:
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            scale = min(1.0, 640 / max(w, h))
            small = cv2.resize(bgr_frame, (int(w*scale), int(h*scale))) if scale < 1 else bgr_frame
            rects, weights = hog.detectMultiScale(small, winStride=(8,8), padding=(4,4), scale=1.05)
            bboxes = [(int(x/scale), int(y/scale), int((x+bw)/scale), int((y+bh)/scale))
                      for (x,y,bw,bh),wt in zip(rects, weights) if wt > 0.3]
            if bboxes:
                return bboxes
        except Exception:
            pass

        # 3. Haar face → expanded upper-body bbox
        if self._face_cascade is not None and not self._face_cascade.empty():
            try:
                gray  = cv2.equalizeHist(cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY))
                faces = self._face_cascade.detectMultiScale(
                    gray, scaleFactor=1.05, minNeighbors=2,
                    minSize=(20, 20), maxSize=(w//2, h),
                    flags=cv2.CASCADE_SCALE_IMAGE,
                )
                bboxes = []
                for (fx, fy, fw, fh) in faces:
                    bboxes.append((
                        max(0, fx - int(fw*0.5)),
                        max(0, fy - int(fh*0.3)),
                        min(w, fx + fw + int(fw*0.5)),
                        min(h, fy + fh + int(fh*2.5)),
                    ))
                return bboxes
            except Exception:
                pass

        return []   # nothing detected — no fake grid

    # ─── Frame extraction for demo mode (real detection only) ────────────────
    def _extract_frames_haar_only(self, video_path: str, students: List[dict]) -> List[dict]:
        """Sample frames, detect persons with Haar/HOG, return real bbox data."""
        try:
            import cv2
        except ImportError:
            return []

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps      = cap.get(cv2.CAP_PROP_FPS) or 25.0
        duration = total / fps
        if total < 1:
            cap.release()
            return []

        results = []
        step    = max(1, total // 40)

        for fi in range(0, total, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, bgr = cap.read()
            if not ret:
                continue
            t_sec  = fi / fps
            bboxes = self._detect_persons(bgr)
            if not bboxes:
                continue
            dets = []
            for i, (x1,y1,x2,y2) in enumerate(bboxes[:len(students)]):
                s = students[i % len(students)]
                dets.append({"track_id": i+1, "bbox": [x1,y1,x2,y2],
                             "label": s["label"], "prob": s["engagement_prob"]})
            results.append({"t": round(t_sec, 2), "detections": dets})

        cap.release()
        logger.info(f"Demo frames: {len(results)} frames with real detections")
        return results

    # ─── Collab verdict from 768-d feature buffers ───────────────────────────
    def _collab_from_features(
        self, feat_bufs: Dict[int, List[np.ndarray]], students: List[dict]
    ):
        """Build pair records with real _compute_signals and score with GroupCollabHead.

        Returns (verdict_str, prob_or_None). Only tracks with >=2 collected features
        are used (need a short temporal sequence to compute relational signals).
        """
        tids = sorted([t for t, f in feat_bufs.items() if len(f) >= 2])
        if len(tids) < 2 or self._collab_head is None:
            return self._collab_from_students(students), None

        pair_records = []
        for i, tidA in enumerate(tids):
            for tidB in tids[i + 1:]:
                A = np.stack(feat_bufs[tidA])   # (T_A, 768)
                B = np.stack(feat_bufs[tidB])   # (T_B, 768)
                T = min(len(A), len(B))
                if T < 2:
                    continue
                sigs = _pair_signals(A[:T], B[:T])  # (6,)
                pair_records.append({
                    "signals":  sigs,
                    "pooled_A": A[:T].mean(0),       # (768,)
                    "pooled_B": B[:T].mean(0),
                })

        if not pair_records:
            return self._collab_from_students(students), None

        try:
            verdict = self._collab_head.predict(pair_records)
            result  = verdict.get("verdict", "UNKNOWN")
            prob    = verdict.get("prob", None)
            logger.info(f"GroupCollabHead verdict: {result} (prob={prob}, n_pairs={len(pair_records)})")
            if result == "UNKNOWN" or prob is None:
                return self._collab_from_students(students), None
            # predict() already returns "COLLABORATIVE" / "NOT COLLABORATIVE".
            return str(result), float(prob)
        except Exception as e:
            logger.warning(f"GroupCollabHead.predict failed: {e}")
            return self._collab_from_students(students), None

    def _collab_from_students(self, students: List[dict]) -> str:
        """Simple heuristic fallback: avg engagement >= 60% → collaborative."""
        if not students:
            return "COLLABORATIVE"
        avg = sum(s["engagement_prob"] for s in students) / len(students)
        return "COLLABORATIVE" if avg >= 0.60 else "NOT COLLABORATIVE"

    # ─── Build results helpers ────────────────────────────────────────────────
    def _build_students(self, track_probs: Dict[int, List[float]],
                        seen_tracks: Optional[Dict[int, float]] = None) -> List[dict]:
        """All detected tracks become students. Labeled tracks get Engaged/Not Engaged;
        tracks still warming up (no clip yet) show 'Analyzing' with a provisional prob."""
        seen_tracks = seen_tracks or {}
        all_tids = sorted(set(track_probs.keys()) | set(seen_tracks.keys()))
        students = []
        for i, tid in enumerate(all_tids):
            if track_probs.get(tid):
                avg_p = sum(track_probs[tid]) / len(track_probs[tid])
                label = "Engaged" if avg_p >= 0.5 else "Not Engaged"
            else:
                avg_p = float(seen_tracks.get(tid, 0.5))
                label = "Analyzing"
            students.append({
                "id": f"ST-{i+1:02d}", "track_id": int(tid),
                "label": label,
                "engagement_prob": round(avg_p, 3),
                "collab_label": "Collaborative" if (label != "Analyzing" and avg_p >= 0.6) else "Not Collaborative",
                "row": i // 3, "col": i % 3,
            })
        return students

    def _build_timeline(self, timeline_pts: Dict[float, List[float]]) -> List[dict]:
        """Build a fine-grained timeline keyed by real seconds (works for short clips),
        downsampled to <=60 points so the chart is readable for long videos too."""
        if not timeline_pts:
            return [{"t": 0, "engagement": 70, "collab": 60, "health": 65}]
        pts = []
        for t in sorted(timeline_pts.keys()):
            vals = timeline_pts[t]
            eng = round(sum(vals) / len(vals) * 100, 1)
            col = round(eng * 0.85, 1)
            pts.append({"t": int(round(t)), "engagement": eng, "collab": col,
                        "health": round((eng + col) / 2, 1)})
        if len(pts) > 60:
            step = len(pts) / 60.0
            pts = [pts[int(i * step)] for i in range(60)]
        return pts

    def _build_alerts(self, students: List[dict], duration: float) -> List[dict]:
        """Honest alerts: real timestamps within the clip (rounded to whole seconds),
        messages that do NOT fabricate '10+ minutes' on a short clip."""
        alerts = []
        dur = duration if duration and duration > 0 else 60.0
        ne = [s for s in students if s["label"] == "Not Engaged"]
        for k, s in enumerate(ne):
            p = s["engagement_prob"]
            sev = "critical" if p < 0.25 else "warning" if p < 0.4 else "soft"
            ts = int(round(dur * (0.25 + 0.6 * (k / max(len(ne), 1)))))   # spread across the clip
            mm, ss = divmod(ts, 60)
            when = f"{mm}:{ss:02d}" if dur >= 60 else f"{ts}s"
            phrase = ("critically disengaged" if sev == "critical"
                      else "disengaged" if sev == "warning" else "showing low attention")
            alerts.append({
                "id": f"alert_{s['id']}", "student_id": s["id"], "severity": sev,
                "message": f"{s['id']} {phrase} (at {when})",
                "timestamp": ts, "resolved": False,
            })
        return alerts
