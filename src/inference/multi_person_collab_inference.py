"""
multi_person_collab_inference.py -- Phase 2: Engagement + Collaboration inference.

Extends Phase 1 inference with:
  1. Persistent person ReID (GlobalID across sessions via SQLite)
  2. Interaction signal computation (proximity, facing, correlation, turn-taking)
  3. CollaborationHead prediction for each person pair
  4. Per-person collaboration label derived from max pairwise score
  5. Full session logging to database
  6. Enhanced video overlay: engagement + collaboration + GlobalID

Architecture (full pipeline):
  Frame → Detection (HOG) → Tracking (SimpleIoU) → ReID (GlobalID assignment)
       → Swin backbone per person → clip features (768-d) + engagement label
       → InteractionSignalComputer → 4-d signals per pair
       → CollaborationHead → collab_prob per pair
       → Max aggregation → per-person (Collaborative / Not Collaborative)
       → DB logging → Video overlay

Usage:
  system = CollabInferenceSystem(
      engagement_model_path="weights/best_clip_model.pth",
      collab_model_path="weights/best_collab_model.pth",
      db_path="database/persons.db",
  )
  system.run(source="videos/VID-20260421-WA0013.mp4",
             save_output="outputs/output_collab.mp4")
"""

import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from collections import deque
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.swin_clip_model import build_clip_model
from src.models.collaboration_head import build_collab_head, build_feature_extractor
from src.inference.engagement_tracker import (
    EngagementTrackerPool,
    LABEL_ENGAGED, LABEL_NOT_ENGAGED, LABEL_UNKNOWN,
    COLOR_ENGAGED, COLOR_NOT_ENGAGED, COLOR_UNKNOWN,
)
from src.inference.interaction_signals import InteractionSignalComputer
from src.tracking.reid_database import ReIDDatabase, AppearanceEmbeddingBuffer


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

LABEL_COLLABORATIVE     = "Collaborative"
LABEL_NOT_COLLABORATIVE = "Not Collaborative"
COLOR_COLLABORATIVE     = (50, 200, 50)      # Green (same as Engaged — good state)
COLOR_NOT_COLLABORATIVE = (200, 150, 50)     # Orange (distinct from engagement red)


def _build_val_transform(image_size: int = 224) -> T.Compose:
    return T.Compose([
        T.Resize(int(image_size * 1.143)),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# SimpleIoU Tracker (same as Phase 1 fallback — proven reliable)
# ---------------------------------------------------------------------------

class SimpleIoUTracker:
    def __init__(self, iou_thresh=0.35, max_lost=8):
        self.iou_thresh = iou_thresh
        self.max_lost   = max_lost
        self._tracks    = {}
        self._next_id   = 1

    def update(self, detections):
        for tid in list(self._tracks.keys()):
            self._tracks[tid]['lost'] += 1
            if self._tracks[tid]['lost'] > self.max_lost:
                del self._tracks[tid]

        if not detections:
            return {tid: t['bbox'] for tid, t in self._tracks.items()}

        det_matched = set()
        for tid, track in list(self._tracks.items()):
            best_iou, best_di = 0.0, -1
            for di, det in enumerate(detections):
                if di in det_matched: continue
                iou = self._iou(track['bbox'], det)
                if iou > best_iou:
                    best_iou, best_di = iou, di
            if best_iou >= self.iou_thresh and best_di >= 0:
                self._tracks[tid]['bbox'] = detections[best_di]
                self._tracks[tid]['lost'] = 0
                det_matched.add(best_di)

        for di, det in enumerate(detections):
            if di not in det_matched:
                self._tracks[self._next_id] = {'bbox': det, 'lost': 0}
                self._next_id += 1

        return {tid: t['bbox'] for tid, t in self._tracks.items()}

    @staticmethod
    def _iou(a, b):
        ax1, ay1, aw, ah = a; ax2, ay2 = ax1+aw, ay1+ah
        bx1, by1, bw, bh = b; bx2, by2 = bx1+bw, by1+bh
        ix1, iy1 = max(ax1,bx1), max(ay1,by1)
        ix2, iy2 = min(ax2,bx2), min(ay2,by2)
        inter = max(0,ix2-ix1)*max(0,iy2-iy1)
        return inter/(aw*ah+bw*bh-inter+1e-6)


# ---------------------------------------------------------------------------
# HOG Person Detector (proven fallback)
# ---------------------------------------------------------------------------

class HOGPersonDetector:
    def __init__(self):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame, min_area=2000):
        h, w = frame.shape[:2]
        scale = min(640/w, 480/h, 1.0)
        small = cv2.resize(frame, (int(w*scale), int(h*scale)))
        boxes, weights = self.hog.detectMultiScale(
            small, winStride=(8,8), padding=(4,4), scale=1.05, hitThreshold=0.3
        )
        results = []
        if len(boxes) == 0: return results
        for (x,y,bw,bh), conf in zip(boxes, weights):
            x,y,bw,bh = int(x/scale), int(y/scale), int(bw/scale), int(bh/scale)
            if bw*bh >= min_area:
                results.append((x,y,bw,bh))
        return results


def try_yolo_detector():
    """Try to load YOLO detector. Returns None if not available."""
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")
        print("  YOLO detector loaded (yolov8n)")
        return model
    except Exception as e:
        print(f"  YOLO not available ({e}) — using HOG fallback")
        return None


def detect_with_yolo(yolo_model, frame, min_conf=0.4, min_area=2000):
    """Run YOLO detection and return list of (x,y,w,h)."""
    try:
        results = yolo_model(frame, classes=[0], conf=min_conf, verbose=False)
        boxes = []
        for r in results:
            for box in r.boxes:
                x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
                bw,bh = x2-x1, y2-y1
                if bw*bh >= min_area:
                    boxes.append((x1,y1,bw,bh))
        return boxes
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Per-person state tracker (extends Phase 1)
# ---------------------------------------------------------------------------

class PersonCollabState:
    """
    Extends engagement tracking with collaboration state per person.
    Maintains:
      - Pairwise collab scores vs each other tracked person
      - Rolling collaborative vs not-collaborative decision history
    """

    def __init__(self, track_id, collab_window=15, collab_threshold=0.40):
        self.track_id          = track_id
        self.collab_threshold  = collab_threshold
        # pair_scores[other_id] → deque of collab_prob values
        self.pair_scores       = {}
        self.window            = collab_window
        self._current_label    = LABEL_UNKNOWN
        self._current_prob     = 0.5

    def update_pair(self, other_id: int, collab_prob: float):
        if other_id not in self.pair_scores:
            self.pair_scores[other_id] = deque(maxlen=self.window)
        self.pair_scores[other_id].append(collab_prob)

        # Update label: collaborative if any pair's smoothed score > threshold
        max_score = self._max_score()
        self._current_prob  = max_score
        self._current_label = (
            LABEL_COLLABORATIVE if max_score >= self.collab_threshold
            else LABEL_NOT_COLLABORATIVE
        )

    def _max_score(self) -> float:
        if not self.pair_scores:
            return 0.0
        return max(
            float(np.mean(list(scores)))
            for scores in self.pair_scores.values()
            if scores
        )

    @property
    def label(self) -> str:
        return self._current_label

    @property
    def prob(self) -> float:
        return self._current_prob

    @property
    def color(self):
        if self._current_label == LABEL_COLLABORATIVE:
            return COLOR_COLLABORATIVE
        elif self._current_label == LABEL_NOT_COLLABORATIVE:
            return COLOR_NOT_COLLABORATIVE
        return COLOR_UNKNOWN


# ---------------------------------------------------------------------------
# CollabInferenceSystem
# ---------------------------------------------------------------------------

class CollabInferenceSystem:
    """
    Phase 2 full inference system:
      Engagement + Collaboration + Persistent ReID + Database logging.
    """

    def __init__(
        self,
        engagement_model_path : str,
        collab_model_path     : str,
        db_path               : str = "database/persons.db",
        device                : str = "cpu",
        eng_threshold         : float = 0.75,   # from Phase 1 Run 8
        collab_threshold      : float = 0.50,   # set after threshold sweep
        ne_threshold          : float = 0.55,
    ):
        self.device            = device
        self.eng_threshold     = eng_threshold
        self.collab_threshold  = collab_threshold

        # ── Load engagement model (Phase 1, FROZEN) ────────────────────────
        print(f"Loading engagement model: {engagement_model_path}")
        eng_model = build_clip_model(num_classes=2, pretrained=False)
        ckpt      = torch.load(engagement_model_path, map_location="cpu")
        sd        = ckpt.get("model_state_dict", ckpt)
        eng_model.load_state_dict(sd, strict=False)
        self.extractor = build_feature_extractor(eng_model).to(device)
        self.extractor.eval()

        # Load collab threshold from checkpoint if available
        if "best_thresh" in ckpt:
            eng_threshold = ckpt["best_thresh"]

        # ── Load collaboration head ────────────────────────────────────────
        self.collab_head = None
        if collab_model_path and os.path.exists(collab_model_path):
            print(f"Loading collaboration head: {collab_model_path}")
            collab_head = build_collab_head()
            collab_ckpt = torch.load(collab_model_path, map_location="cpu")
            collab_head.load_state_dict(collab_ckpt.get("model_state_dict", collab_ckpt))
            # Load threshold
            if "best_thresh" in collab_ckpt:
                self.collab_threshold = collab_ckpt["best_thresh"]
                print(f"  Collab threshold from checkpoint: {self.collab_threshold:.2f}")
            self.collab_head = collab_head.to(device)
            self.collab_head.eval()
        else:
            print(f"  Collab model not found at {collab_model_path}")
            print(f"  Running engagement-only mode (no collaboration prediction)")

        # ── Detection + Tracking ───────────────────────────────────────────
        self.yolo = try_yolo_detector()
        self.hog  = HOGPersonDetector()
        self.tracker = SimpleIoUTracker()

        # ── Per-person state ───────────────────────────────────────────────
        self.engagement_pool = EngagementTrackerPool(ne_threshold=ne_threshold)
        self.collab_states   = {}   # track_id → PersonCollabState
        self.clip_feat_cache = {}   # track_id → latest 768-d clip feat (torch.Tensor)

        # ── Interaction signal computer ────────────────────────────────────
        self.signal_computer = None  # initialized after first frame (need frame dims)

        # ── Appearance embeddings for ReID ─────────────────────────────────
        self.appearance_buffer = AppearanceEmbeddingBuffer(buffer_size=16, min_frames=8)
        self.db           = ReIDDatabase(db_path=db_path)
        self.global_id_map = {}  # track_id → global_id
        self.session_id   = None

        # ── Transform ─────────────────────────────────────────────────────
        self.transform = _build_val_transform()

        print(f"  Detector: {'YOLO' if self.yolo else 'HOG (fallback)'} + SimpleIoUTracker")
        print(f"  Collab mode: {'ACTIVE' if self.collab_head else 'ENGAGEMENT ONLY'}")
        print(f"  ReID database: {db_path}")

    # ── Main run loop ─────────────────────────────────────────────────────

    def run(
        self,
        source       : str,
        save_output  : Optional[str] = None,
        show_window  : bool = False,
        log_db       : bool = True,
        log_interval : int  = 100,   # commit DB every N frames
    ):
        """
        Run Phase 2 inference on a video file or webcam.

        Args:
            source:      Path to video file, or integer (webcam index)
            save_output: Path to save annotated output video (None = don't save)
            show_window: Show live OpenCV window (False for headless)
            log_db:      Log detections to SQLite database
            log_interval: DB commit interval in frames
        """
        cap = cv2.VideoCapture(source if isinstance(source, int) else str(source))
        if not cap.isOpened():
            raise IOError(f"Cannot open video source: {source}")

        fps    = cap.get(cv2.CAP_PROP_FPS) or 15.0
        fw     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Init signal computer now that we know frame dims
        self.signal_computer = InteractionSignalComputer(
            frame_width=fw, frame_height=fh
        )

        print(f"\nStarting collab inference  source={source}  {fw}x{fh} @ {fps:.0f}fps")
        print(f"Press Q to quit.")

        # Start DB session
        if log_db:
            self.session_id = self.db.start_session(video_source=str(source))

        # Video writer
        writer = None
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(save_output), fourcc, fps, (fw, fh))
            print(f"Saving output to: {save_output}")

        frame_num  = 0
        start_time = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_num += 1
                overlay = self.process_frame(frame, frame_num, fps, log_db)

                if writer:
                    writer.write(overlay)

                if show_window:
                    cv2.imshow("Phase 2: Engagement + Collaboration", overlay)
                    if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                        break

                # DB batch commit
                if log_db and frame_num % log_interval == 0:
                    self.db.commit_batch()

                # Progress
                if frame_num % 30 == 0:
                    elapsed = time.time() - start_time
                    curr_fps = frame_num / max(elapsed, 1e-6)
                    summary = self.engagement_pool.class_summary()
                    n_collab = sum(1 for s in self.collab_states.values()
                                   if s.label == LABEL_COLLABORATIVE)
                    print(f"  Frame {frame_num:4d} | {curr_fps:.1f} fps | "
                          f"Tracked={summary['total']} | "
                          f"E={summary['engaged']} NE={summary['not_engaged']} ?={summary['unknown']} | "
                          f"Collab={n_collab}")

        finally:
            cap.release()
            if writer:
                writer.release()
            if show_window:
                cv2.destroyAllWindows()
            if log_db and self.session_id:
                self.db.commit_batch()
                self.db.end_session(self.session_id)
                print(f"\nSession {self.session_id} saved to database.")

        elapsed = time.time() - start_time
        print(f"\nDone. Processed {frame_num} frames in {elapsed:.1f}s")

    def process_frame(
        self,
        frame     : np.ndarray,
        frame_num : int = 0,
        fps       : float = 15.0,
        log_db    : bool = False,
    ) -> np.ndarray:
        """
        Process one BGR frame. Returns annotated overlay frame.

        This is the integration point for external systems:
          detections = system.process_frame(bgr_frame)
        """
        overlay = frame.copy()
        fh, fw  = frame.shape[:2]

        # 1. Detect persons
        if self.yolo is not None:
            boxes = detect_with_yolo(self.yolo, frame)
        else:
            boxes = self.hog.detect(frame)

        # 2. Track
        tracks = self.tracker.update(boxes)   # {track_id → (x,y,w,h)}

        # 3. Prune stale trackers
        self.engagement_pool.prune_stale()
        stale_ids = [tid for tid in list(self.collab_states.keys())
                     if tid not in tracks]
        for tid in stale_ids:
            del self.collab_states[tid]
            self.signal_computer.remove(tid)
            self.appearance_buffer.remove(tid)

        # 4. Per-person processing
        inference_results = {}   # track_id → {eng_label, eng_prob, clip_feat}

        for tid, bbox in tracks.items():
            x, y, w, h = bbox

            # Upper-body crop
            crop_h = int(h * 0.70)
            x1 = max(0, x); y1 = max(0, y)
            x2 = min(fw, x+w); y2 = min(fh, y+crop_h)
            if x2 <= x1 or y2 <= y1 or (x2-x1) < 20:
                continue

            crop_bgr = frame[y1:y2, x1:x2]
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            crop_pil = Image.fromarray(crop_rgb)

            # Get/create engagement tracker
            eng_tracker = self.engagement_pool.get_or_create(tid)

            # Collab state
            if tid not in self.collab_states:
                self.collab_states[tid] = PersonCollabState(
                    tid, collab_threshold=self.collab_threshold
                )

            # Add frame to engagement buffer
            ready = eng_tracker.add_frame(crop_pil)

            if ready:
                # Run SwinClipModel for engagement + clip features
                clip_frames = eng_tracker.get_clip_frames()
                tensors     = [self.transform(f) for f in clip_frames]
                clip_tensor = torch.stack(tensors).unsqueeze(0).to(self.device)  # (1,8,3,224,224)

                with torch.no_grad():
                    eng_logits, clip_feat = self.extractor(clip_tensor)
                    eng_prob = float(torch.softmax(eng_logits, dim=-1)[0, 1].item())

                eng_tracker.update_engagement(eng_prob)

                # Store clip feature for collaboration
                clip_feat_np = clip_feat.squeeze(0).cpu().numpy()
                self.clip_feat_cache[tid] = clip_feat.squeeze(0)

                # Update appearance embedding for ReID
                self.appearance_buffer.update(tid, clip_feat_np)

                # Update interaction signals
                self.signal_computer.update(tid, bbox, eng_tracker.smoothed_prob)

                # ReID assignment
                if self.appearance_buffer.is_ready(tid) and tid not in self.global_id_map:
                    emb = self.appearance_buffer.get_embedding(tid)
                    global_id, sim = self.db.match_or_register(emb)
                    self.global_id_map[tid] = global_id

            inference_results[tid] = {
                'bbox':       bbox,
                'eng_label':  eng_tracker.current_label,
                'eng_prob':   eng_tracker.smoothed_prob,
                'eng_color':  eng_tracker.display_color,
                'global_id':  self.global_id_map.get(tid, None),
            }

        # 5. Collaboration predictions (pairwise)
        if self.collab_head is not None:
            self._compute_collaboration(tracks)

        # 6. DB logging
        if log_db and self.session_id:
            for tid, info in inference_results.items():
                global_id = info.get('global_id')
                if global_id is None:
                    continue
                collab_state = self.collab_states.get(tid)
                self.db.log_detection(
                    session_id    = self.session_id,
                    global_id     = global_id,
                    frame_num     = frame_num,
                    bbox          = info['bbox'],
                    engagement    = info['eng_label'],
                    eng_prob      = info['eng_prob'],
                    collaboration = collab_state.label if collab_state else LABEL_UNKNOWN,
                    collab_prob   = collab_state.prob  if collab_state else 0.5,
                )

        # 7. Draw overlay
        overlay = self._draw_overlay(overlay, inference_results)

        return overlay

    def _compute_collaboration(self, tracks: dict):
        """Compute pairwise collaboration scores for all visible pairs."""
        active_ids = [tid for tid in tracks.keys() if tid in self.clip_feat_cache]

        if len(active_ids) < 2:
            return

        # Batch all pairs
        for i in range(len(active_ids)):
            for j in range(i + 1, len(active_ids)):
                tid_A = active_ids[i]
                tid_B = active_ids[j]

                feat_A = self.clip_feat_cache[tid_A].unsqueeze(0).to(self.device)
                feat_B = self.clip_feat_cache[tid_B].unsqueeze(0).to(self.device)

                # Get interaction signals
                signals_np = self.signal_computer.get_signals(tid_A, tid_B)
                signals    = torch.from_numpy(signals_np).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    logit = self.collab_head(feat_A, feat_B, signals)
                    prob  = float(torch.sigmoid(logit).item())

                # Update both persons' collaboration state
                if tid_A in self.collab_states:
                    self.collab_states[tid_A].update_pair(tid_B, prob)
                if tid_B in self.collab_states:
                    self.collab_states[tid_B].update_pair(tid_A, prob)

    def _draw_overlay(self, frame: np.ndarray, results: dict) -> np.ndarray:
        """Draw bounding boxes and labels for all tracked persons."""
        fh, fw = frame.shape[:2]

        for tid, info in results.items():
            x, y, w, h = info['bbox']
            eng_color   = info['eng_color']
            eng_label   = info['eng_label']
            global_id   = info.get('global_id')
            collab_state = self.collab_states.get(tid)

            # Bounding box — color from engagement
            cv2.rectangle(frame, (x, y), (x+w, y+h), eng_color, 2)

            # GlobalID label (above bbox)
            id_text = f"G{global_id}" if global_id else f"T{tid}"
            cv2.putText(frame, id_text, (x, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Engagement label (top of bbox)
            eng_short = "E" if eng_label == LABEL_ENGAGED else ("NE" if eng_label == LABEL_NOT_ENGAGED else "?")
            cv2.putText(frame, f"{eng_short}:{info['eng_prob']*100:.0f}%",
                        (x+2, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, eng_color, 1)

            # Collaboration label (bottom left of bbox)
            if collab_state and self.collab_head is not None:
                col_short = "COL" if collab_state.label == LABEL_COLLABORATIVE else "NO"
                cv2.putText(frame, f"{col_short}:{collab_state.prob*100:.0f}%",
                            (x+2, y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            collab_state.color, 1)

                # Colored bottom bar (collaboration status)
                cv2.rectangle(frame, (x, y+h-3), (x+w, y+h), collab_state.color, -1)

        # Class-level dashboard (top-left)
        summary   = self.engagement_pool.class_summary()
        n_collab  = sum(1 for s in self.collab_states.values()
                        if s.label == LABEL_COLLABORATIVE)
        n_no_col  = sum(1 for s in self.collab_states.values()
                        if s.label == LABEL_NOT_COLLABORATIVE)
        n_known   = len(self.global_id_map)

        overlay_lines = [
            f"Tracked: {summary['total']}  (GlobalIDs seen: {n_known})",
            f"Engaged: {summary['engaged']}  Not Engaged: {summary['not_engaged']}  ?: {summary['unknown']}",
            f"Collaborative: {n_collab}  Not Collab: {n_no_col}",
        ]

        for i, line in enumerate(overlay_lines):
            y_pos = 20 + i * 20
            cv2.putText(frame, line, (8, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame


# ---------------------------------------------------------------------------
# Run script
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 2: Engagement + Collaboration inference")
    parser.add_argument("--source",     default="temp/test_H264.mp4",
                        help="Video file or webcam index")
    parser.add_argument("--save_output", default="outputs/output_collab.mp4")
    parser.add_argument("--eng_model",   default="weights/best_clip_model.pth")
    parser.add_argument("--collab_model",default="weights/best_collab_model.pth")
    parser.add_argument("--db_path",     default="database/persons.db")
    parser.add_argument("--device",      default="cpu")
    parser.add_argument("--show",        action="store_true")
    parser.add_argument("--no_log",      action="store_true")
    args = parser.parse_args()

    os.makedirs("outputs", exist_ok=True)

    system = CollabInferenceSystem(
        engagement_model_path = args.eng_model,
        collab_model_path     = args.collab_model,
        db_path               = args.db_path,
        device                = args.device,
    )

    system.run(
        source      = args.source,
        save_output = args.save_output,
        show_window = args.show,
        log_db      = not args.no_log,
    )
