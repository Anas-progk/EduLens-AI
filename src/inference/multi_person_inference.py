"""
multi_person_inference.py -- Real-time multi-person engagement analysis.

Full pipeline:
  1. VideoCapture (webcam or video file)
  2. YOLOv8n person/upper-body detection  (ultralytics)
  3. ByteTrack person tracking (built into YOLOv8)  -- persistent IDs
  4. Per-person bounding box crop + resize to 224x224
  5. Per-person frame buffer (deque of 8 frames)
  6. SwinClipModel inference on full buffers
  7. EMA-smoothed engagement probability per person
  8. Temporal window decision: Not Engaged if NE > threshold over rolling window
  9. OpenCV overlay with per-person labels + class-level summary

Why person tracking is mandatory:
  Without tracking: no persistent ID -> cannot build temporal clip per person
  With ByteTrack:  ID_3 stays ID_3 across frames -> 8-frame clip belongs to same person

Multi-person classroom logic:
  - Each student gets an independent engagement timeline
  - Short distractions (< ne_threshold of window) are smoothed out
  - Only sustained inattention triggers "Not Engaged"
  - Class-level dashboard: "12 engaged / 2 not engaged / 1 unknown"

Usage:
  # Real-time webcam
  system = MultiPersonEngagementSystem(model_path="weights/best_clip_model.pth")
  system.run(source=0)

  # Video file
  system.run(source="classroom_recording.mp4", save_output="output.mp4")

  # Single frame inference (for integration)
  detections = system.process_frame(bgr_frame)
  # detections: list of {"track_id", "bbox", "label", "prob", "ne_ratio"}
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.swin_clip_model import build_clip_model
from src.inference.engagement_tracker import (
    EngagementTrackerPool,
    LABEL_ENGAGED, LABEL_NOT_ENGAGED, LABEL_UNKNOWN,
    COLOR_ENGAGED, COLOR_NOT_ENGAGED, COLOR_UNKNOWN,
)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def _build_val_transform(image_size: int = 224) -> T.Compose:
    return T.Compose([
        T.Resize(int(image_size * 1.143)),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Simple IoU tracker (fallback if ultralytics not available)
# ─────────────────────────────────────────────────────────────────────────────

class SimpleIoUTracker:
    """
    Lightweight IoU-based tracker.
    Assigns persistent IDs by matching bboxes frame-to-frame using IoU.
    Not as robust as ByteTrack but works without extra dependencies.

    Used automatically if ultralytics is not installed.
    """

    def __init__(self, iou_threshold: float = 0.30, max_missing: int = 5):
        self.iou_threshold = iou_threshold
        self.max_missing   = max_missing
        self.tracks   = {}    # track_id -> {"bbox": [x1,y1,x2,y2], "missing": 0}
        self._next_id = 1

    def _iou(self, a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
        iw  = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
        inter = iw * ih
        ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
        return inter / max(ua, 1e-6)

    def update(self, detections):
        """
        Args:
            detections: list of [x1, y1, x2, y2] bboxes

        Returns:
            list of (track_id, [x1, y1, x2, y2])
        """
        matched_track_ids = set()
        results = []

        for det in detections:
            best_id, best_iou = None, self.iou_threshold
            for tid, state in self.tracks.items():
                if tid in matched_track_ids:
                    continue
                iou = self._iou(det, state["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_id  = tid

            if best_id is not None:
                self.tracks[best_id]["bbox"]    = det
                self.tracks[best_id]["missing"] = 0
                matched_track_ids.add(best_id)
                results.append((best_id, det))
            else:
                # New person
                tid = self._next_id
                self._next_id += 1
                self.tracks[tid] = {"bbox": det, "missing": 0}
                results.append((tid, det))

        # Age unmatched tracks
        to_remove = []
        for tid in self.tracks:
            if tid not in matched_track_ids:
                self.tracks[tid]["missing"] += 1
                if self.tracks[tid]["missing"] > self.max_missing:
                    to_remove.append(tid)
        for tid in to_remove:
            del self.tracks[tid]

        return results


# ─────────────────────────────────────────────────────────────────────────────
# Main system
# ─────────────────────────────────────────────────────────────────────────────

class MultiPersonEngagementSystem:
    """
    Real-time multi-person engagement classification.

    Parameters
    ----------
    model_path      : str    Path to saved SwinClipModel checkpoint.
    n_frames        : int    Clip length (must match training, default 8).
    image_size      : int    Frame crop size (224).
    device          : str    "cuda" / "cpu" / "auto".
    detection_conf  : float  YOLO detection confidence threshold.
    detection_classes: list  YOLO class IDs to detect (0=person).
    use_yolo        : bool   Use YOLOv8 if available; else SimpleIoUTracker.
    inference_step  : int    Run clip inference every N new frames per person.
    window_seconds  : int    Rolling engagement window in seconds.
    ne_threshold    : float  NE fraction in window to trigger Not Engaged label.
    ema_alpha       : float  Smoothing for probability display.
    fps_hint        : float  Expected video FPS (used for window sizing).
    """

    def __init__(
        self,
        model_path       : str,
        n_frames         : int   = 8,
        image_size       : int   = 224,
        device           : str   = "auto",
        detection_conf   : float = 0.40,
        detection_classes: list  = None,
        use_yolo         : bool  = True,
        inference_step   : int   = 4,
        window_seconds   : int   = 30,
        ne_threshold     : float = 0.55,
        ema_alpha        : float = 0.25,
        fps_hint         : float = 15.0,
    ):
        # ── Device ────────────────────────────────────────────────────────
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device     = device
        self.n_frames   = n_frames
        self.image_size = image_size

        # ── Load engagement model ──────────────────────────────────────────
        print(f"Loading engagement model: {model_path}")
        self.engagement_model, self.threshold = self._load_model(
            model_path, n_frames, image_size, device
        )
        self.transform = _build_val_transform(image_size)
        print(f"  Model loaded | threshold={self.threshold:.2f} | device={device}")

        # ── Detector + tracker ────────────────────────────────────────────
        self.use_yolo = use_yolo and self._try_yolo_import()
        if self.use_yolo:
            from ultralytics import YOLO
            # YOLOv8n is fastest; swap to yolov8s for better accuracy
            self.yolo = YOLO("yolov8n.pt")
            self.detection_conf    = detection_conf
            self.detection_classes = detection_classes or [0]   # 0=person
            print("  Detector: YOLOv8n + ByteTrack (built-in)")
        else:
            print("  Detector: HOG person detector + SimpleIoUTracker (fallback)")
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            self.iou_tracker = SimpleIoUTracker()

        # ── Engagement tracker pool ────────────────────────────────────────
        self.pool = EngagementTrackerPool(
            clip_len       = n_frames,
            inference_step = inference_step,
            window_seconds = window_seconds,
            fps            = fps_hint,
            ne_threshold   = ne_threshold,
            ema_alpha      = ema_alpha,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Model loading
    # ─────────────────────────────────────────────────────────────────────────

    def _load_model(self, path, n_frames, image_size, device):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        threshold = ckpt.get("threshold", 0.5)

        model = build_clip_model(
            num_classes    = 2,
            pretrained     = False,
            n_frames       = n_frames,
        ).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        return model, threshold

    @staticmethod
    def _try_yolo_import() -> bool:
        try:
            import ultralytics  # noqa
            return True
        except ImportError:
            return False

    # ─────────────────────────────────────────────────────────────────────────
    # Detection + tracking
    # ─────────────────────────────────────────────────────────────────────────

    def _detect_and_track(self, bgr_frame):
        """
        Returns list of (track_id, [x1, y1, x2, y2]).
        Coordinates are pixel values in the original frame.
        """
        if self.use_yolo:
            results = self.yolo.track(
                bgr_frame,
                persist    = True,
                conf       = self.detection_conf,
                classes    = self.detection_classes,
                verbose    = False,
                tracker    = "bytetrack.yaml",
            )
            detections = []
            if results and results[0].boxes is not None:
                boxes = results[0].boxes
                for i, box in enumerate(boxes.xyxy):
                    x1, y1, x2, y2 = [int(v) for v in box.cpu().numpy()]
                    tid = (int(boxes.id[i].item())
                           if boxes.id is not None else i + 1)
                    detections.append((tid, [x1, y1, x2, y2]))
            return detections
        else:
            # HOG fallback
            rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            rects, _ = self.hog.detectMultiScale(
                bgr_frame, winStride=(8, 8), padding=(4, 4), scale=1.05
            )
            boxes = [[x, y, x+w, y+h] for (x, y, w, h) in rects] if len(rects) else []
            return self.iou_tracker.update(boxes)

    # ─────────────────────────────────────────────────────────────────────────
    # Per-person crop
    # ─────────────────────────────────────────────────────────────────────────

    def _crop_person(self, bgr_frame, bbox, padding: float = 0.10):
        """
        Crop person bounding box from frame, add proportional padding.

        For engagement analysis we want the upper body (face + torso + workspace).
        If bbox is full-body, we use upper 65% to focus on upper body.
        """
        H, W = bgr_frame.shape[:2]
        x1, y1, x2, y2 = bbox

        bh = y2 - y1
        bw = x2 - x1

        # Use upper 65% of the detected person box (upper body + face focus)
        y2_upper = int(y1 + bh * 0.65)

        # Add padding
        pad_x = int(bw * padding)
        pad_y = int(bh * padding)

        x1p = max(0, x1 - pad_x)
        y1p = max(0, y1 - pad_y)
        x2p = min(W, x2 + pad_x)
        y2p = min(H, y2_upper + pad_y)

        crop = bgr_frame[y1p:y2p, x1p:x2p]
        if crop.size == 0:
            crop = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

        return crop

    # ─────────────────────────────────────────────────────────────────────────
    # Clip inference
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _run_clip_inference(self, frames_list) -> float:
        """
        Run engagement inference on a list of frames.

        Args:
            frames_list: list of PIL Images or numpy BGR arrays (length = n_frames)

        Returns:
            float: P(Engaged) probability
        """
        tensors = []
        for f in frames_list:
            if isinstance(f, np.ndarray):
                # BGR numpy -> RGB PIL
                f = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            tensors.append(self.transform(f))

        # Stack to (T, C, H, W), add batch dim -> (1, T, C, H, W)
        clip_tensor = torch.stack(tensors, dim=0).unsqueeze(0).to(self.device)

        logits = self.engagement_model(clip_tensor)     # (1, 2)
        prob   = F.softmax(logits, dim=1)[0, 1].item()  # P(Engaged)
        return prob

    # ─────────────────────────────────────────────────────────────────────────
    # Process one frame
    # ─────────────────────────────────────────────────────────────────────────

    def process_frame(self, bgr_frame: np.ndarray) -> list:
        """
        Process one BGR video frame.

        Returns:
            List of dicts, one per tracked person:
            {
              "track_id"  : int,
              "bbox"      : [x1, y1, x2, y2],
              "label"     : "Engaged" / "Not Engaged" / "Unknown",
              "prob"      : float (smoothed P(Engaged)),
              "ne_ratio"  : float (fraction of window that was NE),
              "color"     : (B, G, R) tuple,
            }
        """
        # 1. Detect and track persons
        detections = self._detect_and_track(bgr_frame)

        results = []
        for track_id, bbox in detections:
            # 2. Crop upper body
            crop = self._crop_person(bgr_frame, bbox)

            # 3. Get or create per-person tracker
            tracker = self.pool.get_or_create(track_id)

            # 4. Add frame to buffer; check if inference needed
            inference_ready = tracker.add_frame(crop)

            if inference_ready:
                # 5. Run clip inference
                frames = tracker.get_clip_frames()
                engaged_prob = self._run_clip_inference(frames)

                # 6. Update tracker state
                tracker.update_engagement(engaged_prob)

            results.append({
                "track_id" : track_id,
                "bbox"     : bbox,
                "label"    : tracker.current_label,
                "prob"     : round(tracker.smoothed_prob, 3),
                "ne_ratio" : round(tracker.ne_ratio, 3),
                "color"    : tracker.display_color,
            })

        # Prune stale trackers
        self.pool.prune_stale()

        return results

    # ─────────────────────────────────────────────────────────────────────────
    # Overlay drawing
    # ─────────────────────────────────────────────────────────────────────────

    def draw_overlay(self, bgr_frame: np.ndarray, results: list) -> np.ndarray:
        """
        Draw per-person bounding boxes + labels on frame.
        Returns annotated frame (does not modify in place).
        """
        frame = bgr_frame.copy()

        for r in results:
            x1, y1, x2, y2 = r["bbox"]
            color  = r["color"]
            label  = r["label"]
            prob   = r["prob"]
            tid    = r["track_id"]

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label text: "ID:3 | Engaged (87%)"
            text = f"ID:{tid} | {label} ({prob*100:.0f}%)"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)

            # Background rect for readability
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, text, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                        cv2.LINE_AA)

        # Class-level dashboard (top-left)
        summary = self.pool.class_summary()
        dash_lines = [
            f"Students tracked : {summary['total']}",
            f"Engaged          : {summary['engaged']} ({summary['pct_engaged']}%)",
            f"Not Engaged      : {summary['not_engaged']}",
            f"Unknown          : {summary['unknown']}",
        ]
        y_pos = 28
        for line in dash_lines:
            (lw, lh), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (6, y_pos - lh - 4), (12 + lw, y_pos + 4),
                          (0, 0, 0), -1)
            cv2.putText(frame, line, (8, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 220, 200), 1,
                        cv2.LINE_AA)
            y_pos += 24

        return frame

    # ─────────────────────────────────────────────────────────────────────────
    # Main run loop
    # ─────────────────────────────────────────────────────────────────────────

    def run(
        self,
        source       = 0,
        save_output  : str  = None,
        show_window  : bool = True,
        max_frames   : int  = None,
        print_every  : int  = 30,
    ):
        """
        Run the full engagement analysis pipeline.

        Args:
            source:       Camera index (int) or video file path (str).
            save_output:  If set, save annotated video to this path.
            show_window:  Display OpenCV window (requires display).
            max_frames:   Stop after this many frames (None = run until stopped).
            print_every:  Print class summary every N frames.
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

        fps_actual = cap.get(cv2.CAP_PROP_FPS) or 30.0
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(save_output, fourcc, fps_actual, (W, H))
            print(f"Saving output to: {save_output}")

        frame_count = 0
        t_start = time.time()

        print(f"\nStarting engagement analysis | source={source} | "
              f"{W}x{H} @ {fps_actual:.0f}fps")
        print("Press Q to quit.\n")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if max_frames and frame_count > max_frames:
                    break

                # Process
                results = self.process_frame(frame)

                # Draw overlay
                annotated = self.draw_overlay(frame, results)

                # Save
                if writer:
                    writer.write(annotated)

                # Display
                if show_window:
                    cv2.imshow("Engagement Monitor", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("Quit requested.")
                        break

                # Console summary
                if frame_count % print_every == 0:
                    elapsed = time.time() - t_start
                    fps_run = frame_count / elapsed
                    summary = self.pool.class_summary()
                    print(f"  Frame {frame_count:>5} | "
                          f"{fps_run:.1f} fps | "
                          f"Tracked={summary['total']} | "
                          f"E={summary['engaged']} | "
                          f"NE={summary['not_engaged']} | "
                          f"?={summary['unknown']}")

        finally:
            cap.release()
            if writer:
                writer.release()
            if show_window:
                cv2.destroyAllWindows()

        print(f"\nDone. Processed {frame_count} frames in "
              f"{time.time()-t_start:.1f}s")
        return self.pool.class_summary()


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multi-person engagement monitor")
    parser.add_argument("--model",   required=True, help="Path to best_clip_model.pth")
    parser.add_argument("--source",  default=0,     help="Camera index or video path")
    parser.add_argument("--save",    default=None,  help="Save output video to this path")
    parser.add_argument("--no-show", action="store_true", help="Disable display window")
    parser.add_argument("--ne-threshold", type=float, default=0.55,
                        help="NE fraction in window to trigger Not Engaged (0.55)")
    parser.add_argument("--window",  type=int, default=30,
                        help="Rolling window in seconds (30)")
    args = parser.parse_args()

    source = int(args.source) if str(args.source).isdigit() else args.source

    system = MultiPersonEngagementSystem(
        model_path     = args.model,
        ne_threshold   = args.ne_threshold,
        window_seconds = args.window,
    )
    system.run(
        source      = source,
        save_output = args.save,
        show_window = not args.no_show,
    )
