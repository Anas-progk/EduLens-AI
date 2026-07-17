"""
custom_video_processor.py
Extract per-person temporal clips from multi-person classroom videos.

Pipeline:
  1. Reads all .mp4 / .avi / .mov from custom_dataset/raw_videos/
  2. Extracts frames at TARGET_FPS (default 2 FPS)
  3. Detects persons with YOLOv8n (falls back to HOG if ultralytics not installed)
  4. Tracks persons across sampled frames with SimpleIoUTracker
  5. Saves upper-body crops:
       custom_dataset/processed/{video_stem}/{track_id:03d}/frame_{n:04d}.jpg
  6. Windows each track into overlapping 8-frame clips (stride = 4)
  7. Saves two CSVs:
       custom_dataset/tracking_metadata.csv  -- one row per saved crop
       custom_dataset/clips_catalog.csv      -- one row per 8-frame clip

After this script: run annotate_tracks.py to label each clip.

Key design decisions:
  - TARGET_FPS=2: captures behaviour without near-duplicate frames;
    15–23 sec video → 30–46 sampled frames per person
  - IoU tracker on SAMPLED frames: avoids ByteTrack drift from skipped frames;
    students in seated classroom move slowly → IoU matching is reliable
  - Upper-body crop (upper 65% of bbox): focuses on face + torso + workspace,
    which are the actual engagement signals
  - Sliding window stride=4 with window=8: 50% overlap gives more clips
    per track while each clip still has distinct content
  - MIN_TRACK_LEN=8: discards transient detections that can't form even one clip

Usage:
  python src/data/custom_video_processor.py
  python src/data/custom_video_processor.py --fps 1          # slower extract
  python src/data/custom_video_processor.py --fps 3          # denser extract
"""

import os
import sys
import re
import argparse
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import CONFIG


# ──────────────────────────────────────────────────────────────────────────────
# Configurable constants (overridable via CLI args)
# ──────────────────────────────────────────────────────────────────────────────
CUSTOM_RAW_DIR   = CONFIG.get('custom_raw_dir',   'custom_dataset/raw_videos')
CUSTOM_PROC_DIR  = CONFIG.get('custom_proc_dir',  'custom_dataset/processed')
CUSTOM_META_CSV  = CONFIG.get('custom_meta_csv',  'custom_dataset/tracking_metadata.csv')
CUSTOM_CLIPS_CSV = CONFIG.get('custom_clips_csv', 'custom_dataset/clips_catalog.csv')

TARGET_FPS     = 2       # sample frames per second from video
N_FRAMES       = 8       # frames per clip (must match model n_frames_clip)
CLIP_STRIDE    = 4       # sliding window stride (4 = 50% overlap)
MIN_TRACK_LEN  = N_FRAMES  # discard tracks shorter than one full clip
DET_CONF       = 0.40   # YOLO person detection confidence
MIN_BBOX_AREA  = 2500   # pixels^2; ignore tiny detections (< ~50×50 px)
CROP_BODY_FRAC = 0.65   # use top 65% of bbox height (upper body)
CROP_PADDING   = 0.10   # proportional padding around upper-body crop
JPEG_QUALITY   = 85


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def sanitize(name: str) -> str:
    """Convert arbitrary filename stem to a safe folder name."""
    name = re.sub(r'[^a-zA-Z0-9]', '_', name)   # non-alnum → underscore
    name = re.sub(r'_+', '_', name)              # collapse runs of underscores
    return name.strip('_')


# ──────────────────────────────────────────────────────────────────────────────
# SimpleIoUTracker
# Tracks person bounding boxes across sparsely sampled frames using IoU.
# Lightweight and robust for seated/slow-moving subjects.
# ──────────────────────────────────────────────────────────────────────────────

class SimpleIoUTracker:
    """
    IoU-based multi-person tracker for sparsely sampled frames.

    At 2 FPS, 0.5 s gaps between frames.  In a seated classroom setting,
    students move only slightly → IoU ≥ 0.25 reliably links same person.

    Tracks expire after max_missing consecutive missed frames.
    """

    def __init__(self, iou_threshold: float = 0.25, max_missing: int = 3):
        self.iou_thr   = iou_threshold
        self.max_miss  = max_missing
        self.tracks    = {}      # tid → {"bbox": [x1,y1,x2,y2], "missing": int}
        self._next_id  = 1

    @staticmethod
    def _iou(a, b) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
        iw  = max(0.0, ix2 - ix1)
        ih  = max(0.0, iy2 - iy1)
        inter = iw * ih
        union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
        return inter / max(union, 1e-6)

    def update(self, detections: list) -> list:
        """
        Args:
            detections: list of [x1, y1, x2, y2] bboxes for this frame.
        Returns:
            list of (track_id, [x1, y1, x2, y2]) matched or newly created tracks.
        """
        matched  = set()
        results  = []

        for det in detections:
            best_id, best_iou = None, self.iou_thr
            for tid, state in self.tracks.items():
                if tid in matched:
                    continue
                iou = self._iou(det, state['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_id  = tid

            if best_id is not None:
                self.tracks[best_id] = {'bbox': det, 'missing': 0}
                matched.add(best_id)
                results.append((best_id, det))
            else:
                # New person detected → assign new ID
                tid = self._next_id
                self._next_id += 1
                self.tracks[tid] = {'bbox': det, 'missing': 0}
                results.append((tid, det))

        # Age unmatched tracks; prune stale ones
        to_remove = []
        for tid in self.tracks:
            if tid not in matched:
                self.tracks[tid]['missing'] += 1
                if self.tracks[tid]['missing'] > self.max_miss:
                    to_remove.append(tid)
        for tid in to_remove:
            del self.tracks[tid]

        return results


# ──────────────────────────────────────────────────────────────────────────────
# Crop helper
# ──────────────────────────────────────────────────────────────────────────────

def crop_upper_body(
    frame: np.ndarray,
    bbox: list,
    body_frac: float = CROP_BODY_FRAC,
    padding: float   = CROP_PADDING,
) -> np.ndarray:
    """
    Crop the upper body from a full-person bounding box.

    Why upper body?  Face direction, gaze, posture, and workspace interaction
    are the engagement signals.  Legs carry no information.

    Upper body = top (body_frac × height) of the bbox.
    Padding prevents tight crops when bbox is slightly off.
    """
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    bh = y2 - y1
    bw = x2 - x1

    y2_upper = int(y1 + bh * body_frac)

    pad_x = int(bw * padding)
    pad_y = int(bh * padding)

    x1p = max(0, x1 - pad_x)
    y1p = max(0, y1 - pad_y)
    x2p = min(W, x2 + pad_x)
    y2p = min(H, y2_upper + pad_y)

    crop = frame[y1p:y2p, x1p:x2p]
    if crop.size == 0:
        crop = np.zeros((112, 112, 3), dtype=np.uint8)
    return crop


# ──────────────────────────────────────────────────────────────────────────────
# Clip windowing
# ──────────────────────────────────────────────────────────────────────────────

def make_clips(frame_paths: list, clip_len: int = N_FRAMES, stride: int = CLIP_STRIDE) -> list:
    """
    Sliding-window partitioning of a track into overlapping 8-frame clips.

    Example for 30-frame track (stride=4, clip_len=8):
      Clip 0: frames 0–7
      Clip 1: frames 4–11
      ...
      Clip 5: frames 20–27   (total 6 clips)

    50% overlap gives more training samples while each clip retains
    distinct content (different temporal context).
    """
    clips = []
    n = len(frame_paths)
    for start in range(0, n - clip_len + 1, stride):
        clips.append(frame_paths[start : start + clip_len])
    return clips


# ──────────────────────────────────────────────────────────────────────────────
# Detection backends
# ──────────────────────────────────────────────────────────────────────────────

def try_load_yolo():
    """Try to load YOLOv8n; return model or None on failure."""
    try:
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')
        print("  ✅  YOLOv8n loaded (downloads ~6 MB if first run)")
        return model
    except Exception as e:
        print(f"  ⚠️   ultralytics not found ({e}) — using HOG fallback")
        return None


def detect_yolo(model, frame: np.ndarray) -> list:
    """Run YOLO person detection. Returns list of [x1,y1,x2,y2]."""
    results = model(frame, verbose=False, conf=DET_CONF, classes=[0])
    bboxes  = []
    if results and results[0].boxes is not None:
        for box in results[0].boxes.xyxy:
            x1, y1, x2, y2 = [float(v) for v in box.cpu().numpy()]
            if (x2 - x1) * (y2 - y1) < MIN_BBOX_AREA:
                continue
            bboxes.append([x1, y1, x2, y2])
    return bboxes


def detect_hog(hog, frame: np.ndarray) -> list:
    """HOG fallback person detector. Returns list of [x1,y1,x2,y2]."""
    rects, _ = hog.detectMultiScale(
        frame, winStride=(8, 8), padding=(4, 4), scale=1.05
    )
    bboxes = []
    if len(rects):
        for (x, y, w, h) in rects:
            if w * h < MIN_BBOX_AREA:
                continue
            bboxes.append([float(x), float(y), float(x + w), float(y + h)])
    return bboxes


# ──────────────────────────────────────────────────────────────────────────────
# Single video processor
# ──────────────────────────────────────────────────────────────────────────────

def process_video(
    video_path : str,
    proc_dir   : str,
    yolo_model,
    hog,
    target_fps : int = TARGET_FPS,
) -> tuple:
    """
    Process one video file.

    Returns:
        meta_rows  (list of dicts) : one entry per saved crop frame
        clip_rows  (list of dicts) : one entry per generated 8-frame clip
    """
    vpath   = Path(video_path)
    stem    = sanitize(vpath.stem)
    out_root = Path(proc_dir) / stem

    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened():
        print(f"  ❌  Cannot open: {vpath.name}")
        return [], []

    video_fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s   = total_frames / max(video_fps, 1.0)
    frame_step   = max(1, int(round(video_fps / target_fps)))
    n_samples    = total_frames // frame_step

    print(f"\n  [{vpath.name}]  {duration_s:.1f}s @ {video_fps:.0f}fps  →  "
          f"sampling every {frame_step} frames → ~{n_samples} samples")

    tracker = SimpleIoUTracker(iou_threshold=0.25, max_missing=3)

    # track_id → list of (video_frame_idx, sample_idx, crop_ndarray)
    track_data = defaultdict(list)

    sample_idx    = 0
    video_frame_i = 0

    while video_frame_i < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame_i)
        ret, frame = cap.read()
        if not ret:
            break

        # Detect
        if yolo_model is not None:
            bboxes = detect_yolo(yolo_model, frame)
        else:
            bboxes = detect_hog(hog, frame)

        # Track
        tracks = tracker.update(bboxes)

        for tid, bbox in tracks:
            crop = crop_upper_body(frame, bbox)
            track_data[tid].append((video_frame_i, sample_idx, crop))

        sample_idx    += 1
        video_frame_i += frame_step

    cap.release()

    # ── Save crops + build metadata rows ──────────────────────────────────────
    meta_rows = []
    clip_rows = []
    n_valid_tracks = 0

    for tid, frames in track_data.items():
        if len(frames) < MIN_TRACK_LEN:
            continue  # too short to form even one clip → skip

        n_valid_tracks += 1
        track_dir = out_root / f"{tid:03d}"
        track_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []
        for vid_fi, samp_i, crop in frames:
            fpath = track_dir / f"frame_{samp_i:04d}.jpg"
            cv2.imwrite(str(fpath), crop, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            # Always store with forward slashes so paths work on both
            # Windows (where they were created) and Linux/Colab (where
            # ClipDataset reads them).
            fpath_str = fpath.as_posix()
            saved_paths.append(fpath_str)

            meta_rows.append({
                'video_id'    : stem,
                'video_file'  : vpath.name,
                'track_id'    : tid,
                'video_frame' : vid_fi,
                'sample_idx'  : samp_i,
                'frame_path'  : fpath_str,
            })

        # Window the track into 8-frame clips
        person_id = f"{stem}_{tid:03d}"
        clip_list = make_clips(saved_paths, N_FRAMES, CLIP_STRIDE)

        for ci, clip_frames in enumerate(clip_list):
            clip_id = f"{stem}_{tid:03d}_C{ci:02d}"
            clip_rows.append({
                'clip_id'     : clip_id,
                'video_id'    : stem,
                'video_file'  : vpath.name,
                'track_id'    : tid,
                'person_id'   : person_id,
                'clip_idx'    : ci,
                'n_frames'    : len(clip_frames),
                'frame_paths' : ';'.join(clip_frames),   # semicolon-separated
                'label'       : '',    # filled by annotate_tracks.py
            })

    print(f"    Tracks detected: {len(track_data):>3}  |  "
          f"Valid (≥{MIN_TRACK_LEN} frames): {n_valid_tracks:>3}  |  "
          f"Clips generated: {len(clip_rows):>4}")

    return meta_rows, clip_rows


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def process_all(target_fps: int = TARGET_FPS):
    raw_dir  = Path(CUSTOM_RAW_DIR)
    proc_dir = Path(CUSTOM_PROC_DIR)
    proc_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    video_paths = sorted(
        list(raw_dir.glob('*.mp4')) +
        list(raw_dir.glob('*.MP4')) +
        list(raw_dir.glob('*.avi')) +
        list(raw_dir.glob('*.AVI')) +
        list(raw_dir.glob('*.mov'))
    )

    if not video_paths:
        print(f"\n❌  No videos found in: {raw_dir.resolve()}")
        print("    Place your .mp4 / .avi files there and re-run.")
        return

    print(f"\n{'='*65}")
    print(f"  CUSTOM DATASET PROCESSOR")
    print(f"  Videos found : {len(video_paths)}")
    print(f"  Target FPS   : {target_fps}")
    print(f"  Clip length  : {N_FRAMES} frames  |  Stride: {CLIP_STRIDE} frames")
    print(f"  Output dir   : {proc_dir.resolve()}")
    print(f"{'='*65}")

    yolo_model = try_load_yolo()
    hog = None
    if yolo_model is None:
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    all_meta  = []
    all_clips = []

    for vp in video_paths:
        meta_rows, clip_rows = process_video(
            str(vp), str(proc_dir), yolo_model, hog, target_fps
        )
        all_meta.extend(meta_rows)
        all_clips.extend(clip_rows)

    # ── Save CSVs ──────────────────────────────────────────────────────────────
    Path(CUSTOM_META_CSV).parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(all_meta).to_csv(CUSTOM_META_CSV,  index=False)
    pd.DataFrame(all_clips).to_csv(CUSTOM_CLIPS_CSV, index=False)

    total_tracks = len({r['person_id'] for r in all_clips}) if all_clips else 0
    total_clips  = len(all_clips)

    print(f"\n{'='*65}")
    print(f"  ✅  PROCESSING COMPLETE")
    print(f"  Total valid tracks : {total_tracks}")
    print(f"  Total clips        : {total_clips}")
    print(f"  Tracking metadata  : {CUSTOM_META_CSV}")
    print(f"  Clips catalog      : {CUSTOM_CLIPS_CSV}")
    print(f"\n  NEXT STEP:")
    print(f"    python src/data/annotate_tracks.py")
    print(f"{'='*65}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Custom classroom video processor')
    parser.add_argument('--fps', type=int, default=TARGET_FPS,
                        help=f'Frames per second to extract (default: {TARGET_FPS})')
    args = parser.parse_args()
    process_all(target_fps=args.fps)
