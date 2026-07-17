"""
collab_video_processor.py -- Process /videos/ for Phase 2 collaboration dataset.

Takes the 40 WhatsApp classroom videos and:
  1. Extracts frames at 3fps
  2. Detects persons (HOG, same robust approach as Phase 1)
  3. Tracks with SimpleIoU
  4. Saves upper-body crops per person per clip (8-frame windows)
  5. Generates pair_catalog.csv: all (person_A, person_B) pairs
     visible simultaneously in same clip window

Output structure:
  data/collab_raw/
    crops/
      {video_id}/{track_id}/clip_{clip_idx:04d}/
        frame_0000.jpg ... frame_0007.jpg
    pair_catalog.csv   ← pairs ready for annotation
    processing_log.csv ← stats per video

Usage:
  python src/data/collab_video_processor.py
  python src/data/collab_video_processor.py --input_dir videos/ --output_dir data/collab_raw/
  python src/data/collab_video_processor.py --resume   # skip already processed videos
"""

import os
import sys
import cv2
import csv
import argparse
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FPS_EXTRACT   = 3         # Frames per second to sample from video
                          # 3fps: captures natural interaction rhythm without redundancy
CLIP_LEN      = 8        # Frames per clip (same as engagement model)
CLIP_STRIDE   = 4        # Sliding window stride (50% overlap)
CROP_FRAC     = 0.70     # Upper-body crop fraction (same as Phase 1)
MIN_BOX_AREA  = 2000     # Minimum person bbox area in pixels (filter noise)
IOU_THRESHOLD = 0.35     # IoU for track matching (looser than Phase1 for varied scenes)
MAX_LOST_FRAMES = 6      # Frames to keep track alive without detection


# ---------------------------------------------------------------------------
# SimpleIoU Tracker (same proven approach as Phase 1)
# ---------------------------------------------------------------------------

class SimpleIoUTracker:
    """
    Lightweight IoU-based multi-object tracker.
    No external dependencies. Works well for 3fps extraction in seated settings.
    """

    def __init__(self, iou_thresh=IOU_THRESHOLD, max_lost=MAX_LOST_FRAMES):
        self.iou_thresh = iou_thresh
        self.max_lost   = max_lost
        self._tracks    = {}   # track_id → {'bbox': ..., 'lost': int}
        self._next_id   = 1

    def update(self, detections: List[Tuple]) -> Dict[int, Tuple]:
        """
        Args:
            detections: list of (x, y, w, h) bboxes

        Returns:
            dict: {track_id → (x, y, w, h)}
        """
        # Increment lost counter
        for tid in list(self._tracks.keys()):
            self._tracks[tid]['lost'] += 1
            if self._tracks[tid]['lost'] > self.max_lost:
                del self._tracks[tid]

        if not detections:
            return {tid: t['bbox'] for tid, t in self._tracks.items()}

        matched    = set()
        det_matched = set()

        # Match detections to existing tracks
        for tid, track in self._tracks.items():
            best_iou = 0.0
            best_di  = -1
            for di, det in enumerate(detections):
                if di in det_matched:
                    continue
                iou = self._iou(track['bbox'], det)
                if iou > best_iou:
                    best_iou = iou
                    best_di  = di
            if best_iou >= self.iou_thresh and best_di >= 0:
                self._tracks[tid]['bbox'] = detections[best_di]
                self._tracks[tid]['lost'] = 0
                matched.add(tid)
                det_matched.add(best_di)

        # Register new tracks for unmatched detections
        for di, det in enumerate(detections):
            if di not in det_matched:
                self._tracks[self._next_id] = {'bbox': det, 'lost': 0}
                self._next_id += 1

        return {tid: t['bbox'] for tid, t in self._tracks.items()}

    @staticmethod
    def _iou(boxA, boxB) -> float:
        ax1, ay1, aw, ah = boxA
        bx1, by1, bw, bh = boxB
        ax2, ay2 = ax1 + aw, ay1 + ah
        bx2, by2 = bx1 + bw, by1 + bh
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = aw * ah + bw * bh - inter
        return inter / (union + 1e-6)


# ---------------------------------------------------------------------------
# HOG Person Detector (same as Phase 1 fallback — proven reliable)
# ---------------------------------------------------------------------------

class HOGPersonDetector:
    """OpenCV HOG-based person detector. No external deps."""

    def __init__(self):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Returns list of (x, y, w, h) bboxes for detected persons.
        Applies NMS internally.
        """
        h, w = frame.shape[:2]
        scale = min(640 / w, 480 / h, 1.0)
        small = cv2.resize(frame, (int(w * scale), int(h * scale)))

        boxes, weights = self.hog.detectMultiScale(
            small,
            winStride   = (8, 8),
            padding     = (4, 4),
            scale       = 1.05,
            hitThreshold= 0.3,
        )

        results = []
        if len(boxes) == 0:
            return results

        for (x, y, bw, bh), conf in zip(boxes, weights):
            # Scale back to original coordinates
            x  = int(x / scale)
            y  = int(y / scale)
            bw = int(bw / scale)
            bh = int(bh / scale)
            if bw * bh >= MIN_BOX_AREA:
                results.append((x, y, bw, bh))

        return results


# ---------------------------------------------------------------------------
# Crop helper (same as Phase 1)
# ---------------------------------------------------------------------------

def extract_upper_body_crop(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    crop_frac: float = CROP_FRAC,
    target_size: int = 224,
) -> Optional[np.ndarray]:
    """
    Extract upper-body crop and resize to (target_size, target_size).
    Returns None if bbox is out of bounds or too small.
    """
    h, w = frame.shape[:2]
    x, y, bw, bh = bbox

    # Take upper portion
    crop_h = int(bh * crop_frac)
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w, x + bw)
    y2 = min(h, y + crop_h)

    if x2 <= x1 or y2 <= y1 or (x2 - x1) < 20 or (y2 - y1) < 20:
        return None

    crop = frame[y1:y2, x1:x2]
    crop = cv2.resize(crop, (target_size, target_size))
    return crop


# ---------------------------------------------------------------------------
# Main processor
# ---------------------------------------------------------------------------

class CollabVideoProcessor:
    """
    Processes a directory of videos to generate collaboration training data.
    """

    def __init__(
        self,
        input_dir  : str = "videos",
        output_dir : str = "data/collab_raw",
        fps_extract : int = FPS_EXTRACT,
        clip_len    : int = CLIP_LEN,
        clip_stride : int = CLIP_STRIDE,
    ):
        self.input_dir   = Path(input_dir)
        self.output_dir  = Path(output_dir)
        self.fps_extract = fps_extract
        self.clip_len    = clip_len
        self.clip_stride = clip_stride

        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "crops").mkdir(exist_ok=True)

        self.detector = HOGPersonDetector()
        self.pair_catalog   = []   # rows for pair_catalog.csv
        self.processing_log = []   # rows for processing_log.csv

    def process_all(self, resume: bool = False):
        """Process all .mp4 files in input_dir."""
        videos = sorted(self.input_dir.glob("*.mp4"))
        if not videos:
            print(f"No .mp4 files found in {self.input_dir}")
            return

        print(f"Found {len(videos)} videos in {self.input_dir}")
        already_done = self._load_done_videos()

        for vi, video_path in enumerate(videos):
            vid_id = video_path.stem
            if resume and vid_id in already_done:
                print(f"[{vi+1}/{len(videos)}] SKIP (already done): {vid_id}")
                continue

            print(f"\n[{vi+1}/{len(videos)}] Processing: {vid_id}")
            try:
                stats = self._process_one_video(video_path, vid_id)
                self.processing_log.append({'video_id': vid_id, **stats})
            except Exception as e:
                print(f"  ERROR processing {vid_id}: {e}")
                self.processing_log.append({
                    'video_id': vid_id, 'error': str(e),
                    'clips': 0, 'pairs': 0, 'persons': 0
                })

        self._save_catalogs()
        self._print_summary()

    def _process_one_video(self, video_path: Path, vid_id: str) -> dict:
        """
        Process a single video file.
        Returns stats dict.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_skip = max(1, int(round(src_fps / self.fps_extract)))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"  {frame_w}x{frame_h} @ {src_fps:.1f}fps  "
              f"→ extract every {frame_skip} frames (~{self.fps_extract}fps)")

        tracker  = SimpleIoUTracker()
        # track_id → list of frame records: {'frame_rel': int, 'crop': np.array, 'bbox': tuple}
        track_frames: Dict[int, list] = defaultdict(list)
        frame_num  = 0
        extract_num = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_num += 1
            if frame_num % frame_skip != 0:
                continue

            detections = self.detector.detect(frame)
            tracks     = tracker.update(detections)

            for tid, bbox in tracks.items():
                crop = extract_upper_body_crop(frame, bbox)
                if crop is not None:
                    track_frames[tid].append({
                        'frame_rel': extract_num,
                        'crop': crop,
                        'bbox': bbox,
                        'frame_w': frame_w,
                        'frame_h': frame_h,
                    })
            extract_num += 1

        cap.release()

        # Generate clips from each track
        n_clips  = 0
        n_pairs  = 0
        n_persons = len(track_frames)

        # Map: frame_rel → {track_id → bbox}  (for pair generation)
        frame_bbox_map: Dict[int, Dict[int, tuple]] = defaultdict(dict)
        for tid, frames in track_frames.items():
            for rec in frames:
                frame_bbox_map[rec['frame_rel']][tid] = rec['bbox']

        # Clip-level pair tracking: clip_key → [track_ids visible in this clip]
        clip_person_map: Dict[str, List[int]] = {}

        for tid, frames in track_frames.items():
            if len(frames) < self.clip_len:
                continue   # Not enough frames for even one clip

            clip_dir = self.output_dir / "crops" / vid_id / str(tid)
            clip_dir.mkdir(parents=True, exist_ok=True)

            # Sliding window over track frames
            clip_idx = 0
            for start in range(0, len(frames) - self.clip_len + 1, self.clip_stride):
                clip_frames = frames[start : start + self.clip_len]
                clip_key = f"{vid_id}_t{tid:03d}_c{clip_idx:04d}"

                # Save crops
                clip_subdir = clip_dir / f"clip_{clip_idx:04d}"
                clip_subdir.mkdir(exist_ok=True)

                for fi, rec in enumerate(clip_frames):
                    cv2.imwrite(
                        str(clip_subdir / f"frame_{fi:04d}.jpg"),
                        rec['crop'],
                        [cv2.IMWRITE_JPEG_QUALITY, 90]
                    )

                # Which other tracks are visible in this clip window?
                clip_frame_nums = {rec['frame_rel'] for rec in clip_frames}
                covisible = set()
                for fn in clip_frame_nums:
                    for other_tid in frame_bbox_map[fn]:
                        if other_tid != tid:
                            covisible.add(other_tid)

                clip_person_map[clip_key] = {
                    'vid_id': vid_id,
                    'track_id': tid,
                    'clip_idx': clip_idx,
                    'clip_dir': str(clip_subdir),
                    'covisible': list(covisible),
                    'n_frames': len(clip_frames),
                    'start_frame': clip_frames[0]['frame_rel'],
                    'end_frame': clip_frames[-1]['frame_rel'],
                    'frame_w': clip_frames[0]['frame_w'],
                    'frame_h': clip_frames[0]['frame_h'],
                }
                clip_idx += 1
                n_clips += 1

        # Generate pair catalog entries
        # A pair entry is: (clip_A, clip_B) where both tracks are co-visible
        processed_pairs = set()
        for clip_key, info in clip_person_map.items():
            tid_A = info['track_id']
            for tid_B in info['covisible']:
                # Only emit each pair once (A < B convention)
                pair_key = (vid_id, min(tid_A, tid_B), max(tid_A, tid_B),
                            info['start_frame'])
                if pair_key in processed_pairs:
                    continue
                processed_pairs.add(pair_key)

                # Find clip for person B in same window
                b_clip_key = self._find_covisible_clip(
                    clip_person_map, vid_id, tid_B, info['start_frame'], info['end_frame']
                )
                if b_clip_key is None:
                    continue

                self.pair_catalog.append({
                    'pair_id':       f"{vid_id}_A{tid_A:03d}_B{tid_B:03d}_f{info['start_frame']:04d}",
                    'video_id':      vid_id,
                    'track_id_A':    tid_A,
                    'track_id_B':    tid_B,
                    'clip_dir_A':    info['clip_dir'],
                    'clip_dir_B':    clip_person_map[b_clip_key]['clip_dir'],
                    'start_frame':   info['start_frame'],
                    'frame_w':       info['frame_w'],
                    'frame_h':       info['frame_h'],
                    'label':         '',      # ← annotator will fill this: C / N / S
                    'annotated':     False,
                })
                n_pairs += 1

        return {
            'clips': n_clips,
            'pairs': n_pairs,
            'persons': n_persons,
            'total_src_frames': frame_num,
            'extracted_frames': extract_num,
        }

    def _find_covisible_clip(
        self,
        clip_map    : dict,
        vid_id      : str,
        track_id    : int,
        start_frame : int,
        end_frame   : int,
    ) -> Optional[str]:
        """Find a clip for track_id that overlaps [start_frame, end_frame]."""
        for key, info in clip_map.items():
            if info['vid_id'] != vid_id or info['track_id'] != track_id:
                continue
            # Check overlap
            if info['start_frame'] <= end_frame and info['end_frame'] >= start_frame:
                return key
        return None

    def _save_catalogs(self):
        """Write pair_catalog.csv and processing_log.csv."""
        # Pair catalog
        pair_csv = self.output_dir / "pair_catalog.csv"
        if self.pair_catalog:
            fieldnames = list(self.pair_catalog[0].keys())
            with open(pair_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.pair_catalog)
            print(f"\nPair catalog saved: {pair_csv} ({len(self.pair_catalog)} pairs)")

        # Processing log
        log_csv = self.output_dir / "processing_log.csv"
        if self.processing_log:
            fieldnames = ['video_id', 'clips', 'pairs', 'persons',
                          'total_src_frames', 'extracted_frames', 'error']
            with open(log_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(self.processing_log)

    def _load_done_videos(self):
        """Load set of already-processed video IDs from log."""
        log_csv = self.output_dir / "processing_log.csv"
        done = set()
        if log_csv.exists():
            with open(log_csv) as f:
                for row in csv.DictReader(f):
                    if not row.get('error'):
                        done.add(row['video_id'])
        return done

    def _print_summary(self):
        total_clips = sum(r.get('clips', 0) for r in self.processing_log)
        total_pairs = sum(r.get('pairs', 0) for r in self.processing_log)
        total_persons = sum(r.get('persons', 0) for r in self.processing_log)
        errors = sum(1 for r in self.processing_log if r.get('error'))
        print(f"\n{'='*50}")
        print(f"PROCESSING COMPLETE")
        print(f"  Videos processed: {len(self.processing_log)} ({errors} errors)")
        print(f"  Total persons tracked: {total_persons}")
        print(f"  Total clips generated: {total_clips}")
        print(f"  Total pairs for annotation: {total_pairs}")
        print(f"{'='*50}")
        print(f"\nNext step: python src/data/collab_annotator.py")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Process videos for Phase 2 collab dataset")
    parser.add_argument("--input_dir",  default="videos",         help="Directory with .mp4 videos")
    parser.add_argument("--output_dir", default="data/collab_raw", help="Output directory")
    parser.add_argument("--fps",        type=int, default=3,       help="Extraction FPS")
    parser.add_argument("--resume",     action="store_true",        help="Skip already processed videos")
    args = parser.parse_args()

    processor = CollabVideoProcessor(
        input_dir   = args.input_dir,
        output_dir  = args.output_dir,
        fps_extract = args.fps,
    )
    processor.process_all(resume=args.resume)


if __name__ == "__main__":
    main()
