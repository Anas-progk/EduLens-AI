"""
extract_pair_geometry.py -- recover the per-frame bounding boxes the tracker computes but
DISCARDS (only crops are saved). Detection-only re-run; writes a bbox table that joins 1:1
to the existing pairs.

ISOLATION
---------
Imports the EXACT detector + SimpleIoUTracker + extract_upper_body_crop + constants from
collab_video_processor (does NOT modify it) and replays the identical frame loop. Because the
detector (YOLOv8n @ conf 0.40) and the greedy-IoU tracker are deterministic, this reproduces
the SAME track IDs and frame_rel numbering as the original run -- so the bboxes align to the
existing track_id_A/track_id_B + start_frame in pair_catalog_33.csv. No crops are written.

Output CSV columns:
    video_id, track_id, frame_rel, x, y, w, h, frame_w, frame_h
where (x, y, w, h) is the full YOLO person box (top-left + size), frame_rel is the sampled
frame index (extract_num) -- the same space as the catalog's start_frame.

Run on Colab (needs cv2 + ultralytics/torch):
    python src/data/extract_pair_geometry.py --videos videos --out data/collab_raw/bboxes_geom.csv
    # multiple sources:
    python src/data/extract_pair_geometry.py --videos videos custom_dataset/EduAction_E --out ...
"""

import os
import sys
import csv
import argparse
from collections import defaultdict
from pathlib import Path

import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE), os.path.dirname(os.path.dirname(_HERE))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from collab_video_processor import (
        build_detector, SimpleIoUTracker, extract_upper_body_crop,
        FPS_EXTRACT, YOLO_CONF,
    )
except ImportError:
    from src.data.collab_video_processor import (
        build_detector, SimpleIoUTracker, extract_upper_body_crop,
        FPS_EXTRACT, YOLO_CONF,
    )


def iter_videos(paths):
    """Yield .mp4/.MP4 files from a list of dirs and/or files (sorted, deterministic)."""
    seen = []
    for p in paths:
        pp = Path(p)
        if pp.is_dir():
            seen += sorted(list(pp.glob("*.mp4")) + list(pp.glob("*.MP4")))
        elif pp.is_file() and pp.suffix.lower() == ".mp4":
            seen.append(pp)
    # de-dup preserving order
    out, taken = [], set()
    for v in seen:
        if str(v) not in taken:
            out.append(v); taken.add(str(v))
    return out


def extract_one(video_path, detector, fps_extract):
    """Replay collab_video_processor._process_one_video's frame loop, recording bboxes
    instead of crops. Returns list of (track_id, frame_rel, x, y, w, h, frame_w, frame_h)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_skip = max(1, int(round(src_fps / fps_extract)))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tracker = SimpleIoUTracker()
    rows = []
    frame_num = 0
    extract_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1
        if frame_num % frame_skip != 0:
            continue
        detections = detector.detect(frame)
        tracks = tracker.update(detections)
        for tid, bbox in tracks.items():
            # IMPORTANT: replicate the crop-validity gate -- a track only records a frame
            # when extract_upper_body_crop succeeds, exactly as in the original run.
            crop = extract_upper_body_crop(frame, bbox)
            if crop is not None:
                x, y, w, h = bbox
                rows.append((tid, extract_num, int(x), int(y), int(w), int(h), frame_w, frame_h))
        extract_num += 1
    cap.release()
    return rows


def main():
    ap = argparse.ArgumentParser(description="Recover per-frame bboxes (detection-only, aligned to pairs)")
    ap.add_argument("--videos", nargs="+", default=["videos"],
                    help="one or more dirs/files of source videos (same ones used originally)")
    ap.add_argument("--out", default="data/collab_raw/bboxes_geom.csv")
    ap.add_argument("--yolo_model", default="yolov8n.pt")
    ap.add_argument("--conf", type=float, default=YOLO_CONF)
    ap.add_argument("--fps_extract", type=int, default=FPS_EXTRACT)
    args = ap.parse_args()

    videos = iter_videos(args.videos)
    if not videos:
        print(f"[geom] no videos found under {args.videos}"); sys.exit(1)
    detector, dtype = build_detector(args.yolo_model, args.conf)
    print(f"[geom] detector={dtype}  conf={args.conf}  fps_extract={args.fps_extract}  videos={len(videos)}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n_rows = 0
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["video_id", "track_id", "frame_rel", "x", "y", "w", "h", "frame_w", "frame_h"])
        for vi, vp in enumerate(videos):
            vid_id = vp.stem
            try:
                rows = extract_one(vp, detector, args.fps_extract)
            except Exception as e:
                print(f"  [{vi+1}/{len(videos)}] ERROR {vid_id}: {e}")
                continue
            for (tid, fr, x, y, w, h, fw, fh) in rows:
                wr.writerow([vid_id, tid, fr, x, y, w, h, fw, fh])
            n_rows += len(rows)
            print(f"  [{vi+1}/{len(videos)}] {vid_id}: {len(rows)} bbox rows, "
                  f"{len(set(t for t, *_ in rows))} tracks")
    print(f"[geom] wrote {n_rows} rows -> {args.out}")
    print("[geom] next: python src/data/build_pair_geometry.py --bboxes", args.out)


if __name__ == "__main__":
    main()
