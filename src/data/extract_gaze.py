"""
extract_gaze.py -- Phase-2.5 (gaze last-try): per-person HEAD POSE from the existing crops.

MediaPipe FaceMesh is intentionally NOT used: on Colab Python 3.12 its graph init explodes even
after protobuf repair. This module uses dependency-light backends that install cleanly on 3.12:

    --backend insightface   (DEFAULT)  onnxruntime SCRFD detect + buffalo_l pose; no torch needed
    --backend sixdrepnet                torch 6DRepNet (you already run torch on Colab)

Both expose the SAME output, so build_gaze_features.py / eval_gaze.py are unchanged.

INPUT  (already on disk, 20,440 crops):
    data/collab_raw/crops/{video_id}/{track}/clip_xxxx/frame_xxxx.jpg
OUTPUT:
    data/gaze/headpose.csv  with columns:
        video_id, track, clip, frame, yaw, pitch, roll, face_found
    (degrees; NaN when no face. Yaw SIGN convention is fixed per backend -- consistency is all the
     within-scene gate needs, the linear head absorbs the sign.)

The per-video FACE-FOUND COVERAGE printed at the end is the first go/no-go: if faces are rarely
detectable in this 2fps low-res classroom footage, gaze features cannot carry signal -- honest finding.

RUN ON COLAB
    # default (recommended -- no torch/protobuf/graph issues):
    pip install insightface onnxruntime-gpu opencv-python-headless numpy
    python src/data/extract_gaze.py --crops data/collab_raw/crops --out data/gaze/headpose.csv
    # fallback if you prefer torch:
    pip install sixdrepnet opencv-python-headless
    python src/data/extract_gaze.py --backend sixdrepnet --out data/gaze/headpose.csv

SMOKE-TEST ONE IMAGE FIRST (confirm the backend works before the full run):
    from src.data.extract_gaze import make_predictor
    p = make_predictor("insightface")
    print(p.pose_from_path("data/collab_raw/crops/VID_ (4)/1/clip_0000/frame_0000.jpg"))
    # expect (yaw, pitch, roll, 1)
"""

import os
import csv
import glob
import math
import argparse
import numpy as np

NAN = float("nan")


# ===========================================================================
# pluggable head-pose backends  (each: pose_from_path(path) -> (yaw,pitch,roll,found))
# ===========================================================================
class InsightFacePredictor:
    """SCRFD detection + buffalo_l 3D-landmark pose. face.pose = [pitch, yaw, roll] (deg)."""
    name = "insightface"

    def __init__(self, det_size=640, min_side=128):
        import cv2
        from insightface.app import FaceAnalysis
        self.cv2 = cv2
        self.min_side = min_side
        self.app = FaceAnalysis(name="buffalo_l",
                                providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.app.prepare(ctx_id=0, det_size=(det_size, det_size))

    def pose_from_path(self, p):
        img = self.cv2.imread(p)
        if img is None:
            return NAN, NAN, NAN, 0
        h, w = img.shape[:2]
        if min(h, w) < self.min_side:  # upscale tiny crops so SCRFD can find the face
            s = self.min_side / float(min(h, w))
            img = self.cv2.resize(img, (int(round(w * s)), int(round(h * s))),
                                  interpolation=self.cv2.INTER_CUBIC)
        faces = self.app.get(img)
        if not faces:
            return NAN, NAN, NAN, 0
        f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        pose = getattr(f, "pose", None)
        if pose is None or len(pose) < 3:
            return NAN, NAN, NAN, 0
        pitch, yaw, roll = float(pose[0]), float(pose[1]), float(pose[2])
        return yaw, pitch, roll, 1


class SixDRepPredictor:
    """torch 6DRepNet. model.predict(img) -> (pitch, yaw, roll) (deg); has its own detector."""
    name = "sixdrepnet"

    def __init__(self):
        import cv2
        from sixdrepnet import SixDRepNet
        self.cv2 = cv2
        self.model = SixDRepNet()

    def pose_from_path(self, p):
        img = self.cv2.imread(p)
        if img is None:
            return NAN, NAN, NAN, 0
        try:
            pitch, yaw, roll = self.model.predict(img)
            return float(yaw), float(pitch), float(roll), 1
        except Exception:
            return NAN, NAN, NAN, 0


class MockPredictor:
    """Deterministic, no model -- only for self-testing the scaffolding (path walk / CSV / coverage)."""
    name = "mock"

    def pose_from_path(self, p):
        import hashlib
        h = int(hashlib.md5(p.encode()).hexdigest()[:8], 16)
        return float((h % 61) - 30), float(((h >> 4) % 21) - 10), 0.0, 1


def make_predictor(backend):
    backend = (backend or "insightface").lower()
    try:
        if backend == "insightface":
            return InsightFacePredictor()
        if backend == "sixdrepnet":
            return SixDRepPredictor()
        if backend == "mock":
            return MockPredictor()
    except ImportError as e:
        hint = {"insightface": "pip install insightface onnxruntime-gpu opencv-python-headless",
                "sixdrepnet": "pip install sixdrepnet opencv-python-headless"}.get(backend, "")
        raise SystemExit(f"[gaze] backend '{backend}' not importable ({e}).\n[gaze] install: {hint}")
    raise SystemExit(f"[gaze] unknown backend {backend!r} (use insightface | sixdrepnet)")


# ===========================================================================
def _parse_crop_path(p, crops_root):
    """crops/{video}/{track}/clip_xxxx/frame_xxxx.jpg -> (video, track, clip_idx, frame_idx)."""
    rel = os.path.relpath(p, crops_root).replace("\\", "/").split("/")
    if len(rel) < 4:
        return None
    video, track, clip, frame = rel[0], rel[1], rel[2], rel[3]
    clip_i = int("".join(ch for ch in clip if ch.isdigit()) or -1)
    frame_i = int("".join(ch for ch in os.path.splitext(frame)[0] if ch.isdigit()) or -1)
    return video, track, clip_i, frame_i


def main():
    ap = argparse.ArgumentParser(description="Per-person head pose (gaze last-try) from crops")
    ap.add_argument("--crops", default="data/collab_raw/crops")
    ap.add_argument("--out", default="data/gaze/headpose.csv")
    ap.add_argument("--backend", default="insightface",
                    choices=["insightface", "sixdrepnet", "mock"])
    ap.add_argument("--videos", nargs="*", default=None,
                    help="optional subset of video_id folder names (default: all)")
    args = ap.parse_args()

    pred = make_predictor(args.backend)
    print(f"[gaze] head-pose backend = {pred.name}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    paths = sorted(glob.glob(os.path.join(args.crops, "*", "*", "*", "*.jpg")))
    if args.videos:
        keep = set(args.videos)
        paths = [p for p in paths
                 if (_parse_crop_path(p, args.crops) or [None])[0] in keep]
    print(f"[gaze] {len(paths)} crops under {args.crops}"
          + (f"  (subset {args.videos})" if args.videos else ""))

    seen = {}
    n = 0
    with open(args.out, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["video_id", "track", "clip", "frame", "yaw", "pitch", "roll", "face_found"])
        for p in paths:
            meta = _parse_crop_path(p, args.crops)
            if meta is None:
                continue
            video, track, clip_i, frame_i = meta
            try:
                yaw, pitch, roll, found = pred.pose_from_path(p)
            except Exception:
                yaw, pitch, roll, found = NAN, NAN, NAN, 0
            wr.writerow([video, track, clip_i, frame_i,
                         "" if math.isnan(yaw) else round(yaw, 2),
                         "" if math.isnan(pitch) else round(pitch, 2),
                         "" if math.isnan(roll) else round(roll, 2), found])
            s = seen.setdefault(video, [0, 0])
            s[0] += 1
            s[1] += found
            n += 1
            if n % 2000 == 0:
                print(f"  ...{n}/{len(paths)} crops")

    print(f"[gaze] wrote {n} rows -> {args.out}")
    print("[gaze] FACE-FOUND COVERAGE per video  (this IS the go/no-go diagnostic):")
    covs = []
    for v in sorted(seen):
        tot, fnd = seen[v]
        cov = 100.0 * fnd / max(tot, 1)
        covs.append(cov)
        print(f"   {v:24} {fnd:5d}/{tot:5d}  = {cov:5.1f}%")
    if covs:
        print(f"[gaze] overall coverage = {np.mean(covs):.1f}% (median {np.median(covs):.1f}%).")
        print("[gaze] rule of thumb: if most videos are < ~40%, gaze features will be too sparse")
        print("       to separate pairs -> that is the honest answer; do NOT force a pair head.")
    print("[gaze] next: python src/data/build_gaze_features.py --headpose", args.out)


if __name__ == "__main__":
    main()
