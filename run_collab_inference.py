"""
run_collab_inference.py -- Quick launcher for Phase 2 collaboration inference.

Usage:
  python run_collab_inference.py                          # use default test video
  python run_collab_inference.py --source videos/VID-20260421-WA0013.mp4
  python run_collab_inference.py --source 0              # webcam
  python run_collab_inference.py --eng_only              # engagement only (no collab model yet)
"""

from src.inference.multi_person_collab_inference import CollabInferenceSystem
import os
import argparse

os.makedirs("outputs", exist_ok=True)
os.makedirs("database", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--source",      default="temp/test_H264.mp4")
parser.add_argument("--save_output", default=None)  # auto-named below
parser.add_argument("--eng_model",   default="weights/best_clip_model.pth")
parser.add_argument("--collab_model",default="weights/best_collab_model.pth")
parser.add_argument("--db_path",     default="database/persons.db")
parser.add_argument("--device",      default="cpu")
parser.add_argument("--show",        action="store_true")
parser.add_argument("--eng_only",    action="store_true", help="Skip collab model")
parser.add_argument("--no_log",      action="store_true", help="Don't log to DB")
args = parser.parse_args()

# Auto-name output
if args.save_output is None:
    import os
    existing = [f for f in os.listdir("outputs") if f.startswith("collab_output_") and f.endswith(".mp4")]
    next_idx = len(existing)
    args.save_output = f"outputs/collab_output_{next_idx:04d}.mp4"

collab_path = None if args.eng_only else args.collab_model

system = CollabInferenceSystem(
    engagement_model_path = args.eng_model,
    collab_model_path     = collab_path or "",
    db_path               = args.db_path,
    device                = args.device,
)

system.run(
    source      = args.source,
    save_output = args.save_output,
    show_window = args.show,
    log_db      = not args.no_log,
)

print("DONE")
