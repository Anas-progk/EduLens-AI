"""
annotate_tracks.py -- Per-clip engagement annotation tool (OpenCV GUI).

For each 8-frame clip extracted from your classroom videos, this tool:
  - Displays 8 frames as a grid (4 cols × 2 rows)
  - Shows video name, track ID, clip index, and annotation progress
  - Accepts keyboard labels:
        [E]  → Engaged
        [N]  → Not Engaged
        [S]  → Skip (uncertain or ambiguous)
        [B]  → Back (undo last annotation, re-label previous clip)
        [Q]  → Save and quit (safe to resume later)

Progress is auto-saved every 10 annotations and on quit.
Re-running the script resumes from where you left off.

Annotation tips:
  - Engaged:     continuous reading / typing / focused on workspace /
                 attentive posture toward task
  - Not Engaged: phone usage / head-down idle / repeated looking away /
                 sleeping posture / disengaged body language
  - Skip:        person barely visible / highly occluded / posture
                 completely ambiguous even over 8 frames
  - Looking DOWN ≠ Not Engaged automatically.
    If person is reading/writing/coding → Engaged.
    Temporal context (8 frames) should make this clear.

Usage:
  python src/data/annotate_tracks.py
"""

import os
import sys
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import CONFIG


# ──────────────────────────────────────────────────────────────────────────────
# Layout constants
# ──────────────────────────────────────────────────────────────────────────────
CUSTOM_CLIPS_CSV = CONFIG.get('custom_clips_csv', 'custom_dataset/clips_catalog.csv')
CUSTOM_ANNOT_CSV = CONFIG.get('custom_annot_csv', 'custom_dataset/annotations.csv')

THUMB_W    = 160   # width of each frame thumbnail
THUMB_H    = 180   # height (slightly taller than wide for portrait crops)
GRID_COLS  = 4
GRID_ROWS  = 2
GAP        = 5     # gap between thumbnails (pixels)
INFO_H     = 110   # height of top info bar
WIN_NAME   = "Engagement Annotator  |  E=Engaged  N=NotEngaged  S=Skip  B=Back  Q=Quit"


# ──────────────────────────────────────────────────────────────────────────────
# Frame loading + grid construction
# ──────────────────────────────────────────────────────────────────────────────

def load_frames_as_thumbs(frame_paths_str: str) -> list:
    """
    Load 8 frame images from semicolon-separated path string.
    Returns list of (THUMB_H × THUMB_W × 3) BGR arrays.
    Missing / corrupt images → black placeholder.
    """
    paths  = frame_paths_str.split(';')
    thumbs = []
    for p in paths:
        img = cv2.imread(p.strip())
        if img is None:
            img = np.zeros((THUMB_H, THUMB_W, 3), dtype=np.uint8)
        else:
            img = cv2.resize(img, (THUMB_W, THUMB_H))
        thumbs.append(img)

    n_needed = GRID_COLS * GRID_ROWS
    while len(thumbs) < n_needed:
        thumbs.append(np.zeros((THUMB_H, THUMB_W, 3), dtype=np.uint8))
    return thumbs[:n_needed]


def make_grid(thumbs: list) -> np.ndarray:
    """
    Arrange 8 thumbnails into a (GRID_ROWS × GRID_COLS) grid with gaps.
    Overlays frame index on each thumbnail.
    """
    grid_w = GRID_COLS * THUMB_W + (GRID_COLS + 1) * GAP
    grid_h = GRID_ROWS * THUMB_H + (GRID_ROWS + 1) * GAP
    grid   = np.full((grid_h, grid_w, 3), 30, dtype=np.uint8)  # dark background

    for i, thumb in enumerate(thumbs):
        row = i // GRID_COLS
        col = i % GRID_COLS
        y1  = GAP + row * (THUMB_H + GAP)
        x1  = GAP + col * (THUMB_W + GAP)
        grid[y1 : y1 + THUMB_H, x1 : x1 + THUMB_W] = thumb

        # Frame index badge (top-left corner of thumbnail)
        badge = f"F{i}"
        cv2.rectangle(grid, (x1, y1), (x1 + 26, y1 + 18), (0, 0, 0), -1)
        cv2.putText(grid, badge, (x1 + 2, y1 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 220, 80), 1, cv2.LINE_AA)

    return grid


def make_info_bar(
    clip_id   : str,
    video_id  : str,
    track_id  : int,
    clip_idx  : int,
    cur_n     : int,
    total     : int,
    e_cnt     : int,
    n_cnt     : int,
    s_cnt     : int,
    width     : int,
) -> np.ndarray:
    """
    Build the info panel displayed above the frame grid.
    Shows: clip info, progress bar, annotation counts, key guide.
    """
    bar = np.full((INFO_H, width, 3), 20, dtype=np.uint8)

    # Progress bar (bottom strip of info bar)
    filled_w = int(width * cur_n / max(total, 1))
    cv2.rectangle(bar, (0, INFO_H - 7), (filled_w, INFO_H), (0, 160, 80), -1)
    cv2.rectangle(bar, (0, INFO_H - 7), (width, INFO_H),   (60, 60, 60),  1)

    pct = 100 * cur_n / max(total, 1)
    total_done = e_cnt + n_cnt + s_cnt

    lines = [
        (f"Clip: {clip_id}",
         (210, 230, 210)),
        (f"Video: {video_id}   Track: {track_id}   Clip index: {clip_idx}   "
         f"({cur_n}/{total} = {pct:.0f}%)",
         (160, 190, 230)),
        (f"Annotated this session — E: {e_cnt}  NE: {n_cnt}  Skip: {s_cnt}  "
         f"Total done: {total_done}",
         (200, 200, 160)),
        (f"[ E ] Engaged    [ N ] Not Engaged    [ S ] Skip    "
         f"[ B ] Back    [ Q ] Save & Quit",
         (100, 210, 100)),
    ]

    y = 22
    for text, color in lines:
        cv2.putText(bar, text, (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1, cv2.LINE_AA)
        y += 24

    return bar


# ──────────────────────────────────────────────────────────────────────────────
# Annotation state helpers
# ──────────────────────────────────────────────────────────────────────────────

def save_to_csv(existing_df: pd.DataFrame, new_entries: list) -> pd.DataFrame:
    """Append new_entries to existing_df, deduplicate on clip_id, save CSV."""
    if not new_entries:
        return existing_df
    new_df   = pd.DataFrame(new_entries)
    combined = pd.concat([existing_df, new_df], ignore_index=True)
    combined.drop_duplicates(subset='clip_id', keep='last', inplace=True)
    combined.to_csv(CUSTOM_ANNOT_CSV, index=False)
    return combined


# ──────────────────────────────────────────────────────────────────────────────
# Main annotation loop
# ──────────────────────────────────────────────────────────────────────────────

def annotate():
    # ── Load clips catalog ─────────────────────────────────────────────────────
    if not Path(CUSTOM_CLIPS_CSV).exists():
        print(f"\n❌  clips_catalog.csv not found at: {CUSTOM_CLIPS_CSV}")
        print("    Run  python src/data/custom_video_processor.py  first.")
        return

    clips_df = pd.read_csv(CUSTOM_CLIPS_CSV)
    # Sort for consistent ordering: video → track → clip index
    clips_df = clips_df.sort_values(
        ['video_id', 'track_id', 'clip_idx']
    ).reset_index(drop=True)

    print(f"\n{'='*65}")
    print(f"  ENGAGEMENT ANNOTATION TOOL")
    print(f"  Clips catalog : {len(clips_df)} clips from {clips_df['video_id'].nunique()} videos")

    # ── Resume: load existing annotations ─────────────────────────────────────
    if Path(CUSTOM_ANNOT_CSV).exists():
        annot_df = pd.read_csv(CUSTOM_ANNOT_CSV)
        print(f"  Resuming      : {len(annot_df)} already annotated")
    else:
        annot_df = pd.DataFrame(columns=['clip_id', 'label', 'annotated_at'])

    done_ids = set(annot_df['clip_id'].tolist())
    todo_df  = clips_df[~clips_df['clip_id'].isin(done_ids)].reset_index(drop=True)

    print(f"  To annotate   : {len(todo_df)} clips remaining")
    print(f"{'='*65}")

    if len(todo_df) == 0:
        print("\n✅  All clips already annotated!")
        _print_summary(annot_df)
        return

    # Count from existing annotations for running display
    e_cnt = int((annot_df['label'] == 'E').sum())
    n_cnt = int((annot_df['label'] == 'N').sum())
    s_cnt = int((annot_df['label'] == 'S').sum())

    new_entries = []  # annotations added this session (for undo support)
    history     = []  # stack of (clip_id, label) for back navigation

    # ── OpenCV window setup ───────────────────────────────────────────────────
    grid_w = GRID_COLS * THUMB_W + (GRID_COLS + 1) * GAP
    grid_h = GRID_ROWS * THUMB_H + (GRID_ROWS + 1) * GAP
    win_h  = INFO_H + grid_h

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, grid_w, win_h)

    total = len(todo_df)
    idx   = 0

    def _autosave():
        nonlocal annot_df
        annot_df = save_to_csv(annot_df, new_entries)
        new_entries.clear()
        history.clear()

    # ── Annotation loop ───────────────────────────────────────────────────────
    while idx < total:
        row       = todo_df.iloc[idx]
        clip_id   = str(row['clip_id'])
        video_id  = str(row['video_id'])
        track_id  = int(row['track_id'])
        clip_idx  = int(row['clip_idx'])
        fp_str    = str(row['frame_paths'])

        # Build display
        thumbs  = load_frames_as_thumbs(fp_str)
        grid    = make_grid(thumbs)
        info    = make_info_bar(clip_id, video_id, track_id, clip_idx,
                                idx, total, e_cnt, n_cnt, s_cnt, grid_w)
        display = np.vstack([info, grid])
        cv2.imshow(WIN_NAME, display)

        # Wait for valid key
        label = None
        while True:
            key = cv2.waitKey(0) & 0xFF

            if key in (ord('e'), ord('E')):
                label = 'E'; e_cnt += 1
                break

            elif key in (ord('n'), ord('N')):
                label = 'N'; n_cnt += 1
                break

            elif key in (ord('s'), ord('S')):
                label = 'S'; s_cnt += 1
                break

            elif key in (ord('b'), ord('B')):
                # Undo last annotation in this session
                if history:
                    prev_id, prev_label = history.pop()
                    # Remove from new_entries
                    updated = [e for e in new_entries if e['clip_id'] != prev_id]
                    new_entries.clear()
                    new_entries.extend(updated)
                    # Revert counts
                    if prev_label == 'E':   e_cnt -= 1
                    elif prev_label == 'N': n_cnt -= 1
                    elif prev_label == 'S': s_cnt -= 1
                    # Step back in todo_df
                    idx = max(0, idx - 1)
                # label stays None → will re-show clip at new idx
                break

            elif key in (ord('q'), ord('Q')):
                _autosave()
                cv2.destroyAllWindows()
                print(f"\n  💾  Saved annotations → {CUSTOM_ANNOT_CSV}")
                _print_summary(annot_df)
                return

            # Ignore any other key → stay in inner loop waiting

        if label is not None:
            entry = {
                'clip_id'      : clip_id,
                'label'        : label,
                'annotated_at' : datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            }
            new_entries.append(entry)
            history.append((clip_id, label))

            # Auto-save every 10 new annotations (in case of crash)
            if len(new_entries) % 10 == 0:
                annot_df = save_to_csv(annot_df, new_entries)
                new_entries.clear()
                history.clear()
                print(f"  Auto-saved at {idx + 1}/{total}")

            idx += 1
        # If label is None (back was pressed), idx was already adjusted above

    # All done
    _autosave()
    cv2.destroyAllWindows()
    print(f"\n✅  All clips annotated!")
    _print_summary(annot_df)


# ──────────────────────────────────────────────────────────────────────────────
# Summary printer
# ──────────────────────────────────────────────────────────────────────────────

def _print_summary(annot_df: pd.DataFrame):
    e  = int((annot_df['label'] == 'E').sum())
    n  = int((annot_df['label'] == 'N').sum())
    s  = int((annot_df['label'] == 'S').sum())
    total_valid = e + n

    print(f"\n{'='*65}")
    print(f"  ANNOTATION SUMMARY")
    print(f"  Total annotated  : {len(annot_df)}")
    print(f"  Engaged (E)      : {e:>5}  ({100*e/max(total_valid,1):.1f}%)")
    print(f"  Not Engaged (N)  : {n:>5}  ({100*n/max(total_valid,1):.1f}%)")
    print(f"  Skipped (S)      : {s:>5}  (excluded from training)")
    if total_valid < 50:
        print(f"\n  ⚠️   Only {total_valid} usable clips so far.")
        print(f"  Aim for ≥100 clips (≥30 Not Engaged) for meaningful training.")
    print(f"\n  NEXT STEP:")
    print(f"    python src/data/build_custom_splits.py")
    print(f"    python src/data/build_custom_splits.py --merge-daisee")
    print(f"{'='*65}")


if __name__ == '__main__':
    annotate()
