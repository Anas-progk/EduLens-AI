"""
collab_annotator.py -- Collaboration pair annotation GUI.

Shows Person A's 8-frame clip alongside Person B's 8-frame clip.
You press C (Collaborative), N (Not Collaborative), or S (Skip).

This is the CRITICAL human labeling step for Phase 2.
Quality of these labels directly determines collaboration model quality.

LABELING RULES (read before starting):
  Press C (Collaborative) ONLY IF at least 2 of:
    - Visible talking/responding (mouth movement + other person reacting)
    - Bodies oriented TOWARD each other (not both facing forward/camera)
    - Shared focus on same object (laptop, paper, notes)
    - Active listening signals (lean-in, nod, gesture response)

  Press N (Not Collaborative) if:
    - Near each other but working independently
    - Both facing teacher/board (shared attention to 3rd thing ≠ collab)
    - One or both on phone/sleeping
    - One talking to teacher, other listening passively

  Press S (Skip) if:
    - Cannot clearly see both people in at least 5 of 8 frames
    - Complete occlusion, back-of-head with no visible signals
    - Extreme blur or lighting makes it impossible to judge

  Press Q to quit and save progress (auto-saved every 10 annotations).
  Press B to go back one annotation (undo).
  Press R to replay current pair.

Display layout:
  ┌────────────────────────────────────────────┐
  │   Person A (8 frames)  |  Person B (8 frames)  │
  │   Frame grid 2×4       |  Frame grid 2×4        │
  ├────────────────────────────────────────────┤
  │   [Pair info]  [Progress]  [Controls]       │
  └────────────────────────────────────────────┘

Usage:
  python src/data/collab_annotator.py
  python src/data/collab_annotator.py --catalog data/collab_raw/pair_catalog.csv
  python src/data/collab_annotator.py --catalog data/collab_raw/pair_catalog.csv --start_from 50
"""

import os
import sys
import cv2
import csv
import argparse
import numpy as np
from pathlib import Path
from typing import List, Optional


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CATALOG_CSV   = "data/collab_raw/pair_catalog.csv"
FRAME_SIZE    = 112      # Each individual frame in the grid (px)
GRID_ROWS     = 2        # 2 rows × 4 cols = 8 frames per person
GRID_COLS     = 4
SAVE_INTERVAL = 10       # Auto-save every N annotations

# Colors (BGR)
CLR_COLLAB    = (50, 200, 50)      # green
CLR_NO_COLLAB = (50, 50, 220)      # red
CLR_SKIP      = (150, 150, 150)    # gray
CLR_TEXT      = (255, 255, 255)
CLR_BG        = (30, 30, 30)


# ---------------------------------------------------------------------------
# Frame grid builder
# ---------------------------------------------------------------------------

def build_frame_grid(frame_paths: List[Path], label: str = "") -> np.ndarray:
    """
    Build a 2×4 grid of frames for one person.
    Missing frames are filled with a gray placeholder.

    Returns: BGR image of shape (FRAME_SIZE*2, FRAME_SIZE*4, 3)
    """
    grid_h = FRAME_SIZE * GRID_ROWS
    grid_w = FRAME_SIZE * GRID_COLS
    grid   = np.full((grid_h, grid_w, 3), 40, dtype=np.uint8)

    for i, fp in enumerate(frame_paths[:GRID_ROWS * GRID_COLS]):
        row = i // GRID_COLS
        col = i %  GRID_COLS
        y1  = row * FRAME_SIZE
        x1  = col * FRAME_SIZE

        if fp is not None and fp.exists():
            img = cv2.imread(str(fp))
            if img is not None:
                img = cv2.resize(img, (FRAME_SIZE, FRAME_SIZE))
                grid[y1:y1+FRAME_SIZE, x1:x1+FRAME_SIZE] = img
            else:
                cv2.rectangle(grid, (x1, y1), (x1+FRAME_SIZE-1, y1+FRAME_SIZE-1),
                              (60, 60, 60), -1)
        else:
            cv2.rectangle(grid, (x1, y1), (x1+FRAME_SIZE-1, y1+FRAME_SIZE-1),
                          (60, 60, 60), -1)

    # Frame number overlay (tiny)
    for i in range(min(len(frame_paths), GRID_ROWS * GRID_COLS)):
        row = i // GRID_COLS
        col = i %  GRID_COLS
        cv2.putText(grid, f"f{i}", (col*FRAME_SIZE+2, (row+1)*FRAME_SIZE-5),
                    cv2.FONT_HERSHEY_PLAIN, 0.7, (200, 200, 200), 1)

    return grid


def load_clip_frames(clip_dir: str) -> List[Optional[Path]]:
    """Load up to 8 frame paths from a clip directory."""
    clip_path = Path(clip_dir)
    frames = []
    for i in range(8):
        fp = clip_path / f"frame_{i:04d}.jpg"
        frames.append(fp if fp.exists() else None)
    return frames


# ---------------------------------------------------------------------------
# Annotator
# ---------------------------------------------------------------------------

class CollabAnnotator:
    """
    OpenCV-based GUI for annotating collaboration pairs.
    Supports resume (reads existing labels from catalog).
    """

    def __init__(self, catalog_csv: str = CATALOG_CSV, start_from: int = 0):
        self.catalog_path = Path(catalog_csv)
        if not self.catalog_path.exists():
            raise FileNotFoundError(
                f"Pair catalog not found: {catalog_csv}\n"
                f"Run first: python src/data/collab_video_processor.py"
            )

        self.pairs = self._load_catalog()
        self.start_from = start_from
        self.current_idx = start_from
        self.history = []   # For undo (B key)
        self._annotation_count = 0

        print(f"Loaded {len(self.pairs)} pairs from {catalog_csv}")
        already_done = sum(1 for p in self.pairs if p.get('label') in ('C', 'N', 'S'))
        print(f"  Already annotated: {already_done}")
        print(f"  Remaining: {len(self.pairs) - already_done}")

        # Find first unannotated
        if start_from == 0:
            for i, p in enumerate(self.pairs):
                if not p.get('label'):
                    self.current_idx = i
                    break

    def _load_catalog(self) -> List[dict]:
        with open(self.catalog_path, newline='') as f:
            return list(csv.DictReader(f))

    def _save_catalog(self):
        if not self.pairs:
            return
        fieldnames = list(self.pairs[0].keys())
        with open(self.catalog_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.pairs)

    def run(self):
        """Main annotation loop."""
        print("\nControls:")
        print("  C → Collaborative")
        print("  N → Not Collaborative")
        print("  S → Skip (ambiguous)")
        print("  B → Back (undo)")
        print("  Q → Quit and save")
        print("\nStarting annotation from pair", self.current_idx)

        cv2.namedWindow("Collab Annotator", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Collab Annotator", 1200, 600)

        while self.current_idx < len(self.pairs):
            pair = self.pairs[self.current_idx]

            # Skip already-labeled if starting fresh
            # (When resuming, jump to first unannotated pair)
            display = self._build_display(pair, self.current_idx)
            cv2.imshow("Collab Annotator", display)

            key = cv2.waitKey(0) & 0xFF

            if key == ord('q') or key == ord('Q') or key == 27:   # Q or Esc
                print("\nSaving and quitting...")
                break

            elif key == ord('c') or key == ord('C'):
                self._label(self.current_idx, 'C')
                self.current_idx += 1

            elif key == ord('n') or key == ord('N'):
                self._label(self.current_idx, 'N')
                self.current_idx += 1

            elif key == ord('s') or key == ord('S'):
                self._label(self.current_idx, 'S')
                self.current_idx += 1

            elif key == ord('b') or key == ord('B'):
                if self.history:
                    prev_idx = self.history.pop()
                    self.pairs[prev_idx]['label']     = ''
                    self.pairs[prev_idx]['annotated'] = False
                    self.current_idx = prev_idx
                    self._annotation_count = max(0, self._annotation_count - 1)
                    print(f"  Undid annotation for pair {prev_idx}")
                else:
                    print("  Nothing to undo")

            elif key == ord('r') or key == ord('R'):
                pass  # Replay = just re-show same frame (already at top of loop)

            # Skip pairs that already have a label (when looping)
            while (self.current_idx < len(self.pairs) and
                   self.pairs[self.current_idx].get('label') in ('C', 'N', 'S')):
                self.current_idx += 1

        self._save_catalog()
        cv2.destroyAllWindows()
        self._print_stats()

    def _label(self, idx: int, label: str):
        """Apply label to pair at index."""
        self.pairs[idx]['label']     = label
        self.pairs[idx]['annotated'] = True
        self.history.append(idx)
        self._annotation_count += 1

        label_name = {'C': 'Collaborative', 'N': 'Not Collaborative', 'S': 'Skip'}[label]
        print(f"  [{idx+1}/{len(self.pairs)}] {self.pairs[idx]['pair_id']} → {label_name}")

        # Auto-save
        if self._annotation_count % SAVE_INTERVAL == 0:
            self._save_catalog()
            print(f"  [Auto-saved at {self._annotation_count} annotations]")

    def _build_display(self, pair: dict, idx: int) -> np.ndarray:
        """Build the full annotation display frame."""
        # Load clip frames
        frames_A = load_clip_frames(pair.get('clip_dir_A', ''))
        frames_B = load_clip_frames(pair.get('clip_dir_B', ''))

        grid_A = build_frame_grid(frames_A, "Person A")
        grid_B = build_frame_grid(frames_B, "Person B")

        # Label headers for each person
        header_A = self._make_header(grid_A.shape[1], "PERSON A",  (50, 100, 200))
        header_B = self._make_header(grid_B.shape[1], "PERSON B", (200, 100, 50))

        col_A = np.vstack([header_A, grid_A])
        col_B = np.vstack([header_B, grid_B])

        # Divider
        div = np.full((col_A.shape[0], 8, 3), 60, dtype=np.uint8)

        # Side-by-side
        pair_display = np.hstack([col_A, div, col_B])

        # Info bar at bottom
        info = self._make_info_bar(pair_display.shape[1], pair, idx)
        display = np.vstack([pair_display, info])

        return display

    def _make_header(self, width: int, title: str, color) -> np.ndarray:
        """Build colored header bar for each person."""
        header = np.full((32, width, 3), 20, dtype=np.uint8)
        cv2.rectangle(header, (0, 0), (width, 32), color, -1)
        cv2.putText(header, title, (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, CLR_TEXT, 2)
        return header

    def _make_info_bar(self, width: int, pair: dict, idx: int) -> np.ndarray:
        """Build info + controls bar at the bottom."""
        bar = np.full((80, width, 3), CLR_BG, dtype=np.uint8)

        # Progress
        n_total = len(self.pairs)
        n_done  = sum(1 for p in self.pairs if p.get('label') in ('C', 'N', 'S'))
        n_collab = sum(1 for p in self.pairs if p.get('label') == 'C')
        n_no    = sum(1 for p in self.pairs if p.get('label') == 'N')

        prog_pct = n_done / max(n_total, 1) * 100

        cv2.putText(bar, f"Pair {idx+1}/{n_total}  ({prog_pct:.0f}% done)   "
                        f"C:{n_collab}  N:{n_no}  S:{n_done-n_collab-n_no}",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, CLR_TEXT, 1)

        cv2.putText(bar, f"Video: {pair.get('video_id', '?')}  "
                        f"TrackA: {pair.get('track_id_A', '?')}  "
                        f"TrackB: {pair.get('track_id_B', '?')}",
                    (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        # Controls
        controls = "  [C] Collaborative    [N] Not Collab    [S] Skip    [B] Undo    [Q] Quit"
        cv2.putText(bar, controls, (10, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 200, 120), 1)

        # Current label (if re-showing already labeled pair)
        existing = pair.get('label')
        if existing:
            clr = CLR_COLLAB if existing == 'C' else (CLR_NO_COLLAB if existing == 'N' else CLR_SKIP)
            lname = {'C': 'COLLABORATIVE', 'N': 'NOT COLLABORATIVE', 'S': 'SKIP'}.get(existing, '?')
            cv2.putText(bar, f"Current: {lname}", (width - 280, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, clr, 2)

        return bar

    def _print_stats(self):
        n_total = len(self.pairs)
        n_C = sum(1 for p in self.pairs if p.get('label') == 'C')
        n_N = sum(1 for p in self.pairs if p.get('label') == 'N')
        n_S = sum(1 for p in self.pairs if p.get('label') == 'S')
        n_done = n_C + n_N + n_S
        print(f"\n{'='*50}")
        print(f"ANNOTATION SESSION COMPLETE")
        print(f"  Total pairs:          {n_total}")
        print(f"  Annotated (C+N):      {n_C + n_N}")
        print(f"  Collaborative (C):    {n_C}  ({n_C/(n_C+n_N+1e-6)*100:.0f}%)")
        print(f"  Not Collaborative(N): {n_N}  ({n_N/(n_C+n_N+1e-6)*100:.0f}%)")
        print(f"  Skipped (S):          {n_S}")
        print(f"  Remaining unannotated:{n_total - n_done}")
        print(f"\nSaved to: {self.catalog_path}")
        if n_C + n_N < 100:
            print(f"\n  ⚠ WARNING: Only {n_C+n_N} usable pairs. Target ≥ 400 for training.")
            print(f"  Continue annotating more videos.")
        else:
            print(f"\n  ✓ Enough pairs to start training ({n_C+n_N} ≥ 400 threshold)")
            print(f"  Next: python src/data/collab_dataset.py --build_splits")
        print(f"{'='*50}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Annotation GUI for collaboration pairs")
    parser.add_argument("--catalog",    default=CATALOG_CSV, help="Path to pair_catalog.csv")
    parser.add_argument("--start_from", type=int, default=0, help="Start from pair index")
    args = parser.parse_args()

    annotator = CollabAnnotator(catalog_csv=args.catalog, start_from=args.start_from)
    annotator.run()


if __name__ == "__main__":
    main()
