"""
interaction_signals.py -- Compute 4-d interaction signals between tracked persons.

These 4 signals are the KEY addition for Phase 2 over Phase 1.
They capture INTERPERSONAL dynamics that the per-person engagement model alone cannot.

Signal vector: [proximity, facing_score, activity_correlation, turn_taking_score]

All signals are normalized to [0, 1] before being passed to CollaborationHead.

Design principle:
  These signals are NOT learned from data — they are computed from geometry and
  temporal patterns using interpretable rules. This makes the collaboration prediction
  more robust when training data is limited (~400-800 annotated pairs).
  The model learns HOW MUCH to weight each signal for the final decision.

Usage:
  computer = InteractionSignalComputer(frame_width=848, frame_height=480)

  # Per frame: update with current bounding boxes and engagement probs
  for track_id, (bbox, eng_prob) in tracked_persons.items():
      computer.update(track_id, bbox, eng_prob)

  # Compute signals for any pair
  signals = computer.get_signals(track_id_A=1, track_id_B=3)
  # signals: np.array([proximity, facing, correlation, turn_taking])  shape (4,)
"""

import numpy as np
from collections import deque
from typing import Dict, Tuple, Optional
import math


# ---------------------------------------------------------------------------
# Per-person state buffer
# ---------------------------------------------------------------------------

class PersonSignalBuffer:
    """
    Rolling buffer of bbox positions and engagement probabilities for one person.

    Used to compute temporal signals (correlation, turn-taking) that require
    a history of observations.

    Parameters:
        buffer_size: Number of recent observations to keep (16 recommended)
                     At inference_step=4 frames per inference, 16 decisions
                     covers ~64 frames = ~4-5 seconds of video at 15fps.
                     That's enough to observe 2-3 rounds of turn-taking.
    """

    def __init__(self, track_id: int, buffer_size: int = 16):
        self.track_id     = track_id
        self.buffer_size  = buffer_size
        self.bbox_history = deque(maxlen=buffer_size)   # (cx, cy, w, h) normalized
        self.eng_probs    = deque(maxlen=buffer_size)   # P(Engaged) raw

    def update(self, bbox_pixels: Tuple[int,int,int,int], eng_prob: float,
               frame_w: int, frame_h: int):
        """
        Record one observation.

        Args:
            bbox_pixels: (x, y, w, h) in pixel coordinates
            eng_prob: P(Engaged) from SwinClipModel, range [0, 1]
            frame_w, frame_h: Frame dimensions (for normalization)
        """
        x, y, w, h = bbox_pixels
        cx = (x + w / 2) / frame_w
        cy = (y + h / 2) / frame_h
        nw = w / frame_w
        nh = h / frame_h
        self.bbox_history.append((cx, cy, nw, nh))
        self.eng_probs.append(float(eng_prob))

    @property
    def center(self) -> Optional[Tuple[float, float]]:
        """Latest normalized center (cx, cy). None if no history."""
        if not self.bbox_history:
            return None
        return self.bbox_history[-1][0], self.bbox_history[-1][1]

    @property
    def size(self) -> Optional[Tuple[float, float]]:
        """Latest normalized (w, h)."""
        if not self.bbox_history:
            return None
        return self.bbox_history[-1][2], self.bbox_history[-1][3]

    def has_enough_history(self, min_len: int = 5) -> bool:
        return len(self.eng_probs) >= min_len

    def get_eng_prob_array(self) -> np.ndarray:
        return np.array(self.eng_probs, dtype=np.float32)


# ---------------------------------------------------------------------------
# InteractionSignalComputer
# ---------------------------------------------------------------------------

class InteractionSignalComputer:
    """
    Manages signal buffers for all tracked persons and computes pairwise
    interaction signals on demand.

    Signal 1: Proximity Score
      Physical closeness between two persons. Collaboration requires people to be
      near each other. This is the STRONGEST gate: if two people are far apart
      (opposite ends of the classroom), they cannot be collaborating regardless
      of other signals.

    Signal 2: Facing Score
      Whether the two persons are oriented toward each other. Estimated from the
      LEFT-RIGHT spatial arrangement of their bounding box centers.
      IMPORTANT: This is a HEURISTIC — without depth cameras or full 3D pose,
      we cannot know exact head direction. But in classroom settings, students
      facing each other generally have centers arranged consistently relative to
      their body widths.

    Signal 3: Activity Correlation
      Pearson correlation of their engagement probability timeseries.
      High positive: they respond to the same stimuli (both pay attention when
      teacher talks) — not necessarily collaborating.
      Moderate positive + high proximity: likely collaborating on shared task.
      Negative: one is always active while other is passive — unusual, check manually.

    Signal 4: Turn-Taking Score
      Anti-correlation of the CHANGES in engagement probability.
      Classic turn-taking: A's engagement rises while B's falls (A speaking),
      then B's rises while A's falls (B speaking).
      This is the STRONGEST signal of actual verbal interaction.

    Parameters:
        frame_width:  Video frame width in pixels (for normalization)
        frame_height: Video frame height in pixels
        buffer_size:  History length per person (16 recommended)
    """

    def __init__(
        self,
        frame_width  : int = 848,
        frame_height : int = 480,
        buffer_size  : int = 16,
    ):
        self.frame_w     = frame_width
        self.frame_h     = frame_height
        self.buffer_size = buffer_size
        self._frame_diag = math.sqrt(frame_width**2 + frame_height**2)

        # track_id → PersonSignalBuffer
        self._persons: Dict[int, PersonSignalBuffer] = {}

    def update(
        self,
        track_id    : int,
        bbox_pixels : Tuple[int, int, int, int],
        eng_prob    : float,
    ):
        """
        Update state for one tracked person.

        Call this every time inference is run for a person
        (i.e., every inference_step frames).

        Args:
            track_id:    Local tracker ID
            bbox_pixels: (x, y, w, h) in pixels (top-left origin)
            eng_prob:    P(Engaged) from SwinClipModel
        """
        if track_id not in self._persons:
            self._persons[track_id] = PersonSignalBuffer(
                track_id, self.buffer_size
            )
        self._persons[track_id].update(
            bbox_pixels, eng_prob, self.frame_w, self.frame_h
        )

    def remove(self, track_id: int):
        """Remove stale person from buffer (e.g., when tracker loses them)."""
        self._persons.pop(track_id, None)

    def active_ids(self):
        return list(self._persons.keys())

    def get_all_pairs(self) -> list:
        """
        Return list of all unique (id_A, id_B) pairs among active persons.
        Used to compute signals for all visible pairs per frame.
        """
        ids = list(self._persons.keys())
        pairs = []
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                pairs.append((ids[i], ids[j]))
        return pairs

    def get_signals(
        self,
        track_id_A : int,
        track_id_B : int,
    ) -> np.ndarray:
        """
        Compute the 4-d interaction signal vector for pair (A, B).

        Returns np.array([proximity, facing_score, activity_correlation, turn_taking])
        All values in [0, 1].

        Returns zero vector if either person has insufficient history.
        Collaboration head handles zero vectors gracefully (will predict low collab).
        """
        if track_id_A not in self._persons or track_id_B not in self._persons:
            return np.zeros(4, dtype=np.float32)

        buf_A = self._persons[track_id_A]
        buf_B = self._persons[track_id_B]

        signals = np.array([
            self._proximity(buf_A, buf_B),
            self._facing_score(buf_A, buf_B),
            self._activity_correlation(buf_A, buf_B),
            self._turn_taking(buf_A, buf_B),
        ], dtype=np.float32)

        # Clip to [0, 1] (all should be, but safety measure)
        signals = np.clip(signals, 0.0, 1.0)
        return signals

    # ── Signal 1: Proximity ────────────────────────────────────────────────

    def _proximity(self, buf_A: PersonSignalBuffer, buf_B: PersonSignalBuffer) -> float:
        """
        Normalized proximity: 0=far apart (opposite sides of frame), 1=overlapping.

        Formula: 1 - (euclidean_center_distance / frame_diagonal)

        Note: We normalize by frame diagonal (not width or height) because classroom
        videos can be landscape or portrait, and we want consistency.
        """
        cA = buf_A.center
        cB = buf_B.center
        if cA is None or cB is None:
            return 0.0

        # Centers are normalized (0-1 relative to frame), convert back to pixels
        dist_norm = math.sqrt((cA[0] - cB[0])**2 + (cA[1] - cB[1])**2)
        # dist_norm is in [0, sqrt(2)] for normalized coords
        # Map to [0, 1] where 0=far, 1=adjacent
        proximity = 1.0 - min(dist_norm / math.sqrt(2), 1.0)
        return float(proximity)

    # ── Signal 2: Facing Score ─────────────────────────────────────────────

    def _facing_score(self, buf_A: PersonSignalBuffer, buf_B: PersonSignalBuffer) -> float:
        """
        Estimates whether two persons are facing each other based on relative position.

        HEURISTIC REASONING:
        In a seated classroom:
        - If Person A's center is to the LEFT of Person B's center:
          → A would naturally face RIGHT to look at B
          → B would naturally face LEFT to look at A
          → Their body widths and center positions encode this consistently
        - We quantify this by how clearly their centers are separated relative
          to their combined widths (non-overlapping, distinct positions = likely facing)

        This is NOT body pose estimation. It's a spatial arrangement heuristic.
        It will fail for people sitting back-to-back or in unusual orientations.
        The model learns to weight this appropriately given training data.

        Returns:
          0.0 = centers overlap or arrangement unclear (random / non-facing)
          1.0 = clear horizontal separation (likely facing each other)
        """
        cA = buf_A.center
        cB = buf_B.center
        szA = buf_A.size
        szB = buf_B.size
        if cA is None or cB is None or szA is None or szB is None:
            return 0.0

        # Horizontal distance between centers, normalized by average person width
        avg_width = (szA[0] + szB[0]) / 2.0
        if avg_width < 1e-6:
            return 0.0

        h_dist = abs(cA[0] - cB[0])
        # If they're clearly separate (more than 1 person-width apart), facing is plausible
        # If they're overlapping or extremely far, facing is unclear
        facing = np.clip(h_dist / (avg_width * 2.0), 0.0, 1.0)

        # Also consider vertical proximity — face-to-face usually means similar height
        v_dist = abs(cA[1] - cB[1])
        avg_height = (szA[1] + szB[1]) / 2.0
        vertical_penalty = np.clip(v_dist / (avg_height * 2.0), 0.0, 1.0)

        # High facing + low vertical penalty = face-to-face
        return float(facing * (1.0 - 0.5 * vertical_penalty))

    # ── Signal 3: Activity Correlation ────────────────────────────────────

    def _activity_correlation(
        self,
        buf_A : PersonSignalBuffer,
        buf_B : PersonSignalBuffer,
        min_len : int = 5,
    ) -> float:
        """
        Pearson correlation of engagement probability timeseries.
        Mapped from [-1, +1] to [0, 1]: corr=+1 → 1.0, corr=-1 → 0.0.

        Both highly engaged at same times → likely responding to shared stimuli.
        In a collaboration context (high proximity): this suggests joint focus.
        """
        if not buf_A.has_enough_history(min_len) or not buf_B.has_enough_history(min_len):
            return 0.5   # Neutral when insufficient data

        arr_A = buf_A.get_eng_prob_array()
        arr_B = buf_B.get_eng_prob_array()

        # Truncate to same length
        min_len_actual = min(len(arr_A), len(arr_B))
        arr_A = arr_A[-min_len_actual:]
        arr_B = arr_B[-min_len_actual:]

        if np.std(arr_A) < 1e-6 or np.std(arr_B) < 1e-6:
            return 0.5   # One person has constant engagement (no info)

        corr = float(np.corrcoef(arr_A, arr_B)[0, 1])
        corr = np.clip(corr, -1.0, 1.0)

        # Map [-1, +1] → [0, 1]
        return (corr + 1.0) / 2.0

    # ── Signal 4: Turn-Taking ──────────────────────────────────────────────

    def _turn_taking(
        self,
        buf_A   : PersonSignalBuffer,
        buf_B   : PersonSignalBuffer,
        min_len : int = 6,
    ) -> float:
        """
        Turn-taking score: detects alternating activity patterns.

        Classic verbal turn-taking:
          T=0: A speaks → A engagement RISES, B engagement FALLS (B listens)
          T=1: B speaks → B engagement RISES, A engagement FALLS (A listens)

        We detect this by computing the correlation of the DELTAS (frame-to-frame
        changes) in engagement probability. Strong NEGATIVE correlation of deltas
        = they alternate = turn-taking.

        Score: 0 = no turn-taking, 1 = strong alternating pattern

        Why deltas instead of absolute values:
          Two people might both be consistently engaged (high, flat engagement)
          when listening to the teacher. That's NOT turn-taking.
          Deltas filter out the common "baseline" and focus on relative changes.
        """
        if not buf_A.has_enough_history(min_len) or not buf_B.has_enough_history(min_len):
            return 0.0

        arr_A = buf_A.get_eng_prob_array()
        arr_B = buf_B.get_eng_prob_array()

        min_len_actual = min(len(arr_A), len(arr_B))
        arr_A = arr_A[-min_len_actual:]
        arr_B = arr_B[-min_len_actual:]

        if len(arr_A) < 3:
            return 0.0

        delta_A = np.diff(arr_A)
        delta_B = np.diff(arr_B)

        if np.std(delta_A) < 1e-6 or np.std(delta_B) < 1e-6:
            return 0.0

        corr_delta = float(np.corrcoef(delta_A, delta_B)[0, 1])
        corr_delta = np.clip(corr_delta, -1.0, 1.0)

        # Turn-taking = anti-correlation of deltas
        # corr_delta = -1 → perfect alternating → score = 1.0
        # corr_delta = 0  → random                → score = 0.5
        # corr_delta = +1 → same changes           → score = 0.0
        turn_taking = (-corr_delta + 1.0) / 2.0
        return float(np.clip(turn_taking, 0.0, 1.0))

    # ── Batch signal computation ───────────────────────────────────────────

    def get_all_pair_signals(self) -> Dict[Tuple[int, int], np.ndarray]:
        """
        Compute signals for ALL active pairs in one call.

        Returns dict: {(id_A, id_B) → signals (4,)} for all unique pairs.
        Useful for passing to CollaborationHead batch inference.
        """
        pair_signals = {}
        for (id_A, id_B) in self.get_all_pairs():
            pair_signals[(id_A, id_B)] = self.get_signals(id_A, id_B)
        return pair_signals

    def describe_signals(self, signals: np.ndarray) -> str:
        """Human-readable description of a signal vector. For debugging."""
        if signals is None or len(signals) != 4:
            return "invalid signals"
        p, f, c, t = signals
        return (
            f"proximity={p:.2f} | facing={f:.2f} | "
            f"correlation={c:.2f} | turn_taking={t:.2f}"
        )


# ---------------------------------------------------------------------------
# NodDetector (optional enhancement)
# ---------------------------------------------------------------------------

class NodDetector:
    """
    Lightweight head-nod detector based on vertical bbox center movement.

    Nodding is one of the strongest non-verbal collaboration signals — it signals
    active listening and agreement. This detector estimates nodding by tracking
    the vertical position of the upper portion of a person's bounding box.

    NOT used in collaboration signals vector (4-d), but available as
    an OPTIONAL 5th signal for future work or dashboard display.

    Algorithm:
      1. Track vertical center (cy) of person bbox over time
      2. Smooth with short EMA to remove jitter
      3. Detect local minima/maxima separated by 200-800ms (natural nod rhythm)
      4. Nod frequency > 0.3/sec over 3-second window → "nodding"

    Limitations:
      - Only works when upper body is visible (not back-of-head only)
      - Sensitive to camera shake
      - Cannot distinguish nod from slouch/stretch

    Usage:
        nod = NodDetector()
        for frame_num, cy_pixel in enumerate(cy_sequence):
            nod.update(cy_pixel)
        is_nodding = nod.is_nodding()
        nod_freq   = nod.nod_frequency()
    """

    def __init__(self, fps: float = 15.0, window_sec: float = 3.0):
        self.fps        = fps
        self.window     = int(window_sec * fps)
        self._cy_hist   = deque(maxlen=self.window)
        self._smoothed  = None
        self._ema_alpha = 0.3
        self._peaks     = deque(maxlen=20)   # frame indices of detected nods
        self._frame_num = 0

    def update(self, cy_pixel: float, frame_h: float):
        """Update with current normalized vertical center."""
        cy_norm = cy_pixel / frame_h
        if self._smoothed is None:
            self._smoothed = cy_norm
        else:
            self._smoothed = self._ema_alpha * cy_norm + (1 - self._ema_alpha) * self._smoothed

        self._cy_hist.append(self._smoothed)
        self._detect_nod()
        self._frame_num += 1

    def _detect_nod(self):
        """Simple local minimum detection (downward nod = head goes lower)."""
        hist = list(self._cy_hist)
        if len(hist) < 5:
            return
        # Check if current position is a local maximum (head came down then up)
        mid = len(hist) - 3
        if mid < 1:
            return
        if hist[mid] > hist[mid - 1] and hist[mid] > hist[mid + 1]:
            # Significant dip
            amplitude = hist[mid] - min(hist[mid-1], hist[mid+1])
            if amplitude > 0.01:   # at least 1% of frame height
                self._peaks.append(self._frame_num)

    def nod_frequency(self) -> float:
        """Nods per second in recent window."""
        if not self._peaks:
            return 0.0
        recent = [p for p in self._peaks if self._frame_num - p < self.window]
        return len(recent) / (self.window / self.fps)

    def is_nodding(self, threshold: float = 0.2) -> bool:
        """True if nodding more than threshold nods/second."""
        return self.nod_frequency() >= threshold


# ---------------------------------------------------------------------------
# Sanity test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    computer = InteractionSignalComputer(frame_width=848, frame_height=480)

    # Simulate two collaborating people (close, alternating engagement)
    np.random.seed(42)
    for t in range(20):
        # Person 1: left side, higher engagement at even frames
        eng_1 = 0.8 if t % 2 == 0 else 0.4
        computer.update(track_id=1, bbox_pixels=(100, 150, 120, 200), eng_prob=eng_1)

        # Person 2: right side, higher engagement at odd frames (turn-taking!)
        eng_2 = 0.4 if t % 2 == 0 else 0.8
        computer.update(track_id=2, bbox_pixels=(320, 150, 120, 200), eng_prob=eng_2)

    signals_12 = computer.get_signals(1, 2)
    print("Collaborating pair signals:")
    print(f"  {computer.describe_signals(signals_12)}")
    print(f"  Expected: proximity>0.5, facing>0.3, turn_taking≈1.0")

    # Simulate isolated persons (far apart, uncorrelated)
    for t in range(20):
        computer.update(track_id=3, bbox_pixels=(700, 150, 80, 120),
                        eng_prob=float(np.random.rand()))
        computer.update(track_id=4, bbox_pixels=(10, 150, 80, 120),
                        eng_prob=float(np.random.rand()))

    signals_34 = computer.get_signals(3, 4)
    print("\nIsolated persons signals:")
    print(f"  {computer.describe_signals(signals_34)}")
    print(f"  Expected: proximity≈0.0 (far apart)")

    print("\nAll pairs:", computer.get_all_pairs())
