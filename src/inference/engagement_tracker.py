"""
engagement_tracker.py -- Per-person temporal engagement state manager.

One PersonEngagementTracker instance per tracked student.
Manages:
  - Rolling frame buffer (last N frames for clip inference)
  - Engagement decision history over a sliding time window
  - EMA-smoothed probability for stable display
  - Final engagement label: Engaged / Not Engaged / Insufficient Data

Key design principle: engagement is NOT instantaneous.
A student who glances away for 2 seconds is still engaged.
Only SUSTAINED inattention (>= ne_threshold of recent decisions)
triggers a "Not Engaged" label.

Example: 30-second window, 1 FPS, ne_threshold=0.60
  -> "Not Engaged" only if the model predicted NE for >= 18 of last 30 seconds
  -> Short distractions (phone glance, stretch, side-look) do NOT trigger alert
"""

from collections import deque
from typing import Optional, Tuple
import time
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Label constants
# ─────────────────────────────────────────────────────────────────────────────
LABEL_ENGAGED     = "Engaged"
LABEL_NOT_ENGAGED = "Not Engaged"
LABEL_UNKNOWN     = "Unknown"          # not enough data yet

# Display colours (BGR for OpenCV)
COLOR_ENGAGED     = (50, 200, 50)      # green
COLOR_NOT_ENGAGED = (50, 50, 220)      # red
COLOR_UNKNOWN     = (150, 150, 150)    # grey


class PersonEngagementTracker:
    """
    Tracks engagement state for ONE person across time.

    Parameters
    ----------
    person_id      : int     Unique tracking ID assigned by the tracker.
    clip_len       : int     Number of consecutive frames needed for one inference (8).
    inference_step : int     How often to run inference (every N new frames).
                             inference_step=1 -> every new frame triggers inference.
                             inference_step=4 -> inference every 4th frame (faster).
    window_seconds : int     Rolling window length in seconds for NE ratio.
    fps            : float   Expected video FPS (for window_seconds -> frame count).
    ne_threshold   : float   Fraction of window that must be NE to label Not Engaged.
                             0.50 -> if >50% of recent decisions were NE → alert.
    ema_alpha      : float   EMA smoothing weight for probability display.
                             Higher = faster response; lower = smoother.
    min_decisions  : int     Minimum number of inference decisions before label shown.
    """

    def __init__(
        self,
        person_id      : int,
        clip_len       : int   = 8,
        inference_step : int   = 4,
        window_seconds : int   = 30,
        fps            : float = 15.0,
        ne_threshold   : float = 0.55,
        ema_alpha      : float = 0.25,
        min_decisions  : int   = 3,
    ):
        self.person_id      = person_id
        self.clip_len       = clip_len
        self.inference_step = inference_step
        self.ne_threshold   = ne_threshold
        self.ema_alpha      = ema_alpha
        self.min_decisions  = min_decisions

        # Rolling frame buffer for clip inference
        self.frame_buffer   = deque(maxlen=clip_len)
        self.frames_since_inference = 0

        # Decision history over the rolling window
        window_frames       = int(window_seconds * fps)
        self.decision_history = deque(maxlen=window_frames)  # 0=NE, 1=E

        # Smoothed probability: P(Engaged)
        self.smoothed_prob  = 0.5
        self._has_first_inference = False

        # Timestamps for stale-tracker pruning
        self.last_seen_time = time.time()
        self.created_time   = time.time()

        # Latest raw inference result
        self.last_engaged_prob = 0.5
        self.last_inference_label = LABEL_UNKNOWN

    # ─────────────────────────────────────────────────────────────────────────
    # Frame ingestion
    # ─────────────────────────────────────────────────────────────────────────

    def add_frame(self, frame_crop) -> bool:
        """
        Add a new cropped frame (PIL Image or numpy array) for this person.

        Returns True if the buffer is ready for inference (i.e. enough frames
        accumulated AND inference_step condition met).
        """
        self.frame_buffer.append(frame_crop)
        self.last_seen_time = time.time()
        self.frames_since_inference += 1

        ready_for_inference = (
            len(self.frame_buffer) == self.clip_len
            and self.frames_since_inference >= self.inference_step
        )
        if ready_for_inference:
            self.frames_since_inference = 0
        return ready_for_inference

    def get_clip_frames(self):
        """Return current frame buffer as a list (copy)."""
        return list(self.frame_buffer)

    # ─────────────────────────────────────────────────────────────────────────
    # Inference result update
    # ─────────────────────────────────────────────────────────────────────────

    def update_engagement(self, engaged_prob: float):
        """
        Record a new inference result.

        Args:
            engaged_prob: P(Engaged) from the clip model (0.0 to 1.0).
        """
        self.last_engaged_prob = engaged_prob

        # EMA smoothing for stable display
        if not self._has_first_inference:
            self.smoothed_prob = engaged_prob
            self._has_first_inference = True
        else:
            self.smoothed_prob = (self.ema_alpha * engaged_prob
                                  + (1.0 - self.ema_alpha) * self.smoothed_prob)

        # Hard label from smoothed prob (using threshold=0.5 on smoothed)
        hard_label = 1 if self.smoothed_prob >= 0.5 else 0
        self.decision_history.append(hard_label)
        self.last_inference_label = (
            LABEL_ENGAGED if hard_label == 1 else LABEL_NOT_ENGAGED
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Engagement label (final decision)
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def current_label(self) -> str:
        """
        Final engagement decision based on sustained NE ratio in rolling window.

        Returns LABEL_UNKNOWN until min_decisions have been collected.
        Returns LABEL_NOT_ENGAGED if NE ratio >= ne_threshold in recent history.
        Returns LABEL_ENGAGED otherwise.
        """
        if len(self.decision_history) < self.min_decisions:
            return LABEL_UNKNOWN

        ne_ratio = self._ne_ratio()
        if ne_ratio >= self.ne_threshold:
            return LABEL_NOT_ENGAGED
        return LABEL_ENGAGED

    @property
    def display_color(self) -> Tuple[int, int, int]:
        """BGR color for OpenCV overlay."""
        label = self.current_label
        if label == LABEL_ENGAGED:
            return COLOR_ENGAGED
        elif label == LABEL_NOT_ENGAGED:
            return COLOR_NOT_ENGAGED
        return COLOR_UNKNOWN

    @property
    def confidence_display(self) -> str:
        """Human-readable engagement probability string."""
        return f"{self.smoothed_prob * 100:.0f}%"

    def _ne_ratio(self) -> float:
        """Fraction of decisions in window that were Not Engaged."""
        if not self.decision_history:
            return 0.0
        arr = np.array(self.decision_history)
        return 1.0 - float(arr.mean())   # mean=1 means all Engaged

    @property
    def ne_ratio(self) -> float:
        return self._ne_ratio()

    # ─────────────────────────────────────────────────────────────────────────
    # Display summary
    # ─────────────────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        """Dict suitable for logging or API output."""
        return {
            "person_id"    : self.person_id,
            "label"        : self.current_label,
            "engaged_prob" : round(self.smoothed_prob, 4),
            "ne_ratio"     : round(self._ne_ratio(), 4),
            "n_decisions"  : len(self.decision_history),
            "buffer_fill"  : len(self.frame_buffer),
        }

    def is_stale(self, stale_seconds: float = 2.0) -> bool:
        """True if this person hasn't been seen for stale_seconds."""
        return (time.time() - self.last_seen_time) > stale_seconds


class EngagementTrackerPool:
    """
    Manages a pool of PersonEngagementTrackers keyed by track ID.

    Automatically creates new trackers for new IDs and prunes stale ones.
    """

    def __init__(
        self,
        clip_len       : int   = 8,
        inference_step : int   = 4,
        window_seconds : int   = 30,
        fps            : float = 15.0,
        ne_threshold   : float = 0.55,
        ema_alpha      : float = 0.25,
        stale_seconds  : float = 3.0,
    ):
        self.tracker_kwargs = dict(
            clip_len       = clip_len,
            inference_step = inference_step,
            window_seconds = window_seconds,
            fps            = fps,
            ne_threshold   = ne_threshold,
            ema_alpha      = ema_alpha,
        )
        self.stale_seconds = stale_seconds
        self._pool = {}    # track_id -> PersonEngagementTracker

    def get_or_create(self, track_id: int) -> PersonEngagementTracker:
        """Return existing tracker or create a new one for this ID."""
        if track_id not in self._pool:
            self._pool[track_id] = PersonEngagementTracker(
                person_id=track_id, **self.tracker_kwargs
            )
        return self._pool[track_id]

    def prune_stale(self):
        """Remove trackers for persons no longer visible."""
        stale = [tid for tid, t in self._pool.items()
                 if t.is_stale(self.stale_seconds)]
        for tid in stale:
            del self._pool[tid]

    def active_trackers(self):
        """Return list of currently active trackers."""
        return list(self._pool.values())

    def class_summary(self) -> dict:
        """Aggregate counts across all tracked persons."""
        engaged    = sum(1 for t in self._pool.values()
                         if t.current_label == LABEL_ENGAGED)
        not_engaged = sum(1 for t in self._pool.values()
                          if t.current_label == LABEL_NOT_ENGAGED)
        unknown    = sum(1 for t in self._pool.values()
                         if t.current_label == LABEL_UNKNOWN)
        total = len(self._pool)
        return {
            "total"      : total,
            "engaged"    : engaged,
            "not_engaged": not_engaged,
            "unknown"    : unknown,
            "pct_engaged": round(engaged / max(total, 1) * 100, 1),
        }

    def __len__(self):
        return len(self._pool)
