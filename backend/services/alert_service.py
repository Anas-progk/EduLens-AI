"""Alert rule engine for EduLens."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class AlertEngine:
    """Manages engagement-based alerts with escalation logic."""

    SOFT_THRESHOLD_MIN = 3      # Minutes before soft alert
    WARNING_THRESHOLD_MIN = 5   # Minutes before warning alert
    CRITICAL_THRESHOLD_MIN = 10 # Minutes before critical alert
    CLASS_HEALTH_CRITICAL = 40  # Classroom health score below this → escalate to HOD

    def __init__(self):
        self._tracker: dict[str, dict] = {}  # student_id → {consecutive_ne_mins, alerted}

    def process_tick(self, student_id: str, label: str, timestamp_sec: float) -> Optional[dict]:
        """
        Process one engagement tick per student.
        Returns an alert dict if threshold crossed, else None.
        """
        if student_id not in self._tracker:
            self._tracker[student_id] = {"ne_secs": 0, "alerted_severity": None}

        state = self._tracker[student_id]

        if label == "Not Engaged":
            state["ne_secs"] += 30  # tick every 30 seconds
        else:
            state["ne_secs"] = 0
            state["alerted_severity"] = None
            return None

        ne_min = state["ne_secs"] / 60

        if ne_min >= self.CRITICAL_THRESHOLD_MIN and state["alerted_severity"] != "critical":
            state["alerted_severity"] = "critical"
            return self._make_alert(student_id, "critical", ne_min, timestamp_sec)
        elif ne_min >= self.WARNING_THRESHOLD_MIN and state["alerted_severity"] not in ("critical", "warning"):
            state["alerted_severity"] = "warning"
            return self._make_alert(student_id, "warning", ne_min, timestamp_sec)
        elif ne_min >= self.SOFT_THRESHOLD_MIN and state["alerted_severity"] is None:
            state["alerted_severity"] = "soft"
            return self._make_alert(student_id, "soft", ne_min, timestamp_sec)

        return None

    def _make_alert(self, student_id: str, severity: str, ne_min: float, ts: float) -> dict:
        duration = f"{int(ne_min)}+" if severity != "soft" else f"{int(ne_min)}"
        messages = {
            "soft":     f"{student_id} showing low attention — {duration} minutes",
            "warning":  f"{student_id} disengaged for {duration} minutes",
            "critical": f"{student_id} critically disengaged for {duration} minutes — immediate action needed",
        }
        return {
            "id": f"alert_{student_id}_{severity}_{int(ts)}",
            "student_id": student_id,
            "severity": severity,
            "message": messages[severity],
            "timestamp": ts,
            "resolved": False,
        }

    def class_health_alert(self, health_score: float, timestamp_sec: float) -> Optional[dict]:
        """Generate a class-level alert if health drops critically."""
        if health_score < self.CLASS_HEALTH_CRITICAL:
            return {
                "id": f"alert_class_{int(timestamp_sec)}",
                "student_id": None,
                "severity": "critical",
                "message": f"Classroom health critical: {health_score:.0f}% — consider HOD escalation",
                "timestamp": timestamp_sec,
                "resolved": False,
            }
        return None
