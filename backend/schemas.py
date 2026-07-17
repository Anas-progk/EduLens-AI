"""Pydantic schemas for EduLens FastAPI backend."""

from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from datetime import datetime


# ─── Auth ─────────────────────────────────────────────────────────────────────
class LoginRequest(BaseModel):
    email: str
    password: str

class UserOut(BaseModel):
    id: str
    name: str
    email: str
    role: Literal['teacher', 'hod', 'principal']

class LoginResponse(BaseModel):
    token: str
    user: UserOut


# ─── Session ──────────────────────────────────────────────────────────────────
class SessionStatus(BaseModel):
    id: str
    filename: str
    uploaded_at: str
    duration_sec: float
    status: Literal['queued', 'processing', 'done', 'error']
    progress: int = Field(ge=0, le=100, default=0)
    avg_engagement: Optional[float] = None
    avg_collab: Optional[float] = None
    class_health: Optional[float] = None
    collab_verdict: Optional[str] = None
    error_message: Optional[str] = None


# ─── Engagement results ────────────────────────────────────────────────────────
class StudentResult(BaseModel):
    id: str          # ST-01, ST-02...
    track_id: int
    label: Literal['Engaged', 'Not Engaged', 'Unknown']
    engagement_prob: float = Field(ge=0.0, le=1.0)
    collab_label: Literal['Collaborative', 'Not Collaborative', 'Unknown'] = 'Unknown'
    row: int = 0
    col: int = 0

class FrameResult(BaseModel):
    frame_index: int
    timestamp: float     # seconds
    students: List[StudentResult]
    class_health_score: float
    engagement_score: float
    collab_score: float

class SessionTimeline(BaseModel):
    session_id: str
    timeline: List[dict]   # {t, engagement, collab, health}
    students: List[StudentResult]
    alerts: List[dict]
    collab_verdict: str


# ─── Alerts ───────────────────────────────────────────────────────────────────
class Alert(BaseModel):
    id: str
    student_id: Optional[str] = None
    severity: Literal['soft', 'warning', 'critical']
    message: str
    timestamp: float
    resolved: bool = False


# ─── Copilot ──────────────────────────────────────────────────────────────────
class CopilotRequest(BaseModel):
    session_id: Optional[str] = None
    question: str
    history: List[dict] = []

class CopilotAction(BaseModel):
    type: Literal['seek', 'highlight_student', 'show_chart']
    payload: dict

class CopilotResponse(BaseModel):
    id: str
    role: Literal['ai'] = 'ai'
    text: str
    action: Optional[CopilotAction] = None
    timestamp: float


# ─── Analytics ────────────────────────────────────────────────────────────────
class DashboardStats(BaseModel):
    total_sessions: int
    avg_engagement: float
    avg_collab: float
    avg_health: float
    total_alerts: int
    recent_sessions: List[SessionStatus]
