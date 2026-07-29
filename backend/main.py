"""
EduLens FastAPI Backend

Run:
    cd Ai_based_engagement_system
    uvicorn backend.main:app --reload --port 8000

All AI inference runs via backend/services/engagement_service.py
which wraps the existing Swin-Tiny model in src/models/swin_clip_model.py
"""

import sys
import logging
from pathlib import Path

# Make sure project root is on path for src.* imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi import Depends
from backend.dependencies import get_current_user, require_teacher, require_hod, require_principal
# Init DB before importing routers
from backend.database import init_db
init_db()

from backend.routers import video, auth, copilot

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="EduLens API",
    description="AI Classroom Intelligence Backend — Engagement + Collaboration Analytics",
    version="1.0.0",
)

# ─── CORS (allow Next.js dev server) ──────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Routers ──────────────────────────────────────────────────────────────────
app.include_router(video.router)
app.include_router(auth.router)
app.include_router(copilot.router)

# ─── Analytics router (inline for simplicity) ─────────────────────────────────
from fastapi import APIRouter
from backend.database import list_sessions

analytics_router = APIRouter(prefix="/api/analytics", tags=["analytics"])

@analytics_router.get("/dashboard")
async def dashboard_stats(
    current_user=Depends(require_hod),
):
    sessions = list_sessions()
    done = [s for s in sessions if s.get("status") == "done"]

    return {
        "total_sessions": len(sessions),
        "avg_engagement": round(
            sum(s.get("avg_engagement") or 0 for s in done)
            / max(len(done), 1),
            1,
        ),
        "avg_collab": round(
            sum(s.get("avg_collab") or 0 for s in done)
            / max(len(done), 1),
            1,
        ),
        "avg_health": round(
            sum(s.get("class_health") or 0 for s in done)
            / max(len(done), 1),
            1,
        ),
        "total_alerts": 0,
        "recent_sessions": sessions[:5],
    }
app.include_router(analytics_router)

# ─── Alerts router (inline) ───────────────────────────────────────────────────
from backend.database import get_session as db_get_session

alerts_router = APIRouter(prefix="/api/alerts", tags=["alerts"])

@alerts_router.get("")
async def get_alerts(
    session_id: str | None = None,
    current_user=Depends(require_hod),
):
    if session_id:
        session = db_get_session(session_id)
        return session.get("alerts", []) if session else []
    # All alerts across sessions
    sessions = list_sessions()
    alerts = []
    for s in sessions:
        full = db_get_session(s["id"])
        if full:
            alerts.extend(full.get("alerts", []))
    return alerts

app.include_router(alerts_router)


# ─── Health check ─────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "service": "EduLens API", "version": "1.0.0"}


@app.get("/")
async def root():
    return {
        "message": "EduLens AI Classroom Intelligence API",
        "docs": "/docs",
        "endpoints": [
            "POST /api/sessions/upload",
            "POST /api/sessions/{id}/analyze",
            "GET  /api/sessions/{id}",
            "GET  /api/sessions/{id}/timeline",
            "GET  /api/sessions/{id}/report",
            "POST /api/copilot/ask",
            "POST /api/auth/login",
            "GET  /api/analytics/dashboard",
            "GET  /api/alerts",
        ]
    }


# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
