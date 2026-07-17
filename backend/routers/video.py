"""Video upload and analysis endpoints."""

import uuid
import json
import logging
import threading
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse

from backend.schemas import SessionStatus, SessionTimeline
from backend.database import create_session, get_session, update_session, list_sessions, get_session_frames

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/sessions", tags=["sessions"])

UPLOAD_DIR = Path(__file__).parent.parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


def _run_analysis(session_id: str, video_path: str):
    """Background thread: run engagement inference and update DB."""
    try:
        from backend.services.engagement_service import EngagementService
        service = EngagementService()

        def on_progress(pct: int):
            update_session(session_id, status="processing", progress=pct)

        update_session(session_id, status="processing", progress=5)
        result = service.analyze_video(video_path, session_id, progress_callback=on_progress)

        update_session(
            session_id,
            status="done",
            progress=100,
            avg_engagement=result["avg_engagement"],
            avg_collab=result["avg_collab"],
            class_health=result["class_health"],
            collab_verdict=result["collab_verdict"],
            timeline_json=json.dumps(result["timeline"]),
            students_json=json.dumps(result["students"]),
            alerts_json=json.dumps(result["alerts"]),
            frames_json=json.dumps(result.get("frames", [])),
        )
        logger.info(f"Session {session_id} analysis complete")
    except Exception as e:
        logger.error(f"Analysis failed for {session_id}: {e}", exc_info=True)
        update_session(session_id, status="error", error_message=str(e))


@router.post("/upload")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload a classroom video and create a new session."""
    if not file.content_type or not file.content_type.startswith("video/"):
        # Also accept if content type detection fails
        if not (file.filename or "").lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            raise HTTPException(400, "Please upload a video file (MP4, AVI, MOV)")

    session_id = f"sess_{uuid.uuid4().hex[:12]}"
    save_path = UPLOAD_DIR / f"{session_id}_{file.filename}"

    # Save file
    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    # Create DB record
    create_session(session_id=session_id, filename=file.filename or "video.mp4")
    update_session(session_id, file_path=str(save_path), status="queued")

    logger.info(f"Video uploaded: {file.filename} → session {session_id}")
    return {"sessionId": session_id, "filename": file.filename}


@router.post("/{session_id}/analyze")
async def start_analysis(session_id: str):
    """Start background analysis for an uploaded session."""
    session = get_session(session_id)
    if not session:
        raise HTTPException(404, f"Session {session_id} not found")
    if session["status"] == "processing":
        return {"message": "Already processing"}

    video_path = session.get("file_path", "")
    thread = threading.Thread(target=_run_analysis, args=(session_id, video_path), daemon=True)
    thread.start()
    update_session(session_id, status="processing", progress=1)
    return {"message": "Analysis started", "sessionId": session_id}


@router.get("/{session_id}", response_model=SessionStatus)
async def get_session_status(session_id: str):
    """Get current status and progress of a session."""
    session = get_session(session_id)
    if not session:
        raise HTTPException(404, f"Session {session_id} not found")
    return {
        "id": session["id"],
        "filename": session["filename"],
        "uploaded_at": session["uploaded_at"],
        "duration_sec": session.get("duration_sec", 0),
        "status": session["status"],
        "progress": session.get("progress", 0),
        "avg_engagement": session.get("avg_engagement"),
        "avg_collab": session.get("avg_collab"),
        "class_health": session.get("class_health"),
        "collab_verdict": session.get("collab_verdict"),
        "error_message": session.get("error_message"),
    }


@router.get("", response_model=list)
async def get_sessions():
    """List all sessions, most recent first."""
    sessions = list_sessions()
    return [
        {
            "id": s["id"],
            "filename": s["filename"],
            "uploaded_at": s["uploaded_at"],
            "duration_sec": s.get("duration_sec", 0),
            "status": s["status"],
            "progress": s.get("progress", 0),
            "avg_engagement": s.get("avg_engagement"),
            "avg_collab": s.get("avg_collab"),
            "class_health": s.get("class_health"),
        }
        for s in sessions
    ]


@router.get("/{session_id}/timeline")
async def get_timeline(session_id: str):
    """Get full session timeline, students, and alerts."""
    session = get_session(session_id)
    if not session:
        raise HTTPException(404, f"Session {session_id} not found")
    if session["status"] != "done":
        raise HTTPException(400, f"Session not yet complete (status: {session['status']})")
    return {
        "session_id": session_id,
        "timeline": session.get("timeline", []),
        "students": session.get("students", []),
        "alerts": session.get("alerts", []),
        "collab_verdict": session.get("collab_verdict", "UNKNOWN"),
    }


@router.get("/{session_id}/frames")
async def get_frames(session_id: str):
    """
    Return per-frame detection data for bbox canvas overlay.
    Format: [{t: float, detections: [{track_id, bbox:[x1,y1,x2,y2], label, prob}]}]
    """
    session = get_session(session_id)
    if not session:
        raise HTTPException(404, f"Session {session_id} not found")
    if session["status"] != "done":
        return []
    return get_session_frames(session_id)


@router.get("/{session_id}/report")
async def get_report(session_id: str):
    """Generate and return a PDF session report."""
    session = get_session(session_id)
    if not session:
        raise HTTPException(404, f"Session {session_id} not found")

    try:
        from backend.services.report_service import generate_pdf_report
        pdf_path = generate_pdf_report(session)
        return FileResponse(pdf_path, media_type="application/pdf", filename=f"edulens_report_{session_id}.pdf")
    except Exception as e:
        raise HTTPException(500, f"Report generation failed: {e}")
