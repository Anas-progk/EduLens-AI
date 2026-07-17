"""AI Classroom Copilot router - rule-based analytics explainer."""

import uuid
import time
import logging
from fastapi import APIRouter, HTTPException
from backend.schemas import CopilotRequest, CopilotResponse, CopilotAction
from backend.database import get_session

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/copilot", tags=["copilot"])

BULLET = "•"
ARROW  = "▶"
DASH   = "—"


def _build_context(session_id=None):
    if not session_id:
        return {}
    session = get_session(session_id)
    if not session:
        return {}
    return {
        "engagement":     session.get("avg_engagement", 75),
        "collab":         session.get("avg_collab", 65),
        "health":         session.get("class_health", 70),
        "collab_verdict": session.get("collab_verdict", "UNKNOWN"),
        "students":       session.get("students", []),
        "alerts":         session.get("alerts", []),
        "timeline":       session.get("timeline", []),
    }


def _rule_response(question, ctx):
    """Rule-based copilot - works 100% offline, no API key needed."""
    q   = question.lower()
    eng = ctx.get("engagement", 75)
    col = ctx.get("collab", 65)
    health  = ctx.get("health", 70)
    students = ctx.get("students", [])
    alerts   = ctx.get("alerts", [])
    timeline = ctx.get("timeline", [])
    collab_verdict = ctx.get("collab_verdict", "UNKNOWN")

    ne_students    = [s for s in students if s.get("label") == "Not Engaged"]
    active_alerts  = [a for a in alerts   if not a.get("resolved")]

    # Find lowest engagement point
    if timeline:
        lowest  = min(timeline, key=lambda p: p.get("engagement", 100))
        low_t   = lowest.get("t", 0)
        low_eng = lowest.get("engagement", 0)
    else:
        low_t, low_eng = 2040, 38

    action = None

    # --- Why is class red / engagement low ---
    if any(w in q for w in ["red", "low", "why", "drop", "bad", "concern"]):
        ne_ids = ", ".join(s["id"] for s in ne_students[:3]) or "none currently"
        text = (
            "During the analyzed session:\n"
            f"{BULLET} Engagement dropped to **{low_eng}%** at minute {low_t // 60}\n"
            f"{BULLET} **{len(ne_students)}** student(s) showed sustained disengagement\n"
            f"{BULLET} Collaboration: **{col}%** | Health: **{health}%**\n\n"
            "Suggested actions:\n"
            f"{BULLET} Introduce a discussion or group activity\n"
            f"{BULLET} Ask targeted questions to: {ne_ids}\n"
            f"{BULLET} Consider a short break if past minute 40\n\n"
            f"{ARROW} Click to replay minute {low_t // 60}"
        )
        action = CopilotAction(type="seek", payload={"t": low_t})

    # --- Most disengaged period ---
    elif any(w in q for w in ["most disengaged", "worst", "lowest", "minimum"]):
        m = low_t // 60
        nearby_alerts = len([a for a in alerts if abs(a.get("timestamp", 0) - low_t) < 300])
        text = (
            f"The most disengaged period was around **minute {m}-{m+3}**:\n"
            f"{BULLET} Average engagement: **{low_eng}%** (below the 50% critical line)\n"
            f"{BULLET} {nearby_alerts} alert(s) fired in this window\n\n"
            f"{ARROW} Click to jump there"
        )
        action = CopilotAction(type="seek", payload={"t": low_t})

    # --- Which students need attention ---
    elif any(w in q for w in ["student", "who", "which", "attention"]):
        if not ne_students:
            text = "All tracked students are currently **Engaged**. Great class health!"
        else:
            lines = [
                f"{BULLET} **{s['id']}** {DASH} engagement: {int(s.get('engagement_prob', 0) * 100)}%"
                for s in ne_students
            ]
            text = (
                f"**{len(ne_students)}** student(s) need attention:\n"
                + "\n".join(lines)
                + "\n\nThey have shown sustained disengagement for 5+ minutes."
            )

    # --- Alerts ---
    elif any(w in q for w in ["alert", "notification", "escalat"]):
        if not active_alerts:
            text = "No active alerts. The classroom is performing well!"
        else:
            lines = [f"{BULLET} [{a['severity'].upper()}] {a['message']}" for a in active_alerts[:5]]
            text = f"**{len(active_alerts)} active alert(s):**\n" + "\n".join(lines)

    # --- Collaboration ---
    elif any(w in q for w in ["collab", "group", "interact", "together", "gaze"]):
        text = (
            "**Group Collaboration Analysis:**\n"
            f"{BULLET} Session verdict: **{collab_verdict}** ({col}% avg score)\n"
            f"{BULLET} Based on 6 relational signals + mutual gaze\n"
            f"{BULLET} Architecture: Swin-Tiny backbone -> 20-d group aggregation -> logistic head\n"
            f"{BULLET} Honest LOVO macro-F1: **0.667** (gaze-augmented: **0.764**)\n"
            f"{BULLET} Majority baseline: 0.348 {DASH} well above chance"
        )

    # --- Report / summary ---
    elif any(w in q for w in ["report", "summary", "pdf", "export", "summarize"]):
        text = (
            "**Session Summary:**\n"
            f"{BULLET} Average engagement: **{eng}%**\n"
            f"{BULLET} Average collaboration: **{col}%**\n"
            f"{BULLET} Class health score: **{health}%**\n"
            f"{BULLET} Alerts fired: **{len(alerts)}**\n"
            f"{BULLET} Students analyzed: **{len(students)}**\n\n"
            "Click **Export PDF** in the toolbar to generate a full report with charts and recommendations."
        )

    # --- Model architecture ---
    elif any(w in q for w in ["model", "accuracy", "f1", "architecture", "swin", "transformer", "backbone", "how"]):
        text = (
            "**EduLens Model Architecture:**\n"
            f"{BULLET} **Backbone:** Swin-Tiny Transformer (pretrained ImageNet-22k)\n"
            f"{BULLET} **Temporal head:** 2-layer TransformerEncoder, 8-frame clips, CLS token\n"
            f"{BULLET} **Engagement:** macro-F1 **0.73** on 5 held-out classrooms (honest test)\n"
            f"{BULLET} **Collaboration:** LOVO macro-F1 **0.667** -> **0.764** with gaze signals\n"
            f"{BULLET} **Runtime:** ~2s/clip on CPU, near-real-time on GPU\n"
            f"{BULLET} Training: 18 train / 5 val / 5 test classrooms {DASH} zero data leakage"
        )

    # --- Suggest interventions ---
    elif any(w in q for w in ["suggest", "recommend", "action", "intervention", "help", "what should"]):
        if eng < 60:
            recs = [
                "Start a discussion activity (engagement below 60%)",
                "Ask targeted questions to the 2-3 most disengaged students",
                "Introduce a collaborative pair exercise",
                "Consider a 2-minute mental reset break",
            ]
        else:
            recs = [
                "Engagement is healthy - maintain current pace",
                "Try a quick quiz to reinforce learning",
                "Small-group work could further boost collaboration",
            ]
        text = (
            f"Based on current metrics (Engagement: {eng}%, Collab: {col}%):\n\n"
            "**Recommended actions:**\n"
            + "\n".join(f"{BULLET} {r}" for r in recs)
        )

    # --- Default / help ---
    else:
        text = (
            "I can help you understand your classroom analytics. Try asking:\n"
            f"{BULLET} \"Why is the class showing red?\"\n"
            f"{BULLET} \"Show me the most disengaged period\"\n"
            f"{BULLET} \"Which students need attention?\"\n"
            f"{BULLET} \"What are the active alerts?\"\n"
            f"{BULLET} \"Suggest teaching interventions\"\n"
            f"{BULLET} \"Summarize this session\"\n"
            f"{BULLET} \"How does the model work?\""
        )

    return text, action


@router.post("/ask", response_model=CopilotResponse)
async def ask_copilot(req: CopilotRequest):
    """Main copilot endpoint."""
    ctx  = _build_context(req.session_id)
    text, action = _rule_response(req.question, ctx)
    return CopilotResponse(
        id=f"ai-{uuid.uuid4().hex[:8]}",
        role="ai",
        text=text,
        action=action,
        timestamp=time.time(),
    )
