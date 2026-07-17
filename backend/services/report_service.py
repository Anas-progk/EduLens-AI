"""PDF report generation using fpdf2."""

import os
from pathlib import Path
from datetime import datetime

REPORT_DIR = Path(__file__).parent.parent / "reports"
REPORT_DIR.mkdir(exist_ok=True)


def generate_pdf_report(session: dict) -> str:
    """Generate a PDF report for a completed session. Returns path to PDF."""
    try:
        from fpdf import FPDF
    except ImportError:
        raise RuntimeError("fpdf2 not installed — run: pip install fpdf2")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Colors
    NAVY = (6, 12, 26)
    BLUE = (79, 127, 255)
    GREEN = (34, 211, 166)
    RED = (255, 78, 78)
    GRAY = (148, 163, 184)

    # Header
    pdf.set_fill_color(*NAVY)
    pdf.rect(0, 0, 210, 30, 'F')
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_xy(10, 8)
    pdf.cell(0, 8, "EduLens — AI Classroom Analytics Report", ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_xy(10, 18)
    pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  Session: {session['id']}", ln=True)

    pdf.set_y(38)
    pdf.set_text_color(30, 30, 50)

    # Summary box
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Session Summary", ln=True)
    pdf.set_font("Helvetica", "", 10)

    metrics = [
        ("File", session.get("filename", "N/A")),
        ("Average Engagement", f"{session.get('avg_engagement', 0):.1f}%"),
        ("Average Collaboration", f"{session.get('avg_collab', 0):.1f}%"),
        ("Class Health Score", f"{session.get('class_health', 0):.1f}%"),
        ("Collaboration Verdict", session.get("collab_verdict", "UNKNOWN")),
        ("Students Analyzed", str(len(session.get("students", [])))),
        ("Alerts Triggered", str(len(session.get("alerts", [])))),
    ]

    for label, val in metrics:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(70, 7, label + ":", border=0)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 7, val, ln=True)

    # Student table
    pdf.ln(5)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Student Engagement Summary", ln=True)

    pdf.set_fill_color(220, 230, 245)
    pdf.set_font("Helvetica", "B", 9)
    for col, width in [("Student ID", 30), ("Status", 45), ("Engagement %", 45), ("Collaboration", 60)]:
        pdf.cell(width, 7, col, border=1, fill=True)
    pdf.ln()

    pdf.set_font("Helvetica", "", 9)
    for s in session.get("students", []):
        pdf.cell(30, 6, s.get("id", ""), border=1)
        pdf.cell(45, 6, s.get("label", ""), border=1)
        pdf.cell(45, 6, f"{s.get('engagement_prob', 0)*100:.0f}%", border=1)
        pdf.cell(60, 6, s.get("collab_label", ""), border=1)
        pdf.ln()

    # Alerts table
    if session.get("alerts"):
        pdf.ln(5)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Alerts Log", ln=True)
        pdf.set_font("Helvetica", "", 9)
        for a in session["alerts"]:
            sev = a.get("severity", "").upper()
            msg = a.get("message", "")
            t = a.get("timestamp", 0)
            m, s_sec = divmod(int(t), 60)
            pdf.cell(0, 6, f"[{sev}] @{m}:{s_sec:02d} — {msg}", ln=True)

    # Recommendations
    pdf.ln(5)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Recommendations", ln=True)
    pdf.set_font("Helvetica", "", 10)
    eng = session.get("avg_engagement", 75)
    recs = [
        "Introduce discussion activities when engagement drops below 60%",
        "Use collaborative pair exercises to boost collaboration scores",
        "Monitor high-risk students (low engagement probability) proactively",
        "Schedule breaks after 40+ minutes of sustained instruction",
    ] if eng < 75 else [
        "Maintain current teaching pace — engagement is strong",
        "Try quiz-based reinforcement to test retention",
        "Small-group work could further improve collaboration",
    ]
    for r in recs:
        pdf.cell(0, 7, f"  • {r}", ln=True)

    # Footer
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(*GRAY)
    pdf.set_y(-15)
    pdf.cell(0, 5, "EduLens AI Classroom Intelligence Platform · Swin-Tiny Transformer · RTRP 2026", align="C", ln=True)

    # Save
    out_path = REPORT_DIR / f"report_{session['id']}.pdf"
    pdf.output(str(out_path))
    return str(out_path)
