# EduLens — AI Classroom Intelligence Platform

## Quick Start (2 terminals)

### Terminal 1 — Backend (FastAPI)
```bash
cd Ai_based_engagement_system

# Install backend deps (if not done)
pip install fastapi uvicorn[standard] python-multipart pydantic fpdf2

# Start backend on port 8000
python -m uvicorn backend.main:app --reload --port 8000
```

Or on Windows, just double-click **`start_backend.bat`**

Backend will be live at: http://localhost:8000  
API docs (Swagger UI): http://localhost:8000/docs
    
---

### Terminal 2 — Frontend (Next.js)
```bash
cd Ai_based_engagement_system/frontend

# Install Node deps (first time only — takes ~1 minute)
npm install

# Start dev server on port 3000
npm run dev
```

Or on Windows, just double-click **`start_frontend.bat`**

App will be live at: **http://localhost:3000**

---

## Demo Login Credentials

| Role      | Email                    | Password | Access |
|-----------|--------------------------|----------|--------|
| Teacher   | teacher@edulens.ai       | demo123  | Full classroom |
| HOD       | hod@edulens.ai           | demo123  | Dept summaries + alerts |
| Principal | principal@edulens.ai     | demo123  | Institution analytics |

---

## Project Structure

```
Ai_based_engagement_system/
├── frontend/                    ← Next.js 14 + TailwindCSS + Framer Motion
│   ├── app/
│   │   ├── page.tsx             ← Landing page (animated, hero + features)
│   │   ├── monitor/page.tsx     ← Live Monitor (video upload + classroom map)
│   │   ├── dashboard/page.tsx   ← Analytics Dashboard (charts + AI Copilot)
│   │   └── login/page.tsx       ← Role-based auth
│   ├── components/
│   │   ├── CopilotPanel.tsx     ← ⭐ AI Classroom Copilot (standout feature)
│   │   ├── ClassroomMap.tsx     ← Digital classroom twin grid
│   │   ├── HealthPulse.tsx      ← Animated health gauge
│   │   ├── EngagementChart.tsx  ← Recharts timeline
│   │   ├── CollabGraph.tsx      ← D3 interaction network
│   │   ├── TimelineReplay.tsx   ← Clickable session replay timeline
│   │   └── AlertPanel.tsx       ← Tiered alert system
│   └── lib/
│       ├── api.ts               ← Backend API calls + demo data
│       ├── types.ts             ← TypeScript types
│       └── hooks.ts             ← Shared React hooks
│
├── backend/                     ← FastAPI + SQLite
│   ├── main.py                  ← App entry, all routes registered
│   ├── database.py              ← SQLite setup, CRUD operations
│   ├── schemas.py               ← Pydantic request/response models
│   ├── routers/
│   │   ├── video.py             ← Upload + analyze + timeline endpoints
│   │   ├── auth.py              ← Login / logout
│   │   └── copilot.py          ← AI Copilot rule engine
│   └── services/
│       ├── engagement_service.py ← SwinClipModel inference wrapper
│       ├── alert_service.py      ← 3-tier alert engine
│       └── report_service.py    ← PDF report generator (fpdf2)
│
├── src/                         ← Existing ML code (untouched)
│   ├── models/swin_clip_model.py
│   ├── inference/multi_person_inference.py
│   └── inference/group_collab.py
│
└── weights/
    ├── best_clip_model.pth           ← Engagement model
    └── best_collab_group_fresh.npz  ← Collaboration model
```

---

## Features Implemented

### ⭐ Standout Feature: AI Classroom Copilot
- Contextual Q&A about live analytics
- Questions like: *"Why is the class red?"*, *"Show most disengaged period"*
- One-click timeline replay from AI responses
- Browser voice synthesis (Text-to-Speech narrator)

### Live Monitor Page
- Drag-and-drop video upload
- YOLO + Swin-Tiny inference pipeline (or demo mode if no GPU)
- Real-time bounding box overlay on video
- **Digital Classroom Twin** — animated grid showing every student's state
- **Animated Health Pulse** gauge (green → amber → red)
- Tiered alert panel (Soft / Warning / Critical)

### Analytics Dashboard
- Engagement + Collaboration timeline (Recharts AreaChart)
- Per-student engagement table with probability bars
- **Interaction Network Graph** (D3 force-directed, draggable nodes)
- **Timeline Replay** — clickable markers jump to incident moments
- AI Copilot panel (contextual, analytics-aware)
- Voice AI Narrator (summarizes session aloud)
- One-click PDF report generation

### Alert System
- **Soft (3 min)** → Teacher notification
- **Warning (5 min)** → Escalated alert
- **Critical (10 min)** → HOD escalation
- Class-level health alert when score < 40%

### Auth + Privacy
- Role-based login (Teacher / HOD / Principal)
- Audit logging to SQLite
- Privacy badge: anonymous IDs, no facial recognition, 24h video deletion

---

## Model Integration Details

**Engagement (Phase 1)**
- Model: `SwinClipModel` (Swin-Tiny + TemporalTransformer)
- Weights: `weights/best_clip_model.pth`
- Load fix: `torch.load(..., map_location='cpu', weights_only=False)`
- Honest test metric: macro-F1 **0.73** on 5 held-out classrooms

**Collaboration (Phase 2)**
- Model: `GroupCollabHead` (logistic head on 20-d group aggregate)
- Weights: `weights/best_collab_group_fresh.npz`
- Session-level LOVO macro-F1 **0.667** → **0.764** with gaze signals

**CPU Demo Mode**
- If weights not found or GPU unavailable → realistic demo data
- All UI features work identically in demo mode

---

## Deployment

| Component | Platform | Command |
|-----------|----------|---------|
| Frontend  | Vercel   | `vercel --prod` from `frontend/` |
| Backend   | Render   | Connect GitHub repo, set start command: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT` |
| Database  | Supabase | Swap SQLite for PostgreSQL by updating `database.py` connection string |

---

## Sanity Checklist for Demo

- [ ] `npm run build` completes without errors  
- [ ] Backend `/health` returns `{"status": "ok"}`
- [ ] Upload a sample video → progress bar fills → metrics appear
- [ ] Copilot answers "Why is the class showing red?"
- [ ] Timeline markers are clickable
- [ ] Alert panel shows tiered alerts
- [ ] Voice narrator speaks on "Narrate Session" click
- [ ] PDF generates on "Export PDF" click (requires backend)
- [ ] Login works for all 3 demo roles
