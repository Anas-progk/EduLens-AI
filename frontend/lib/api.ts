import axios from 'axios';
import type { Session, CopilotMessage, StudentState, TimelinePoint, AlertEvent, AuthUser, LoginResponse } from './types';

const BASE = "https://edulens-ai-backend.onrender.com/api";
console.log("NEXT_PUBLIC_API_URL =", process.env.NEXT_PUBLIC_API_URL);
console.log("BASE =", BASE);
const LS_ACCESS = 'edulens_access_token';
const LS_REFRESH = 'edulens_refresh_token';
const LS_USER = 'edulens_user';

const client = axios.create({ baseURL: BASE, timeout: 120000 });

client.interceptors.request.use((config) => {
  if (typeof window !== 'undefined') {
    const token = localStorage.getItem(LS_ACCESS);
    if (token) config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

let isRefreshing = false;
let failedQueue: Array<{ resolve: (token: string) => void; reject: (err: unknown) => void }> = [];

function processQueue(error: unknown, token: string | null = null) {
  failedQueue.forEach((prom) => {
    if (error) prom.reject(error);
    else if (token) prom.resolve(token);
  });
  failedQueue = [];
}

client.interceptors.response.use(
  (response) => response,
  async (error) => {
    const originalRequest = error.config;

    if (error.response?.status === 401 && !originalRequest._retry) {
      if (isRefreshing) {
        return new Promise<string>((resolve, reject) => {
          failedQueue.push({ resolve, reject });
        }).then((token) => {
          originalRequest.headers.Authorization = `Bearer ${token}`;
          return client(originalRequest);
        });
      }

      originalRequest._retry = true;
      isRefreshing = true;

      try {
        const refreshToken = localStorage.getItem(LS_REFRESH);
        if (!refreshToken) throw new Error('No refresh token');

        const { data } = await axios.post(`${BASE}/auth/refresh`, { refresh_token: refreshToken });
        const { access_token, refresh_token, user } = data as LoginResponse;

        localStorage.setItem(LS_ACCESS, access_token);
        localStorage.setItem(LS_REFRESH, refresh_token);
        localStorage.setItem(LS_USER, JSON.stringify(user));

        processQueue(null, access_token);

        originalRequest.headers.Authorization = `Bearer ${access_token}`;
        return client(originalRequest);
      } catch (err) {
        processQueue(err, null);
        localStorage.removeItem(LS_ACCESS);
        localStorage.removeItem(LS_REFRESH);
        localStorage.removeItem(LS_USER);
        if (typeof window !== 'undefined') {
          window.location.href = '/login';
        }
        return Promise.reject(err);
      } finally {
        isRefreshing = false;
      }
    }

    return Promise.reject(error);
  }
);

// ---- snake_case -> camelCase mappers (backend uses snake, frontend uses camel) ----
function mapStudent(d: Record<string, unknown>): StudentState {
  return {
    id:             String(d.id ?? ''),
    trackId:        Number(d.trackId ?? d.track_id ?? 0),
    label:          ((d.label ?? 'Unknown') as StudentState['label']),
    engagementProb: Number(d.engagementProb ?? d.engagement_prob ?? 0),
    collabLabel:    ((d.collabLabel ?? d.collab_label ?? 'Unknown') as StudentState['collabLabel']),
    lastSeen:       Number(d.lastSeen ?? d.last_seen ?? 0),
    row:            Number(d.row ?? 0),
    col:            Number(d.col ?? 0),
  };
}

function mapAlert(d: Record<string, unknown>): AlertEvent {
  return {
    id:        String(d.id ?? ''),
    studentId: (d.studentId ?? d.student_id) as string | undefined,
    severity:  ((d.severity ?? 'soft') as AlertEvent['severity']),
    message:   String(d.message ?? ''),
    timestamp: Number(d.timestamp ?? 0),
    resolved:  Boolean(d.resolved ?? false),
  };
}

function mapSession(d: Record<string, unknown>): Session {
  const rawStudents = d.students as Record<string, unknown>[] | undefined;
  const rawAlerts   = d.alerts   as Record<string, unknown>[] | undefined;
  return {
    id:            String(d.id ?? ''),
    filename:      String(d.filename ?? ''),
    uploadedAt:    String(d.uploaded_at ?? ''),
    durationSec:   Number(d.duration_sec ?? 0),
    status:        (d.status as Session['status']) ?? 'queued',
    progress:      Number(d.progress ?? 0),
    avgEngagement: d.avg_engagement as number | undefined,
    avgCollab:     d.avg_collab     as number | undefined,
    classHealth:   d.class_health   as number | undefined,
    collabVerdict: d.collab_verdict as Session['collabVerdict'],
    students:      rawStudents ? rawStudents.map(mapStudent) : undefined,
    timeline:      d.timeline  as TimelinePoint[] | undefined,
    alerts:        rawAlerts   ? rawAlerts.map(mapAlert)   : undefined,
  };
}

// ---- localStorage helpers ----
const LS_KEY = 'edulens_last_session';

export function saveSessionToLocalStorage(session: Session & { frames?: unknown[] }) {
  try {
    if (typeof window === 'undefined') return;
    localStorage.setItem(LS_KEY, JSON.stringify({
      id:           session.id,
      filename:     session.filename,
      avgEngagement: session.avgEngagement,
      avgCollab:    session.avgCollab,
      classHealth:  session.classHealth,
      collabVerdict: session.collabVerdict,
      students:     session.students,
      timeline:     session.timeline,
      alerts:       session.alerts,
      frames:       session.frames,
      savedAt:      Date.now(),
    }));
  } catch { /* quota exceeded or SSR */ }
}

export function loadSessionFromLocalStorage(): (Session & { frames?: unknown[] }) | null {
  try {
    if (typeof window === 'undefined') return null;
    const raw = localStorage.getItem(LS_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    // Discard sessions older than 24 hours
    if (Date.now() - (parsed.savedAt ?? 0) > 86400000) {
      localStorage.removeItem(LS_KEY);
      return null;
    }
    return parsed;
  } catch { return null; }
}

// ---- Sessions ----
export async function uploadVideo(file: File): Promise<{ sessionId: string }> {
  const form = new FormData();
  form.append('file', file);
  const { data } = await client.post('/sessions/upload', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return data;
}

export async function triggerAnalysis(sessionId: string): Promise<void> {
  await client.post(`/sessions/${sessionId}/analyze`);
}

export async function getSession(sessionId: string): Promise<Session> {
  const { data } = await client.get(`/sessions/${sessionId}`);
  return mapSession(data as Record<string, unknown>);
}

export async function getSessionTimeline(sessionId: string): Promise<Session> {
  const { data } = await client.get(`/sessions/${sessionId}/timeline`);
  // timeline endpoint returns: { session_id, timeline, students, alerts, collab_verdict }
  const d = data as Record<string, unknown>;
  const rawStudents = d.students as Record<string, unknown>[] | undefined;
  const rawAlerts   = d.alerts   as Record<string, unknown>[] | undefined;
  return {
    id:            sessionId,
    filename:      '',
    uploadedAt:    '',
    durationSec:   0,
    status:        'done',
    progress:      100,
    collabVerdict: d.collab_verdict as Session['collabVerdict'],
    timeline:      d.timeline as TimelinePoint[] | undefined,
    students:      rawStudents ? rawStudents.map(mapStudent) : undefined,
    alerts:        rawAlerts   ? rawAlerts.map(mapAlert)     : undefined,
  };
}

export async function listSessions(): Promise<Session[]> {
  const { data } = await client.get('/sessions');
  return (data as Record<string, unknown>[]).map(mapSession);
}

export async function getSessionFrames(sessionId: string): Promise<unknown[]> {
  const { data } = await client.get(`/sessions/${sessionId}/frames`);
  return Array.isArray(data) ? data : [];
}

// ---- Copilot ----
export async function askCopilot(
  sessionId: string,
  question: string,
  history: CopilotMessage[]
): Promise<CopilotMessage> {
  const { data } = await client.post('/copilot/ask', {
    session_id: sessionId,
    question,
    history: history.slice(-6),
  });
  return data as CopilotMessage;
}

// ---- Reports ----
export async function generateReport(sessionId: string): Promise<Blob> {
  const response = await client.get(`/sessions/${sessionId}/report`, {
    responseType: 'blob',
  });
  return response.data as Blob;
}

// ---- Auth ----
export async function login(email: string, password: string, captchaToken?: string): Promise<LoginResponse> {
  const { data } = await client.post('/auth/login', { email, password, captcha_token: captchaToken });
  return data as LoginResponse;
}

export async function logoutApi(refreshToken: string): Promise<void> {
  await client.post('/auth/logout', { refresh_token: refreshToken });
}

export async function refreshTokenApi(refreshToken: string): Promise<LoginResponse> {
  const { data } = await axios.post(`${BASE}/auth/refresh`, { refresh_token: refreshToken });
  return data as LoginResponse;
}

// ---- Dashboard stats ----
export async function getDashboardStats() {
  const { data } = await client.get('/analytics/dashboard');
  return data;
}

// ---- Demo data (always available, varies per import but stable reference) ----
export const DEMO_TIMELINE: TimelinePoint[] = Array.from({ length: 61 }, (_, t) => ({
  t: t * 60,
  engagement: Math.round(70 + 15 * Math.sin((t / 60) * Math.PI * 2) - (t > 32 && t < 40 ? 30 : 0) + (((t * 17 + 3) % 17) - 8)),
  collab:     Math.round(60 + 10 * Math.sin((t / 60) * Math.PI)     - (t > 32 && t < 40 ? 20 : 0) + (((t * 13 + 5) % 13) - 6)),
  health:     0,
})).map(p => ({ ...p, health: Math.round((p.engagement + p.collab) / 2) }));

export const DEMO_STUDENTS = [
  { id:'ST-01',trackId:1,label:'Engaged'     as const,engagementProb:0.88,collabLabel:'Collaborative'     as const,lastSeen:0,row:0,col:0 },
  { id:'ST-02',trackId:2,label:'Not Engaged' as const,engagementProb:0.21,collabLabel:'Not Collaborative' as const,lastSeen:0,row:0,col:1 },
  { id:'ST-03',trackId:3,label:'Engaged'     as const,engagementProb:0.79,collabLabel:'Collaborative'     as const,lastSeen:0,row:0,col:2 },
  { id:'ST-04',trackId:4,label:'Engaged'     as const,engagementProb:0.91,collabLabel:'Collaborative'     as const,lastSeen:0,row:1,col:0 },
  { id:'ST-05',trackId:5,label:'Not Engaged' as const,engagementProb:0.34,collabLabel:'Not Collaborative' as const,lastSeen:0,row:1,col:1 },
  { id:'ST-06',trackId:6,label:'Engaged'     as const,engagementProb:0.82,collabLabel:'Collaborative'     as const,lastSeen:0,row:1,col:2 },
  { id:'ST-07',trackId:7,label:'Engaged'     as const,engagementProb:0.75,collabLabel:'Collaborative'     as const,lastSeen:0,row:2,col:0 },
  { id:'ST-08',trackId:8,label:'Engaged'     as const,engagementProb:0.69,collabLabel:'Not Collaborative' as const,lastSeen:0,row:2,col:1 },
  { id:'ST-09',trackId:9,label:'Not Engaged' as const,engagementProb:0.28,collabLabel:'Not Collaborative' as const,lastSeen:0,row:2,col:2 },
];

export const DEMO_ALERTS = [
  { id:'a1',studentId:'ST-02',severity:'warning'  as const,message:'ST-02 disengaged for 5+ minutes',            timestamp:2100,resolved:false },
  { id:'a2',studentId:'ST-05',severity:'soft'     as const,message:'ST-05 showing low attention',                timestamp:1800,resolved:false },
  { id:'a3',studentId:'ST-09',severity:'critical' as const,message:'ST-09 disengaged 10+ minutes — escalating', timestamp:3000,resolved:false },
  { id:'a4',studentId:undefined,severity:'warning' as const,message:'Class health dropped below 60%',            timestamp:2400,resolved:false },
];
