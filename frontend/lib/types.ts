// ─── Core types for EduLens ────────────────────────────────────────────────────

export type EngagementLabel = 'Engaged' | 'Not Engaged' | 'Unknown';
export type CollabLabel = 'Collaborative' | 'Not Collaborative' | 'Unknown';
export type AlertSeverity = 'soft' | 'warning' | 'critical';
export type UserRole = 'teacher' | 'hod' | 'principal';

export interface StudentState {
  id: string;         // e.g. "ST-01"
  trackId: number;
  label: EngagementLabel;
  engagementProb: number;   // 0–1
  collabLabel: CollabLabel;
  lastSeen: number;          // timestamp ms
  row: number;
  col: number;
}

export interface FrameResult {
  frameIndex: number;
  timestamp: number;       // seconds
  students: StudentState[];
  classHealthScore: number; // 0–100
  engagementScore: number;  // 0–100
  collabScore: number;      // 0–100
}

export interface TimelinePoint {
  t: number;            // seconds
  engagement: number;   // 0–100
  collab: number;
  health: number;
}

export interface AlertEvent {
  id: string;
  studentId?: string;   // null = classroom-level
  severity: AlertSeverity;
  message: string;
  timestamp: number;    // seconds into video
  resolved: boolean;
}

export interface Session {
  id: string;
  filename: string;
  uploadedAt: string;     // ISO
  durationSec: number;
  status: 'queued' | 'processing' | 'done' | 'error';
  progress: number;       // 0–100
  avgEngagement?: number;
  avgCollab?: number;
  classHealth?: number;
  timeline?: TimelinePoint[];
  alerts?: AlertEvent[];
  students?: StudentState[];
  collabVerdict?: CollabLabel;
}

export interface CopilotMessage {
  id: string;
  role: 'user' | 'ai';
  text: string;
  action?: CopilotAction;
  timestamp: number;
}

export interface CopilotAction {
  type: 'seek' | 'highlight_student' | 'show_chart';
  payload: Record<string, unknown>;
}

export interface AuthUser {
  id: string;
  name: string;
  email: string;
  role: UserRole;
  token: string;
}
