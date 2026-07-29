'use client';

import { useState, useCallback, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { motion } from 'framer-motion';
import {
  BarChart3, Brain, Users, Bell, Download, RefreshCw,
  Activity, TrendingUp, TrendingDown, AlertTriangle, Volume2
} from 'lucide-react';
import { Sidebar, TopNav } from '@/components/Navbar';
import { AuthGuard } from '@/components/AuthGuard';
import { EngagementChart } from '@/components/EngagementChart';
import { TimelineReplay } from '@/components/TimelineReplay';
import { CopilotPanel } from '@/components/CopilotPanel';
import { ClassroomMap } from '@/components/ClassroomMap';
import { AlertPanel } from '@/components/AlertPanel';
import { HealthPulse } from '@/components/HealthPulse';
import { CollabGraph } from '@/components/CollabGraph';
import {
  DEMO_TIMELINE, DEMO_STUDENTS, DEMO_ALERTS,
  loadSessionFromLocalStorage, getSession, getSessionTimeline, generateReport,
} from '@/lib/api';
import type { StudentState, AlertEvent, TimelinePoint, Session } from '@/lib/types';
import toast from 'react-hot-toast';

// ─── Voice narrator ────────────────────────────────────────────────────────────
function useNarrator() {
  const [speaking, setSpeaking] = useState(false);
  const narrate = useCallback((text: string) => {
    if (!window.speechSynthesis) return;
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.rate = 0.85;
    u.onend = () => setSpeaking(false);
    setSpeaking(true);
    window.speechSynthesis.speak(u);
  }, []);
  return { narrate, speaking };
}

// ─── Stat card ─────────────────────────────────────────────────────────────────
interface StatCardProps {
  label: string;
  value: string | number;
  sub?: string;
  color: string;
  icon: React.ElementType;
  trend?: 'up' | 'down' | null;
}

function StatCard({ label, value, sub, color, icon: Icon, trend }: StatCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass-card rounded-xl p-4 flex flex-col gap-2"
    >
      <div className="flex items-start justify-between">
        <div className="w-9 h-9 rounded-lg flex items-center justify-center"
          style={{ background: `${color}15`, border: `1px solid ${color}25` }}>
          <Icon size={16} style={{ color }} />
        </div>
        {trend && (
          <div className={`flex items-center gap-1 text-[10px] font-medium ${trend === 'up' ? 'text-status-engaged' : 'text-status-notEngaged'}`}>
            {trend === 'up' ? <TrendingUp size={11} /> : <TrendingDown size={11} />}
            {trend === 'up' ? '+4%' : '-3%'}
          </div>
        )}
      </div>
      <div>
        <div className="text-2xl font-black" style={{ color }}>{value}</div>
        <div className="text-xs text-text-muted">{label}</div>
        {sub && <div className="text-[10px] text-text-muted mt-0.5 opacity-70">{sub}</div>}
      </div>
    </motion.div>
  );
}

// ─── Dashboard page ────────────────────────────────────────────────────────────
export default function DashboardPage() {
  const [currentT, setCurrentT] = useState(0);
  const [selectedStudent, setSelectedStudent] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'engagement' | 'collab' | 'students'>('engagement');
  const [isReal, setIsReal] = useState(false);
  const [refreshing, setRefreshing] = useState(false);

  // ── Session data: real from localStorage, or demo fallback ──
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [timeline, setTimeline] = useState<TimelinePoint[]>(DEMO_TIMELINE);
  const [students, setStudents] = useState<StudentState[]>([...DEMO_STUDENTS] as unknown as StudentState[]);
  const [alerts, setAlerts] = useState<AlertEvent[]>([...DEMO_ALERTS] as unknown as AlertEvent[]);
  const [collabVerdict, setCollabVerdict] = useState<Session['collabVerdict']>('Collaborative');
  const [sessionDate, setSessionDate] = useState<string>('Demo Session');

  // Load real session from localStorage on mount (set by monitor page after analysis)
  useEffect(() => {
    const stored = loadSessionFromLocalStorage();
    if (stored && stored.id) {
      setSessionId(stored.id);
      setIsReal(true);
      if (stored.timeline?.length) setTimeline(stored.timeline);
      if (stored.students?.length) setStudents(stored.students as StudentState[]);
      if (stored.alerts?.length)   setAlerts(stored.alerts as AlertEvent[]);
      if (stored.collabVerdict)    setCollabVerdict(stored.collabVerdict);
      if (stored.filename)         setSessionDate(stored.filename);
      toast.success('Loaded real session data', { duration: 2000 });
    }
  }, []);

  // Refresh: re-fetch from backend by session id
  const handleRefresh = async () => {
    if (!sessionId) { toast('No session to refresh — upload a video first', { icon: 'ℹ️' }); return; }
    setRefreshing(true);
    try {
      const full = await getSessionTimeline(sessionId);
      if (full.timeline?.length) setTimeline(full.timeline);
      if (full.students?.length) setStudents(full.students as StudentState[]);
      if (full.alerts?.length)   setAlerts(full.alerts as AlertEvent[]);
      if (full.collabVerdict)    setCollabVerdict(full.collabVerdict);
      toast.success('Dashboard refreshed');
    } catch {
      toast.error('Could not refresh — backend may be offline');
    } finally {
      setRefreshing(false);
    }
  };

  // Export PDF from backend
  const handleExportPDF = async () => {
    if (!sessionId) { toast('No session — upload a video first', { icon: 'ℹ️' }); return; }
    toast.loading('Generating PDF…', { id: 'pdf' });
    try {
      const blob = await generateReport(sessionId);
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url; a.download = `edulens_report_${sessionId}.pdf`; a.click();
      URL.revokeObjectURL(url);
      toast.success('PDF downloaded', { id: 'pdf' });
    } catch {
      toast.error('PDF requires backend to be running', { id: 'pdf' });
    }
  };

  const handleExportJSON = () => {
    const data = { timeline, students, alerts, collabVerdict, avgEngagement, avgCollab, avgHealth };
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = `edulens_data_${sessionId ?? 'demo'}.json`; a.click();
    URL.revokeObjectURL(url);
    toast.success('JSON data exported');
  };

  const avgEngagement = Math.round(timeline.reduce((s, p) => s + p.engagement, 0) / timeline.length);
  const avgCollab = Math.round(timeline.reduce((s, p) => s + p.collab, 0) / timeline.length);
  const avgHealth = Math.round(timeline.reduce((s, p) => s + p.health, 0) / timeline.length);
  const activeAlerts = alerts.filter((a) => !a.resolved).length;
  const sessionDuration = Math.max(30, ...timeline.map((p) => p.t || 0), ...alerts.map((a) => a.timestamp || 0));

  const { narrate, speaking } = useNarrator();
  const router = useRouter();

  const handleSeek = (t: number) => {
    setCurrentT(t);
    const ts = Math.round(t);
    toast(`Opening video at ${Math.floor(ts / 60)}:${String(ts % 60).padStart(2, '0')}`, { icon: '▶', duration: 1500 });
    router.push(`/monitor?seek=${ts}`);   // jump to that moment on the live monitor video
  };

  const handleNarrate = () => {
    const text = `Session summary. Average engagement was ${avgEngagement} percent. Collaboration remained at ${avgCollab} percent. Class health score was ${avgHealth} percent. ${activeAlerts} alerts were triggered during the session.`;
    narrate(text);
  };

  const dismissAlert = (id: string) =>
    setAlerts((prev) => prev.map((a) => (a.id === id ? { ...a, resolved: true } : a)));

  const context = {
    timeline,
    alerts,
    students,
    engagement: avgEngagement,
    collab: avgCollab,
    health: avgHealth,
  };

  return (
    <AuthGuard>
    <div className="flex h-screen bg-bg-primary overflow-hidden">
      <Sidebar />

      <main className="flex-1 flex flex-col overflow-hidden">
        <TopNav title="Analytics Dashboard" />

        {/* Toolbar */}
        <div className="flex items-center gap-3 px-6 py-3 border-b border-white/5">
          <div className="live-dot text-xs text-text-muted">{sessionDate}</div>
          {isReal && (
            <span className="text-[10px] px-2 py-0.5 rounded-full bg-status-engaged/12 text-status-engaged border border-status-engaged/20 font-medium">
              ✓ Real Analysis
            </span>
          )}
          {!isReal && (
            <span className="text-[10px] px-2 py-0.5 rounded-full bg-white/6 text-text-muted border border-white/10">
              Demo Mode
            </span>
          )}
          <div className="ml-auto flex items-center gap-2">
            <button
              onClick={handleNarrate}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs border border-white/8 transition-all ${speaking ? 'text-accent-blue border-accent-blue/30' : 'text-text-secondary hover:text-text-primary hover:bg-white/4'}`}
            >
              <Volume2 size={13} />
              {speaking ? 'Narrating…' : 'Narrate Session'}
            </button>
            <button
              onClick={handleExportJSON}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs bg-white/5 border border-white/8 text-text-secondary hover:text-text-primary hover:bg-white/8 transition-all"
            >
              <Download size={13} />
              Export JSON
            </button>
            <button
              onClick={handleExportPDF}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs bg-white/5 border border-white/8 text-text-secondary hover:text-text-primary hover:bg-white/8 transition-all"
            >
              <Download size={13} />
              Export PDF
            </button>
            <button
              onClick={handleRefresh}
              disabled={refreshing}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs bg-accent-blue/10 border border-accent-blue/20 text-accent-blue hover:bg-accent-blue/15 transition-all disabled:opacity-50"
            >
              <RefreshCw size={13} className={refreshing ? 'animate-spin' : ''} />
              {refreshing ? 'Refreshing…' : 'Refresh'}
            </button>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto">
          <div className="p-6 space-y-6">
            {/* Top stats */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
              <StatCard label="Class Health" value={`${avgHealth}%`} color={avgHealth >= 70 ? '#22D3A6' : '#F59E0B'} icon={Activity} trend="up" />
              <StatCard label="Avg Engagement" value={`${avgEngagement}%`} sub="macro-F1 0.73 test" color="#4F7FFF" icon={Brain} trend="up" />
              <StatCard label="Avg Collaboration" value={`${avgCollab}%`} sub="LOVO F1 0.764" color="#38BDF8" icon={Users} trend="down" />
              <StatCard label="Alerts Fired" value={activeAlerts} sub={`${alerts.length} total`} color={activeAlerts > 0 ? '#FF4E4E' : '#22D3A6'} icon={Bell} trend={null} />
            </div>

            {/* Main content grid */}
            <div className="grid xl:grid-cols-3 gap-6">
              {/* Left: Charts + Timeline */}
              <div className="xl:col-span-2 space-y-5">
                {/* Tabbed charts */}
                <div className="glass-card rounded-2xl overflow-hidden">
                  {/* Tab bar */}
                  <div className="flex border-b border-white/6">
                    {[
                      { key: 'engagement', label: 'Engagement Timeline', icon: Brain },
                      { key: 'collab', label: 'Collaboration', icon: Users },
                      { key: 'students', label: 'Student View', icon: Activity },
                    ].map(({ key, label, icon: Icon }) => (
                      <button
                        key={key}
                        onClick={() => setActiveTab(key as any)}
                        className={`flex items-center gap-2 px-4 py-3 text-xs font-medium border-b-2 transition-all ${
                          activeTab === key
                            ? 'border-accent-blue text-white'
                            : 'border-transparent text-text-secondary hover:text-text-primary'
                        }`}
                      >
                        <Icon size={13} />
                        {label}
                      </button>
                    ))}
                  </div>

                  <div className="p-5">
                    {activeTab === 'engagement' && (
                      <div>
                        <div className="text-xs text-text-muted mb-3">Click on chart to jump to that moment in the video</div>
                        <EngagementChart data={timeline} onSeek={handleSeek} showCollab={false} height={220} />
                      </div>
                    )}
                    {activeTab === 'collab' && (
                      <div>
                        <div className="text-xs text-text-muted mb-3">Session-level collaboration verdict + engagement overlay</div>
                        <EngagementChart data={timeline} onSeek={handleSeek} showCollab height={220} />
                        <div className="mt-4 flex items-center gap-3 text-xs">
                          <div className={`px-3 py-1.5 rounded-lg font-medium ${collabVerdict === 'Not Collaborative' ? 'status-not-engaged' : 'status-collab'}`}>
                            Group Verdict: {collabVerdict ?? 'COLLABORATIVE'}
                          </div>
                          <span className="text-text-muted">LOVO F1: 0.764 (gaze-augmented)</span>
                        </div>
                      </div>
                    )}
                    {activeTab === 'students' && (
                      <div className="space-y-2">
                        <div className="text-xs text-text-muted mb-3">Per-student engagement probability</div>
                        <table className="w-full data-table">
                          <thead>
                            <tr>
                              <th className="text-left">Student ID</th>
                              <th className="text-left">Status</th>
                              <th className="text-left">Engagement</th>
                              <th className="text-left">Collaboration</th>
                            </tr>
                          </thead>
                          <tbody>
                            {students.map((s) => (
                              <tr key={s.id} onClick={() => setSelectedStudent(s.id === selectedStudent ? null : s.id)}
                                className={`cursor-pointer ${selectedStudent === s.id ? 'bg-accent-blue/5' : ''}`}>
                                <td className="font-mono text-xs font-medium">{s.id}</td>
                                <td>
                                  <span className={`text-[10px] px-2 py-0.5 rounded-full font-medium ${
                                    s.label === 'Engaged' ? 'status-engaged' : 'status-not-engaged'
                                  }`}>
                                    {s.label}
                                  </span>
                                </td>
                                <td>
                                  <div className="flex items-center gap-2">
                                    <div className="w-20 h-1.5 rounded-full bg-white/6 overflow-hidden">
                                      <div
                                        className="h-full rounded-full"
                                        style={{
                                          width: `${s.engagementProb * 100}%`,
                                          background: s.label === 'Engaged' ? '#22D3A6' : '#FF4E4E',
                                        }}
                                      />
                                    </div>
                                    <span className="text-xs text-text-muted">{(s.engagementProb * 100).toFixed(0)}%</span>
                                  </div>
                                </td>
                                <td>
                                  <span className={`text-[10px] ${s.collabLabel === 'Collaborative' ? 'text-status-collab' : 'text-text-muted'}`}>
                                    {s.collabLabel}
                                  </span>
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    )}
                  </div>
                </div>

                {/* Timeline Replay */}
                <div className="glass-card rounded-2xl p-5">
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-2">
                      <Activity size={14} className="text-accent-blue" />
                      <span className="text-sm font-semibold">Session Timeline Replay</span>
                    </div>
                    <span className="text-xs text-text-muted">Click markers to jump to incidents</span>
                  </div>
                  <TimelineReplay
                    timeline={timeline}
                    alerts={alerts}
                    duration={sessionDuration}
                    onSeek={handleSeek}
                  />
                </div>

                {/* Interaction Network */}
                <div className="glass-card rounded-2xl p-5">
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-2">
                      <Users size={14} className="text-accent-blue" />
                      <span className="text-sm font-semibold">Live Interaction Network</span>
                    </div>
                    <span className={`text-xs px-2 py-0.5 rounded-full ${collabVerdict === 'Not Collaborative' ? 'status-not-engaged' : 'status-collab'}`}>
                      Group: {collabVerdict ?? 'Collaborative'}
                    </span>
                  </div>
                  <CollabGraph students={students} />
                </div>
              </div>

              {/* Right: Copilot + Health + Alerts + Map */}
              <div className="space-y-4">
                {/* AI Copilot — STANDOUT FEATURE */}
                <CopilotPanel context={context} onSeek={handleSeek} />

                {/* Health gauge */}
                <div className="glass-card rounded-2xl p-5 flex flex-col items-center">
                  <div className="flex items-center gap-2 mb-3 self-start">
                    <Activity size={13} className="text-accent-blue" />
                    <span className="text-xs font-semibold">Class Health</span>
                  </div>
                  <HealthPulse score={avgHealth} size="md" />
                </div>

                {/* Alerts */}
                <div className="glass-card rounded-2xl p-4">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <Bell size={13} className="text-status-warning" />
                      <span className="text-xs font-semibold">Alerts Center</span>
                    </div>
                    <span className="text-[10px] text-text-muted">{activeAlerts} active</span>
                  </div>
                  <AlertPanel alerts={alerts} onDismiss={dismissAlert} onSeek={handleSeek} />
                </div>

                {/* Classroom Map */}
                <div className="glass-card rounded-2xl p-4">
                  <div className="flex items-center gap-2 mb-3">
                    <Brain size={13} className="text-accent-blue" />
                    <span className="text-xs font-semibold">Digital Twin</span>
                  </div>
                  <ClassroomMap students={students} showCollab />
                </div>

                {/* Privacy note */}
                <div className="glass-card rounded-xl p-4 border border-white/8">
                  <div className="text-[10px] font-bold text-text-muted uppercase tracking-wider mb-2 flex items-center gap-1.5">
                    <div className="w-1.5 h-1.5 rounded-full bg-status-engaged" />
                    Privacy Status
                  </div>
                  {[
                    '✓ Anonymous Student IDs',
                    '✓ No Facial Recognition',
                    '✓ Video Auto-Deleted 24h',
                    '✓ Role-Based Access Control',
                  ].map((t) => (
                    <div key={t} className="text-[10px] text-text-muted py-0.5">{t}</div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
    </AuthGuard>
  );
}
