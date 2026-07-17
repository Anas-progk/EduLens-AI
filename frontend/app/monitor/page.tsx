'use client';

import { useState, useRef, useCallback, useEffect } from 'react';
import { motion } from 'framer-motion';
import Link from 'next/link';
import { Upload, BarChart3, Brain, Bell, Users, Activity, ChevronRight, RefreshCw, Video, CheckCircle2 } from 'lucide-react';
import { Sidebar, TopNav } from '@/components/Navbar';
import { HealthPulse, MiniStat } from '@/components/HealthPulse';
import { ClassroomMap } from '@/components/ClassroomMap';
import { AlertPanel } from '@/components/AlertPanel';
import {
  uploadVideo, triggerAnalysis, getSession, getSessionTimeline, getSessionFrames,
  saveSessionToLocalStorage, loadSessionFromLocalStorage,
} from '@/lib/api';
import { sessionStore } from '@/lib/sessionStore';
import type { StudentState, AlertEvent, Session } from '@/lib/types';
import toast from 'react-hot-toast';

// ---- Types ----
interface Detection { track_id: number; bbox: [number,number,number,number]; label: string; prob: number; }
interface FrameData  { t: number; detections: Detection[]; }

// (Removed dead useLiveSim demo ticker — it referenced un-imported DEMO_STUDENTS/DEMO_ALERTS
//  and broke `npm run build`. The monitor now shows ONLY real analysis results.)

// ---- Canvas bbox overlay ----
function BboxOverlay({ videoRef, frameData, demoStudents }: {
  videoRef: React.RefObject<HTMLVideoElement>;
  frameData: FrameData[];
  demoStudents: StudentState[];
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const draw = useCallback((currentTime: number) => {
    const canvas = canvasRef.current;
    const video  = videoRef.current;
    if (!canvas || !video || !video.videoWidth) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const rect = video.getBoundingClientRect();
    if (canvas.width !== rect.width || canvas.height !== rect.height) {
      canvas.width  = rect.width;
      canvas.height = rect.height;
    }
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const vW = video.videoWidth;   // intrinsic video dimensions
    const vH = video.videoHeight;

    // ── Account for object-contain letterboxing/pillarboxing ──────────────────
    // The video element may have black bars. Calculate the actual rendered area.
    const cW = canvas.width;
    const cH = canvas.height;
    const vAspect = vW / vH;
    const cAspect = cW / cH;

    let renderW: number, renderH: number, offsetX: number, offsetY: number;
    if (vAspect > cAspect) {
      // Video is wider than container → black bars top/bottom (letterbox)
      renderW = cW;
      renderH = cW / vAspect;
      offsetX = 0;
      offsetY = (cH - renderH) / 2;
    } else {
      // Video is taller than container → black bars left/right (pillarbox)
      renderH = cH;
      renderW = cH * vAspect;
      offsetX = (cW - renderW) / 2;
      offsetY = 0;
    }
    const sx = renderW / vW;
    const sy = renderH / vH;

    // Use real backend frame detections — no fake grid fallback
    let detections: Detection[] = [];
    if (frameData.length > 0) {
      // Find the closest stored frame within a 3s window of current video time
      const frame = frameData.reduce<FrameData | null>((best, f) => {
        const d = Math.abs(f.t - currentTime);
        if (d > 0.8) return best;   // tight window so boxes follow the video
        return (!best || d < Math.abs(best.t - currentTime)) ? f : best;
      }, null);
      if (frame) detections = frame.detections;
    }
    // NO demo grid fallback — bboxes come from real Haar/HOG/YOLO detection only

    for (const det of detections) {
      const [x1, y1, x2, y2] = det.bbox;
      // Scale from intrinsic video space → rendered video area → canvas space
      const cx1 = x1 * sx + offsetX;
      const cy1 = y1 * sy + offsetY;
      const bw  = (x2 - x1) * sx;
      const bh  = (y2 - y1) * sy;
      const pending = det.label === 'Analyzing' || det.label === 'Unknown';
      const color   = pending ? '#8B95A7' : (det.label === 'Not Engaged' ? '#FF4E4E' : '#22D3A6');
      const alpha   = pending ? '18' : (det.label === 'Not Engaged' ? '28' : '20');

      ctx.fillStyle   = color + alpha;
      ctx.fillRect(cx1, cy1, bw, bh);
      ctx.strokeStyle = color;
      ctx.lineWidth   = 2;
      ctx.strokeRect(cx1, cy1, bw, bh);

      // Corner accents
      const cs = 10;
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(cx1, cy1 + cs); ctx.lineTo(cx1, cy1); ctx.lineTo(cx1 + cs, cy1);
      ctx.moveTo(cx1 + bw - cs, cy1); ctx.lineTo(cx1 + bw, cy1); ctx.lineTo(cx1 + bw, cy1 + cs);
      ctx.moveTo(cx1, cy1 + bh - cs); ctx.lineTo(cx1, cy1 + bh); ctx.lineTo(cx1 + cs, cy1 + bh);
      ctx.moveTo(cx1 + bw - cs, cy1 + bh); ctx.lineTo(cx1 + bw, cy1 + bh); ctx.lineTo(cx1 + bw, cy1 + bh - cs);
      ctx.stroke();

      // Label pill
      const id  = `ST-${String(det.track_id).padStart(2, '0')}`;
      const pct = `${(det.prob * 100).toFixed(0)}%`;
      ctx.font = 'bold 11px Inter, sans-serif';
      const tw  = ctx.measureText(`${id}  ${pct}`).width + 12;
      const pH  = 18;
      const pY  = Math.max(0, cy1 - pH - 2);
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.roundRect(cx1, pY, tw, pH, 4);
      ctx.fill();
      ctx.fillStyle = 'rgba(0,0,0,0.8)';
      ctx.font = 'bold 10px Inter, sans-serif';
      ctx.fillText(id, cx1 + 5, pY + 13);
      ctx.fillStyle = 'rgba(255,255,255,0.9)';
      ctx.fillText(pct, cx1 + ctx.measureText(id).width + 8, pY + 13);
    }
  }, [frameData, demoStudents, videoRef]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    const onTime = () => draw(video.currentTime);
    const onMeta = () => draw(0);
    video.addEventListener('timeupdate', onTime);
    video.addEventListener('loadedmetadata', onMeta);
    // Also draw when frame data changes
    if (video.readyState >= 1) draw(video.currentTime);
    return () => {
      video.removeEventListener('timeupdate', onTime);
      video.removeEventListener('loadedmetadata', onMeta);
    };
  }, [draw, videoRef]);

  // Redraw when frameData changes
  useEffect(() => {
    const video = videoRef.current;
    if (video && video.readyState >= 1) draw(video.currentTime);
  }, [frameData, demoStudents, draw, videoRef]);

  return (
    <canvas ref={canvasRef} className="absolute inset-0 pointer-events-none" style={{ width: '100%', height: '100%' }} />
  );
}

// ---- Drop zone ----
function DropZone({ onFile, processing, progress }: { onFile: (f: File) => void; processing: boolean; progress: number; }) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  return (
    <div
      className={`drop-zone rounded-2xl p-10 text-center cursor-pointer ${dragging ? 'drag-over' : ''}`}
      onDragOver={e => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={e => { e.preventDefault(); setDragging(false); const f = e.dataTransfer.files[0]; if (f) onFile(f); else toast.error('Drop a video file'); }}
      onClick={() => inputRef.current?.click()}
    >
      <input ref={inputRef} type="file" accept="video/*" className="hidden" onChange={e => { const f = e.target.files?.[0]; if (f) onFile(f); }} />
      {processing ? (
        <div className="space-y-4">
          <div className="w-12 h-12 mx-auto rounded-xl bg-accent-blue/10 flex items-center justify-center"><RefreshCw size={22} className="text-accent-blue animate-spin" /></div>
          <div>
            <p className="text-sm font-semibold mb-1">Analyzing classroom video...</p>
            <p className="text-xs text-text-muted">Swin-Tiny backbone · 8-frame clips · Face detection</p>
          </div>
          <div className="max-w-xs mx-auto">
            <div className="flex justify-between text-xs text-text-muted mb-1.5"><span>Processing</span><span>{progress}%</span></div>
            <div className="h-1 rounded-full bg-white/8"><motion.div className="h-full rounded-full progress-bar" animate={{ width: `${progress}%` }} transition={{ duration: 0.4 }} /></div>
          </div>
        </div>
      ) : (
        <div className="space-y-3">
          <div className="w-14 h-14 mx-auto rounded-2xl bg-accent-blue/8 border border-accent-blue/20 flex items-center justify-center"><Upload size={24} className="text-accent-blue" /></div>
          <div><p className="text-sm font-semibold mb-1">Drop classroom video here</p><p className="text-xs text-text-muted">or click to browse · MP4, AVI, MOV</p></div>
          <div className="flex justify-center gap-3 text-[10px] text-text-muted">
            <span>✓ Face detection</span><span>✓ Swin-Tiny</span><span>✓ Group collab</span>
          </div>
        </div>
      )}
    </div>
  );
}

// ---- Main page ----
export default function MonitorPage() {
  const [sessionId,   setSessionId]   = useState<string | null>(null);
  const [session,     setSession]     = useState<Session | null>(null);
  const [processing,  setProcessing]  = useState(false);
  const [progress,    setProgress]    = useState(0);
  const [demoMode,    setDemoMode]    = useState(true);
  const [videoUrl,    setVideoUrl]    = useState<string | null>(null);
  const [frameData,   setFrameData]   = useState<FrameData[]>([]);
  const videoRef = useRef<HTMLVideoElement>(null);
  const [pendingSeek, setPendingSeek] = useState<number | null>(null);

  // Restore the last analysis when navigating back from the dashboard (module state
  // survives client-side navigation), and honor a ?seek= param for "jump to moment".
  useEffect(() => {
    if (typeof window === 'undefined') return;
    const seek = new URLSearchParams(window.location.search).get('seek');
    if (seek) setPendingSeek(Number(seek));
    if (!videoUrl && sessionStore.videoUrl) {
      setVideoUrl(sessionStore.videoUrl);
      const stored = loadSessionFromLocalStorage();
      if (stored && stored.id) {
        // saved session has no `status`; force 'done' so the metrics render on restore
        setSession({ ...(stored as Record<string, unknown>), status: 'done' } as unknown as Session);
        setSessionId(stored.id);
        if (Array.isArray(stored.frames)) setFrameData(stored.frames as FrameData[]);
        setProcessing(false);
        setDemoMode(false);
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Seek the restored video to the requested moment once it can play.
  useEffect(() => {
    const v = videoRef.current;
    if (!v || pendingSeek == null) return;
    const doSeek = () => { try { v.currentTime = pendingSeek; } catch { /* not ready */ } };
    if (v.readyState >= 1) doSeek();
    else v.addEventListener('loadedmetadata', doSeek, { once: true });
  }, [pendingSeek, videoUrl]);

  // Poll backend while processing
  useEffect(() => {
    if (!sessionId || !processing) return;
    let active = true;

    async function poll() {
      if (!active) return;
      try {
        const s = await getSession(sessionId!);
        // s now has camelCase fields after our mapper fix
        setProgress(s.progress ?? 0);

        if (s.status === 'done') {
          setProcessing(false);
          setDemoMode(false);

          // Fetch full timeline data (students, alerts, timeline)
          try {
            const full = await getSessionTimeline(sessionId!);
            const enriched = { ...s, ...full };
            setSession(enriched);

            // Fetch frame data for bbox overlay
            const frames = await getSessionFrames(sessionId!) as FrameData[];
            if (frames?.length) setFrameData(frames);

            // Persist to localStorage so dashboard can read it
            saveSessionToLocalStorage({ ...enriched, frames });
            toast.success(`Analysis complete! Engagement: ${enriched.avgEngagement?.toFixed(0)}%`);
          } catch {
            setSession(s);
            saveSessionToLocalStorage(s);
            toast.success('Analysis complete!');
          }
        } else if (s.status === 'error') {
          setProcessing(false);
          toast.error('Analysis failed — showing demo data');
          setDemoMode(true);
        } else {
          setTimeout(poll, 2000);
        }
      } catch { setTimeout(poll, 3000); }
    }

    setTimeout(poll, 1000);
    return () => { active = false; };
  }, [sessionId, processing]);

  const handleFile = async (file: File) => {
    const url = URL.createObjectURL(file);
    setVideoUrl(url);
    sessionStore.videoUrl = url;        // persist for back-navigation / seek
    setFrameData([]);
    setSession(null);
    setProcessing(true);
    setDemoMode(false);
    setProgress(0);

    let prog = 0;
    const progId = setInterval(() => {
      prog = Math.min(88, prog + Math.random() * 5);
      setProgress(Math.round(prog));
    }, 700);

    try {
      const { sessionId: sid } = await uploadVideo(file);
      setSessionId(sid);
      sessionStore.sessionId = sid;
      await triggerAnalysis(sid);
      clearInterval(progId);
    } catch {
      clearInterval(progId);
      // No backend — simulate after delay
      setTimeout(() => {
        setProgress(100);
        setProcessing(false);
        setDemoMode(true);
        toast.success('Analysis complete (demo mode — start backend for real results)');
      }, 6000);
    }
  };

  // Only show stats after a video has been uploaded and analysis is done (or running)
  const hasVideo = !!videoUrl;
  const analysisDone = session?.status === 'done';

  // Real session data when done; null/placeholder before upload
  const displayStudents   = analysisDone ? (session?.students ?? []) as StudentState[]
                          : hasVideo ? [] : [];
  const displayHealth     = analysisDone ? (session?.classHealth   ?? 0) : null;
  const displayEngagement = analysisDone ? (session?.avgEngagement ?? 0) : null;
  const displayCollab     = analysisDone ? (session?.avgCollab     ?? 0) : null;
  const displayAlerts     = analysisDone ? (session?.alerts        ?? []) as AlertEvent[] : [];
  const activeAlerts      = displayAlerts.filter(a => !a.resolved).length;

  return (
    <div className="flex h-screen bg-bg-primary overflow-hidden">
      <Sidebar />
      <main className="flex-1 flex flex-col overflow-hidden">
        <TopNav title="Live Monitor" />
        <div className="flex-1 overflow-y-auto">
          <div className="p-6 space-y-5">

            {/* Top metric cards — show real values only after analysis, placeholder before */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
              {(() => {
                const hColor = displayHealth != null
                  ? (displayHealth >= 70 ? '#22D3A6' : displayHealth >= 45 ? '#F59E0B' : '#FF4E4E')
                  : '#4B5563';
                return [
                  { label:'Class Health',  val: displayHealth  != null ? `${Math.round(displayHealth)}%`  : processing ? '…' : '—', color: hColor, icon:Activity },
                  { label:'Engagement',    val: displayEngagement != null ? `${Math.round(displayEngagement)}%` : processing ? '…' : '—', color:'#4F7FFF', icon:Brain },
                  { label:'Collaboration', val: displayCollab   != null ? `${Math.round(displayCollab)}%`  : processing ? '…' : '—', color:'#38BDF8', icon:Users },
                  { label:'Active Alerts', val: analysisDone ? activeAlerts : '—', color: activeAlerts > 0 ? '#FF4E4E' : '#22D3A6', icon:Bell },
                ].map(({ label, val, color, icon: Icon }) => (
                  <motion.div key={label} initial={{ opacity:0, y:12 }} animate={{ opacity:1, y:0 }}
                    className="glass-card rounded-xl p-4 flex items-center gap-3">
                    <div className="w-10 h-10 rounded-xl flex items-center justify-center" style={{ background:`${color}15`, border:`1px solid ${color}25` }}>
                      <Icon size={18} style={{ color }} />
                    </div>
                    <div>
                      <div className="text-xl font-black" style={{ color }}>{val}</div>
                      <div className="text-xs text-text-muted">{label}</div>
                    </div>
                  </motion.div>
                ));
              })()}
            </div>

            <div className="grid lg:grid-cols-3 gap-5">
              {/* Video + metrics */}
              <div className="lg:col-span-2 space-y-4">
                <div className="glass-card rounded-2xl overflow-hidden">
                  <div className="flex items-center justify-between px-5 py-3.5 border-b border-white/6">
                    <div className="flex items-center gap-2">
                      <Video size={15} className="text-accent-blue" />
                      <span className="text-sm font-semibold">Classroom Feed</span>
                    </div>
                    <div className="flex items-center gap-3">
                      {demoMode && !videoUrl && <div className="live-dot text-xs text-text-muted">Demo Mode</div>}
                      {!demoMode && session?.status === 'done' && (
                        <div className="flex items-center gap-1.5 text-xs text-status-engaged"><CheckCircle2 size={12} /> Real Analysis</div>
                      )}
                      {session?.collabVerdict && (
                        <span className="text-xs px-2 py-0.5 rounded-full" style={{ background: session.collabVerdict === 'COLLABORATIVE' ? 'rgba(56,189,248,0.12)' : 'rgba(255,78,78,0.12)', color: session.collabVerdict === 'COLLABORATIVE' ? '#38BDF8' : '#FF4E4E' }}>
                          {session.collabVerdict === 'COLLABORATIVE' ? 'Collaborative' : 'Not Collaborative'}
                        </span>
                      )}
                      <Link href="/dashboard" className="flex items-center gap-1 text-xs text-accent-blue hover:underline">
                        Detailed Analytics <ChevronRight size={12} />
                      </Link>
                    </div>
                  </div>
                  <div className="p-4">
                    {videoUrl && !processing ? (
                      <div className="relative rounded-xl overflow-hidden bg-black aspect-video">
                        <video ref={videoRef} src={videoUrl} controls className="w-full h-full object-contain" />
                        <BboxOverlay videoRef={videoRef} frameData={frameData} demoStudents={displayStudents} />
                        <div className="absolute top-3 left-3 pointer-events-none flex gap-2">
                          <div className="flex items-center gap-1.5 px-2 py-1 rounded-lg bg-black/60 backdrop-blur-sm text-[10px] text-white">
                            <span className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
                            AI Active
                          </div>
                        </div>
                      </div>
                    ) : (
                      <DropZone onFile={handleFile} processing={processing} progress={progress} />
                    )}
                  </div>
                </div>

                <div className="glass-card rounded-2xl p-5">
                  <h3 className="text-sm font-semibold mb-4 flex items-center gap-2"><Activity size={14} className="text-accent-blue" /> Live Metrics</h3>
                  {analysisDone ? (
                    <div className="grid grid-cols-3 gap-6">
                      <MiniStat label="Engagement"    value={displayEngagement ?? 0} color="#4F7FFF" />
                      <MiniStat label="Collaboration"  value={displayCollab    ?? 0} color="#38BDF8" />
                      <MiniStat label="Class Health"   value={displayHealth    ?? 0} color={(displayHealth ?? 0) >= 70 ? '#22D3A6' : '#F59E0B'} />
                    </div>
                  ) : (
                    <div className="text-center py-4 text-text-muted text-xs">
                      {processing ? 'Analyzing…' : 'Upload a video to see live metrics'}
                    </div>
                  )}
                </div>
              </div>

              {/* Right panel */}
              <div className="space-y-4">
                <div className="glass-card rounded-2xl p-5 flex flex-col items-center">
                  <div className="flex items-center gap-2 mb-4 self-start"><Activity size={14} className="text-accent-blue" /><span className="text-sm font-semibold">Class Health Score</span></div>
                  {analysisDone ? (
                    <>
                      <HealthPulse score={displayHealth ?? 0} size="lg" label="Pulsing when classroom is healthy" />
                      <div className="mt-4 grid grid-cols-2 gap-2 w-full text-center">
                        <div className="rounded-lg bg-white/4 py-2"><div className="text-lg font-bold text-status-engaged">{Math.round(displayEngagement ?? 0)}%</div><div className="text-[10px] text-text-muted">Engaged</div></div>
                        <div className="rounded-lg bg-white/4 py-2"><div className="text-lg font-bold text-status-collab">{Math.round(displayCollab ?? 0)}%</div><div className="text-[10px] text-text-muted">Collab</div></div>
                      </div>
                    </>
                  ) : (
                    <div className="w-full flex flex-col items-center py-6 text-text-muted">
                      <div className="w-24 h-24 rounded-full border-2 border-dashed border-white/10 flex items-center justify-center mb-3">
                        <Activity size={28} className="opacity-20" />
                      </div>
                      <p className="text-xs">{processing ? 'Analyzing…' : 'Awaiting video'}</p>
                    </div>
                  )}
                </div>

                <div className="glass-card rounded-2xl p-5">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2"><Bell size={14} className="text-status-warning" /><span className="text-sm font-semibold">Alerts</span></div>
                    {activeAlerts > 0 && <span className="text-xs px-2 py-0.5 rounded-full bg-status-notEngaged/12 text-status-notEngaged border border-status-notEngaged/20">{activeAlerts} active</span>}
                  </div>
                  {analysisDone
                    ? <AlertPanel alerts={displayAlerts} onDismiss={() => {}} compact />
                    : <p className="text-xs text-text-muted py-2">{processing ? 'Detection running…' : 'No alerts yet'}</p>
                  }
                </div>

                <div className="glass-card rounded-2xl p-5">
                  <div className="flex items-center gap-2 mb-4"><Users size={14} className="text-accent-blue" /><span className="text-sm font-semibold">Digital Classroom Twin</span></div>
                  {analysisDone && displayStudents.length > 0
                    ? <ClassroomMap students={displayStudents} />
                    : <p className="text-xs text-text-muted py-2">{processing ? 'Detecting students…' : 'Upload a video to see the classroom map'}</p>
                  }
                </div>
              </div>
            </div>

            <div className="flex justify-center">
              <Link href="/dashboard" className="flex items-center gap-2 px-6 py-3 rounded-xl text-sm font-semibold text-white" style={{ background:'linear-gradient(135deg, #4F7FFF, #7C6FFF)' }}>
                <BarChart3 size={16} /> View Detailed Analytics + Copilot <ChevronRight size={16} />
              </Link>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
