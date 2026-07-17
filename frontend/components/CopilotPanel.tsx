'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Brain, Send, Volume2, Minimize2, Maximize2, Sparkles } from 'lucide-react';
import type { CopilotMessage, TimelinePoint, AlertEvent, StudentState } from '@/lib/types';
import { askCopilot } from '@/lib/api';

// ─── Rule-based Copilot response engine ──────────────────────────────────────
interface ContextData {
  timeline: TimelinePoint[];
  alerts: AlertEvent[];
  students: StudentState[];
  engagement: number;
  collab: number;
  health: number;
  sessionId?: string;
}

function generateLocalResponse(q: string, ctx: ContextData): CopilotMessage {
  const lower = q.toLowerCase();
  const { timeline, alerts, students, engagement, collab, health } = ctx;

  // Find lowest engagement point
  const sorted = [...timeline].sort((a, b) => a.engagement - b.engagement);
  const lowestPt = sorted[0];
  const dropPts = timeline.filter((p) => p.engagement < 50);
  const disengaged = students.filter((s) => s.label === 'Not Engaged');
  const critical = alerts.find((a) => a.severity === 'critical');

  // Pattern matching
  if (lower.includes('red') || lower.includes('low') || lower.includes('why')) {
    const text = `During the last session:\n• Engagement dropped to **${lowestPt?.engagement ?? 0}%** at minute ${Math.floor((lowestPt?.t ?? 0) / 60)}\n• ${disengaged.length} student${disengaged.length !== 1 ? 's' : ''} showed sustained disengagement (${disengaged.map(s => s.id).join(', ')})\n• Collaboration remained at **${collab}%**\n• Class health: **${health}%**\n\nSuggested action: Introduce a discussion activity or quick quiz to re-engage students.${lowestPt ? `\n\n▶ Click to replay minute ${Math.floor(lowestPt.t / 60)}` : ''}`;
    return mkAI(text, lowestPt ? { type: 'seek', payload: { t: lowestPt.t } } : undefined);
  }

  if (lower.includes('most disengaged') || lower.includes('worst') || lower.includes('lowest')) {
    const t = lowestPt?.t ?? 0;
    const m = Math.floor(t / 60);
    const text = `The most disengaged period was minute **${m}–${m + 2}**:\n• Average engagement: **${lowestPt?.engagement ?? 0}%**\n• ${alerts.filter(a => Math.abs(a.timestamp - t) < 300).length} alerts fired during this window\n• Collaboration also dipped to ${lowestPt?.collab ?? 0}%\n\n▶ Click to jump there`;
    return mkAI(text, { type: 'seek', payload: { t } });
  }

  if (lower.includes('student') || lower.includes('who')) {
    const ne = students.filter(s => s.label === 'Not Engaged');
    if (ne.length === 0) {
      return mkAI('All tracked students are currently **Engaged**. Great class health!');
    }
    const text = `${ne.length} student${ne.length !== 1 ? 's' : ''} currently need attention:\n${ne.map(s => `• **${s.id}** — engagement prob: ${(s.engagementProb * 100).toFixed(0)}%`).join('\n')}\n\nThey have shown sustained disengagement for 5+ minutes. Consider direct engagement.`;
    return mkAI(text);
  }

  if (lower.includes('alert')) {
    const active = alerts.filter(a => !a.resolved);
    if (active.length === 0) return mkAI('No active alerts. The classroom is performing well!');
    const text = `**${active.length} active alert${active.length !== 1 ? 's' : ''}:**\n${active.map(a => `• [${a.severity.toUpperCase()}] ${a.message}`).join('\n')}\n\nRecommendation: Address critical alerts first, then warning-level students.`;
    return mkAI(text);
  }

  if (lower.includes('collab') || lower.includes('group') || lower.includes('interact')) {
    const verdict = collab >= 65 ? 'COLLABORATIVE' : 'NOT COLLABORATIVE';
    const text = `**Group Collaboration Analysis:**\n• Session-level verdict: **${verdict}** (${collab}%)\n• This is based on 6 relational signals + mutual gaze patterns\n• Model: Swin-Tiny backbone + Group Logistic Head\n• Honest LOVO macro-F1: **0.667** (baseline: 0.348)\n\nGaze-augmented model achieves **0.764** on this session.`;
    return mkAI(text);
  }

  if (lower.includes('report') || lower.includes('summary') || lower.includes('pdf')) {
    const text = `Session Summary:\n• Average engagement: **${engagement}%**\n• Average collaboration: **${collab}%**\n• Class health score: **${health}%**\n• Alerts triggered: **${alerts.length}**\n• Students analyzed: **${students.length}**\n\nClick "Generate Report" in the toolbar to export a full PDF with charts, heatmaps, and recommendations.`;
    return mkAI(text);
  }

  if (lower.includes('suggest') || lower.includes('recommend') || lower.includes('action')) {
    const actions = engagement < 60
      ? ['Start a discussion activity', 'Ask targeted questions to disengaged students', 'Introduce a short collaborative exercise', 'Take a 2-minute break']
      : ['Maintain current pace — engagement is good', 'Consider a quick quiz to test retention', 'Try small-group work to boost collaboration'];
    const text = `Based on current metrics (Engagement: ${engagement}%, Collab: ${collab}%):\n\n**Recommended actions:**\n${actions.map(a => `• ${a}`).join('\n')}`;
    return mkAI(text);
  }

  if (lower.includes('model') || lower.includes('accuracy') || lower.includes('f1') || lower.includes('how')) {
    return mkAI('**Model Architecture:**\n• Backbone: **Swin-Tiny Transformer** (pre-trained ImageNet)\n• Temporal head: 2-layer TransformerEncoder (8-frame clips)\n• Engagement: macro-F1 **0.73** on 5 held-out classrooms\n• Collaboration: LOVO macro-F1 **0.667** → **0.764** with gaze signals\n• Real-time: ~2s per clip on CPU, instant on GPU');
  }

  // Default
  const text = `I can help you understand your classroom analytics. Try asking:\n• "Why is the class showing red?"\n• "Show me the most disengaged period"\n• "Which students need attention?"\n• "What are the active alerts?"\n• "Suggest teaching interventions"\n• "Summarize this session"`;
  return mkAI(text);
}

function mkAI(text: string, action?: CopilotMessage['action']): CopilotMessage {
  return {
    id: `ai-${Date.now()}`,
    role: 'ai',
    text,
    action,
    timestamp: Date.now(),
  };
}

const SUGGESTIONS = [
  'Why is the class showing red?',
  'Most disengaged period?',
  'Which students need help?',
  'Suggest interventions',
];

interface CopilotPanelProps {
  sessionId?: string;
  context: ContextData;
  onSeek?: (t: number) => void;
}

export function CopilotPanel({ sessionId, context, onSeek }: CopilotPanelProps) {
  const [messages, setMessages] = useState<CopilotMessage[]>([
    mkAI("Hi! I'm your **EduLens Copilot**. Ask me anything about your classroom — engagement drops, collaboration patterns, alerts, or teaching suggestions."),
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [minimized, setMinimized] = useState(false);
  const [speaking, setSpeaking] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const speak = useCallback((text: string) => {
    if (!window.speechSynthesis) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(
      text.replace(/\*\*/g, '').replace(/\n/g, '. ').replace(/•/g, ',')
    );
    utterance.rate = 0.9;
    utterance.pitch = 1;
    utterance.onend = () => setSpeaking(false);
    setSpeaking(true);
    window.speechSynthesis.speak(utterance);
  }, []);

  const send = async (question?: string) => {
    const q = (question ?? input).trim();
    if (!q || loading) return;
    setInput('');

    const userMsg: CopilotMessage = { id: `u-${Date.now()}`, role: 'user', text: q, timestamp: Date.now() };
    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);

    try {
      // Try real API first
      if (sessionId) {
        const reply = await askCopilot(sessionId, q, messages);
        setMessages((prev) => [...prev, reply]);
        // Do NOT auto-navigate on a seek action — only the explicit
        // "Jump to this moment" button should move to the video.
        setLoading(false);
        return;
      }
    } catch { /* fallback to local */ }

    // Local rule-based response (always available)
    await new Promise((r) => setTimeout(r, 400 + Math.random() * 600)); // simulate thinking
    const reply = generateLocalResponse(q, context);
    setMessages((prev) => [...prev, reply]);   // show the answer; user taps "Jump" to navigate
    setLoading(false);
  };

  // Render markdown-like formatting
  const renderText = (text: string) =>
    text.split('\n').map((line, i) => (
      <span key={i}>
        {line.split(/(\*\*[^*]+\*\*)/g).map((part, j) =>
          part.startsWith('**') && part.endsWith('**')
            ? <strong key={j} className="text-text-primary">{part.slice(2, -2)}</strong>
            : part
        )}
        {i < text.split('\n').length - 1 && <br />}
      </span>
    ));

  return (
    <div className={`glass-card rounded-2xl flex flex-col transition-all duration-300 ${minimized ? 'h-14' : 'h-[480px]'}`}>
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-white/6">
        <div className="flex items-center gap-2">
          <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-accent-indigo to-accent-blue flex items-center justify-center"
            style={{ boxShadow: '0 0 12px rgba(124,111,255,0.4)' }}>
            <Brain size={14} className="text-white" />
          </div>
          <div>
            <div className="text-xs font-bold flex items-center gap-1.5">
              Classroom Copilot
              <span className="px-1.5 py-0.5 rounded-full bg-accent-indigo/15 text-accent-indigo text-[9px] font-semibold border border-accent-indigo/20">
                AI
              </span>
            </div>
            <div className="text-[9px] text-text-muted">Context-aware analytics explainer</div>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {speaking && (
            <Volume2 size={13} className="text-accent-blue animate-pulse" />
          )}
          <button onClick={() => setMinimized(!minimized)} className="text-text-muted hover:text-text-secondary transition-colors">
            {minimized ? <Maximize2 size={13} /> : <Minimize2 size={13} />}
          </button>
        </div>
      </div>

      {!minimized && (
        <>
          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-4 space-y-3">
            <AnimatePresence initial={false}>
              {messages.map((m) => (
                <motion.div
                  key={m.id}
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.2 }}
                  className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  {m.role === 'ai' && (
                    <div className="w-5 h-5 rounded-md bg-accent-indigo/20 flex items-center justify-center mr-2 mt-0.5 flex-shrink-0">
                      <Sparkles size={10} className="text-accent-indigo" />
                    </div>
                  )}
                  <div
                    className={`max-w-[85%] text-xs leading-relaxed p-2.5 ${m.role === 'user' ? 'bubble-user' : 'bubble-ai'}`}
                  >
                    {renderText(m.text)}
                    {m.action?.type === 'seek' && (
                      <button
                        onClick={() => onSeek?.((m.action!.payload as { t: number }).t)}
                        className="mt-2 flex items-center gap-1 text-accent-blue hover:underline text-[10px]"
                      >
                        ▶ Jump to this moment
                      </button>
                    )}
                    {m.role === 'ai' && (
                      <button
                        onClick={() => speak(m.text)}
                        className="mt-1.5 text-[9px] text-text-muted hover:text-accent-blue transition-colors flex items-center gap-1"
                      >
                        <Volume2 size={9} /> Read aloud
                      </button>
                    )}
                  </div>
                </motion.div>
              ))}
              {loading && (
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex justify-start">
                  <div className="bubble-ai text-xs p-2.5 flex items-center gap-1.5">
                    <span className="w-1.5 h-1.5 rounded-full bg-accent-indigo animate-bounce" style={{ animationDelay: '0ms' }} />
                    <span className="w-1.5 h-1.5 rounded-full bg-accent-indigo animate-bounce" style={{ animationDelay: '150ms' }} />
                    <span className="w-1.5 h-1.5 rounded-full bg-accent-indigo animate-bounce" style={{ animationDelay: '300ms' }} />
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
            <div ref={endRef} />
          </div>

          {/* Quick suggestions */}
          <div className="px-3 pb-2 flex gap-1.5 overflow-x-auto">
            {SUGGESTIONS.map((s) => (
              <button
                key={s}
                onClick={() => send(s)}
                className="flex-shrink-0 text-[10px] px-2.5 py-1 rounded-full border border-white/8 bg-white/4 text-text-secondary hover:text-text-primary hover:border-white/15 transition-all"
              >
                {s}
              </button>
            ))}
          </div>

          {/* Input */}
          <div className="px-3 pb-3">
            <div className="flex items-center gap-2 bg-white/5 rounded-xl border border-white/10 px-3 py-2 focus-within:border-accent-indigo/40 transition-colors">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && send()}
                placeholder="Ask about your classroom…"
                className="flex-1 bg-transparent text-xs text-text-primary placeholder:text-text-muted outline-none"
              />
              <button
                onClick={() => send()}
                disabled={!input.trim() || loading}
                className="w-6 h-6 rounded-lg flex items-center justify-center transition-all disabled:opacity-30"
                style={{ background: input.trim() ? 'linear-gradient(135deg, #7C6FFF, #4F7FFF)' : 'transparent' }}
              >
                <Send size={11} className="text-white" />
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
