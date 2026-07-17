'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import type { AlertEvent, TimelinePoint } from '@/lib/types';

interface TimelineReplayProps {
  timeline: TimelinePoint[];
  alerts: AlertEvent[];
  duration: number;  // seconds
  onSeek?: (t: number) => void;
}

function getMarkerColor(severity: AlertEvent['severity']) {
  return { soft: '#F59E0B', warning: '#FF8C00', critical: '#FF4E4E' }[severity];
}

function formatTime(sec: number) {
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return `${m}:${s.toString().padStart(2, '0')}`;
}

export function TimelineReplay({ timeline, alerts, duration, onSeek }: TimelineReplayProps) {
  const [currentT, setCurrentT] = useState(0);
  const [hovered, setHovered] = useState<AlertEvent | null>(null);

  const seek = (t: number) => {
    setCurrentT(t);
    onSeek?.(t);
  };

  const currentData = timeline.find((p) => p.t >= currentT) ?? timeline[timeline.length - 1];

  return (
    <div className="space-y-3">
      {/* Metric readout at current time */}
      <div className="flex items-center gap-4 text-xs">
        <span className="font-mono text-text-muted">{formatTime(currentT)}</span>
        <div className="flex items-center gap-1.5">
          <div className="w-2 h-2 rounded-full bg-accent-blue" />
          <span className="text-text-muted">Engagement:</span>
          <span className="font-bold text-accent-blue">{currentData?.engagement ?? 0}%</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-2 h-2 rounded-full bg-status-collab" />
          <span className="text-text-muted">Collab:</span>
          <span className="font-bold text-status-collab">{currentData?.collab ?? 0}%</span>
        </div>
        <div className="ml-auto flex items-center gap-1.5">
          <div className="w-2 h-2 rounded-full bg-status-engaged" />
          <span className="text-text-muted">Health:</span>
          <span className="font-bold text-status-engaged">{currentData?.health ?? 0}%</span>
        </div>
      </div>

      {/* Timeline scrubber */}
      <div className="relative">
        {/* Track */}
        <div className="relative h-8 flex items-center">
          <div className="absolute inset-x-0 top-1/2 -translate-y-1/2 h-1 rounded-full bg-white/8" />

          {/* Progress fill */}
          <div
            className="absolute top-1/2 -translate-y-1/2 h-1 rounded-full bg-gradient-to-r from-accent-blue to-accent-indigo"
            style={{ width: `${(currentT / duration) * 100}%` }}
          />

          {/* Engagement mini-chart overlay */}
          <svg className="absolute inset-x-0 top-0 h-8 w-full" preserveAspectRatio="none">
            <defs>
              <linearGradient id="tlGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#4F7FFF" stopOpacity={0.15} />
                <stop offset="100%" stopColor="#4F7FFF" stopOpacity={0} />
              </linearGradient>
            </defs>
            <polyline
              points={timeline.map((p) => `${(p.t / duration) * 100},${32 - (p.engagement / 100) * 28}`).join(' ')}
              fill="none"
              stroke="rgba(79,127,255,0.3)"
              strokeWidth={1}
            />
          </svg>

          {/* Alert markers */}
          {alerts.map((alert) => {
            const pct = (alert.timestamp / duration) * 100;
            const color = getMarkerColor(alert.severity);
            return (
              <div
                key={alert.id}
                className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-3 h-3 rounded-full cursor-pointer z-10 transition-transform hover:scale-150"
                style={{ left: `${pct}%`, background: color, boxShadow: `0 0 6px ${color}` }}
                onClick={() => seek(alert.timestamp)}
                onMouseEnter={() => setHovered(alert)}
                onMouseLeave={() => setHovered(null)}
              />
            );
          })}

          {/* Current position cursor */}
          <motion.div
            className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-4 h-4 rounded-full bg-white z-20 cursor-grab shadow-lg"
            style={{ left: `${(currentT / duration) * 100}%`, boxShadow: '0 0 12px rgba(255,255,255,0.3)' }}
            drag="x"
            dragConstraints={{ left: 0, right: 0 }}
            dragMomentum={false}
          />

          {/* Clickable overlay */}
          <div
            className="absolute inset-0 cursor-pointer z-30"
            onClick={(e) => {
              const rect = e.currentTarget.getBoundingClientRect();
              const pct = (e.clientX - rect.left) / rect.width;
              seek(Math.round(pct * duration));
            }}
          />
        </div>
      </div>

      {/* Time labels */}
      <div className="flex justify-between text-[10px] text-text-muted font-mono">
        {Array.from({ length: 7 }, (_, i) => (
          <span key={i}>{formatTime(Math.round((i / 6) * duration))}</span>
        ))}
      </div>

      {/* Alert tooltip */}
      {hovered && (
        <motion.div
          initial={{ opacity: 0, y: 4 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass-card rounded-lg px-3 py-2 text-xs border border-white/10"
        >
          <span className="font-medium" style={{ color: getMarkerColor(hovered.severity) }}>
            {hovered.severity.toUpperCase()}
          </span>
          {' '}{hovered.message} — <span className="text-text-muted font-mono">@{formatTime(hovered.timestamp)}</span>
        </motion.div>
      )}

      {/* Quick-jump markers */}
      <div className="flex flex-wrap gap-2">
        {alerts.map((a) => (
          <button
            key={a.id}
            onClick={() => seek(a.timestamp)}
            className="flex items-center gap-1.5 px-2 py-1 rounded-lg text-[10px] font-medium transition-all hover:opacity-90"
            style={{
              background: `${getMarkerColor(a.severity)}15`,
              border: `1px solid ${getMarkerColor(a.severity)}30`,
              color: getMarkerColor(a.severity),
            }}
          >
            <div className="w-1.5 h-1.5 rounded-full" style={{ background: getMarkerColor(a.severity) }} />
            {formatTime(a.timestamp)} — {a.message.split(' ').slice(0, 3).join(' ')}…
          </button>
        ))}
      </div>
    </div>
  );
}
