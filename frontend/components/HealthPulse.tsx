'use client';

import { motion } from 'framer-motion';

interface HealthPulseProps {
  score: number;  // 0-100
  label?: string;
  size?: 'sm' | 'md' | 'lg';
}

function getColor(score: number) {
  if (score >= 70) return { stroke: '#22D3A6', fill: 'rgba(34,211,166,0.08)', text: '#22D3A6', label: 'Healthy' };
  if (score >= 45) return { stroke: '#F59E0B', fill: 'rgba(245,158,11,0.08)', text: '#F59E0B', label: 'Moderate' };
  return { stroke: '#FF4E4E', fill: 'rgba(255,78,78,0.08)', text: '#FF4E4E', label: 'Critical' };
}

export function HealthPulse({ score, label, size = 'md' }: HealthPulseProps) {
  const dims = { sm: 100, md: 140, lg: 180 }[size];
  const r = (dims / 2) - 12;
  const cx = dims / 2;
  const cy = dims / 2;
  const circumference = 2 * Math.PI * r;
  const strokeDash = (score / 100) * circumference;
  const color = getColor(score);

  return (
    <div className="flex flex-col items-center gap-3">
      <div className="relative" style={{ width: dims, height: dims }}>
        {/* Pulse ring (only when healthy) */}
        {score >= 70 && (
          <div
            className="absolute inset-0 rounded-full pulse-ring"
            style={{ border: `2px solid ${color.stroke}`, opacity: 0.3 }}
          />
        )}

        {/* SVG gauge */}
        <svg width={dims} height={dims} viewBox={`0 0 ${dims} ${dims}`}>
          {/* Background track */}
          <circle
            cx={cx} cy={cy} r={r}
            fill={color.fill}
            stroke="rgba(148,163,184,0.1)"
            strokeWidth={8}
          />
          {/* Animated progress arc */}
          <motion.circle
            cx={cx} cy={cy} r={r}
            fill="none"
            stroke={color.stroke}
            strokeWidth={8}
            strokeLinecap="round"
            strokeDasharray={circumference}
            initial={{ strokeDashoffset: circumference }}
            animate={{ strokeDashoffset: circumference - strokeDash }}
            transition={{ duration: 1.2, ease: 'easeOut' }}
            transform={`rotate(-90 ${cx} ${cy})`}
            style={{ filter: `drop-shadow(0 0 6px ${color.stroke})` }}
          />
          {/* Score text */}
          <text x={cx} y={cy - 6} textAnchor="middle" dominantBaseline="middle"
            style={{ fill: color.text, fontSize: dims === 100 ? 22 : dims === 140 ? 28 : 36, fontWeight: 800, fontFamily: 'Inter, sans-serif' }}>
            {score}
          </text>
          <text x={cx} y={cy + (dims === 100 ? 14 : 18)} textAnchor="middle" dominantBaseline="middle"
            style={{ fill: '#94A3B8', fontSize: 11, fontFamily: 'Inter, sans-serif' }}>
            {color.label}
          </text>
        </svg>
      </div>
      {label && <p className="text-xs text-text-muted text-center">{label}</p>}
    </div>
  );
}

// Mini bar version for sidebar stats
export function MiniStat({ label, value, max = 100, color }: {
  label: string; value: number; max?: number; color: string;
}) {
  const pct = (value / max) * 100;
  return (
    <div className="space-y-1.5">
      <div className="flex justify-between items-center">
        <span className="text-xs text-text-secondary">{label}</span>
        <span className="text-xs font-bold" style={{ color }}>{value}{max === 100 ? '%' : ''}</span>
      </div>
      <div className="h-1.5 rounded-full bg-white/6 overflow-hidden">
        <motion.div
          className="h-full rounded-full"
          style={{ background: color }}
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 1, ease: 'easeOut' }}
        />
      </div>
    </div>
  );
}
