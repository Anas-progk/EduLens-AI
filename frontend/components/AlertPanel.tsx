'use client';

import { motion, AnimatePresence } from 'framer-motion';
import { Bell, AlertTriangle, X, CheckCircle2 } from 'lucide-react';
import type { AlertEvent } from '@/lib/types';

function formatTime(sec: number) {
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return `${m}:${s.toString().padStart(2, '0')}`;
}

const SEVERITY_CONFIG = {
  soft: {
    icon: Bell,
    color: '#F59E0B',
    bg: 'rgba(245,158,11,0.08)',
    border: 'rgba(245,158,11,0.25)',
    label: 'Soft',
  },
  warning: {
    icon: AlertTriangle,
    color: '#FF8C00',
    bg: 'rgba(255,140,0,0.08)',
    border: 'rgba(255,140,0,0.25)',
    label: 'Warning',
  },
  critical: {
    icon: AlertTriangle,
    color: '#FF4E4E',
    bg: 'rgba(255,78,78,0.08)',
    border: 'rgba(255,78,78,0.3)',
    label: 'Critical',
  },
};

interface AlertPanelProps {
  alerts: AlertEvent[];
  onDismiss?: (id: string) => void;
  onSeek?: (timestamp: number) => void;
  compact?: boolean;
}

export function AlertPanel({ alerts, onDismiss, onSeek, compact = false }: AlertPanelProps) {
  const active = alerts.filter((a) => !a.resolved);

  if (active.length === 0) {
    return (
      <div className="flex flex-col items-center gap-2 py-6 text-center">
        <CheckCircle2 size={20} className="text-status-engaged" />
        <p className="text-xs text-text-muted">No active alerts</p>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <AnimatePresence initial={false}>
        {active.map((alert) => {
          const cfg = SEVERITY_CONFIG[alert.severity];
          const Icon = cfg.icon;

          return (
            <motion.div
              key={alert.id}
              initial={{ opacity: 0, x: 20, height: 0 }}
              animate={{ opacity: 1, x: 0, height: 'auto' }}
              exit={{ opacity: 0, x: 20, height: 0 }}
              transition={{ duration: 0.2 }}
              className="rounded-xl p-3 flex items-start gap-3 cursor-pointer"
              style={{ background: cfg.bg, border: `1px solid ${cfg.border}` }}
              onClick={() => onSeek?.(alert.timestamp)}
            >
              <div
                className="w-6 h-6 rounded-lg flex items-center justify-center flex-shrink-0 mt-0.5"
                style={{ background: `${cfg.color}18` }}
              >
                <Icon size={12} style={{ color: cfg.color }} />
              </div>

              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-0.5">
                  <span className="text-[10px] font-bold uppercase tracking-wider" style={{ color: cfg.color }}>
                    {cfg.label}
                  </span>
                  {!compact && (
                    <span className="text-[10px] text-text-muted font-mono">@{formatTime(alert.timestamp)}</span>
                  )}
                </div>
                <p className="text-xs text-text-secondary leading-relaxed">{alert.message}</p>
                {!compact && onSeek && (
                  <button
                    className="text-[10px] mt-1 hover:underline"
                    style={{ color: cfg.color }}
                  >
                    → Click to jump to this moment
                  </button>
                )}
              </div>

              {onDismiss && (
                <button
                  onClick={(e) => { e.stopPropagation(); onDismiss(alert.id); }}
                  className="flex-shrink-0 opacity-50 hover:opacity-100 transition-opacity"
                >
                  <X size={12} className="text-text-muted" />
                </button>
              )}
            </motion.div>
          );
        })}
      </AnimatePresence>
    </div>
  );
}

// Summary badge for navbar
export function AlertBadge({ count }: { count: number }) {
  if (count === 0) return null;
  return (
    <motion.span
      initial={{ scale: 0 }}
      animate={{ scale: 1 }}
      className="absolute -top-1 -right-1 w-4 h-4 rounded-full bg-status-notEngaged text-white text-[9px] font-bold flex items-center justify-center"
    >
      {count > 9 ? '9+' : count}
    </motion.span>
  );
}
