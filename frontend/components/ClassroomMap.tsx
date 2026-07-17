'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { StudentState } from '@/lib/types';

function getStatusColor(label: StudentState['label'], prob: number) {
  if (label === 'Unknown') return { bg: '#1a2234', border: 'rgba(148,163,184,0.2)', dot: '#64748B', text: '#94A3B8' };
  if (label === 'Engaged')  return { bg: 'rgba(34,211,166,0.08)', border: 'rgba(34,211,166,0.3)', dot: '#22D3A6', text: '#22D3A6' };
  return { bg: 'rgba(255,78,78,0.08)', border: 'rgba(255,78,78,0.3)', dot: '#FF4E4E', text: '#FF4E4E' };
}

function getCollabIcon(label: StudentState['collabLabel']) {
  if (label === 'Collaborative') return '🔗';
  if (label === 'Not Collaborative') return '⛓️';
  return '❓';
}

interface ClassroomMapProps {
  students: StudentState[];
  rows?: number;
  cols?: number;
  showCollab?: boolean;
}

export function ClassroomMap({ students, rows = 3, cols = 3, showCollab = true }: ClassroomMapProps) {
  const [hovered, setHovered] = useState<string | null>(null);

  // Fill grid
  const grid: (StudentState | null)[][] = Array.from({ length: rows }, (_, r) =>
    Array.from({ length: cols }, (_, c) =>
      students.find((s) => s.row === r && s.col === c) ?? null
    )
  );

  return (
    <div className="space-y-3">
      {/* Teacher row */}
      <div className="flex justify-center">
        <div className="px-6 py-2 rounded-xl border border-accent-blue/20 bg-accent-blue/6 text-xs text-accent-blue font-medium">
          📋 Teacher / Whiteboard
        </div>
      </div>

      {/* Grid */}
      <div className="grid gap-3" style={{ gridTemplateColumns: `repeat(${cols}, 1fr)` }}>
        {grid.flat().map((student, idx) => {
          const r = Math.floor(idx / cols);
          const c = idx % cols;
          if (!student) {
            return (
              <div key={`empty-${r}-${c}`} className="aspect-square rounded-xl border border-dashed border-white/6 opacity-30" />
            );
          }
          const colors = getStatusColor(student.label, student.engagementProb);
          const isHovered = hovered === student.id;

          return (
            <motion.div
              key={student.id}
              initial={{ scale: 0, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ delay: idx * 0.05, type: 'spring', stiffness: 300 }}
              whileHover={{ scale: 1.06 }}
              onHoverStart={() => setHovered(student.id)}
              onHoverEnd={() => setHovered(null)}
              className="aspect-square rounded-xl flex flex-col items-center justify-center gap-1.5 cursor-pointer relative transition-all"
              style={{
                background: colors.bg,
                border: `1px solid ${colors.border}`,
                boxShadow: isHovered ? `0 0 20px ${colors.dot}40` : undefined,
              }}
            >
              {/* Status dot */}
              <div className="relative">
                <div className="w-3 h-3 rounded-full" style={{ background: colors.dot }} />
                {student.label === 'Not Engaged' && (
                  <div className="absolute inset-0 w-3 h-3 rounded-full animate-ping-slow" style={{ background: colors.dot, opacity: 0.3 }} />
                )}
              </div>

              {/* ID */}
              <div className="text-[10px] font-bold" style={{ color: colors.text }}>{student.id}</div>

              {/* Collab icon */}
              {showCollab && (
                <div className="text-[9px]">{getCollabIcon(student.collabLabel)}</div>
              )}

              {/* Probability bar */}
              <div className="w-4/5 h-0.5 rounded-full bg-white/10 overflow-hidden">
                <div
                  className="h-full rounded-full transition-all duration-500"
                  style={{ width: `${student.engagementProb * 100}%`, background: colors.dot }}
                />
              </div>

              {/* Tooltip */}
              <AnimatePresence>
                {isHovered && (
                  <motion.div
                    initial={{ opacity: 0, scale: 0.9, y: 4 }}
                    animate={{ opacity: 1, scale: 1, y: 0 }}
                    exit={{ opacity: 0, scale: 0.9, y: 4 }}
                    className="absolute -top-20 left-1/2 -translate-x-1/2 z-20 glass-card rounded-lg px-3 py-2 text-[10px] space-y-1 shadow-xl w-32 pointer-events-none"
                  >
                    <div className="font-semibold" style={{ color: colors.text }}>{student.label}</div>
                    <div className="text-text-muted">Prob: {(student.engagementProb * 100).toFixed(0)}%</div>
                    <div className="text-text-muted">{student.collabLabel}</div>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          );
        })}
      </div>

      {/* Legend */}
      <div className="flex items-center justify-center gap-4 text-[10px] text-text-muted pt-1">
        {[
          { color: '#22D3A6', label: 'Engaged' },
          { color: '#FF4E4E', label: 'Not Engaged' },
          { color: '#64748B', label: 'Unknown' },
        ].map((l) => (
          <div key={l.label} className="flex items-center gap-1.5">
            <div className="w-2 h-2 rounded-full" style={{ background: l.color }} />
            {l.label}
          </div>
        ))}
      </div>
    </div>
  );
}
