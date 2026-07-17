'use client';

import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine, Legend, Area, AreaChart
} from 'recharts';
import type { TimelinePoint } from '@/lib/types';

interface EngagementChartProps {
  data: TimelinePoint[];
  onSeek?: (t: number) => void;
  showCollab?: boolean;
  height?: number;
}

function formatTime(sec: number) {
  const m = Math.floor(sec / 60);
  return `${m}m`;
}

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null;
  const t = typeof label === 'number' ? label : 0;
  return (
    <div className="glass-card rounded-xl px-3 py-2.5 text-xs space-y-1.5 shadow-xl border border-white/10">
      <div className="text-text-muted font-medium mb-1">
        {Math.floor(t / 60)}:{String(t % 60).padStart(2, '0')} min
      </div>
      {payload.map((p: any) => (
        <div key={p.dataKey} className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full" style={{ background: p.color }} />
          <span className="text-text-muted capitalize">{p.name}:</span>
          <span className="font-bold" style={{ color: p.color }}>{Math.round(p.value)}%</span>
        </div>
      ))}
    </div>
  );
};

export function EngagementChart({ data, onSeek, showCollab = true, height = 200 }: EngagementChartProps) {
  // Find the drop zone (engagement < 50 for 3+ consecutive points)
  const dropZones = data.reduce<number[]>((acc, p, i) => {
    if (p.engagement < 50 && i > 0) acc.push(p.t);
    return acc;
  }, []);

  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={data} margin={{ top: 5, right: 10, bottom: 5, left: -20 }}
        onClick={(e) => e?.activePayload && onSeek?.(e.activePayload[0]?.payload?.t)}>
        <defs>
          <linearGradient id="engGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#4F7FFF" stopOpacity={0.2} />
            <stop offset="95%" stopColor="#4F7FFF" stopOpacity={0} />
          </linearGradient>
          <linearGradient id="collGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#38BDF8" stopOpacity={0.2} />
            <stop offset="95%" stopColor="#38BDF8" stopOpacity={0} />
          </linearGradient>
          <linearGradient id="healthGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#22D3A6" stopOpacity={0.15} />
            <stop offset="95%" stopColor="#22D3A6" stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.06)" />
        <XAxis
          dataKey="t"
          tickFormatter={formatTime}
          tick={{ fill: '#64748B', fontSize: 10 }}
          axisLine={false}
          tickLine={false}
        />
        <YAxis
          domain={[0, 100]}
          tick={{ fill: '#64748B', fontSize: 10 }}
          axisLine={false}
          tickLine={false}
        />
        <Tooltip content={<CustomTooltip />} />
        {/* Critical threshold line */}
        <ReferenceLine y={50} stroke="rgba(255,78,78,0.3)" strokeDasharray="4 4" label={{ value: 'Critical', fill: '#FF4E4E', fontSize: 9, position: 'right' }} />

        <Area type="monotone" dataKey="engagement" name="Engagement" stroke="#4F7FFF" strokeWidth={2} fill="url(#engGrad)" dot={false} activeDot={{ r: 4, strokeWidth: 0 }} />
        {showCollab && (
          <Area type="monotone" dataKey="collab" name="Collaboration" stroke="#38BDF8" strokeWidth={2} fill="url(#collGrad)" dot={false} activeDot={{ r: 4, strokeWidth: 0 }} />
        )}
        <Area type="monotone" dataKey="health" name="Health" stroke="#22D3A6" strokeWidth={1.5} strokeDasharray="4 3" fill="url(#healthGrad)" dot={false} activeDot={{ r: 3, strokeWidth: 0 }} />
        <Legend
          iconType="circle"
          iconSize={7}
          wrapperStyle={{ fontSize: 11, paddingTop: 8, color: '#94A3B8' }}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}
