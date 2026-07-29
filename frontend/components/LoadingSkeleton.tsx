'use client';

export function SkeletonCard({ lines = 3 }: { lines?: number }) {
  return (
    <div className="glass-card rounded-xl p-4 space-y-3 animate-pulse">
      <div className="w-9 h-9 rounded-lg bg-white/6" />
      {Array.from({ length: lines }).map((_, i) => (
        <div key={i} className={`h-3 rounded-full bg-white/6 ${i === 0 ? 'w-1/2' : i === 1 ? 'w-2/3' : 'w-1/3'}`} />
      ))}
    </div>
  );
}

export function SkeletonChart() {
  return (
    <div className="glass-card rounded-2xl p-5 space-y-4 animate-pulse">
      <div className="flex gap-2">
        {Array.from({ length: 3 }).map((_, i) => (
          <div key={i} className="h-8 rounded-lg bg-white/6 flex-1" />
        ))}
      </div>
      <div className="h-48 rounded-xl bg-white/6" />
      <div className="flex justify-between">
        {Array.from({ length: 5 }).map((_, i) => (
          <div key={i} className="h-4 w-8 rounded-full bg-white/6" />
        ))}
      </div>
    </div>
  );
}

export function SkeletonTimeline() {
  return (
    <div className="glass-card rounded-2xl p-5 space-y-3 animate-pulse">
      <div className="h-4 w-40 rounded-full bg-white/6" />
      <div className="h-16 rounded-xl bg-white/6" />
    </div>
  );
}
