'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import type { Session, AuthUser } from './types';
import { getSession } from './api';
import { useAuthStore } from './store';

// ─── Poll session until done ────────────────────────────────────────────────
export function usePollSession(sessionId: string | null, intervalMs = 2000) {
  const [session, setSession] = useState<Session | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!sessionId) return;
    let active = true;

    const poll = async () => {
      try {
        const s = await getSession(sessionId);
        if (!active) return;
        setSession(s);
        if (s.status === 'done' || s.status === 'error') return; // stop
        setTimeout(poll, intervalMs);
      } catch (err) {
        if (!active) return;
        setError(String(err));
      }
    };

    poll();
    return () => { active = false; };
  }, [sessionId, intervalMs]);

  return { session, error };
}

// ─── Auth state ────────────────────────────────────────────────────────────
export function useAuth(): { user: AuthUser | null; logout: () => void; isLoading: boolean } {
  const store = useAuthStore();

  useEffect(() => {
    store.hydrate();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return { user: store.user, logout: store.logout, isLoading: store.isLoading };
}

// ─── Animated counter ────────────────────────────────────────────────────────
export function useCountUp(to: number, duration = 1500) {
  const [value, setValue] = useState(0);
  const frameRef = useRef<number>(0);

  useEffect(() => {
    const start = performance.now();
    const tick = (now: number) => {
      const elapsed = now - start;
      const progress = Math.min(elapsed / duration, 1);
      // Ease out cubic
      const eased = 1 - Math.pow(1 - progress, 3);
      setValue(Math.round(eased * to));
      if (progress < 1) frameRef.current = requestAnimationFrame(tick);
    };
    frameRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frameRef.current);
  }, [to, duration]);

  return value;
}

// ─── Viewport intersection (scroll reveal) ──────────────────────────────────
export function useIntersection(threshold = 0.1) {
  const ref = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const obs = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting) { setVisible(true); obs.disconnect(); }
    }, { threshold });
    obs.observe(el);
    return () => obs.disconnect();
  }, [threshold]);

  return { ref, visible };
}

// ─── Live simulation ticker ─────────────────────────────────────────────────
export function useLiveTicker(intervalMs = 4000, active = true) {
  const [tick, setTick] = useState(0);
  useEffect(() => {
    if (!active) return;
    const id = setInterval(() => setTick((t) => t + 1), intervalMs);
    return () => clearInterval(id);
  }, [intervalMs, active]);
  return tick;
}
