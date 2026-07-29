'use client';

import { useState, useEffect, useRef } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { motion } from 'framer-motion';
import { Brain, Eye, EyeOff, ArrowRight, Shield, AlertCircle } from 'lucide-react';
import toast from 'react-hot-toast';
import { useAuthStore } from '@/lib/store';

// Turnstile is opt-in locally; keys are domain-bound and must not be hard-coded.
const TURNSTILE_ENABLED = process.env.NEXT_PUBLIC_TURNSTILE_ENABLED === 'true';
const TURNSTILE_SITE_KEY = process.env.NEXT_PUBLIC_TURNSTILE_SITE_KEY ?? '';

const DEMO_USERS = [
  { email: 'teacher@edulens.ai', password: 'demo123', role: 'Teacher', desc: 'Full classroom access' },
  { email: 'hod@edulens.ai',     password: 'demo123', role: 'HOD', desc: 'Department summaries + alerts' },
  { email: 'principal@edulens.ai', password: 'demo123', role: 'Principal', desc: 'Institution-level analytics' },
];

declare global {
  interface Window {
    turnstile?: {
      render: (container: string | HTMLElement, options: Record<string, unknown>) => string;
      reset: (widgetId: string) => void;
      getResponse: (widgetId: string) => string;
    };
    onTurnstileLoad?: () => void;
  }
}

export default function LoginPage() {
  const router = useRouter();
  const login = useAuthStore((s) => s.login);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPw, setShowPw] = useState(false);
  const [loading, setLoading] = useState(false);
  const [captchaToken, setCaptchaToken] = useState<string | null>(null);
  const captchaRef = useRef<string | null>(null);
  const captchaContainerRef = useRef<HTMLDivElement>(null);
  const [captchaError, setCaptchaError] = useState(false);
  const [loginError, setLoginError] = useState<string | null>(null);

  useEffect(() => {
    if (!TURNSTILE_ENABLED || !TURNSTILE_SITE_KEY) return;

    const checkTurnstile = () => {
      if (typeof window !== 'undefined' && window.turnstile && captchaContainerRef.current) {
        const id = window.turnstile.render(captchaContainerRef.current, {
          sitekey: TURNSTILE_SITE_KEY,
          theme: 'dark',
          callback: (token: string) => {
            setCaptchaToken(token);
            setCaptchaError(false);
          },
          'expired-callback': () => {
            setCaptchaToken(null);
          },
          'error-callback': () => {
            setCaptchaToken(null);
            setCaptchaError(true);
          },
        });
        captchaRef.current = id;
      } else {
        setTimeout(checkTurnstile, 200);
      }
    };
    checkTurnstile();
  }, []);

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoginError(null);
    setLoading(true);

    // Demo mode auto-fill check
    const match = DEMO_USERS.find((u) => u.email === email && u.password === password);
    if (match) {
      try {
        await login(email, password);
        toast.success(`Welcome back, ${match.role}!`);
        router.push('/monitor');
      } catch {
        toast.error('Backend is not running — start the server to continue');
      }
      setLoading(false);
      return;
    }

    try {
      await login(email, password, TURNSTILE_ENABLED ? captchaToken ?? undefined : undefined);
      toast.success('Logged in successfully');
      router.push('/monitor');
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Login failed';
      if (msg.includes('CAPTCHA') || msg.includes('captcha')) {
        setCaptchaError(true);
        if (captchaRef.current && window.turnstile) {
          window.turnstile.reset(captchaRef.current);
        }
        setCaptchaToken(null);
        setLoginError('CAPTCHA verification failed. Please try again.');
      } else if (msg.includes('401') || msg.includes('Invalid') || msg.includes('invalid')) {
        setLoginError('Invalid email or password');
      } else {
        setLoginError('Login failed — check the backend is running');
      }
    } finally {
      setLoading(false);
    }
  };

  const quickLogin = async (u: (typeof DEMO_USERS)[0]) => {
    setEmail(u.email);
    setPassword(u.password);
    setLoginError(null);
    setLoading(true);
    try {
      await login(u.email, u.password);
      toast.success(`Welcome back, ${u.role}!`);
      router.push('/monitor');
    } catch {
      toast.error('Backend is not running — start the server to continue');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-bg-primary p-6 relative overflow-hidden">
      <div className="absolute top-1/4 left-1/4 w-96 h-96 rounded-full blur-3xl opacity-8"
        style={{ background: 'radial-gradient(circle, #4F7FFF 0%, transparent 70%)' }} />
      <div className="absolute bottom-1/4 right-1/4 w-80 h-80 rounded-full blur-3xl opacity-6"
        style={{ background: 'radial-gradient(circle, #22D3A6 0%, transparent 70%)' }} />

      <motion.div
        initial={{ opacity: 0, y: 24 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="w-full max-w-md"
      >
        <div className="flex justify-center mb-8">
          <Link href="/" className="flex items-center gap-2.5">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-accent-blue to-accent-indigo flex items-center justify-center shadow-lg">
              <Brain size={20} className="text-white" />
            </div>
            <span className="text-xl font-black">Edu<span className="text-gradient">Lens</span></span>
          </Link>
        </div>

        <div className="glass-card rounded-2xl p-8 shadow-2xl">
          <h1 className="text-2xl font-bold mb-1 text-center">Welcome back</h1>
          <p className="text-text-secondary text-sm text-center mb-8">Sign in to your EduLens account</p>

          <form onSubmit={handleLogin} className="space-y-4">
            {loginError && (
              <motion.div
                initial={{ opacity: 0, y: -8 }}
                animate={{ opacity: 1, y: 0 }}
                className="flex items-center gap-2 px-3 py-2 rounded-lg bg-status-notEngaged/10 border border-status-notEngaged/20 text-xs text-status-notEngaged"
              >
                <AlertCircle size={13} />
                {loginError}
              </motion.div>
            )}

            <div>
              <label className="block text-xs font-medium text-text-secondary mb-1.5">Email</label>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="teacher@edulens.ai"
                required
                className="w-full px-4 py-2.5 rounded-xl bg-white/5 border border-white/10 text-sm text-text-primary placeholder:text-text-muted focus:outline-none focus:border-accent-blue/50 focus:bg-white/8 transition-all"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-text-secondary mb-1.5">Password</label>
              <div className="relative">
                <input
                  type={showPw ? 'text' : 'password'}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="••••••••"
                  required
                  className="w-full px-4 py-2.5 pr-10 rounded-xl bg-white/5 border border-white/10 text-sm text-text-primary placeholder:text-text-muted focus:outline-none focus:border-accent-blue/50 focus:bg-white/8 transition-all"
                />
                <button
                  type="button"
                  onClick={() => setShowPw(!showPw)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-text-muted hover:text-text-secondary transition-colors"
                >
                  {showPw ? <EyeOff size={15} /> : <Eye size={15} />}
                </button>
              </div>
            </div>

            <div ref={captchaContainerRef} className={`flex justify-center ${captchaError ? 'ring-1 ring-status-notEngaged/50 rounded-lg' : ''}`} />

            <button
              type="submit"
              disabled={loading}
              className="w-full flex items-center justify-center gap-2 py-3 rounded-xl font-semibold text-white text-sm transition-all disabled:opacity-60"
              style={{ background: 'linear-gradient(135deg, #4F7FFF, #7C6FFF)' }}
            >
              {loading ? (
                <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              ) : (
                <>Sign In <ArrowRight size={15} /></>
              )}
            </button>
          </form>

          <div className="mt-6 pt-6 border-t border-white/8">
            <p className="text-xs text-text-muted text-center mb-3 flex items-center justify-center gap-1.5">
              <Shield size={11} />
              Demo accounts — click to auto-fill & login
            </p>
            <div className="space-y-2">
              {DEMO_USERS.map((u) => (
                <button
                  key={u.role}
                  onClick={() => quickLogin(u)}
                  disabled={loading}
                  className="w-full flex items-center gap-3 px-3 py-2 rounded-lg border border-white/6 bg-white/3 hover:bg-white/6 hover:border-white/12 transition-all text-left disabled:opacity-50"
                >
                  <div className="w-7 h-7 rounded-full bg-gradient-to-br from-accent-blue to-accent-indigo flex items-center justify-center text-xs font-bold text-white flex-shrink-0">
                    {u.role[0]}
                  </div>
                  <div>
                    <div className="text-xs font-medium">{u.role}</div>
                    <div className="text-[10px] text-text-muted">{u.desc}</div>
                  </div>
                  <div className="ml-auto text-[10px] text-text-muted font-mono">{u.email}</div>
                </button>
              ))}
            </div>
          </div>
        </div>

        <p className="text-center text-xs text-text-muted mt-6">
          <Link href="/" className="hover:text-text-secondary transition-colors">← Back to home</Link>
        </p>
      </motion.div>
    </div>
  );
}
