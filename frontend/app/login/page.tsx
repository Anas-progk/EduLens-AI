'use client';

import { useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { motion } from 'framer-motion';
import { Brain, Eye, EyeOff, ArrowRight, Shield } from 'lucide-react';
import toast from 'react-hot-toast';

const DEMO_USERS = [
  { email: 'teacher@edulens.ai', password: 'demo123', role: 'Teacher', desc: 'Full classroom access' },
  { email: 'hod@edulens.ai',     password: 'demo123', role: 'HOD', desc: 'Department summaries + alerts' },
  { email: 'principal@edulens.ai', password: 'demo123', role: 'Principal', desc: 'Institution-level analytics' },
];

export default function LoginPage() {
  const router = useRouter();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPw, setShowPw] = useState(false);
  const [loading, setLoading] = useState(false);

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
      // Demo mode — just check against local users
      const match = DEMO_USERS.find((u) => u.email === email && u.password === password);
      if (match) {
        if (typeof window !== 'undefined') {
          localStorage.setItem('edulens_token', 'demo-token');
          localStorage.setItem('edulens_user', JSON.stringify({ email, role: match.role }));
        }
        toast.success(`Welcome back, ${match.role}!`);
        setTimeout(() => router.push('/monitor'), 800);
      } else {
        // Try actual API
        const res = await fetch('/api/auth/login', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ email, password }),
        });
        if (res.ok) {
          const data = await res.json();
          localStorage.setItem('edulens_token', data.token);
          localStorage.setItem('edulens_user', JSON.stringify(data.user));
          toast.success('Logged in successfully');
          router.push('/monitor');
        } else {
          toast.error('Invalid credentials');
        }
      }
    } catch {
      toast.error('Login failed — check the backend is running');
    } finally {
      setLoading(false);
    }
  };

  const quickLogin = (u: (typeof DEMO_USERS)[0]) => {
    setEmail(u.email);
    setPassword(u.password);
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-bg-primary p-6 relative overflow-hidden">
      {/* Background orbs */}
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
        {/* Logo */}
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

          {/* Demo accounts */}
          <div className="mt-6 pt-6 border-t border-white/8">
            <p className="text-xs text-text-muted text-center mb-3 flex items-center justify-center gap-1.5">
              <Shield size={11} />
              Demo accounts — click to auto-fill
            </p>
            <div className="space-y-2">
              {DEMO_USERS.map((u) => (
                <button
                  key={u.role}
                  onClick={() => quickLogin(u)}
                  className="w-full flex items-center gap-3 px-3 py-2 rounded-lg border border-white/6 bg-white/3 hover:bg-white/6 hover:border-white/12 transition-all text-left"
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
