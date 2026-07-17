'use client';

import { useEffect, useRef, useState } from 'react';
import Link from 'next/link';
import { motion, useInView, useMotionValue, useSpring } from 'framer-motion';
import {
  Brain, Eye, Users, Bell, BarChart3, Shield,
  ArrowRight, Play, ChevronRight, Zap, Activity,
  Camera, FileText, MessageSquare, Globe
} from 'lucide-react';

// ─── Animated counter ─────────────────────────────────────────────────────────
function Counter({ to, decimals = 0, suffix = '' }: { to: number; decimals?: number; suffix?: string }) {
  const ref = useRef<HTMLSpanElement>(null);
  const inView = useInView(ref, { once: true });
  const motionVal = useMotionValue(0);
  const spring = useSpring(motionVal, { duration: 2000, bounce: 0 });

  useEffect(() => {
    if (inView) motionVal.set(to);
  }, [inView, motionVal, to]);

  useEffect(() => {
    return spring.on('change', (v) => {
      if (ref.current) ref.current.textContent = v.toFixed(decimals) + suffix;
    });
  }, [spring, decimals, suffix]);

  return <span ref={ref}>0{suffix}</span>;
}

// ─── Particle canvas ───────────────────────────────────────────────────────────
function ParticleCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d')!;
    let raf: number;
    let w = 0, h = 0;

    const resize = () => {
      w = canvas.width = canvas.offsetWidth;
      h = canvas.height = canvas.offsetHeight;
    };

    const N = 60;
    const particles = Array.from({ length: N }, () => ({
      x: Math.random(), y: Math.random(),
      vx: (Math.random() - 0.5) * 0.0003,
      vy: (Math.random() - 0.5) * 0.0003,
      r: Math.random() * 1.5 + 0.5,
      a: Math.random() * 0.4 + 0.1,
    }));

    const draw = () => {
      ctx.clearRect(0, 0, w, h);
      for (const p of particles) {
        p.x = (p.x + p.vx + 1) % 1;
        p.y = (p.y + p.vy + 1) % 1;
        ctx.beginPath();
        ctx.arc(p.x * w, p.y * h, p.r, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(79,127,255,${p.a})`;
        ctx.fill();
      }
      // Draw connections
      for (let i = 0; i < N; i++) {
        for (let j = i + 1; j < N; j++) {
          const dx = (particles[i].x - particles[j].x) * w;
          const dy = (particles[i].y - particles[j].y) * h;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < 120) {
            ctx.beginPath();
            ctx.moveTo(particles[i].x * w, particles[i].y * h);
            ctx.lineTo(particles[j].x * w, particles[j].y * h);
            ctx.strokeStyle = `rgba(79,127,255,${0.15 * (1 - dist / 120)})`;
            ctx.lineWidth = 0.5;
            ctx.stroke();
          }
        }
      }
      raf = requestAnimationFrame(draw);
    };

    resize();
    window.addEventListener('resize', resize);
    draw();
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener('resize', resize);
    };
  }, []);

  return <canvas ref={canvasRef} className="absolute inset-0 w-full h-full opacity-40" />;
}

// ─── Dashboard mockup component ────────────────────────────────────────────────
function DashboardMockup() {
  return (
    <div className="relative w-full max-w-xl mx-auto">
      {/* Glow behind */}
      <div className="absolute inset-0 blur-3xl opacity-20 bg-gradient-to-br from-blue-500 via-indigo-500 to-teal-500 rounded-3xl" />

      {/* Browser window */}
      <motion.div
        animate={{ y: [0, -10, 0] }}
        transition={{ duration: 5, repeat: Infinity, ease: 'easeInOut' }}
        className="relative rounded-2xl overflow-hidden shadow-2xl border border-white/10"
        style={{ background: 'rgba(10,18,40,0.95)' }}
      >
        {/* Chrome bar */}
        <div className="flex items-center gap-2 px-4 py-3 border-b border-white/8 bg-white/3">
          <div className="w-3 h-3 rounded-full bg-red-400/70" />
          <div className="w-3 h-3 rounded-full bg-yellow-400/70" />
          <div className="w-3 h-3 rounded-full bg-green-400/70" />
          <div className="flex-1 mx-3 h-5 rounded bg-white/5 text-xs text-text-muted flex items-center px-3 font-mono">
            localhost:3000/dashboard
          </div>
        </div>

        {/* App body */}
        <div className="flex h-56">
          {/* Sidebar */}
          <div className="w-14 border-r border-white/6 flex flex-col items-center gap-3 pt-4 bg-white/2">
            {[Brain, Activity, BarChart3, Bell, Shield].map((Icon, i) => (
              <div key={i} className={`p-2 rounded-lg ${i === 0 ? 'bg-accent-blue/20 text-accent-blue' : 'text-text-muted hover:text-white'}`}>
                <Icon size={14} />
              </div>
            ))}
          </div>

          {/* Content */}
          <div className="flex-1 p-4 overflow-hidden">
            {/* Top stat row */}
            <div className="grid grid-cols-3 gap-2 mb-3">
              {[
                { label: 'Health', val: '84%', color: '#22D3A6' },
                { label: 'Engaged', val: '87%', color: '#4F7FFF' },
                { label: 'Collab', val: '76%', color: '#38BDF8' },
              ].map((s) => (
                <div key={s.label} className="bg-white/5 rounded-lg p-2">
                  <div className="text-xs text-text-muted mb-1">{s.label}</div>
                  <div className="text-sm font-bold" style={{ color: s.color }}>{s.val}</div>
                </div>
              ))}
            </div>

            {/* Fake chart bars */}
            <div className="flex items-end gap-1 h-16 mb-3">
              {[55, 72, 48, 88, 65, 79, 91, 60, 83, 70, 55, 85].map((h, i) => (
                <motion.div
                  key={i}
                  className="flex-1 rounded-sm"
                  style={{ background: `rgba(79,127,255,${0.3 + h / 200})` }}
                  initial={{ height: 0 }}
                  animate={{ height: `${h}%` }}
                  transition={{ delay: i * 0.06, duration: 0.6, ease: 'easeOut' }}
                />
              ))}
            </div>

            {/* Student rows */}
            <div className="space-y-1.5">
              {[
                { id: 'ST-01', label: 'Engaged', color: '#22D3A6' },
                { id: 'ST-02', label: 'Not Engaged', color: '#FF4E4E' },
                { id: 'ST-03', label: 'Engaged', color: '#22D3A6' },
              ].map((s) => (
                <div key={s.id} className="flex items-center gap-2">
                  <div className="w-4 h-4 rounded-full bg-white/10 text-[8px] flex items-center justify-center font-mono text-text-muted">{s.id.split('-')[1]}</div>
                  <div className="flex-1 h-1.5 bg-white/5 rounded-full overflow-hidden">
                    <motion.div
                      className="h-full rounded-full"
                      style={{ background: s.color, width: s.label === 'Engaged' ? '80%' : '35%' }}
                      initial={{ width: 0 }}
                      animate={{ width: s.label === 'Engaged' ? '80%' : '35%' }}
                      transition={{ delay: 0.8, duration: 0.8, ease: 'easeOut' }}
                    />
                  </div>
                  <div className="text-[9px] font-medium" style={{ color: s.color }}>{s.label}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </motion.div>

      {/* Floating alert card */}
      <motion.div
        className="absolute -bottom-4 -right-6 glass-card rounded-xl px-3 py-2 shadow-xl border border-yellow-400/20"
        animate={{ y: [0, -6, 0] }}
        transition={{ duration: 3, repeat: Infinity, ease: 'easeInOut', delay: 1.5 }}
      >
        <div className="flex items-center gap-2">
          <Bell size={12} className="text-yellow-400" />
          <span className="text-xs text-text-secondary">ST-02 disengaged <span className="text-yellow-400">5m</span></span>
        </div>
      </motion.div>

      {/* Floating copilot card */}
      <motion.div
        className="absolute -top-4 -left-6 glass-card rounded-xl px-3 py-2 shadow-xl border border-accent-blue/20"
        animate={{ y: [0, -5, 0] }}
        transition={{ duration: 4, repeat: Infinity, ease: 'easeInOut', delay: 0.5 }}
      >
        <div className="flex items-center gap-2">
          <Brain size={12} className="text-accent-blue" />
          <span className="text-xs text-text-secondary">Copilot: <span className="text-accent-blue">Why is room red?</span></span>
        </div>
      </motion.div>
    </div>
  );
}

// ─── Feature card ──────────────────────────────────────────────────────────────
const features = [
  {
    icon: Eye,
    title: 'Engagement Detection',
    desc: 'Swin Transformer + Temporal Transformer classifies Engaged vs Not Engaged with 0.73 macro-F1 on unseen classrooms.',
    color: '#4F7FFF',
    tag: 'Phase 1',
  },
  {
    icon: Users,
    title: 'Collaboration Analysis',
    desc: 'Group-level collaboration verdict via relational signals + mutual gaze. Session-level macro-F1 0.764.',
    color: '#38BDF8',
    tag: 'Phase 2',
  },
  {
    icon: Brain,
    title: 'AI Classroom Copilot',
    desc: 'Ask "Why is Room 201 red?" and get instant analytics-aware explanations with one-click timeline replay.',
    color: '#7C6FFF',
    tag: 'Standout',
    highlight: true,
  },
  {
    icon: Activity,
    title: 'Digital Classroom Twin',
    desc: 'Live grid showing every student\'s engagement state. Instant spatial awareness with animated health pulse.',
    color: '#22D3A6',
    tag: 'Live',
  },
  {
    icon: Bell,
    title: 'Smart Alert System',
    desc: 'Tiered alerts: 3-min soft → 5-min warning → 10-min critical. HOD escalation when whole class stays disengaged.',
    color: '#F59E0B',
    tag: 'Alerts',
  },
  {
    icon: FileText,
    title: 'AI Session Reports',
    desc: 'One-click PDF reports with engagement trends, collaboration heatmap, alerts, and teaching recommendations.',
    color: '#22D3A6',
    tag: 'Reports',
  },
];

const steps = [
  {
    n: '01',
    title: 'Upload Classroom Video',
    desc: 'Drag-drop any classroom recording. The pipeline auto-detects and tracks each student with persistent IDs.',
    icon: Camera,
  },
  {
    n: '02',
    title: 'AI Analysis Pipeline',
    desc: 'Swin-Tiny backbone extracts temporal features. Group-level gaze signals detect collaboration in real-time.',
    icon: Brain,
  },
  {
    n: '03',
    title: 'Live Analytics + Copilot',
    desc: 'Explore the dashboard, ask your AI Copilot questions, replay problem moments, and generate reports.',
    icon: BarChart3,
  },
];

// ─── Main page ─────────────────────────────────────────────────────────────────
const fadeUp = {
  hidden: { opacity: 0, y: 24 },
  show: { opacity: 1, y: 0 },
};
const stagger = { show: { transition: { staggerChildren: 0.1 } } };

export default function LandingPage() {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 20);
    window.addEventListener('scroll', onScroll);
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  return (
    <div className="min-h-screen bg-bg-primary overflow-x-hidden">
      {/* ── Navbar ── */}
      <header
        className={`fixed top-0 inset-x-0 z-50 transition-all duration-300 ${
          scrolled ? 'glass-card border-b border-white/5 py-3' : 'py-5'
        }`}
      >
        <div className="max-w-7xl mx-auto px-6 flex items-center justify-between">
          {/* Logo */}
          <Link href="/" className="flex items-center gap-2.5 group">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-accent-blue to-accent-indigo flex items-center justify-center shadow-lg">
              <Brain size={16} className="text-white" />
            </div>
            <span className="text-base font-bold tracking-tight">
              Edu<span className="text-gradient">Lens</span>
            </span>
          </Link>

          {/* Nav links */}
          <nav className="hidden md:flex items-center gap-6 text-sm text-text-secondary">
            {['Features', 'How It Works', 'Analytics', 'Dashboard'].map((item) => (
              <Link
                key={item}
                href={item === 'Dashboard' ? '/dashboard' : `#${item.toLowerCase().replace(' ', '-')}`}
                className="hover:text-text-primary transition-colors"
              >
                {item}
              </Link>
            ))}
          </nav>

          {/* CTAs */}
          <div className="flex items-center gap-3">
            <Link href="/login" className="hidden md:block text-sm text-text-secondary hover:text-text-primary transition-colors">
              Sign In
            </Link>
            <Link
              href="/monitor"
              className="flex items-center gap-1.5 px-4 py-2 rounded-lg text-sm font-medium text-white"
              style={{ background: 'linear-gradient(135deg, #4F7FFF, #7C6FFF)' }}
            >
              Live Demo
              <ArrowRight size={14} />
            </Link>
          </div>
        </div>
      </header>

      {/* ── Hero section ── */}
      <section className="relative min-h-screen flex items-center pt-24 pb-20 overflow-hidden">
        {/* Background */}
        <div className="absolute inset-0">
          <ParticleCanvas />
          {/* Gradient orbs */}
          <div className="absolute top-1/4 -left-1/4 w-[600px] h-[600px] rounded-full blur-3xl opacity-10"
            style={{ background: 'radial-gradient(circle, #4F7FFF 0%, transparent 70%)' }} />
          <div className="absolute bottom-0 right-0 w-[500px] h-[500px] rounded-full blur-3xl opacity-8"
            style={{ background: 'radial-gradient(circle, #7C6FFF 0%, transparent 70%)' }} />
          <div className="absolute top-3/4 left-1/2 w-[400px] h-[400px] rounded-full blur-3xl opacity-6"
            style={{ background: 'radial-gradient(circle, #22D3A6 0%, transparent 70%)' }} />
        </div>

        <div className="relative max-w-7xl mx-auto px-6 grid lg:grid-cols-2 gap-16 items-center">
          {/* Text side */}
          <motion.div
            initial="hidden"
            animate="show"
            variants={stagger}
          >
            {/* Pill badge */}
            <motion.div variants={fadeUp} className="inline-flex items-center gap-2 mb-6">
              <div className="flex items-center gap-2 px-3 py-1.5 rounded-full border border-accent-blue/30 bg-accent-blue/8 text-xs text-accent-blue font-medium">
                <span className="w-1.5 h-1.5 rounded-full bg-accent-blue animate-pulse" />
                Research Preview · RTRP 2026
                <ChevronRight size={12} />
              </div>
            </motion.div>

            {/* Headline */}
            <motion.h1
              variants={fadeUp}
              className="text-5xl lg:text-6xl font-black tracking-tight leading-[1.05] mb-6"
            >
              AI Classroom
              <br />
              <span className="text-gradient">Intelligence</span>
              <br />
              Platform.
            </motion.h1>

            {/* Subtitle */}
            <motion.p variants={fadeUp} className="text-lg text-text-secondary leading-relaxed mb-8 max-w-md">
              Real-time engagement detection, group collaboration analysis, and an AI Copilot that explains your classroom — all in one platform.
            </motion.p>

            {/* CTAs */}
            <motion.div variants={fadeUp} className="flex flex-wrap gap-4 mb-10">
              <Link
                href="/monitor"
                className="flex items-center gap-2 px-6 py-3 rounded-xl font-semibold text-white shadow-lg transition-transform hover:scale-105"
                style={{ background: 'linear-gradient(135deg, #4F7FFF, #7C6FFF)', boxShadow: '0 4px 24px rgba(79,127,255,0.35)' }}
              >
                <Play size={16} fill="white" />
                Try Live Demo
              </Link>
              <Link
                href="/dashboard"
                className="flex items-center gap-2 px-6 py-3 rounded-xl font-semibold text-text-primary border border-white/10 bg-white/5 hover:bg-white/8 transition-all"
              >
                <BarChart3 size={16} />
                View Dashboard
              </Link>
            </motion.div>

            {/* Metrics pills */}
            <motion.div variants={fadeUp} className="flex flex-wrap gap-3">
              {[
                { label: 'Engagement Tracking', val: 'Real-Time', color: '#4F7FFF' },
                { label: 'Collaboration Verdict', val: 'Live', color: '#22D3A6' },
                { label: 'Explains Every Drop', val: 'AI Copilot', color: '#38BDF8' },
              ].map((m) => (
                <div key={m.label} className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-white/5 border border-white/8 text-xs">
                  <span className="font-bold" style={{ color: m.color }}>{m.val}</span>
                  <span className="text-text-muted">{m.label}</span>
                </div>
              ))}
            </motion.div>
          </motion.div>

          {/* Mockup side */}
          <motion.div
            initial={{ opacity: 0, x: 40 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.4, duration: 0.8, ease: 'easeOut' }}
          >
            <DashboardMockup />
          </motion.div>
        </div>

        {/* Scroll indicator */}
        <motion.div
          className="absolute bottom-8 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2 text-text-muted text-xs"
          animate={{ y: [0, 6, 0] }}
          transition={{ duration: 2, repeat: Infinity }}
        >
          <span>Scroll to explore</span>
          <div className="w-px h-8 bg-gradient-to-b from-text-muted to-transparent" />
        </motion.div>
      </section>

      {/* ── Stats section ── */}
      <section className="py-16 border-y border-white/5">
        <div className="max-w-5xl mx-auto px-6">
          <motion.div
            initial="hidden"
            whileInView="show"
            viewport={{ once: true }}
            variants={stagger}
            className="grid grid-cols-2 md:grid-cols-4 gap-8 text-center"
          >
            {[
              { val: 6, dec: 0, suffix: '+', label: 'Students Tracked Live', color: '#4F7FFF' },
              { val: 3, dec: 0, suffix: '', label: 'Smart Alert Tiers', color: '#22D3A6' },
              { val: 3, dec: 0, suffix: '', label: 'Role Dashboards', color: '#38BDF8' },
              { val: 100, dec: 0, suffix: '%', label: 'Private — No Face ID', color: '#F59E0B' },
            ].map((s) => (
              <motion.div key={s.label} variants={fadeUp} className="flex flex-col items-center gap-1">
                <div className="text-4xl font-black font-mono" style={{ color: s.color }}>
                  <Counter to={s.val} decimals={s.dec} suffix={s.suffix} />
                </div>
                <div className="text-xs text-text-muted">{s.label}</div>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* ── Features section ── */}
      <section id="features" className="py-24 max-w-7xl mx-auto px-6">
        <motion.div
          initial="hidden"
          whileInView="show"
          viewport={{ once: true, margin: '-80px' }}
          variants={stagger}
          className="text-center mb-16"
        >
          <motion.div variants={fadeUp} className="inline-block px-3 py-1 rounded-full border border-accent-blue/30 bg-accent-blue/8 text-xs text-accent-blue font-medium mb-4">
            Platform Features
          </motion.div>
          <motion.h2 variants={fadeUp} className="text-4xl font-black mb-4">
            Everything your classroom needs
          </motion.h2>
          <motion.p variants={fadeUp} className="text-text-secondary max-w-lg mx-auto">
            From real-time detection to AI-powered explanations — built for teachers who need answers, not just numbers.
          </motion.p>
        </motion.div>

        <motion.div
          initial="hidden"
          whileInView="show"
          viewport={{ once: true, margin: '-60px' }}
          variants={stagger}
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5"
        >
          {features.map((f) => (
            <motion.div
              key={f.title}
              variants={fadeUp}
              whileHover={{ y: -4, scale: 1.01 }}
              className={`relative glass-card rounded-2xl p-6 cursor-default transition-all duration-300 ${
                f.highlight ? 'glow-blue border-accent-blue/20' : 'hover:border-white/15'
              }`}
            >
              {f.highlight && (
                <div className="absolute -top-px left-6 right-6 h-px bg-gradient-to-r from-transparent via-accent-blue to-transparent" />
              )}
              {/* Tag */}
              <div className="flex items-center justify-between mb-4">
                <div
                  className="w-10 h-10 rounded-xl flex items-center justify-center"
                  style={{ background: `${f.color}18`, border: `1px solid ${f.color}30` }}
                >
                  <f.icon size={18} style={{ color: f.color }} />
                </div>
                <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full border"
                  style={{ borderColor: `${f.color}30`, color: f.color, background: `${f.color}10` }}>
                  {f.tag}
                </span>
              </div>
              <h3 className="text-base font-semibold mb-2 text-text-primary">{f.title}</h3>
              <p className="text-sm text-text-secondary leading-relaxed">{f.desc}</p>
            </motion.div>
          ))}
        </motion.div>
      </section>

      {/* ── How it works ── */}
      <section id="how-it-works" className="py-24 border-t border-white/5">
        <div className="max-w-5xl mx-auto px-6">
          <motion.div
            initial="hidden"
            whileInView="show"
            viewport={{ once: true }}
            variants={stagger}
            className="text-center mb-16"
          >
            <motion.div variants={fadeUp} className="inline-block px-3 py-1 rounded-full border border-white/10 text-xs text-text-muted mb-4">
              How It Works
            </motion.div>
            <motion.h2 variants={fadeUp} className="text-4xl font-black mb-4">
              From video to insights in 3 steps
            </motion.h2>
          </motion.div>

          <motion.div
            initial="hidden"
            whileInView="show"
            viewport={{ once: true }}
            variants={stagger}
            className="grid md:grid-cols-3 gap-8 relative"
          >
            {/* Connector lines (hidden on mobile) */}
            <div className="hidden md:block absolute top-8 left-1/3 right-1/3 h-px bg-gradient-to-r from-transparent via-accent-blue/30 to-transparent" />

            {steps.map((s, i) => (
              <motion.div key={s.n} variants={fadeUp} className="flex flex-col items-center text-center">
                <div className="relative mb-6">
                  <div className="w-16 h-16 rounded-2xl flex items-center justify-center border border-accent-blue/20 bg-accent-blue/8"
                    style={{ boxShadow: '0 0 30px rgba(79,127,255,0.1)' }}>
                    <s.icon size={24} className="text-accent-blue" />
                  </div>
                  <div className="absolute -top-2 -right-2 w-6 h-6 rounded-full bg-bg-secondary border border-accent-blue/30 flex items-center justify-center text-[10px] font-bold text-accent-blue">
                    {i + 1}
                  </div>
                </div>
                <h3 className="text-base font-semibold mb-2">{s.title}</h3>
                <p className="text-sm text-text-secondary leading-relaxed">{s.desc}</p>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* ── Copilot highlight ── */}
      <section className="py-24 max-w-7xl mx-auto px-6">
        <motion.div
          initial="hidden"
          whileInView="show"
          viewport={{ once: true }}
          variants={stagger}
          className="glass-card rounded-3xl p-12 relative overflow-hidden border border-accent-indigo/20"
          style={{ boxShadow: '0 0 80px rgba(124,111,255,0.08)' }}
        >
          {/* Decorative orb */}
          <div className="absolute -top-20 -right-20 w-64 h-64 rounded-full blur-3xl opacity-15"
            style={{ background: 'radial-gradient(circle, #7C6FFF 0%, transparent 70%)' }} />

          <div className="relative grid lg:grid-cols-2 gap-12 items-center">
            {/* Text */}
            <div>
              <motion.div variants={fadeUp} className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full border border-accent-indigo/30 bg-accent-indigo/8 text-xs text-accent-indigo font-medium mb-6">
                <Brain size={12} />
                Standout Feature
              </motion.div>
              <motion.h2 variants={fadeUp} className="text-3xl font-black mb-4">
                AI Classroom Copilot
              </motion.h2>
              <motion.p variants={fadeUp} className="text-text-secondary mb-6 leading-relaxed">
                Ask your classroom anything. The Copilot understands your analytics in real-time and explains drops, identifies patterns, and can jump to any incident in the video — all with a single question.
              </motion.p>
              <motion.div variants={fadeUp} className="space-y-3">
                {[
                  '"Why is Room 201 showing red?"',
                  '"Show me the most disengaged period"',
                  '"What happened at minute 34?"',
                  '"Which students need attention today?"',
                ].map((q) => (
                  <div key={q} className="flex items-center gap-3 text-sm text-text-secondary">
                    <MessageSquare size={14} className="text-accent-indigo flex-shrink-0" />
                    <span className="italic">{q}</span>
                  </div>
                ))}
              </motion.div>
              <motion.div variants={fadeUp} className="mt-8">
                <Link
                  href="/dashboard"
                  className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-semibold text-white"
                  style={{ background: 'linear-gradient(135deg, #7C6FFF, #4F7FFF)' }}
                >
                  Try the Copilot
                  <ArrowRight size={14} />
                </Link>
              </motion.div>
            </div>

            {/* Chat preview */}
            <motion.div variants={fadeUp} className="space-y-3">
              {[
                { role: 'user', text: 'Why is Room 201 showing red?' },
                {
                  role: 'ai',
                  text: 'During the last 20 minutes: engagement dropped from 78% → 42%. 8 students showed sustained disengagement starting at minute 34. Collaboration remained low (31%). Click ▶ to replay minute 34.',
                },
                { role: 'user', text: 'Show me the most disengaged period' },
                { role: 'ai', text: 'Minute 32–38 had the lowest engagement: 38% avg. 3 alerts fired. Click here to jump to that moment.' },
              ].map((m, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, x: m.role === 'user' ? 20 : -20 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.15 }}
                  viewport={{ once: true }}
                  className={`max-w-sm text-sm p-3 leading-relaxed ${
                    m.role === 'user' ? 'ml-auto bubble-user' : 'bubble-ai'
                  }`}
                >
                  {m.role === 'ai' && (
                    <div className="flex items-center gap-1.5 mb-1.5">
                      <Brain size={11} className="text-accent-indigo" />
                      <span className="text-[10px] font-semibold text-accent-indigo">EduLens Copilot</span>
                    </div>
                  )}
                  {m.text}
                </motion.div>
              ))}
            </motion.div>
          </div>
        </motion.div>
      </section>

      {/* ── CTA ── */}
      <section className="py-24 border-t border-white/5">
        <div className="max-w-3xl mx-auto px-6 text-center">
          <motion.div
            initial="hidden"
            whileInView="show"
            viewport={{ once: true }}
            variants={stagger}
          >
            <motion.h2 variants={fadeUp} className="text-4xl font-black mb-4">
              Ready to transform your classroom?
            </motion.h2>
            <motion.p variants={fadeUp} className="text-text-secondary mb-8">
              Upload a classroom video and see engagement + collaboration analytics in minutes.
            </motion.p>
            <motion.div variants={fadeUp} className="flex justify-center gap-4">
              <Link
                href="/monitor"
                className="flex items-center gap-2 px-8 py-4 rounded-xl font-semibold text-white shadow-xl transition-transform hover:scale-105"
                style={{ background: 'linear-gradient(135deg, #4F7FFF, #7C6FFF)', boxShadow: '0 4px 30px rgba(79,127,255,0.4)' }}
              >
                <Zap size={18} />
                Start Live Analysis
              </Link>
              <Link
                href="/dashboard"
                className="flex items-center gap-2 px-8 py-4 rounded-xl font-semibold text-text-primary border border-white/10 bg-white/5 hover:bg-white/8 transition-all"
              >
                <Globe size={18} />
                Explore Demo
              </Link>
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* ── Footer ── */}
      <footer className="border-t border-white/5 py-10">
        <div className="max-w-7xl mx-auto px-6 flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2.5">
            <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-accent-blue to-accent-indigo flex items-center justify-center">
              <Brain size={14} className="text-white" />
            </div>
            <span className="text-sm font-bold">Edu<span className="text-gradient">Lens</span></span>
          </div>
          <p className="text-xs text-text-muted">
            Built with Swin-Tiny Transformer · Temporal Attention · Gaze Signals · RTRP 2026
          </p>
          <div className="flex items-center gap-4 text-xs text-text-muted">
            <Link href="/monitor" className="hover:text-text-primary transition-colors">Monitor</Link>
            <Link href="/dashboard" className="hover:text-text-primary transition-colors">Dashboard</Link>
            <Link href="/login" className="hover:text-text-primary transition-colors">Login</Link>
          </div>
        </div>
      </footer>
    </div>
  );
}
