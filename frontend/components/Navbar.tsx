'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Brain, Activity, BarChart3, Bell, Shield, LogOut, ChevronRight, Video } from 'lucide-react';
import { clsx } from 'clsx';

const navItems = [
  { href: '/monitor',   icon: Video,      label: 'Live Monitor' },
  { href: '/dashboard', icon: BarChart3,  label: 'Dashboard' },
  { href: '/alerts',    icon: Bell,       label: 'Alerts' },
  { href: '/privacy',   icon: Shield,     label: 'Privacy' },
];

export function Sidebar() {
  const pathname = usePathname();
  return (
    <aside className="w-56 flex-shrink-0 h-screen sticky top-0 flex flex-col border-r border-white/6 bg-bg-secondary">
      {/* Logo */}
      <Link href="/" className="flex items-center gap-2.5 px-5 py-5 border-b border-white/6">
        <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-accent-blue to-accent-indigo flex items-center justify-center">
          <Brain size={14} className="text-white" />
        </div>
        <span className="text-sm font-bold">Edu<span className="text-gradient">Lens</span></span>
      </Link>

      {/* Nav */}
      <nav className="flex-1 py-4 px-3 space-y-1 overflow-y-auto">
        {navItems.map(({ href, icon: Icon, label }) => (
          <Link
            key={href}
            href={href}
            className={clsx(
              'flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all',
              pathname === href
                ? 'bg-accent-blue/10 text-white border-l-2 border-accent-blue pl-[10px]'
                : 'text-text-secondary hover:text-text-primary hover:bg-white/4'
            )}
          >
            <Icon size={16} className={pathname === href ? 'text-accent-blue' : ''} />
            {label}
          </Link>
        ))}
      </nav>

      {/* User area */}
      <div className="px-3 py-4 border-t border-white/6">
        <div className="flex items-center gap-3 px-3 py-2.5 rounded-xl hover:bg-white/4 cursor-pointer group">
          <div className="w-7 h-7 rounded-full bg-gradient-to-br from-accent-blue to-accent-indigo flex items-center justify-center text-xs font-bold text-white">
            T
          </div>
          <div className="flex-1 min-w-0">
            <div className="text-xs font-medium truncate">Teacher</div>
            <div className="text-[10px] text-text-muted truncate">Room 201</div>
          </div>
          <LogOut size={13} className="text-text-muted group-hover:text-text-secondary" />
        </div>
      </div>
    </aside>
  );
}

export function TopNav({ title }: { title: string }) {
  const pathname = usePathname();
  return (
    <header className="h-14 border-b border-white/6 flex items-center px-6 justify-between bg-bg-secondary/50 backdrop-blur-sm sticky top-0 z-30">
      {/* Breadcrumb */}
      <div className="flex items-center gap-2 text-sm">
        <Link href="/" className="text-text-muted hover:text-text-secondary transition-colors">EduLens</Link>
        <ChevronRight size={14} className="text-text-muted" />
        <span className="text-text-primary font-medium">{title}</span>
      </div>

      {/* Live indicator */}
      <div className="live-dot text-xs text-text-muted">Live Analysis</div>
    </header>
  );
}
