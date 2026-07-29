'use client';

import { useEffect } from 'react';
import Link from 'next/link';
import { usePathname, useRouter } from 'next/navigation';
import { Brain, BarChart3, Shield, LogOut, ChevronRight, Video } from 'lucide-react';
import { clsx } from 'clsx';
import { useAuthStore } from '@/lib/store';
import type { UserRole } from '@/lib/types';

const ALL_NAV_ITEMS: { href: string; icon: React.ElementType; label: string; roles: UserRole[] }[] = [
  { href: '/monitor',   icon: Video,     label: 'Upload & Monitor', roles: ['teacher', 'hod', 'principal'] },
  { href: '/dashboard', icon: BarChart3, label: 'Dashboard',        roles: ['teacher', 'hod', 'principal'] },
  { href: '/privacy',   icon: Shield,    label: 'Privacy',          roles: ['teacher', 'hod', 'principal'] },
];

export function Sidebar() {
  const pathname = usePathname();
  const router = useRouter();
  const { user, logout, hydrate } = useAuthStore();

  useEffect(() => { hydrate(); }, [hydrate]);

  const navItems = ALL_NAV_ITEMS.filter((item) => item.roles.includes(user?.role ?? 'teacher'));

  const handleLogout = async () => {
    await logout();
    router.push('/login');
  };

  return (
    <aside className="w-56 flex-shrink-0 h-screen sticky top-0 flex flex-col border-r border-white/6 bg-bg-secondary">
      <Link href="/" className="flex items-center gap-2.5 px-5 py-5 border-b border-white/6">
        <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-accent-blue to-accent-indigo flex items-center justify-center">
          <Brain size={14} className="text-white" />
        </div>
        <span className="text-sm font-bold">Edu<span className="text-gradient">Lens</span></span>
      </Link>

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

      <div className="px-3 py-4 border-t border-white/6">
        <button onClick={handleLogout} className="w-full flex items-center gap-3 px-3 py-2.5 rounded-xl hover:bg-white/4 group transition-all">
          <div className="w-7 h-7 rounded-full bg-gradient-to-br from-accent-blue to-accent-indigo flex items-center justify-center text-xs font-bold text-white flex-shrink-0">
            {user?.name?.charAt(0)?.toUpperCase() ?? '?'}
          </div>
          <div className="flex-1 min-w-0 text-left">
            <div className="text-xs font-medium truncate">{user?.name ?? 'User'}</div>
            <div className="text-[10px] text-text-muted truncate capitalize">{user?.role ?? '—'}</div>
          </div>
          <LogOut size={13} className="text-text-muted group-hover:text-text-secondary flex-shrink-0" />
        </button>
      </div>
    </aside>
  );
}

export function TopNav({ title }: { title: string }) {
  const pathname = usePathname();
  return (
    <header className="h-14 border-b border-white/6 flex items-center px-6 justify-between bg-bg-secondary/50 backdrop-blur-sm sticky top-0 z-30">
      <div className="flex items-center gap-2 text-sm">
        <Link href="/" className="text-text-muted hover:text-text-secondary transition-colors">EduLens</Link>
        <ChevronRight size={14} className="text-text-muted" />
        <span className="text-text-primary font-medium">{title}</span>
      </div>

      <div className="live-dot text-xs text-text-muted">Live Analysis</div>
    </header>
  );
}
