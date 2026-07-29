'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useAuthStore } from '@/lib/store';

export function AuthGuard({ children, requiredRole }: { children: React.ReactNode; requiredRole?: 'teacher' | 'hod' | 'principal' }) {
  const router = useRouter();
  const { accessToken, user, isHydrated, hydrate } = useAuthStore();

  useEffect(() => {
    hydrate();
  }, [hydrate]);

  useEffect(() => {
    if (!isHydrated) return;
    if (!accessToken || !user) {
      router.replace('/login');
      return;
    }
    if (requiredRole) {
      const roleHierarchy: Record<string, number> = { teacher: 1, hod: 2, principal: 3 };
      if (roleHierarchy[user.role] < roleHierarchy[requiredRole]) {
        router.replace('/dashboard');
      }
    }
  }, [accessToken, user, isHydrated, requiredRole, router]);

  if (!isHydrated || !accessToken || !user) {
    return (
      <div className="min-h-screen bg-bg-primary flex items-center justify-center">
        <div className="flex flex-col items-center gap-3">
          <div className="w-8 h-8 border-2 border-accent-blue/30 border-t-accent-blue rounded-full animate-spin" />
          <span className="text-sm text-text-muted">Verifying access...</span>
        </div>
      </div>
    );
  }

  return <>{children}</>;
}
