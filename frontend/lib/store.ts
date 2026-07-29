'use client';

import { create } from 'zustand';
import type { AuthUser, LoginResponse } from './types';
import { login as apiLogin } from './api';
import axios from 'axios';

const LS_ACCESS = 'edulens_access_token';
const LS_REFRESH = 'edulens_refresh_token';
const LS_USER = 'edulens_user';

interface AuthState {
  accessToken: string | null;
  refreshToken: string | null;
  user: AuthUser | null;
  isLoading: boolean;
  isHydrated: boolean;
  login: (email: string, password: string, captchaToken?: string) => Promise<LoginResponse>;
  logout: () => Promise<void>;
  refresh: () => Promise<boolean>;
  setTokens: (access: string, refresh: string) => void;
  hydrate: () => void;
}

export const useAuthStore = create<AuthState>((set, get) => ({
  accessToken: null,
  refreshToken: null,
  user: null,
  isLoading: false,
  isHydrated: false,

  hydrate: () => {
    if (typeof window === 'undefined') return;
    const access = localStorage.getItem(LS_ACCESS);
    const refresh = localStorage.getItem(LS_REFRESH);
    const raw = localStorage.getItem(LS_USER);
    let user: AuthUser | null = null;
    if (raw) {
      try { user = JSON.parse(raw); } catch { /* ignore */ }
    }
    set({ accessToken: access, refreshToken: refresh, user, isHydrated: true });
  },

  login: async (email: string, password: string, captchaToken?: string) => {
    set({ isLoading: true });
    try {
      const res = await apiLogin(email, password, captchaToken);
      const { access_token, refresh_token, user } = res;
      localStorage.setItem(LS_ACCESS, access_token);
      localStorage.setItem(LS_REFRESH, refresh_token);
      localStorage.setItem(LS_USER, JSON.stringify(user));
      set({ accessToken: access_token, refreshToken: refresh_token, user, isLoading: false });
      return res;
    } catch (err) {
      set({ isLoading: false });
      throw err;
    }
  },

  logout: async () => {
    const { refreshToken } = get();
    try {
      if (refreshToken) {
        await axios.post('/api/auth/logout', { refresh_token: refreshToken });
      }
    } catch { /* ignore */ }
    localStorage.removeItem(LS_ACCESS);
    localStorage.removeItem(LS_REFRESH);
    localStorage.removeItem(LS_USER);
    set({ accessToken: null, refreshToken: null, user: null });
  },

  refresh: async () => {
    const { refreshToken } = get();
    if (!refreshToken) return false;
    try {
      const { data } = await axios.post('/api/auth/refresh', { refresh_token: refreshToken });
      const { access_token, refresh_token, user } = data as LoginResponse;
      localStorage.setItem(LS_ACCESS, access_token);
      localStorage.setItem(LS_REFRESH, refresh_token);
      localStorage.setItem(LS_USER, JSON.stringify(user));
      set({ accessToken: access_token, refreshToken: refresh_token, user });
      return true;
    } catch {
      localStorage.removeItem(LS_ACCESS);
      localStorage.removeItem(LS_REFRESH);
      localStorage.removeItem(LS_USER);
      set({ accessToken: null, refreshToken: null, user: null });
      return false;
    }
  },

  setTokens: (access: string, refresh: string) => {
    localStorage.setItem(LS_ACCESS, access);
    localStorage.setItem(LS_REFRESH, refresh);
    set({ accessToken: access, refreshToken: refresh });
  },
}));
