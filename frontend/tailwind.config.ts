import type { Config } from 'tailwindcss';

const config: Config = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        // EduLens Design System
        bg: {
          primary: '#060C1A',
          secondary: '#0A1628',
          tertiary: '#0F1E38',
          card: '#0C1830',
        },
        accent: {
          blue: '#4F7FFF',
          indigo: '#7C6FFF',
          purple: '#9B59FF',
        },
        status: {
          engaged: '#22D3A6',
          'engaged-bg': 'rgba(34,211,166,0.12)',
          notEngaged: '#FF4E4E',
          'notEngaged-bg': 'rgba(255,78,78,0.12)',
          collab: '#38BDF8',
          'collab-bg': 'rgba(56,189,248,0.12)',
          warning: '#F59E0B',
          'warning-bg': 'rgba(245,158,11,0.12)',
          unknown: '#94A3B8',
        },
        border: {
          DEFAULT: 'rgba(148,163,184,0.1)',
          hover: 'rgba(148,163,184,0.2)',
          accent: 'rgba(79,127,255,0.3)',
        },
        text: {
          primary: '#E2E8F0',
          secondary: '#94A3B8',
          muted: '#64748B',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'spin-slow': 'spin 8s linear infinite',
        'float': 'float 6s ease-in-out infinite',
        'gradient': 'gradient 8s ease infinite',
        'count-up': 'countUp 0.5s ease-out',
        'slide-up': 'slideUp 0.4s ease-out',
        'fade-in': 'fadeIn 0.6s ease-out',
        'glow': 'glow 2s ease-in-out infinite',
        'ping-slow': 'ping 3s cubic-bezier(0, 0, 0.2, 1) infinite',
      },
      keyframes: {
        float: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-12px)' },
        },
        gradient: {
          '0%, 100%': { backgroundPosition: '0% 50%' },
          '50%': { backgroundPosition: '100% 50%' },
        },
        countUp: {
          from: { opacity: '0', transform: 'translateY(8px)' },
          to: { opacity: '1', transform: 'translateY(0)' },
        },
        slideUp: {
          from: { opacity: '0', transform: 'translateY(16px)' },
          to: { opacity: '1', transform: 'translateY(0)' },
        },
        fadeIn: {
          from: { opacity: '0' },
          to: { opacity: '1' },
        },
        glow: {
          '0%, 100%': { boxShadow: '0 0 20px rgba(79,127,255,0.3)' },
          '50%': { boxShadow: '0 0 40px rgba(79,127,255,0.6)' },
        },
      },
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'gradient-conic': 'conic-gradient(from 180deg at 50% 50%, var(--tw-gradient-stops))',
        'grid-pattern': "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='40' height='40'%3E%3Cpath d='M 40 0 L 0 0 0 40' fill='none' stroke='rgba(148,163,184,0.04)' stroke-width='1'/%3E%3C/svg%3E\")",
      },
      boxShadow: {
        'card': '0 4px 24px rgba(0,0,0,0.4), 0 1px 4px rgba(0,0,0,0.3)',
        'card-hover': '0 8px 40px rgba(0,0,0,0.5), 0 2px 8px rgba(0,0,0,0.4)',
        'glow-blue': '0 0 30px rgba(79,127,255,0.25)',
        'glow-green': '0 0 20px rgba(34,211,166,0.25)',
        'glow-red': '0 0 20px rgba(255,78,78,0.25)',
        'inner-card': 'inset 0 1px 0 rgba(255,255,255,0.05)',
      },
      backdropBlur: {
        'xs': '4px',
      },
    },
  },
  plugins: [],
};

export default config;
