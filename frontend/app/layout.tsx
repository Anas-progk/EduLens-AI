import type { Metadata } from 'next';
import './globals.css';
import { Toaster } from 'react-hot-toast';

export const metadata: Metadata = {
  title: 'EduLens — AI Classroom Intelligence',
  description: 'Real-time AI-powered engagement and collaboration analytics for modern classrooms.',
  keywords: ['AI', 'classroom', 'engagement', 'collaboration', 'analytics', 'education'],
  openGraph: {
    title: 'EduLens — AI Classroom Intelligence',
    description: 'Real-time AI-powered engagement and collaboration analytics for modern classrooms.',
    type: 'website',
  },
};

const turnstileEnabled = process.env.NEXT_PUBLIC_TURNSTILE_ENABLED === 'true';

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="scroll-smooth">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap"
          rel="stylesheet"
        />
        {turnstileEnabled && (
          <script src="https://challenges.cloudflare.com/turnstile/v0/api.js" async defer />
        )}
      </head>
      <body className="min-h-screen bg-bg-primary text-text-primary antialiased grid-bg">
        {children}
        <Toaster
          position="top-right"
          toastOptions={{
            style: {
              background: '#0C1830',
              color: '#E2E8F0',
              border: '1px solid rgba(148,163,184,0.15)',
              borderRadius: '10px',
              fontSize: '13px',
            },
          }}
        />
      </body>
    </html>
  );
}
