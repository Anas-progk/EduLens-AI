// Module-level cache that survives client-side navigation (Next.js SPA route changes
// do NOT reload the page, so this persists when going dashboard <-> monitor).
// It is lost on a full page reload (F5) — that's expected.
export const sessionStore: {
  videoUrl: string | null;
  sessionId: string | null;
} = {
  videoUrl: null,
  sessionId: null,
};
