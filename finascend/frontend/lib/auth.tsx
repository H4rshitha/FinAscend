"use client";

/**
 * Session state for the whole app.
 *
 * One provider owns the session and every page reads it. The important
 * property is the THREE-state model: `loading`, `signed out`, `signed in`.
 * Collapsing loading into signed-out is the classic bug here — the app
 * momentarily believes you are logged out while it is still reading storage,
 * bounces you to /login, and destroys the page you were on. So `ready` is
 * tracked separately and the guard refuses to redirect until it is true.
 *
 * On boot the stored token is VALIDATED against `/auth/me` rather than
 * trusted. A token can be expired, signed with a rotated secret, or belong to
 * a user who has since been deactivated; rendering a signed-in shell around
 * any of those produces a page where every panel fails separately instead of
 * one clean redirect to sign in.
 */

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import {
  api,
  ApiError,
  loadStoredToken,
  setAuthToken,
  type Capability,
  type Session,
  type SignupPayload,
} from "@/lib/api";

interface AuthValue {
  session: Session | null;
  /** False until the stored token has been checked. Never redirect before this. */
  ready: boolean;
  signIn: (email: string, password: string) => Promise<void>;
  signUp: (payload: SignupPayload) => Promise<void>;
  signOut: () => void;
  /** Entitlement check. Always ask this, never compare plan names. */
  can: (capability: Capability) => boolean;
}

const AuthContext = createContext<AuthValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [session, setSession] = useState<Session | null>(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    const stored = loadStoredToken();
    if (!stored) {
      setReady(true);
      return;
    }
    let live = true;
    api
      .get<Session>("/auth/me")
      .then((s) => {
        if (!live) return;
        setAuthToken(s.access_token); // /auth/me reissues, refreshing the expiry
        setSession(s);
      })
      .catch(() => {
        // Any failure here means the stored token is unusable. Clear it rather
        // than leaving a dead credential that makes every later call 401.
        if (live) {
          setAuthToken(null);
          setSession(null);
        }
      })
      .finally(() => live && setReady(true));
    return () => {
      live = false;
    };
  }, []);

  const adopt = useCallback((s: Session) => {
    setAuthToken(s.access_token);
    setSession(s);
  }, []);

  const signIn = useCallback(
    async (email: string, password: string) => {
      adopt(await api.postJson<Session>("/auth/login", { email, password }));
    },
    [adopt]
  );

  const signUp = useCallback(
    async (payload: SignupPayload) => {
      adopt(await api.postJson<Session>("/auth/signup", payload));
    },
    [adopt]
  );

  const signOut = useCallback(() => {
    setAuthToken(null);
    setSession(null);
  }, []);

  const can = useCallback(
    (capability: Capability) => !!session?.capabilities.includes(capability),
    [session]
  );

  const value = useMemo(
    () => ({ session, ready, signIn, signUp, signOut, can }),
    [session, ready, signIn, signUp, signOut, can]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used inside <AuthProvider>");
  return ctx;
}

/** True when the error means "your session is gone", so the UI can sign out. */
export function isAuthFailure(e: unknown): boolean {
  return (
    (e instanceof ApiError && e.status === 401) ||
    (e instanceof Error && e.name === "NotAuthenticatedError")
  );
}
