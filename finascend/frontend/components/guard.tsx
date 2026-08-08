"use client";

/**
 * Route guard.
 *
 * Wraps everything except /login. Three rules:
 *
 *  1. Never redirect before `ready`. Until the stored token has been checked
 *     the app does not yet know whether you are signed in, and bouncing on a
 *     "not signed in" that is really "not checked yet" throws away the page
 *     the user asked for on every refresh.
 *
 *  2. Remember where they were going. Someone opening a bookmarked /plan while
 *     signed out should land on /plan after signing in, not on the home page.
 *
 *  3. Render nothing while redirecting. Flashing a page's chrome for one frame
 *     before navigating away looks broken, and on a finance app it looks like
 *     data leaked.
 */

import { useEffect } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useAuth } from "@/lib/auth";
import { Skeleton } from "@/components/ui";

export const PUBLIC_ROUTES = ["/login"];

export function AuthGuard({ children }: { children: React.ReactNode }) {
  const { session, ready } = useAuth();
  const pathname = usePathname();
  const router = useRouter();
  const isPublic = PUBLIC_ROUTES.includes(pathname);

  useEffect(() => {
    if (!ready) return;
    if (!session && !isPublic) {
      const next = encodeURIComponent(pathname);
      router.replace(`/login?next=${next}`);
    }
    if (session && isPublic) {
      // Honour ?next= here too. The login page issues its own redirect on a
      // successful sign-in, and this effect fires on the same state change —
      // whichever lands second wins. Sending both to the same destination
      // removes the race instead of hoping it resolves the right way, which is
      // what dropped users on the home page after signing up from a deep link.
      //
      // Read from window rather than useSearchParams: this component renders in
      // the root layout, and that hook would force the entire app out of static
      // rendering unless every page were wrapped in its own Suspense boundary.
      router.replace(safeNext(window.location.search));
    }
  }, [ready, session, isPublic, pathname, router]);

  if (isPublic) return <>{children}</>;

  if (!ready || !session) {
    // Shaped like the app shell so the transition into the real page is a
    // fill-in rather than a jump.
    return (
      <div className="stack" aria-busy="true" aria-live="polite">
        <span className="sr-only">Checking your session…</span>
        <Skeleton h={120} />
        <Skeleton h={220} />
        <Skeleton h={180} />
      </div>
    );
  }

  return <>{children}</>;
}

/**
 * Validate a `next` destination.
 *
 * Only same-site relative paths are allowed. An attacker-supplied absolute URL
 * would turn the login page into an open redirect — a standard phishing
 * primitive, because the link genuinely begins on the real domain and only
 * jumps elsewhere after the victim has signed in. `//evil.com` is rejected
 * alongside `https://evil.com`: the browser reads a protocol-relative URL as
 * absolute, so checking only for a leading `/` would let it straight through.
 */
export function safeNext(search: string): string {
  const next = new URLSearchParams(search).get("next");
  if (!next || !next.startsWith("/") || next.startsWith("//")) return "/";
  return next;
}

/** Hook form, for the login page. */
export function useNextPath(): string {
  const params = useSearchParams();
  const next = params.get("next");
  if (!next || !next.startsWith("/") || next.startsWith("//")) return "/";
  return next;
}
