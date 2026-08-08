"use client";

/**
 * App chrome.
 *
 * The login page renders bare — no top bar, no nav, no footer. A signed-out
 * visitor should not see navigation to six pages they cannot open, and the
 * split-pane sign-in layout needs the full viewport.
 */

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth";
import { PUBLIC_ROUTES } from "@/components/guard";

const LINKS = [
  { href: "/", label: "Cash health" },
  { href: "/plan", label: "Action plan" },
  { href: "/risk", label: "Risk explorer" },
  { href: "/counterparties", label: "Customers" },
  { href: "/receipt", label: "Add a receipt" },
  { href: "/transparency", label: "How this works" },
];

export function Shell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const router = useRouter();
  const { session, signOut } = useAuth();

  if (PUBLIC_ROUTES.includes(pathname)) return <>{children}</>;

  return (
    <div className="shell">
      <header className="topbar">
        <div className="topbar-inner">
          <Link className="wordmark" href="/">
            <span className="wordmark-dot" aria-hidden="true" />
            FinAscend
          </Link>

          <nav className="nav" aria-label="Main">
            {LINKS.map((l) => (
              <Link
                key={l.href}
                href={l.href}
                aria-current={pathname === l.href ? "page" : undefined}
              >
                {l.label}
              </Link>
            ))}
          </nav>

          {session ? (
            <div className="acct">
              <span className="acct-id">
                <span className="acct-name">{session.user.full_name}</span>
                <span className="acct-org">{session.organization.name}</span>
              </span>
              <span
                className="acct-plan"
                title={session.organization.plan_tagline}
              >
                {session.organization.plan_label}
              </span>
              <button
                type="button"
                className="btn btn-secondary"
                style={{ padding: "6px 12px", minHeight: 34 }}
                onClick={() => {
                  signOut();
                  router.replace("/login");
                }}
              >
                Sign out
              </button>
            </div>
          ) : null}
        </div>
      </header>

      <main className="main">{children}</main>

      <footer className="footer">
        <div className="footer-inner">
          Every figure on this site is fetched live from the FinAscend API when the
          page loads. Nothing is stored in the page, and no number is estimated in
          the browser — if a value cannot be fetched, the page says so instead of
          showing a placeholder.
        </div>
      </footer>
    </div>
  );
}
