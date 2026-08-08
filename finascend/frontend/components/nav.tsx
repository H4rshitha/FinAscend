"use client";

import { usePathname } from "next/navigation";
import Link from "next/link";

/**
 * Six destinations, always visible, horizontally scrollable on narrow screens.
 *
 * Deliberately not a hamburger: the whole set fits, and burying navigation
 * behind a tap on the page a worried owner opens first is a cost with no
 * matching benefit at this count.
 */
const LINKS = [
  { href: "/", label: "Cash health" },
  { href: "/plan", label: "Action plan" },
  { href: "/risk", label: "Risk explorer" },
  { href: "/counterparties", label: "Customers" },
  { href: "/receipt", label: "Add a receipt" },
  { href: "/transparency", label: "How this works" },
];

export function Nav() {
  const path = usePathname();
  return (
    <nav className="nav" aria-label="Main">
      {LINKS.map((l) => (
        <Link
          key={l.href}
          href={l.href}
          aria-current={path === l.href ? "page" : undefined}
        >
          {l.label}
        </Link>
      ))}
    </nav>
  );
}
