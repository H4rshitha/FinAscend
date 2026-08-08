"use client";

/** Shared shell pieces: nav, cards, tiles, and the load/error states. */

import Link from "next/link";
import { usePathname } from "next/navigation";
import React, { useEffect, useState } from "react";
import { ApiError, API_BASE } from "@/lib/api";

const PAGES = [
  { href: "/", n: "01", label: "Overview" },
  { href: "/solvers", n: "02", label: "Solver comparison" },
  { href: "/credit", n: "03", label: "Credit risk" },
  { href: "/ocr", n: "04", label: "OCR ingestion" },
  { href: "/backtest", n: "05", label: "Backtest report" },
];

export function Sidebar() {
  const path = usePathname();
  return (
    <nav className="sidebar">
      <div className="brand">
        <div className="brand-name">FinAscend</div>
        <div className="brand-sub">liquidity terminal</div>
      </div>
      {PAGES.map((p) => (
        <Link key={p.href} href={p.href} className="nav-item"
              data-active={path === p.href}>
          <span className="nav-index">{p.n}</span>
          {p.label}
        </Link>
      ))}
      <div style={{ padding: "24px", marginTop: "16px", borderTop: "1px solid var(--border)" }}>
        <div style={{ fontSize: "11px", color: "var(--text-muted)", lineHeight: 1.7 }}>
          Every figure on every page is fetched live from
          <br />
          <code style={{ fontFamily: "var(--font-mono)", fontSize: "10px", wordBreak: "break-all" }}>
            {API_BASE}
          </code>
          <br />
          Nothing is hard-coded.
        </div>
      </div>
    </nav>
  );
}

export function Card({
  title,
  note,
  children,
  right,
}: {
  title?: string;
  note?: React.ReactNode;
  children: React.ReactNode;
  right?: React.ReactNode;
}) {
  return (
    <section className="card">
      {(title || note) && (
        <header className="card-head">
          <div style={{ display: "flex", justifyContent: "space-between", gap: 16, alignItems: "baseline" }}>
            {title && <h2 className="card-title">{title}</h2>}
            {right}
          </div>
          {note && <div className="card-note">{note}</div>}
        </header>
      )}
      {children}
    </section>
  );
}

export function Tile({
  label,
  value,
  unit,
  sub,
  tone,
}: {
  label: string;
  value: React.ReactNode;
  unit?: string;
  sub?: React.ReactNode;
  tone?: "good" | "warn" | "critical";
}) {
  const color =
    tone === "good" ? "var(--status-good)"
    : tone === "warn" ? "var(--status-warn)"
    : tone === "critical" ? "var(--status-critical)"
    : "var(--text-primary)";
  return (
    <div className="tile">
      <div className="tile-label">{label}</div>
      <div className="tile-value num" style={{ color }}>
        {value}
        {unit && <span className="tile-unit">{unit}</span>}
      </div>
      {sub && <div className="tile-sub">{sub}</div>}
    </div>
  );
}

export function Chip({
  tone,
  children,
}: {
  tone?: "good" | "warn" | "critical";
  children: React.ReactNode;
}) {
  return <span className="chip" data-tone={tone}>{children}</span>;
}

export function Legend({ items }: { items: { label: string; color: string }[] }) {
  return (
    <div className="legend">
      {items.map((i) => (
        <span key={i.label} className="legend-item">
          <span className="swatch" style={{ background: i.color }} />
          {i.label}
        </span>
      ))}
    </div>
  );
}

export function Loading({ rows = 3 }: { rows?: number }) {
  return (
    <div>
      {Array.from({ length: rows }).map((_, i) => (
        <div key={i} className="skeleton"
             style={{ height: i === 0 ? 40 : 20, marginBottom: 10, width: i === 0 ? "45%" : "100%" }} />
      ))}
      <div className="state">computing — the API runs the real model on request</div>
    </div>
  );
}

/**
 * The error state. It never falls back to sample data: it says what failed and
 * how to fix it, because a chart drawn from a fallback is indistinguishable
 * from a chart drawn from a measurement.
 */
export function ErrorState({ error }: { error: ApiError | Error }) {
  const api = error as ApiError;
  const unreachable = api.status === 0;
  const detail = api.detail as { details?: Record<string, string> } | null;
  const how = detail?.details?.how_to_generate;

  return (
    <div className="state-error">
      <strong>{unreachable ? "Backend unreachable" : `Request failed (HTTP ${api.status})`}</strong>
      <div style={{ marginTop: 8 }}>{error.message}</div>
      {unreachable && (
        <>
          <div style={{ marginTop: 12, color: "var(--text-secondary)" }}>
            Start the API from <code>finascend/backend</code>:
          </div>
          <code>../.venv/Scripts/python -m uvicorn app.main:app --reload</code>
        </>
      )}
      {how && (
        <>
          <div style={{ marginTop: 12, color: "var(--text-secondary)" }}>
            Generate the missing artifact:
          </div>
          <code>{how}</code>
        </>
      )}
      <div style={{ marginTop: 12, fontSize: 11, color: "var(--text-muted)" }}>
        No placeholder values are shown in place of the real ones.
      </div>
    </div>
  );
}

/** Fetch-once-on-mount hook with explicit loading and error states. */
export function useApi<T>(fn: () => Promise<T>, deps: React.DependencyList = []) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let alive = true;
    setLoading(true);
    setError(null);
    fn()
      .then((d) => alive && setData(d))
      .catch((e) => alive && setError(e as Error))
      .finally(() => alive && setLoading(false));
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return { data, error, loading };
}

export function PageHead({ title, sub }: { title: string; sub: React.ReactNode }) {
  return (
    <header className="page-head">
      <h1>{title}</h1>
      <p className="page-sub">{sub}</p>
    </header>
  );
}
