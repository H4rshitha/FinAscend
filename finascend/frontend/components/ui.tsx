"use client";

/**
 * Shared primitives.
 *
 * The two that carry the product's argument:
 *
 *   <Method>  — progressive disclosure. Every plain-language claim can be
 *               opened to show the real method underneath. Collapsed by
 *               default, never absent, never a different page. One audience
 *               reads the summary; the other reads the derivation; neither is
 *               sent somewhere else to do it.
 *
 *   <Status>  — a risk signal is ALWAYS colour + icon + words. Colour alone
 *               fails for colourblind users and disappears in greyscale print,
 *               and this is a product where misreading a risk state has real
 *               consequences.
 */

import { ReactNode, useEffect, useId, useRef, useState } from "react";
import { ApiError, API_BASE } from "@/lib/api";

// ---------------------------------------------------------------------------
// data fetching
// ---------------------------------------------------------------------------

export interface Async<T> {
  data: T | null;
  error: ApiError | null;
  loading: boolean;
  reload: () => void;
}

/** Runs `fn` on mount. Every page's data comes through here — no other path. */
export function useApi<T>(fn: () => Promise<T>, deps: unknown[] = []): Async<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [loading, setLoading] = useState(true);
  const [nonce, setNonce] = useState(0);
  const fnRef = useRef(fn);
  fnRef.current = fn;

  useEffect(() => {
    let live = true;
    setLoading(true);
    setError(null);
    fnRef
      .current()
      .then((d) => live && (setData(d), setError(null)))
      .catch((e) => live && setError(e instanceof ApiError ? e : new ApiError(String(e))))
      .finally(() => live && setLoading(false));
    return () => {
      live = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, nonce]);

  return { data, error, loading, reload: () => setNonce((n) => n + 1) };
}

// ---------------------------------------------------------------------------
// icons — every status pairs one of these with its colour and its label
// ---------------------------------------------------------------------------

const ico = (path: ReactNode, size = 14) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 16 16"
    fill="none"
    stroke="currentColor"
    strokeWidth="1.7"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden="true"
  >
    {path}
  </svg>
);

export const Icon = {
  check: (s?: number) => ico(<path d="M13.5 4.5 6 12 2.5 8.5" />, s),
  alert: (s?: number) =>
    ico(
      <>
        <path d="M8 1.8 15 14H1L8 1.8Z" />
        <path d="M8 6.5v3" />
        <circle cx="8" cy="11.8" r="0.6" fill="currentColor" stroke="none" />
      </>,
      s
    ),
  info: (s?: number) =>
    ico(
      <>
        <circle cx="8" cy="8" r="6.4" />
        <path d="M8 7.4v4" />
        <circle cx="8" cy="5" r="0.6" fill="currentColor" stroke="none" />
      </>,
      s
    ),
  clock: (s?: number) => ico(<><circle cx="8" cy="8" r="6.4" /><path d="M8 4.4V8l2.4 1.6" /></>, s),
  pause: (s?: number) => ico(<><path d="M6 4v8M10 4v8" /></>, s),
  chevron: (s?: number) => ico(<path d="M6 3.5 10.5 8 6 12.5" />, s),
  offline: (s?: number) =>
    ico(<><path d="M2 2l12 12" /><path d="M5.5 10.5a3.5 3.5 0 0 1 4.2-3.4" /><path d="M2.6 7.2a8 8 0 0 1 3-2" /><path d="M13.4 7.2a8 8 0 0 0-3.6-2.2" /><circle cx="8" cy="13" r="0.7" fill="currentColor" stroke="none" /></>, s),
  empty: (s?: number) => ico(<><path d="M2.5 5.5h11v8h-11z" /><path d="M2.5 5.5 4.5 2.5h7l2 3" /><path d="M6.2 9h3.6" /></>, s),
  lock: (s?: number) => ico(<><rect x="3" y="7" width="10" height="7" rx="1.4" /><path d="M5.5 7V5a2.5 2.5 0 0 1 5 0v2" /></>, s),
  upload: (s?: number) => ico(<><path d="M8 11V3" /><path d="M4.8 6.2 8 3l3.2 3.2" /><path d="M2.5 11v2.5h11V11" /></>, s),
};

// ---------------------------------------------------------------------------
// status — colour + icon + label, never colour alone
// ---------------------------------------------------------------------------

export type Tone = "good" | "warning" | "serious" | "critical" | "neutral";

const TONE_ICON: Record<Tone, () => ReactNode> = {
  good: () => Icon.check(),
  warning: () => Icon.alert(),
  serious: () => Icon.alert(),
  critical: () => Icon.alert(),
  neutral: () => Icon.info(),
};

export function Status({
  tone,
  children,
  icon,
}: {
  tone: Tone;
  children: ReactNode;
  icon?: ReactNode;
}) {
  return (
    <span className={`pill pill-${tone}`}>
      {icon ?? TONE_ICON[tone]()}
      <span>{children}</span>
    </span>
  );
}

// ---------------------------------------------------------------------------
// layout
// ---------------------------------------------------------------------------

export function Card({
  children,
  className = "",
  ...rest
}: { children: ReactNode; className?: string } & React.HTMLAttributes<HTMLDivElement>) {
  return (
    <section className={`card ${className}`} {...rest}>
      {children}
    </section>
  );
}

export function CardHead({
  title,
  note,
  aside,
}: {
  title: ReactNode;
  note?: ReactNode;
  aside?: ReactNode;
}) {
  return (
    <div className="card-head">
      <div className="row-between">
        <h2>{title}</h2>
        {aside}
      </div>
      {note ? <p className="card-note">{note}</p> : null}
    </div>
  );
}

export function Tile({
  label,
  value,
  note,
  tone,
}: {
  label: string;
  value: ReactNode;
  note?: ReactNode;
  tone?: Tone;
}) {
  const color =
    tone && tone !== "neutral" ? { color: `var(--${tone})` } : undefined;
  return (
    <div className="tile">
      <span className="tile-label">{label}</span>
      <span className="tile-value num" style={color}>
        {value}
      </span>
      {note ? <span className="tile-note">{note}</span> : null}
    </div>
  );
}

// ---------------------------------------------------------------------------
// progressive disclosure
// ---------------------------------------------------------------------------

/**
 * The expandable "how we calculated this" panel.
 *
 * `id` is stable so the Transparency page can deep-link straight to a specific
 * method panel (`/risk#method-copula`) and have it open on arrival — the audit
 * trail points at the working, not at the page that contains it.
 */
export function Method({
  label = "How we calculated this",
  hint,
  id,
  children,
  defaultOpen = false,
}: {
  label?: string;
  hint?: string;
  id?: string;
  children: ReactNode;
  defaultOpen?: boolean;
}) {
  const auto = useId();
  const panelId = id ?? `method-${auto.replace(/:/g, "")}`;
  const [open, setOpen] = useState(defaultOpen);

  // Opening from a hash link: /risk#method-copula
  useEffect(() => {
    if (typeof window === "undefined") return;
    const check = () => {
      if (window.location.hash === `#${panelId}`) {
        setOpen(true);
        document.getElementById(panelId)?.scrollIntoView({ block: "start" });
      }
    };
    check();
    window.addEventListener("hashchange", check);
    return () => window.removeEventListener("hashchange", check);
  }, [panelId]);

  return (
    <div className="method" id={panelId}>
      <button
        type="button"
        className="method-toggle"
        aria-expanded={open}
        aria-controls={`${panelId}-panel`}
        onClick={() => setOpen((o) => !o)}
      >
        <span className="method-chev">{Icon.chevron(13)}</span>
        <span>{label}</span>
        {hint ? <span className="method-hint">{hint}</span> : null}
      </button>
      {open ? (
        <div className="method-panel" id={`${panelId}-panel`}>
          {children}
        </div>
      ) : null}
    </div>
  );
}

// ---------------------------------------------------------------------------
// states — designed, shaped like the content they replace
// ---------------------------------------------------------------------------

export function Skeleton({
  h = 16,
  w = "100%",
  style,
}: {
  h?: number | string;
  w?: number | string;
  style?: React.CSSProperties;
}) {
  return <div className="skel" style={{ height: h, width: w, ...style }} />;
}

/** Mirrors the hero's shape so the page does not jump when the number lands. */
export function HeroSkeleton() {
  return (
    <div className="stack-sm" aria-busy="true" aria-live="polite">
      <span className="sr-only">Working out your cash position…</span>
      <Skeleton h={12} w={140} />
      <Skeleton h={56} w="min(420px, 80%)" />
      <Skeleton h={20} w="min(520px, 95%)" />
    </div>
  );
}

export function TileSkeleton({ n = 3 }: { n?: number }) {
  return (
    <div className="grid grid-3" aria-busy="true">
      {Array.from({ length: n }).map((_, i) => (
        <div key={i} className="stack-sm">
          <Skeleton h={10} w={72} />
          <Skeleton h={28} w={120} />
          <Skeleton h={12} w={150} />
        </div>
      ))}
    </div>
  );
}

export function ChartSkeleton({ h = 200 }: { h?: number }) {
  return (
    <div className="chart" aria-busy="true">
      <Skeleton h={h} />
    </div>
  );
}

export function RowsSkeleton({ n = 4 }: { n?: number }) {
  return (
    <div className="stack-sm" aria-busy="true">
      {Array.from({ length: n }).map((_, i) => (
        <div key={i} className="row" style={{ gap: "var(--s-4)" }}>
          <Skeleton h={38} w={38} style={{ borderRadius: 9, flexShrink: 0 }} />
          <div className="stack-sm" style={{ flex: 1, gap: 6 }}>
            <Skeleton h={14} w={`${45 + ((i * 13) % 30)}%`} />
            <Skeleton h={12} w={`${65 + ((i * 7) % 25)}%`} />
          </div>
        </div>
      ))}
    </div>
  );
}

/**
 * The error state names the actual failure and gives the actual fix. It never
 * substitutes data. If the API is unreachable it says so and prints the command
 * that starts it, because during local use that is genuinely the whole problem.
 */
export function ErrorState({ error, onRetry }: { error: ApiError; onRetry?: () => void }) {
  const offline = error.isOffline;
  return (
    <div className="state state-error" role="alert">
      <div className="state-icon">{offline ? Icon.offline(20) : Icon.alert(20)}</div>
      <h3>{offline ? "Can’t reach the FinAscend service" : "That didn’t load"}</h3>
      {offline ? (
        <>
          <p>
            Nothing is shown here rather than something approximate — this app never
            displays a number it did not get from the live service.
          </p>
          <p className="tiny muted">
            Expecting the API at <code>{API_BASE}</code>. Start it with:
            <br />
            <code>cd finascend/backend &amp;&amp; ../.venv/Scripts/python -m uvicorn app.main:app</code>
          </p>
        </>
      ) : (
        <>
          <p>{error.message}</p>
          {error.status ? <p className="tiny muted">HTTP {error.status}</p> : null}
        </>
      )}
      {onRetry ? (
        <button type="button" className="btn btn-secondary" onClick={onRetry}>
          Try again
        </button>
      ) : null}
    </div>
  );
}

export function EmptyState({
  title,
  children,
  action,
}: {
  title: string;
  children?: ReactNode;
  action?: ReactNode;
}) {
  return (
    <div className="state state-empty">
      <div className="state-icon">{Icon.empty(20)}</div>
      <h3>{title}</h3>
      {children ? <p>{children}</p> : null}
      {action}
    </div>
  );
}

/**
 * One place that decides what to render for a request: skeleton, error, or the
 * data. Pages never hand-roll this, so no page can accidentally render a
 * placeholder number while `loading` is true.
 */
export function Loaded<T>({
  q,
  skeleton,
  children,
}: {
  q: Async<T>;
  skeleton?: ReactNode;
  children: (data: T) => ReactNode;
}) {
  if (q.error) return <ErrorState error={q.error} onRetry={q.reload} />;
  if (q.loading || q.data === null) return <>{skeleton ?? <RowsSkeleton />}</>;
  return <>{children(q.data)}</>;
}
