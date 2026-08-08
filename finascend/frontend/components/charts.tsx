"use client";

/**
 * Inline-SVG charts built to one spec, with no charting library.
 *
 * Written by hand because the mark rules this project cares about — 2px
 * strokes, recessive grid, a 2px surface ring where marks overlap, selective
 * direct labels rather than a number on every point — are easier to state
 * directly than to coax out of a library's theme layer. It also keeps the
 * frontend at three dependencies.
 *
 * SERIES COLOR IS BOUND TO THE ENTITY, NOT ITS RANK. `SERIES` maps a strategy
 * name to a fixed slot, so `rules_baseline` is amber on every page and a chart
 * that drops a strategy never repaints the others.
 */

import React, { useId, useState } from "react";

export const SERIES: Record<string, string> = {
  rules_baseline: "var(--series-1)",
  lp_optimizer: "var(--series-2)",
  dp_knapsack: "var(--series-3)",
  chance_constrained: "var(--series-4)",
};

export const SLOTS = [
  "var(--series-1)",
  "var(--series-2)",
  "var(--series-3)",
  "var(--series-4)",
];

/** Dash patterns as a secondary encoding, so identity is never colour-alone. */
export const DASHES = ["", "6 3", "2 3", "9 3 2 3"];

export const seriesColor = (name: string, i = 0) =>
  SERIES[name] ?? SLOTS[i % SLOTS.length];

export function fmt(n: number, digits = 0): string {
  if (!Number.isFinite(n)) return "—";
  const abs = Math.abs(n);
  if (abs >= 1e7) return `${(n / 1e7).toFixed(2)}Cr`;
  if (abs >= 1e5) return `${(n / 1e5).toFixed(2)}L`;
  if (abs >= 1e3) return `${(n / 1e3).toFixed(abs >= 1e4 ? 0 : 1)}k`;
  return n.toFixed(digits);
}

export const pct = (n: number, d = 1) =>
  Number.isFinite(n) ? `${(n * 100).toFixed(d)}%` : "—";

interface Tip {
  x: number;
  y: number;
  lines: string[];
}

function Tooltip({ tip }: { tip: Tip | null }) {
  if (!tip) return null;
  return (
    <div className="tooltip" style={{ left: tip.x + 12, top: tip.y - 8 }}>
      {tip.lines.map((l, i) => (
        <div key={i}>{l}</div>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Sparkline — the default for "how did this move", per the brief's preference
// for sparklines over heavy widgets.
// ---------------------------------------------------------------------------

export function Sparkline({
  values,
  width = 120,
  height = 28,
  color = "var(--accent)",
  showLast = true,
}: {
  values: number[];
  width?: number;
  height?: number;
  color?: string;
  showLast?: boolean;
}) {
  if (!values.length) return null;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const pad = 3;
  const x = (i: number) => (i / Math.max(values.length - 1, 1)) * (width - pad * 2) + pad;
  const y = (v: number) => height - pad - ((v - min) / span) * (height - pad * 2);
  const d = values.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");

  return (
    <svg
      className="spark"
      width={width}
      height={height}
      role="img"
      aria-label={`sparkline, ${values.length} points, latest ${fmt(values[values.length - 1])}`}
    >
      <path d={d} fill="none" stroke={color} strokeWidth={1.5} strokeLinejoin="round" />
      {showLast && (
        <circle
          cx={x(values.length - 1)}
          cy={y(values[values.length - 1])}
          r={2.5}
          fill={color}
          stroke="var(--surface-1)"
          strokeWidth={1.5}
        />
      )}
    </svg>
  );
}

// ---------------------------------------------------------------------------
// Fan chart — point forecast plus its prediction interval
// ---------------------------------------------------------------------------

export function FanChart({
  points,
  height = 260,
  label = "forecast",
}: {
  points: { as_of_date: string; point: number; lower: number; upper: number }[];
  height?: number;
  label?: string;
}) {
  const [tip, setTip] = useState<Tip | null>(null);
  const gid = useId().replace(/:/g, "");
  const W = 860;
  const M = { t: 12, r: 16, b: 26, l: 62 };
  const iw = W - M.l - M.r;
  const ih = height - M.t - M.b;
  if (!points.length) return null;

  const lo = Math.min(...points.map((p) => p.lower));
  const hi = Math.max(...points.map((p) => p.upper));
  const pad = (hi - lo) * 0.06 || 1;
  const yMin = lo - pad;
  const yMax = hi + pad;

  const X = (i: number) => M.l + (i / Math.max(points.length - 1, 1)) * iw;
  const Y = (v: number) => M.t + ih - ((v - yMin) / (yMax - yMin)) * ih;

  const band =
    points.map((p, i) => `${i ? "L" : "M"}${X(i).toFixed(1)},${Y(p.upper).toFixed(1)}`).join(" ") +
    " " +
    points
      .slice()
      .reverse()
      .map((p, i) => `L${X(points.length - 1 - i).toFixed(1)},${Y(p.lower).toFixed(1)}`)
      .join(" ") +
    " Z";
  const line = points
    .map((p, i) => `${i ? "L" : "M"}${X(i).toFixed(1)},${Y(p.point).toFixed(1)}`)
    .join(" ");

  const ticks = 5;
  const yTicks = Array.from({ length: ticks }, (_, i) => yMin + ((yMax - yMin) * i) / (ticks - 1));

  const onMove = (e: React.MouseEvent<SVGRectElement>) => {
    const box = e.currentTarget.getBoundingClientRect();
    const rel = (e.clientX - box.left) / box.width;
    const i = Math.round(rel * (points.length - 1));
    const p = points[Math.max(0, Math.min(points.length - 1, i))];
    if (!p) return;
    setTip({
      x: e.clientX,
      y: e.clientY,
      lines: [
        p.as_of_date,
        `point  ${fmt(p.point)}`,
        `upper  ${fmt(p.upper)}`,
        `lower  ${fmt(p.lower)}`,
        `width  ${fmt(p.upper - p.lower)}`,
      ],
    });
  };

  return (
    <div className="chart-wrap">
      <svg className="chart" viewBox={`0 0 ${W} ${height}`} width="100%" role="img"
           aria-label={`${label} with prediction interval over ${points.length} days`}>
        <defs>
          <linearGradient id={`fan-${gid}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="var(--series-2)" stopOpacity="0.26" />
            <stop offset="100%" stopColor="var(--series-2)" stopOpacity="0.09" />
          </linearGradient>
        </defs>

        {yTicks.map((t, i) => (
          <g key={i}>
            <line className="grid-line" x1={M.l} x2={W - M.r} y1={Y(t)} y2={Y(t)} />
            <text x={M.l - 8} y={Y(t) + 3} textAnchor="end">{fmt(t)}</text>
          </g>
        ))}
        {/* zero line is meaningful here: net operating flow crosses it */}
        {yMin < 0 && yMax > 0 && (
          <line x1={M.l} x2={W - M.r} y1={Y(0)} y2={Y(0)}
                stroke="var(--border-strong)" strokeWidth={1} strokeDasharray="3 3" />
        )}

        <path d={band} fill={`url(#fan-${gid})`} />
        <path d={line} className="series-line" stroke="var(--series-2)" />

        <line className="axis-line" x1={M.l} x2={W - M.r} y1={height - M.b} y2={height - M.b} />
        {[0, Math.floor(points.length / 2), points.length - 1].map((i) => (
          <text key={i} x={X(i)} y={height - M.b + 14}
                textAnchor={i === 0 ? "start" : i === points.length - 1 ? "end" : "middle"}>
            {points[i]?.as_of_date.slice(5)}
          </text>
        ))}

        <rect x={M.l} y={M.t} width={iw} height={ih} fill="transparent"
              onMouseMove={onMove} onMouseLeave={() => setTip(null)} />
      </svg>
      <Tooltip tip={tip} />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Multi-series line chart
// ---------------------------------------------------------------------------

export interface Series {
  name: string;
  color: string;
  points: { x: string; y: number }[];
  dash?: string;
}

export function LineChart({
  series,
  height = 260,
  yFormat = fmt,
  yLabel,
  markers,
}: {
  series: Series[];
  height?: number;
  yFormat?: (n: number) => string;
  yLabel?: string;
  /** Horizontal reference lines, e.g. a nominal coverage level. */
  markers?: { y: number; label: string; color?: string }[];
}) {
  const [tip, setTip] = useState<Tip | null>(null);
  const W = 860;
  const M = { t: 14, r: 16, b: 28, l: 64 };
  const iw = W - M.l - M.r;
  const ih = height - M.t - M.b;

  const all = series.flatMap((s) => s.points.map((p) => p.y));
  if (!all.length) return null;
  const extra = (markers ?? []).map((m) => m.y);
  let yMin = Math.min(...all, ...extra);
  let yMax = Math.max(...all, ...extra);
  const pad = (yMax - yMin) * 0.08 || Math.abs(yMax) * 0.1 || 1;
  yMin -= pad;
  yMax += pad;

  const n = Math.max(...series.map((s) => s.points.length));
  const X = (i: number) => M.l + (i / Math.max(n - 1, 1)) * iw;
  const Y = (v: number) => M.t + ih - ((v - yMin) / (yMax - yMin)) * ih;

  const ticks = 5;
  const yTicks = Array.from({ length: ticks }, (_, i) => yMin + ((yMax - yMin) * i) / (ticks - 1));
  const labels = series[0]?.points.map((p) => p.x) ?? [];

  const onMove = (e: React.MouseEvent<SVGRectElement>) => {
    const box = e.currentTarget.getBoundingClientRect();
    const rel = (e.clientX - box.left) / box.width;
    const i = Math.max(0, Math.min(n - 1, Math.round(rel * (n - 1))));
    setTip({
      x: e.clientX,
      y: e.clientY,
      lines: [
        labels[i] ?? `#${i}`,
        ...series.map((s) => `${s.name.padEnd(20, " ")} ${yFormat(s.points[i]?.y ?? NaN)}`),
      ],
    });
  };

  return (
    <div className="chart-wrap">
      <svg className="chart" viewBox={`0 0 ${W} ${height}`} width="100%" role="img"
           aria-label={`${yLabel ?? "series"} over time, ${series.length} series`}>
        {yTicks.map((t, i) => (
          <g key={i}>
            <line className="grid-line" x1={M.l} x2={W - M.r} y1={Y(t)} y2={Y(t)} />
            <text x={M.l - 8} y={Y(t) + 3} textAnchor="end">{yFormat(t)}</text>
          </g>
        ))}

        {(markers ?? []).map((m, i) => (
          <g key={`m${i}`}>
            <line x1={M.l} x2={W - M.r} y1={Y(m.y)} y2={Y(m.y)}
                  stroke={m.color ?? "var(--status-warn)"} strokeWidth={1.5}
                  strokeDasharray="5 4" />
            <text x={W - M.r} y={Y(m.y) - 5} textAnchor="end"
                  fill={m.color ?? "var(--status-warn)"}>{m.label}</text>
          </g>
        ))}

        {series.map((s) => (
          <path key={s.name} className="series-line" stroke={s.color}
                strokeDasharray={s.dash || undefined}
                d={s.points.map((p, i) => `${i ? "L" : "M"}${X(i).toFixed(1)},${Y(p.y).toFixed(1)}`).join(" ")} />
        ))}

        <line className="axis-line" x1={M.l} x2={W - M.r} y1={height - M.b} y2={height - M.b} />
        {[0, Math.floor(n / 2), n - 1].map((i) => (
          <text key={i} x={X(i)} y={height - M.b + 15}
                textAnchor={i === 0 ? "start" : i === n - 1 ? "end" : "middle"}>
            {(labels[i] ?? "").slice(0, 10)}
          </text>
        ))}

        <rect x={M.l} y={M.t} width={iw} height={ih} fill="transparent"
              onMouseMove={onMove} onMouseLeave={() => setTip(null)} />
      </svg>
      <Tooltip tip={tip} />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Horizontal bar chart — magnitude comparison across a few named entities
// ---------------------------------------------------------------------------

export function BarChart({
  bars,
  height,
  valueFormat = fmt,
  note,
}: {
  bars: { label: string; value: number; color: string; sub?: string }[];
  height?: number;
  valueFormat?: (n: number) => string;
  note?: string;
}) {
  const [tip, setTip] = useState<Tip | null>(null);
  const W = 860;
  const rowH = 34;
  const gap = 2; // the 2px surface gap between adjacent fills
  const M = { t: 8, r: 130, b: 8, l: 150 };
  const h = height ?? bars.length * rowH + M.t + M.b;
  const iw = W - M.l - M.r;
  const max = Math.max(...bars.map((b) => Math.abs(b.value)), 1);

  return (
    <div className="chart-wrap">
      <svg className="chart" viewBox={`0 0 ${W} ${h}`} width="100%" role="img"
           aria-label={note ?? "comparison"}>
        {bars.map((b, i) => {
          const y = M.t + i * rowH;
          const bw = (Math.abs(b.value) / max) * iw;
          return (
            <g key={b.label}
               onMouseMove={(e) => setTip({ x: e.clientX, y: e.clientY,
                 lines: [b.label, valueFormat(b.value), ...(b.sub ? [b.sub] : [])] })}
               onMouseLeave={() => setTip(null)}>
              <text x={M.l - 10} y={y + rowH / 2 + 3} textAnchor="end"
                    fill="var(--text-secondary)">{b.label}</text>
              <rect x={M.l} y={y + gap} width={iw} height={rowH - gap * 2}
                    fill="var(--surface-2)" rx={3} />
              {/* 4px rounded data-end, anchored to the baseline */}
              <rect x={M.l} y={y + gap} width={Math.max(bw, 3)} height={rowH - gap * 2}
                    fill={b.color} rx={4} />
              <text x={M.l + iw + 10} y={y + rowH / 2 + 3}
                    fill="var(--text-primary)">{valueFormat(b.value)}</text>
              {b.sub && (
                <text x={M.l + iw + 10} y={y + rowH / 2 + 15} fill="var(--text-muted)"
                      fontSize={9}>{b.sub}</text>
              )}
            </g>
          );
        })}
      </svg>
      <Tooltip tip={tip} />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Calibration curve — predicted vs observed against the identity line
// ---------------------------------------------------------------------------

export function CalibrationCurve({
  bins,
  height = 300,
}: {
  bins: { mean_predicted: number; observed_rate: number; n: number }[];
  height?: number;
}) {
  const [tip, setTip] = useState<Tip | null>(null);
  const S = 300;
  const M = { t: 14, r: 14, b: 38, l: 46 };
  const iw = S - M.l - M.r;
  const ih = height - M.t - M.b;
  const lim = Math.max(0.3, ...bins.map((b) => Math.max(b.mean_predicted, b.observed_rate))) * 1.15;
  const X = (v: number) => M.l + (v / lim) * iw;
  const Y = (v: number) => M.t + ih - (v / lim) * ih;
  const maxN = Math.max(...bins.map((b) => b.n), 1);

  return (
    <div className="chart-wrap">
      <svg className="chart" viewBox={`0 0 ${S} ${height}`} width="100%"
           style={{ maxWidth: 340 }} role="img"
           aria-label="calibration curve: mean predicted probability versus observed default rate">
        {[0, 0.25, 0.5, 0.75, 1].map((f) => (
          <g key={f}>
            <line className="grid-line" x1={M.l} x2={S - M.r} y1={Y(lim * f)} y2={Y(lim * f)} />
            <text x={M.l - 6} y={Y(lim * f) + 3} textAnchor="end">{(lim * f).toFixed(2)}</text>
            <text x={X(lim * f)} y={height - M.b + 13} textAnchor="middle">
              {(lim * f).toFixed(2)}
            </text>
          </g>
        ))}

        {/* perfect calibration */}
        <line x1={X(0)} y1={Y(0)} x2={X(lim)} y2={Y(lim)}
              stroke="var(--text-muted)" strokeWidth={1.5} strokeDasharray="4 4" />
        <text x={X(lim) - 4} y={Y(lim) + 14} textAnchor="end" fill="var(--text-muted)">
          perfect
        </text>

        <path className="series-line" stroke="var(--series-2)"
              d={bins.map((b, i) => `${i ? "L" : "M"}${X(b.mean_predicted)},${Y(b.observed_rate)}`).join(" ")} />
        {bins.map((b, i) => (
          <circle key={i} cx={X(b.mean_predicted)} cy={Y(b.observed_rate)}
                  r={4 + (b.n / maxN) * 4}
                  fill="var(--series-2)"
                  stroke="var(--surface-1)" strokeWidth={2}
                  onMouseMove={(e) => setTip({ x: e.clientX, y: e.clientY, lines: [
                    `predicted ${b.mean_predicted.toFixed(3)}`,
                    `observed  ${b.observed_rate.toFixed(3)}`,
                    `n = ${b.n}`,
                  ] })}
                  onMouseLeave={() => setTip(null)} />
        ))}

        <line className="axis-line" x1={M.l} x2={S - M.r} y1={height - M.b} y2={height - M.b} />
        <line className="axis-line" x1={M.l} x2={M.l} y1={M.t} y2={height - M.b} />
        <text x={M.l + iw / 2} y={height - 6} textAnchor="middle">mean predicted</text>
      </svg>
      <Tooltip tip={tip} />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Diverging contribution bars — SHAP / coefficient attributions
// ---------------------------------------------------------------------------

export function ContributionChart({
  items,
  height,
}: {
  items: { feature: string; value: number; contribution: number }[];
  height?: number;
}) {
  const [tip, setTip] = useState<Tip | null>(null);
  const W = 700;
  const rowH = 26;
  const M = { t: 8, r: 74, b: 24, l: 190 };
  const h = height ?? items.length * rowH + M.t + M.b;
  const iw = W - M.l - M.r;
  const max = Math.max(...items.map((d) => Math.abs(d.contribution)), 1e-9);
  const mid = M.l + iw / 2;
  const half = iw / 2;

  return (
    <div className="chart-wrap">
      <svg className="chart" viewBox={`0 0 ${W} ${h}`} width="100%" role="img"
           aria-label="per-feature contributions, signed">
        <line x1={mid} x2={mid} y1={M.t} y2={h - M.b} stroke="var(--border-strong)" strokeWidth={1} />
        {items.map((d, i) => {
          const y = M.t + i * rowH;
          const w = (Math.abs(d.contribution) / max) * half;
          const pos = d.contribution >= 0;
          return (
            <g key={d.feature}
               onMouseMove={(e) => setTip({ x: e.clientX, y: e.clientY, lines: [
                 d.feature,
                 `value        ${d.value.toFixed(4)}`,
                 `contribution ${d.contribution >= 0 ? "+" : ""}${d.contribution.toFixed(4)}`,
                 pos ? "raises default probability" : "lowers default probability",
               ] })}
               onMouseLeave={() => setTip(null)}>
              <text x={M.l - 12} y={y + rowH / 2 + 3} textAnchor="end"
                    fill="var(--text-secondary)">{d.feature}</text>
              <rect x={pos ? mid : mid - w} y={y + 3} width={Math.max(w, 2)} height={rowH - 8}
                    rx={3}
                    fill={pos ? "var(--series-4)" : "var(--series-2)"} />
              <text x={W - M.r + 8} y={y + rowH / 2 + 3} fill="var(--text-primary)">
                {d.contribution >= 0 ? "+" : ""}{d.contribution.toFixed(3)}
              </text>
            </g>
          );
        })}
        <text x={mid - half / 2} y={h - 8} textAnchor="middle">← lowers risk</text>
        <text x={mid + half / 2} y={h - 8} textAnchor="middle">raises risk →</text>
      </svg>
      <Tooltip tip={tip} />
    </div>
  );
}
