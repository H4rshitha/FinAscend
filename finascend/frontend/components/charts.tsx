"use client";

/**
 * Charts.
 *
 * Rules applied throughout, not per-chart:
 *  - Recessive grid and axes; the data is the only thing with weight.
 *  - 2px lines, >=8px hit targets, 4px rounded data-ends on bars anchored to
 *    the baseline, a 2px surface gap between adjacent fills.
 *  - A legend whenever there are >=2 series (a single series is named by the
 *    title instead), plus direct labels on <=4 series so identity is never
 *    carried by colour alone.
 *  - Amber (--series-4) measured 2.21:1 against the chart surface, below the
 *    3:1 floor. The validator's relief rule allows it only with a visible
 *    label, so `NEEDS_DIRECT_LABEL` forces one. That is enforced here rather
 *    than trusted to each caller.
 *  - Every chart ships a table view. It is the accessible representation and
 *    the one that survives greyscale printing.
 *  - Hover is default, not an enhancement: an SVG chart in a browser is an
 *    interactive object and reading an exact value should not require guessing
 *    against an axis.
 */

import { ReactNode, useCallback, useMemo, useRef, useState } from "react";
import { compact, NEEDS_DIRECT_LABEL } from "@/lib/format";

const PAD = { t: 14, r: 16, b: 26, l: 46 };

// ---------------------------------------------------------------------------
// shared shell: title, note, legend, tooltip, table view
// ---------------------------------------------------------------------------

interface Tip {
  x: number;
  y: number;
  rows: { color?: string; label: string; value: string }[];
  head?: string;
}

function Tooltip({ tip, host }: { tip: Tip | null; host: HTMLDivElement | null }) {
  if (!tip || !host) return null;
  const w = host.clientWidth;
  // Flip before the tooltip can leave the frame, rather than letting it clip.
  const flip = tip.x > w - 150;
  return (
    <div
      className="tip"
      style={{
        left: flip ? undefined : tip.x + 12,
        right: flip ? w - tip.x + 12 : undefined,
        top: Math.max(0, tip.y - 10),
      }}
    >
      {tip.head ? <div style={{ opacity: 0.75, marginBottom: 3 }}>{tip.head}</div> : null}
      {tip.rows.map((r, i) => (
        <div className="tip-row" key={i}>
          {r.color ? <span className="tip-key" style={{ background: r.color }} /> : null}
          <span style={{ opacity: 0.8 }}>{r.label}</span>
          <b style={{ marginLeft: "auto" }}>{r.value}</b>
        </div>
      ))}
    </div>
  );
}

export function ChartFrame({
  title,
  note,
  legend,
  table,
  children,
}: {
  title?: ReactNode;
  note?: ReactNode;
  legend?: { color: string; label: string; dash?: boolean; square?: boolean }[];
  table?: ReactNode;
  children: ReactNode;
}) {
  return (
    <figure className="chart" style={{ margin: 0 }}>
      {title ? <figcaption className="chart-title">{title}</figcaption> : null}
      {note ? <div className="chart-note">{note}</div> : null}
      {children}
      {legend && legend.length > 1 ? (
        <div className="legend">
          {legend.map((l) => (
            <span className="legend-item" key={l.label}>
              <span
                className={`legend-key${l.square ? " sq" : ""}`}
                style={{
                  background: l.dash
                    ? `repeating-linear-gradient(90deg, ${l.color} 0 4px, transparent 4px 7px)`
                    : l.color,
                }}
              />
              {l.label}
            </span>
          ))}
        </div>
      ) : null}
      {table ? (
        <details style={{ marginTop: "var(--s-3)" }}>
          <summary
            className="tiny"
            style={{ cursor: "pointer", color: "var(--brand-800)", fontWeight: 600 }}
          >
            Show as table
          </summary>
          <div className="table-wrap" style={{ marginTop: "var(--s-2)" }}>
            {table}
          </div>
        </details>
      ) : null}
    </figure>
  );
}

const s_hasPoints = (s: { points: unknown[] }) => s.points.length > 0;

function useTip() {
  const ref = useRef<HTMLDivElement>(null);
  const [tip, setTip] = useState<Tip | null>(null);
  return { ref, tip, setTip, clear: useCallback(() => setTip(null), []) };
}

// ---------------------------------------------------------------------------
// scales / axes
// ---------------------------------------------------------------------------

function ticks(lo: number, hi: number, n = 4): number[] {
  if (!Number.isFinite(lo) || !Number.isFinite(hi) || lo === hi) return [lo];
  const raw = (hi - lo) / n;
  const mag = Math.pow(10, Math.floor(Math.log10(Math.abs(raw))));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? mag * 10;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(v);
  return out;
}

function Grid({
  w,
  h,
  yTicks,
  yOf,
  fmt,
}: {
  w: number;
  h: number;
  yTicks: number[];
  yOf: (v: number) => number;
  fmt: (v: number) => string;
}) {
  return (
    <g>
      {yTicks.map((t) => (
        <g key={t}>
          <line
            x1={PAD.l}
            x2={w - PAD.r}
            y1={yOf(t)}
            y2={yOf(t)}
            stroke="var(--grid)"
            strokeWidth="1"
          />
          <text
            x={PAD.l - 8}
            y={yOf(t)}
            textAnchor="end"
            dominantBaseline="middle"
            fontSize="10"
            fill="var(--ink-muted)"
            style={{ fontVariantNumeric: "tabular-nums" }}
          >
            {fmt(t)}
          </text>
        </g>
      ))}
      <line
        x1={PAD.l}
        x2={w - PAD.r}
        y1={h - PAD.b}
        y2={h - PAD.b}
        stroke="var(--border-strong)"
        strokeWidth="1"
      />
    </g>
  );
}

// ---------------------------------------------------------------------------
// FanChart — forecast point + prediction interval
// ---------------------------------------------------------------------------

export function FanChart({
  points,
  confidence,
  title,
  note,
}: {
  points: { as_of_date: string; point: number; lower: number; upper: number }[];
  confidence: number;
  title?: ReactNode;
  note?: ReactNode;
}) {
  const W = 720;
  const H = 260;
  const { ref, tip, setTip, clear } = useTip();

  const geom = useMemo(() => {
    const lo = Math.min(...points.map((p) => p.lower));
    const hi = Math.max(...points.map((p) => p.upper));
    const padY = (hi - lo) * 0.08 || 1;
    const y0 = lo - padY;
    const y1 = hi + padY;
    const xOf = (i: number) =>
      PAD.l + (i / Math.max(1, points.length - 1)) * (W - PAD.l - PAD.r);
    const yOf = (v: number) =>
      H - PAD.b - ((v - y0) / (y1 - y0)) * (H - PAD.t - PAD.b);
    return { xOf, yOf, y0, y1 };
  }, [points]);

  if (!points.length) return null;

  const band =
    points.map((p, i) => `${i ? "L" : "M"}${geom.xOf(i)},${geom.yOf(p.upper)}`).join(" ") +
    " " +
    points
      .map((p, i) => `L${geom.xOf(points.length - 1 - i)},${geom.yOf(points[points.length - 1 - i].lower)}`)
      .join(" ") +
    " Z";
  const line = points
    .map((p, i) => `${i ? "L" : "M"}${geom.xOf(i)},${geom.yOf(p.point)}`)
    .join(" ");

  const zeroY = geom.y0 <= 0 && geom.y1 >= 0 ? geom.yOf(0) : null;

  const onMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const host = ref.current;
    if (!host) return;
    const r = host.getBoundingClientRect();
    const px = ((e.clientX - r.left) / r.width) * W;
    const i = Math.round(
      ((px - PAD.l) / (W - PAD.l - PAD.r)) * (points.length - 1)
    );
    const p = points[Math.max(0, Math.min(points.length - 1, i))];
    if (!p) return;
    setTip({
      x: ((geom.xOf(Math.max(0, Math.min(points.length - 1, i))) / W) * r.width),
      y: ((geom.yOf(p.point) / H) * r.height),
      head: new Date(p.as_of_date).toLocaleDateString("en-IN", {
        day: "numeric",
        month: "short",
      }),
      rows: [
        { color: "var(--series-1)", label: "Expected", value: compact(p.point) },
        { label: `Range (${Math.round(confidence * 100)}%)`, value: `${compact(p.lower)} — ${compact(p.upper)}` },
      ],
    });
  };

  return (
    <ChartFrame
      title={title}
      note={note}
      legend={[
        { color: "var(--series-1)", label: "Most likely" },
        { color: "var(--brand-100)", label: `${Math.round(confidence * 100)}% range`, square: true },
      ]}
      table={
        <table className="data">
          <thead>
            <tr><th>Date</th><th>Low</th><th>Expected</th><th>High</th></tr>
          </thead>
          <tbody>
            {points.filter((_, i) => i % 7 === 0).map((p) => (
              <tr key={p.as_of_date}>
                <td>{new Date(p.as_of_date).toLocaleDateString("en-IN", { day: "numeric", month: "short" })}</td>
                <td>{compact(p.lower)}</td>
                <td>{compact(p.point)}</td>
                <td>{compact(p.upper)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      }
    >
      <div className="chart-scroll">
      <div className="chart-frame" ref={ref}>
        <svg viewBox={`0 0 ${W} ${H}`} role="img" onMouseMove={onMove} onMouseLeave={clear}>
          <Grid w={W} h={H} yTicks={ticks(geom.y0, geom.y1, 4)} yOf={geom.yOf} fmt={compact} />
          {/* A neutral reference line, not a danger threshold. This series is
              operating flow excluding customer payments, so sitting below zero
              is normal rather than alarming — painting it in the critical
              colour would teach the reader to panic at the expected case. */}
          {zeroY !== null ? (
            <>
              <line x1={PAD.l} x2={W - PAD.r} y1={zeroY} y2={zeroY}
                stroke="var(--border-strong)" strokeWidth="1" strokeDasharray="4 3" />
              <text x={PAD.l + 3} y={zeroY - 5} fontSize="10" fill="var(--ink-muted)">
                break-even
              </text>
            </>
          ) : null}
          <path d={band} fill="var(--brand-100)" opacity="0.75" />
          <path d={line} fill="none" stroke="var(--series-1)" strokeWidth="2"
            strokeLinejoin="round" strokeLinecap="round" />
          {tip ? (
            <line
              x1={(tip.x / (ref.current?.clientWidth ?? W)) * W}
              x2={(tip.x / (ref.current?.clientWidth ?? W)) * W}
              y1={PAD.t} y2={H - PAD.b}
              stroke="var(--ink-muted)" strokeWidth="1" strokeDasharray="3 3"
            />
          ) : null}
          {points.map((p, i) =>
            i % 14 === 0 ? (
              <text key={i} x={geom.xOf(i)} y={H - PAD.b + 14} textAnchor="middle"
                fontSize="10" fill="var(--ink-muted)">
                {new Date(p.as_of_date).toLocaleDateString("en-IN", { day: "numeric", month: "short" })}
              </text>
            ) : null
          )}
        </svg>
        <Tooltip tip={tip} host={ref.current} />
      </div>
      </div>
    </ChartFrame>
  );
}

// ---------------------------------------------------------------------------
// LineChart — multi-series over an ordered x
// ---------------------------------------------------------------------------

export interface Series {
  name: string;
  color: string;
  points: { x: number | string; y: number }[];
  dash?: boolean;
}

export function LineChart({
  series,
  title,
  note,
  yFmt = compact,
  xLabels,
  yDomain,
  reference,
  table,
  height = 220,
}: {
  series: Series[];
  title?: ReactNode;
  note?: ReactNode;
  yFmt?: (v: number) => string;
  xLabels?: string[];
  yDomain?: [number, number];
  reference?: { y: number; label: string };
  table?: ReactNode;
  height?: number;
}) {
  const W = 720;
  const H = height;
  const { ref, tip, setTip, clear } = useTip();
  const n = Math.max(...series.map((s) => s.points.length), 1);

  const all = series.flatMap((s) => s.points.map((p) => p.y));
  if (reference) all.push(reference.y);
  const lo = yDomain ? yDomain[0] : Math.min(...all);
  const hi = yDomain ? yDomain[1] : Math.max(...all);
  const pad = (hi - lo) * 0.1 || Math.abs(hi) * 0.1 || 1;
  const y0 = yDomain ? lo : lo - pad;
  const y1 = yDomain ? hi : hi + pad;

  const xOf = (i: number) => PAD.l + (i / Math.max(1, n - 1)) * (W - PAD.l - PAD.r);
  const yOf = (v: number) => H - PAD.b - ((v - y0) / (y1 - y0)) * (H - PAD.t - PAD.b);

  const onMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const host = ref.current;
    if (!host) return;
    const r = host.getBoundingClientRect();
    const px = ((e.clientX - r.left) / r.width) * W;
    const i = Math.max(0, Math.min(n - 1, Math.round(((px - PAD.l) / (W - PAD.l - PAD.r)) * (n - 1))));
    setTip({
      x: (xOf(i) / W) * r.width,
      y: 8,
      head: xLabels?.[i] ?? String(series[0]?.points[i]?.x ?? i),
      rows: series
        .filter((s) => s.points[i])
        .map((s) => ({ color: s.color, label: s.name, value: yFmt(s.points[i].y) })),
    });
  };

  return (
    <ChartFrame
      title={title}
      note={note}
      legend={series.map((s) => ({ color: s.color, label: s.name, dash: s.dash }))}
      table={table}
    >
      <div className="chart-scroll">
      <div className="chart-frame" ref={ref}>
        <svg viewBox={`0 0 ${W} ${H}`} role="img" onMouseMove={onMove} onMouseLeave={clear}>
          <Grid w={W} h={H} yTicks={ticks(y0, y1, 4)} yOf={yOf} fmt={yFmt} />
          {reference ? (
            <>
              <line x1={PAD.l} x2={W - PAD.r} y1={yOf(reference.y)} y2={yOf(reference.y)}
                stroke="var(--ink-muted)" strokeWidth="1.5" strokeDasharray="5 4" />
              <text x={W - PAD.r} y={yOf(reference.y) - 5} textAnchor="end"
                fontSize="10" fill="var(--ink-muted)">{reference.label}</text>
            </>
          ) : null}
          {series.map((s) => (
            <path
              key={s.name}
              d={s.points.map((p, i) => `${i ? "L" : "M"}${xOf(i)},${yOf(p.y)}`).join(" ")}
              fill="none"
              stroke={s.color}
              strokeWidth="2"
              strokeDasharray={s.dash ? "5 4" : undefined}
              strokeLinejoin="round"
              strokeLinecap="round"
            />
          ))}
          {/* Direct labels at the series end — identity without relying on
              colour. Series that converge (regret lines all near zero, say)
              would overprint into unreadable mush, so a label is dropped when
              it cannot clear the one already placed. The legend still carries
              identity for anything dropped.

              Two series are never dropped: any below the 3:1 contrast floor,
              where the label IS the required relief. Those are placed first so
              they always win the space. */}
          {series.length <= 4
            ? (() => {
                const ends = series
                  .map((s, i) => ({ s, i, y: yOf(s.points[s.points.length - 1]?.y ?? 0) }))
                  .filter((e) => s_hasPoints(e.s))
                  .sort((a, b) => {
                    const ap = NEEDS_DIRECT_LABEL.has(a.s.color) ? 0 : 1;
                    const bp = NEEDS_DIRECT_LABEL.has(b.s.color) ? 0 : 1;
                    return ap - bp || a.y - b.y;
                  });
                const placed: number[] = [];
                return ends.map((e) => {
                  const mustLabel = NEEDS_DIRECT_LABEL.has(e.s.color);
                  const collides = placed.some((p) => Math.abs(p - e.y) < 13);
                  if (collides && !mustLabel) return null;
                  placed.push(e.y);
                  return (
                    <text
                      key={e.s.name}
                      x={xOf(e.s.points.length - 1) + 4}
                      y={e.y - 6}
                      fontSize="10"
                      fontWeight="600"
                      fill={e.s.color}
                      textAnchor="end"
                    >
                      {e.s.name}
                    </text>
                  );
                });
              })()
            : null}
          {tip ? (
            <line x1={(tip.x / (ref.current?.clientWidth ?? W)) * W}
              x2={(tip.x / (ref.current?.clientWidth ?? W)) * W}
              y1={PAD.t} y2={H - PAD.b}
              stroke="var(--ink-muted)" strokeWidth="1" strokeDasharray="3 3" />
          ) : null}
          {xLabels
            ? xLabels.map((l, i) =>
                i % Math.ceil(n / 6) === 0 ? (
                  <text key={i} x={xOf(i)} y={H - PAD.b + 14} textAnchor="middle"
                    fontSize="10" fill="var(--ink-muted)">{l}</text>
                ) : null
              )
            : null}
        </svg>
        <Tooltip tip={tip} host={ref.current} />
      </div>
      </div>
    </ChartFrame>
  );
}

// ---------------------------------------------------------------------------
// BarChart — horizontal, categorical
// ---------------------------------------------------------------------------

export function BarChart({
  bars,
  title,
  note,
  fmt = compact,
  table,
}: {
  bars: { label: string; value: number; color: string; sub?: string }[];
  title?: ReactNode;
  note?: ReactNode;
  fmt?: (v: number) => string;
  table?: ReactNode;
}) {
  const max = Math.max(...bars.map((b) => Math.abs(b.value)), 1);
  return (
    <ChartFrame title={title} note={note} table={table}>
      <div className="stack-sm" style={{ gap: "var(--s-3)" }}>
        {bars.map((b) => {
          const w = (Math.abs(b.value) / max) * 100;
          // Amber is below the 3:1 contrast floor; the value label beside the
          // bar is its required relief, so it is never optional.
          const mustLabel = NEEDS_DIRECT_LABEL.has(b.color);
          return (
            <div key={b.label}>
              <div className="row-between" style={{ gap: "var(--s-2)", marginBottom: 4 }}>
                <span className="small" style={{ fontWeight: 560 }}>
                  <span className="swatch" style={{ background: b.color }} />
                  {b.label}
                </span>
                <span className="small num" style={{ fontWeight: 640 }}>
                  {fmt(b.value)}
                </span>
              </div>
              <div
                style={{
                  height: 12,
                  background: "var(--surface-sunken)",
                  borderRadius: 4,
                  overflow: "hidden",
                }}
              >
                <div
                  style={{
                    width: `${w}%`,
                    height: "100%",
                    background: b.color,
                    borderRadius: 4,
                    // 2px surface gap so adjacent fills never touch.
                    boxShadow: "inset -2px 0 0 var(--surface-chart)",
                  }}
                  title={`${b.label}: ${fmt(b.value)}`}
                />
              </div>
              {b.sub || mustLabel ? (
                <div className="tiny muted" style={{ marginTop: 3 }}>{b.sub}</div>
              ) : null}
            </div>
          );
        })}
      </div>
    </ChartFrame>
  );
}

// ---------------------------------------------------------------------------
// CalibrationCurve — predicted vs observed against the ideal diagonal
// ---------------------------------------------------------------------------

export function CalibrationCurve({
  bins,
  title,
  note,
}: {
  bins: { mean_predicted: number; observed_rate: number; n: number }[];
  title?: ReactNode;
  note?: ReactNode;
}) {
  const W = 340;
  const H = 300;
  const { ref, tip, setTip, clear } = useTip();
  const hi = Math.max(0.35, ...bins.map((b) => Math.max(b.mean_predicted, b.observed_rate))) * 1.1;
  const xOf = (v: number) => PAD.l + (v / hi) * (W - PAD.l - PAD.r);
  const yOf = (v: number) => H - PAD.b - (v / hi) * (H - PAD.t - PAD.b);
  const maxN = Math.max(...bins.map((b) => b.n), 1);

  return (
    <ChartFrame
      title={title}
      note={note}
      legend={[
        { color: "var(--series-3)", label: "Observed", square: true },
        { color: "var(--ink-muted)", label: "Perfect calibration", dash: true },
      ]}
      table={
        <table className="data">
          <thead><tr><th>Predicted</th><th>Observed</th><th>n</th></tr></thead>
          <tbody>
            {bins.map((b, i) => (
              <tr key={i}>
                <td>{(b.mean_predicted * 100).toFixed(1)}%</td>
                <td>{(b.observed_rate * 100).toFixed(1)}%</td>
                <td>{b.n}</td>
              </tr>
            ))}
          </tbody>
        </table>
      }
    >
      <div className="chart-scroll">
      <div className="chart-frame" ref={ref}>
        <svg viewBox={`0 0 ${W} ${H}`} role="img" onMouseLeave={clear}>
          <Grid w={W} h={H} yTicks={ticks(0, hi, 4)} yOf={yOf} fmt={(v) => `${Math.round(v * 100)}%`} />
          <line x1={xOf(0)} y1={yOf(0)} x2={xOf(hi)} y2={yOf(hi)}
            stroke="var(--ink-muted)" strokeWidth="1.5" strokeDasharray="5 4" />
          <path
            d={bins.map((b, i) => `${i ? "L" : "M"}${xOf(b.mean_predicted)},${yOf(b.observed_rate)}`).join(" ")}
            fill="none" stroke="var(--series-3)" strokeWidth="2" strokeLinejoin="round"
          />
          {bins.map((b, i) => (
            <circle
              key={i}
              cx={xOf(b.mean_predicted)}
              cy={yOf(b.observed_rate)}
              r={Math.max(4, 4 + (b.n / maxN) * 5)}
              fill="var(--series-3)"
              stroke="var(--surface-chart)"
              strokeWidth="2"
              onMouseEnter={(e) => {
                const host = ref.current;
                if (!host) return;
                const r = host.getBoundingClientRect();
                setTip({
                  x: (xOf(b.mean_predicted) / W) * r.width,
                  y: (yOf(b.observed_rate) / H) * r.height,
                  rows: [
                    { label: "Predicted", value: `${(b.mean_predicted * 100).toFixed(1)}%` },
                    { label: "Actually happened", value: `${(b.observed_rate * 100).toFixed(1)}%` },
                    { label: "Cases", value: String(b.n) },
                  ],
                });
              }}
            />
          ))}
          <text x={(W + PAD.l) / 2} y={H - 4} textAnchor="middle" fontSize="10" fill="var(--ink-muted)">
            predicted probability
          </text>
        </svg>
        <Tooltip tip={tip} host={ref.current} />
      </div>
      </div>
    </ChartFrame>
  );
}

// ---------------------------------------------------------------------------
// ContributionChart — diverging bars (SHAP / coefficient attribution)
// ---------------------------------------------------------------------------

export function ContributionChart({
  items,
  title,
  note,
}: {
  items: { feature: string; value: number; contribution: number }[];
  title?: ReactNode;
  note?: ReactNode;
}) {
  const max = Math.max(...items.map((i) => Math.abs(i.contribution)), 1e-9);
  return (
    <ChartFrame
      title={title}
      note={note}
      legend={[
        { color: "var(--serious)", label: "Raises risk", square: true },
        { color: "var(--good)", label: "Lowers risk", square: true },
      ]}
      table={
        <table className="data">
          <thead><tr><th>Factor</th><th>Value</th><th>Effect</th></tr></thead>
          <tbody>
            {items.map((i) => (
              <tr key={i.feature}>
                <td>{i.feature}</td>
                <td>{i.value.toFixed(3)}</td>
                <td>{i.contribution >= 0 ? "+" : ""}{i.contribution.toFixed(4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      }
    >
      <div className="stack-sm" style={{ gap: 10 }}>
        {items.map((it) => {
          const up = it.contribution >= 0;
          const w = (Math.abs(it.contribution) / max) * 50;
          return (
            <div key={it.feature}>
              <div className="row-between" style={{ marginBottom: 3 }}>
                <span className="tiny secondary">{it.feature}</span>
                <span className="tiny num muted">{it.value.toFixed(2)}</span>
              </div>
              <div style={{ position: "relative", height: 12, background: "var(--surface-sunken)", borderRadius: 4 }}>
                <div style={{ position: "absolute", left: "50%", top: -2, bottom: -2, width: 1, background: "var(--border-strong)" }} />
                <div
                  style={{
                    position: "absolute",
                    left: up ? "50%" : `${50 - w}%`,
                    width: `${w}%`,
                    top: 0,
                    bottom: 0,
                    background: up ? "var(--serious)" : "var(--good)",
                    borderRadius: 4,
                  }}
                  title={`${it.feature}: ${up ? "raises" : "lowers"} risk by ${Math.abs(it.contribution).toFixed(4)}`}
                />
              </div>
            </div>
          );
        })}
      </div>
    </ChartFrame>
  );
}

// ---------------------------------------------------------------------------
// Sparkline — inline trend, no axes
// ---------------------------------------------------------------------------

export function Sparkline({
  values,
  color = "var(--series-1)",
  w = 96,
  h = 26,
  label,
}: {
  values: number[];
  color?: string;
  w?: number;
  h?: number;
  label?: string;
}) {
  if (values.length < 2) return null;
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const span = hi - lo || 1;
  const d = values
    .map((v, i) => `${i ? "L" : "M"}${(i / (values.length - 1)) * w},${h - ((v - lo) / span) * h}`)
    .join(" ");
  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} role="img"
      aria-label={label ?? "trend"} style={{ overflow: "visible", display: "block" }}>
      <path d={d} fill="none" stroke={color} strokeWidth="1.75" strokeLinejoin="round" strokeLinecap="round" />
      <circle cx={w} cy={h - ((values[values.length - 1] - lo) / span) * h} r="2.5" fill={color} />
    </svg>
  );
}

// ---------------------------------------------------------------------------
// ConvergencePlot — log-log, for the Monte Carlo error decay
// ---------------------------------------------------------------------------

export function ConvergencePlot({
  points,
  title,
  note,
}: {
  points: { n: number; se: number }[];
  title?: ReactNode;
  note?: ReactNode;
}) {
  const W = 420;
  const H = 240;
  const usable = points.filter((p) => p.se > 0);
  if (usable.length < 2) return null;
  const lx = usable.map((p) => Math.log10(p.n));
  const ly = usable.map((p) => Math.log10(p.se));
  const x0 = Math.min(...lx), x1 = Math.max(...lx);
  const y0 = Math.min(...ly) - 0.1, y1 = Math.max(...ly) + 0.1;
  const xOf = (v: number) => PAD.l + ((v - x0) / (x1 - x0 || 1)) * (W - PAD.l - PAD.r);
  const yOf = (v: number) => H - PAD.b - ((v - y0) / (y1 - y0 || 1)) * (H - PAD.t - PAD.b);

  return (
    <ChartFrame
      title={title}
      note={note}
      legend={[
        { color: "var(--series-1)", label: "Measured error" },
        { color: "var(--ink-muted)", label: "Theory: N^-1/2", dash: true },
      ]}
      table={
        <table className="data">
          <thead><tr><th>Iterations</th><th>Standard error</th></tr></thead>
          <tbody>
            {points.map((p) => (
              <tr key={p.n}><td>{p.n.toLocaleString()}</td><td>{p.se.toFixed(4)}</td></tr>
            ))}
          </tbody>
        </table>
      }
    >
      <svg viewBox={`0 0 ${W} ${H}`} role="img">
        <Grid w={W} h={H} yTicks={ticks(y0, y1, 4)} yOf={yOf} fmt={(v) => Math.pow(10, v).toFixed(2)} />
        <line
          x1={xOf(x0)} y1={yOf(ly[0])}
          x2={xOf(x1)} y2={yOf(ly[0] - 0.5 * (x1 - x0))}
          stroke="var(--ink-muted)" strokeWidth="1.5" strokeDasharray="5 4"
        />
        <path d={usable.map((p, i) => `${i ? "L" : "M"}${xOf(lx[i])},${yOf(ly[i])}`).join(" ")}
          fill="none" stroke="var(--series-1)" strokeWidth="2" strokeLinejoin="round" />
        {usable.map((p, i) => (
          <circle key={p.n} cx={xOf(lx[i])} cy={yOf(ly[i])} r="4"
            fill="var(--series-1)" stroke="var(--surface-chart)" strokeWidth="2">
            <title>{`${p.n.toLocaleString()} iterations → SE ${p.se.toFixed(4)}`}</title>
          </circle>
        ))}
        <text x={(W + PAD.l) / 2} y={H - 4} textAnchor="middle" fontSize="10" fill="var(--ink-muted)">
          iterations (log scale)
        </text>
      </svg>
    </ChartFrame>
  );
}
