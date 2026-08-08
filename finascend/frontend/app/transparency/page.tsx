"use client";

/**
 * 06 — How this works.
 *
 * A first-class page, not a settings tab. The product's stated problem is a
 * trust barrier: owners don't act on advice they can't interrogate. So this
 * page does three things a settings screen would not.
 *
 *  1. States the results that make the product look WORSE, at the top, in
 *     plain language, with links to the working.
 *  2. Shows the tamper-evident decision log, including the caveat that a hash
 *     chain proves tamper-evidence and not tamper-resistance.
 *  3. Links directly into every method panel elsewhere in the app, so the
 *     audit trail points at the working rather than at the page containing it.
 */

import {
  api,
  AuditChain,
  AuditVerify,
  BacktestSummary,
  SolverComparison,
} from "@/lib/api";
import {
  Card,
  CardHead,
  EmptyState,
  Loaded,
  Method,
  RowsSkeleton,
  Status,
  Tile,
  TileSkeleton,
  useApi,
} from "@/components/ui";
import { LineChart } from "@/components/charts";
import { compact, modelLabel, pct, shortDate, STRATEGY_COLOR } from "@/lib/format";

/** Deep links into the collapsed method panels on the other pages. */
const PANELS = [
  { href: "/#method-runway", label: "How the runway number is simulated", page: "Cash health" },
  { href: "/#method-forecast", label: "How we forecast, and how honest the range is", page: "Cash health" },
  { href: "/plan#method-solvers", label: "How the payment order is decided — and where the optimiser lost", page: "Action plan" },
  { href: "/risk#method-convergence", label: "How many simulations, and why that many", page: "Risk explorer" },
  { href: "/risk#method-copula", label: "The dependence assumption between customers", page: "Risk explorer" },
  { href: "/counterparties#method-credit", label: "Credit model: calibration, attribution, vs rules", page: "Customers" },
  { href: "/receipt#method-ocr", label: "How receipt reading works, and where it fails", page: "Add a receipt" },
];

export default function Transparency() {
  const chain = useApi(() => api.get<AuditChain>("/audit/chain"));
  const verify = useApi(() => api.get<AuditVerify>("/audit/verify"));
  const bt = useApi(() => api.get<BacktestSummary>("/backtest/summary"));
  const solvers = useApi(() => api.get<SolverComparison>("/backtest/solvers"));
  const report = useApi(() => api.get<{ markdown: string; source: string }>("/backtest/report"));

  return (
    <div className="stack">
      <div className="page-head">
        <h1>How this works</h1>
        <p className="lede">
          Every number in this app can be opened up and checked. This page collects the
          places where our own testing produced an uncomfortable answer, and the record of
          what was decided and when.
        </p>
      </div>

      {/* ------------------------------------------------------------------ */}
      {/* the uncomfortable results, first                                    */}
      {/* ------------------------------------------------------------------ */}
      <Card>
        <CardHead
          title="Where our own testing says we fall short"
          note="These are here because a tool that only reports its wins hasn't told you enough to trust it."
        />
        <div className="card-body stack-sm">
          <Loaded q={solvers} skeleton={<RowsSkeleton n={2} />}>
            {(s) => {
              const rules = s.strategies.find((x) => x.name === "rules_baseline");
              const lp = s.strategies.find((x) => x.name === "lp_optimizer");
              return (
                <div className="callout callout-warning">
                  <div className="row" style={{ marginBottom: 6 }}>
                    <Status tone="warning">The clever optimiser lost to a simple rule</Status>
                  </div>
                  Over {rules?.n_steps ?? 49} replayed decisions, the simple priority rule
                  incurred {compact(rules?.total_realized_penalty ?? 0)} in late fees and the
                  exact optimiser {compact(lp?.total_realized_penalty ?? 0)} — the optimiser
                  cost {pct(lp?.vs_rules_baseline ?? 0)} more. It plans perfectly against a
                  cash forecast, and when that forecast is optimistic it commits money that
                  never arrives.{" "}
                  <a href="/plan#method-solvers">See the full comparison →</a>
                </div>
              );
            }}
          </Loaded>

          <Loaded q={bt} skeleton={<RowsSkeleton n={2} />}>
            {(b) => (
              <div className="callout callout-warning" id="method-coverage">
                <div className="row" style={{ marginBottom: 6 }}>
                  <Status tone="warning">Our forecast ranges were once too narrow</Status>
                </div>
                An earlier build published{" "}
                <strong>{pct(b.calibration.previous_build.pooled)}</strong> of outcomes
                falling inside a range claimed to hold 95% of them. Too-narrow ranges make
                runway look safer than it is — the exact failure this product exists to
                prevent. After diagnosis and a rebuilt interval construction it now
                measures <strong>{pct(b.calibration.empirical)}</strong> over{" "}
                {b.calibration.n_observations.toLocaleString()} checks.
              </div>
            )}
          </Loaded>

          <div className="callout callout-warning">
            <div className="row" style={{ marginBottom: 6 }}>
              <Status tone="serious">Receipt reading fails unsafely on bad photos</Status>
            </div>
            On poor-quality images the total amount was read <em>wrong</em> half the time
            while never once declining to answer. That is the dangerous direction, and it
            is why the app makes you confirm the amount.{" "}
            <a href="/receipt#method-ocr">See the per-quality numbers →</a>
          </div>

          <div className="callout">
            <strong>And the biggest caveat of all.</strong> This runs on generated data
            with a known answer, which is what makes the methods checkable — but it means
            none of it validates real-world customer behaviour. Every accuracy figure here
            is an upper bound on what the same approach would achieve against real
            payments.
          </div>
        </div>

        <Method id="method-coverage-detail" label="The interval fix, in full" hint="before / after by horizon">
          <Loaded q={bt}>
            {(b) => (
              <>
                <p className="method-lede">
                  The shortfall was not spread across the model — it lived entirely in one
                  branch. The diagnosis came before any change was made.
                </p>
                <div className="table-wrap">
                  <table className="data">
                    <thead>
                      <tr><th>Measured on</th><th>Before</th><th>After</th></tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td>Pooled coverage</td>
                        <td>{pct(b.calibration.previous_build.diagnostic_before_pooled)}</td>
                        <td>{pct(b.calibration.previous_build.diagnostic_after_pooled)}</td>
                      </tr>
                      <tr>
                        <td>Seasonal ARIMA branch</td>
                        <td>{pct(b.calibration.previous_build.sarimax_branch)}</td>
                        <td>—</td>
                      </tr>
                      <tr>
                        <td>Holt-Winters branch</td>
                        <td>{pct(b.calibration.previous_build.holt_winters_branch)}</td>
                        <td>—</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
                <p className="tiny muted">{b.calibration.previous_build.note}</p>

                <div className="callout">
                  <strong>Why this is a fix and not just widening the ranges.</strong> The
                  correction is measured from held-out error and was never tuned against
                  the coverage target. It lands within a few percent of what a correctly
                  specified model needs anyway. And coverage is now flat across the
                  horizon — blanket widening would have over-covered the near term to
                  rescue the far term, which shows up as a tilt.
                </div>

                <LineChart
                  title="Coverage by forecast horizon, after the fix"
                  height={190}
                  series={[{
                    name: "Coverage",
                    color: "var(--series-1)",
                    points: b.calibration.by_horizon.map((h) => ({ x: h.label, y: h.coverage })),
                  }]}
                  xLabels={b.calibration.by_horizon.map((h) => h.label)}
                  yDomain={[0.9, 1]}
                  yFmt={(v) => `${(v * 100).toFixed(0)}%`}
                  reference={{ y: b.calibration.nominal, label: "95% claimed" }}
                />

                <div className="callout callout-warning">
                  <strong>Still not fixed.</strong> Range width is constant across days of
                  the week while real volatility is not — weekends are quieter. The result
                  is over-coverage at weekends and slightly less on midweek days. It errs
                  safe, and it is second-order next to what was fixed, but it is not
                  solved.
                </div>
              </>
            )}
          </Loaded>
        </Method>
      </Card>

      {/* ------------------------------------------------------------------ */}
      {/* regret over time                                                    */}
      {/* ------------------------------------------------------------------ */}
      <Card>
        <CardHead
          title="How each method performed over time"
          note="Regret is the extra cost versus a planner who knew exactly what cash would arrive. It separates the cost of not knowing the future from the cost of a bad method."
        />
        <div className="card-body">
          <Loaded q={bt} skeleton={<div className="skel" style={{ height: 220 }} />}>
            {(b) => (
              <LineChart
                title="Cumulative regret across the replay"
                note="Lower and flatter is better. The simple rule stays lowest — that is the honest result, not a presentation choice."
                yFmt={compact}
                series={b.strategies.map((s) => {
                  let run = 0;
                  return {
                    name: modelLabel(s.name),
                    color: STRATEGY_COLOR[s.name] ?? "var(--series-1)",
                    points: s.regret_series.map((p, i) => {
                      run += p.regret;
                      return { x: i, y: run };
                    }),
                  };
                })}
                xLabels={b.strategies[0]?.regret_series.map((p) => shortDate(p.as_of)) ?? []}
                table={
                  <table className="data">
                    <thead>
                      <tr>
                        <th>Method</th><th>Total fees</th><th>Mean regret</th>
                        <th>p95 regret</th><th>Over-commitment</th>
                      </tr>
                    </thead>
                    <tbody>
                      {b.strategies.map((s) => (
                        <tr key={s.name}>
                          <td>
                            <span className="swatch" style={{ background: STRATEGY_COLOR[s.name] }} />
                            {modelLabel(s.name)}
                          </td>
                          <td>{compact(s.total_realized_penalty)}</td>
                          <td>{compact(s.mean_regret)}</td>
                          <td>{compact(s.p95_regret)}</td>
                          <td>{s.over_commitment_steps}/{s.n_steps}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                }
              />
            )}
          </Loaded>
        </div>
      </Card>

      {/* ------------------------------------------------------------------ */}
      {/* audit log                                                           */}
      {/* ------------------------------------------------------------------ */}
      <Card>
        <CardHead
          title="Decision log"
          note="Every approval is recorded in a chain where each entry seals the one before it. Change any past entry and the seals stop matching."
          aside={
            <Loaded q={verify} skeleton={<span />}>
              {(v) => (
                <Status tone={v.valid ? "good" : "critical"}>
                  {v.valid ? "Chain intact" : `Broken at #${v.first_broken_sequence}`}
                </Status>
              )}
            </Loaded>
          }
        />
        <div className="card-body">
          <Loaded q={chain} skeleton={<RowsSkeleton n={3} />}>
            {(c) =>
              c.n_entries === 0 ? (
                <EmptyState
                  title="Nothing recorded yet"
                  action={
                    <a className="btn btn-secondary" href="/plan">
                      Approve a plan to create the first entry
                    </a>
                  }
                >
                  The log is genuinely empty — we are not showing sample rows to make it
                  look populated. Approve a payment plan and it will appear here.
                </EmptyState>
              ) : (
                <>
                  <div className="grid grid-3" style={{ marginBottom: "var(--s-4)" }}>
                    <Tile label="Entries" value={c.n_entries} />
                    <Tile
                      label="Head hash"
                      value={<span className="hash">{c.head_hash?.slice(0, 12)}…</span>}
                      note="seals every entry before it"
                    />
                  </div>
                  <div className="table-wrap">
                    <table className="data">
                      <thead>
                        <tr><th>#</th><th>When</th><th>Who</th><th>What</th><th>Hash</th></tr>
                      </thead>
                      <tbody>
                        {c.entries.map((e) => (
                          <tr key={e.sequence}>
                            <td>{e.sequence}</td>
                            <td>{new Date(e.timestamp).toLocaleString("en-IN")}</td>
                            <td>{e.actor_id}</td>
                            <td style={{ textAlign: "left" }}>
                              {e.action.replace(/_/g, " ")} · {e.entity_id}
                            </td>
                            <td className="hash">{e.entry_hash.slice(0, 10)}…</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </>
              )
            }
          </Loaded>

          <Loaded q={verify} skeleton={<span />}>
            {(v) => (
              <div className="callout callout-warning" style={{ marginTop: "var(--s-4)" }}>
                <strong>What this does and does not prove.</strong> {v.caveat}
              </div>
            )}
          </Loaded>
        </div>
      </Card>

      {/* ------------------------------------------------------------------ */}
      {/* index of method panels                                              */}
      {/* ------------------------------------------------------------------ */}
      <Card>
        <CardHead
          title="Every calculation, in one list"
          note="Each link opens the relevant working directly, already expanded."
        />
        <div className="card-body">
          <div className="stack-sm">
            {PANELS.map((p) => (
              <a
                key={p.href}
                href={p.href}
                className="row-between"
                style={{
                  padding: "var(--s-3)",
                  border: "1px solid var(--border)",
                  borderRadius: "var(--radius-sm)",
                  textDecoration: "none",
                  color: "var(--ink)",
                  gap: "var(--s-3)",
                }}
              >
                <span style={{ fontWeight: 560 }}>{p.label}</span>
                <span className="tiny muted nowrap">{p.page} →</span>
              </a>
            ))}
          </div>
        </div>
      </Card>

      {/* ------------------------------------------------------------------ */}
      {/* full report                                                         */}
      {/* ------------------------------------------------------------------ */}
      <Card>
        <CardHead title="The full backtest report" note="Generated by the test harness. Every figure in it is produced by that run." />
        <div className="card-body">
          <Loaded q={report} skeleton={<RowsSkeleton n={5} />}>
            {(r) => (
              <details>
                <summary style={{ cursor: "pointer", color: "var(--brand-800)", fontWeight: 600 }}>
                  Read the raw report ({r.markdown.split("\n").length} lines)
                </summary>
                <pre
                  className="tiny"
                  style={{
                    whiteSpace: "pre-wrap", marginTop: "var(--s-3)", padding: "var(--s-4)",
                    background: "var(--surface-sunken)", border: "1px solid var(--border)",
                    borderRadius: "var(--radius-sm)", maxHeight: 460, overflow: "auto",
                  }}
                >
                  {r.markdown}
                </pre>
              </details>
            )}
          </Loaded>
        </div>
      </Card>
    </div>
  );
}
