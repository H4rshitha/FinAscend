"use client";

/**
 * 02 — Solver comparison.
 *
 * The finding goes in the page subtitle, not three scrolls down: the LP did
 * NOT beat the naive rules baseline. Burying a negative result under a table
 * is how it stops being reported.
 */

import { api, BacktestSummary, SolverComparison } from "@/lib/api";
import { Card, Chip, ErrorState, Legend, Loading, PageHead, Tile, useApi } from "@/components/ui";
import { BarChart, LineChart, SERIES, Sparkline, fmt, pct, DASHES } from "@/components/charts";

const LABEL: Record<string, string> = {
  rules_baseline: "Rules baseline",
  lp_optimizer: "LP / MILP",
  dp_knapsack: "Knapsack DP",
  chance_constrained: "Chance-constrained",
};

export default function SolversPage() {
  const cmp = useApi(() => api.get<SolverComparison>("/backtest/solvers"));
  const sum = useApi(() => api.get<BacktestSummary>("/backtest/summary"));
  const err = cmp.error ?? sum.error;

  const best = cmp.data
    ? [...cmp.data.strategies].sort((a, b) => a.total_realized_penalty - b.total_realized_penalty)[0]
    : null;

  return (
    <>
      <PageHead
        title="Solver comparison"
        sub={
          cmp.data ? (
            <>
              <strong style={{ color: "var(--status-warn)" }}>
                The optimizer did not beat the naive baseline.
              </strong>{" "}
              {cmp.data.finding}
            </>
          ) : (
            "Rules baseline versus LP/MILP versus knapsack DP versus the chance-constrained formulation, replayed over the same decision points."
          )
        }
      />

      {err ? (
        <ErrorState error={err} />
      ) : cmp.loading || !cmp.data ? (
        <Loading rows={5} />
      ) : (
        <>
          <div className="grid grid-4">
            <Tile label="Decision points" value={cmp.data.config.n_steps as number}
                  sub={`${cmp.data.config.step_days}-day steps over ${cmp.data.config.n_days} days`} />
            <Tile label="Cheapest strategy" value={LABEL[best!.name]}
                  sub={`${fmt(best!.total_realized_penalty)} realized penalty`} />
            <Tile label="Hindsight-optimal floor"
                  value={fmt(cmp.data.strategies[0].total_hindsight_penalty)}
                  sub="what a planner with perfect foresight would have paid — even it cannot reach zero, because obligations exceed cash" />
            <Tile label="Regime" value={String(cmp.data.config.regime)}
                  sub={`seed ${cmp.data.config.seed}, one world, one realization`} />
          </div>

          <Card
            title="Realized penalty and over-commitment, side by side"
            note="These two columns must be read together. Realized penalty alone rewards whoever spent most aggressively in the periods where the forecast happened to be right; the over-commitment count — planning to spend more cash than actually materialized — is what exposes the risk taken to earn it."
          >
            <Legend items={Object.keys(LABEL).map((k) => ({ label: LABEL[k], color: SERIES[k] }))} />
            <BarChart
              bars={cmp.data.strategies.map((s) => ({
                label: LABEL[s.name],
                value: s.total_realized_penalty,
                color: SERIES[s.name],
                sub: `over-commit ${s.over_commitment_steps}/${s.n_steps}`,
              }))}
              valueFormat={(n) => fmt(n)}
              note="total realized penalty by strategy"
            />

            <table className="data" style={{ marginTop: 24 }}>
              <thead>
                <tr>
                  <th>Strategy</th>
                  <th className="n">Realized penalty</th>
                  <th className="n">vs rules</th>
                  <th className="n">Relative regret</th>
                  <th className="n">Over-commitment</th>
                  <th className="n">Mean regret</th>
                  <th className="n">p95 regret</th>
                </tr>
              </thead>
              <tbody>
                {cmp.data.strategies.map((s) => (
                  <tr key={s.name}>
                    <td>
                      <span className="row-key">
                        <span className="swatch" style={{ background: SERIES[s.name] }} />
                        {LABEL[s.name]}
                      </span>
                    </td>
                    <td className="n">{s.total_realized_penalty.toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
                    <td className="n" style={{
                      color: s.name === "rules_baseline" ? "var(--text-muted)"
                        : s.vs_rules_baseline > 0 ? "var(--status-critical)" : "var(--status-good)",
                    }}>
                      {s.name === "rules_baseline" ? "—" : `${s.vs_rules_baseline > 0 ? "+" : ""}${pct(s.vs_rules_baseline)}`}
                    </td>
                    <td className="n">{pct(s.relative_regret)}</td>
                    <td className="n">
                      {s.over_commitment_steps}/{s.n_steps}
                      {s.over_commitment_steps === 0 && <> <Chip tone="good">none</Chip></>}
                    </td>
                    <td className="n">{fmt(s.mean_regret)}</td>
                    <td className="n">{fmt(s.p95_regret)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card
            title="Why the exact solver lost"
            note="The mechanism is visible in the over-commitment column, not in the penalty column."
          >
            <div style={{ display: "grid", gap: 16, gridTemplateColumns: "repeat(auto-fit,minmax(280px,1fr))" }}>
              <div>
                <div className="tile-label">The trap</div>
                <p className="card-note">
                  The LP solves the allocation <em>exactly</em> — against a <em>forecast</em> of
                  available cash. When that forecast is optimistic it commits money that never
                  arrives, and the unfunded obligations then incur their full penalty. Solving
                  the wrong problem precisely is worse than solving roughly the right problem
                  approximately: the greedy baseline is myopic, but its myopia happens to leave
                  slack that absorbs forecast error.
                </p>
              </div>
              <div>
                <div className="tile-label">The DP agrees, which is the point</div>
                <p className="card-note">
                  The knapsack DP is an independently written solver over a grid subset of the
                  LP&apos;s feasible region. It lands within a rounding error of the LP on both
                  columns — so the LP&apos;s result is a property of <em>optimizing against an
                  uncertain forecast</em>, not a bug in one solver.
                </p>
              </div>
              <div>
                <div className="tile-label">What the chance constraint buys</div>
                <p className="card-note">
                  It eliminated over-commitment entirely — and paid for that safety in penalty,
                  because reserving cash against a 5% shortfall probability means declining
                  obligations that would usually have been affordable. That is a risk-appetite
                  decision, not a modelling one, which is why the engine reports all four rather
                  than silently picking the most sophisticated.
                </p>
              </div>
            </div>
          </Card>

          <Card
            title="Regret over time, by strategy"
            note="Regret is the extra penalty versus perfect foresight. It is near zero for long stretches and then spikes at the moments the forecast missed a turning point — which is exactly the structure a single average hides. Series are distinguished by dash pattern as well as hue, so identity never rests on colour alone."
          >
            {sum.loading || !sum.data ? (
              <Loading rows={3} />
            ) : (
              <>
                <Legend items={sum.data.strategies.map((s) => ({ label: LABEL[s.name], color: SERIES[s.name] }))} />
                <LineChart
                  height={280}
                  yLabel="regret"
                  series={sum.data.strategies.map((s, i) => ({
                    name: LABEL[s.name],
                    color: SERIES[s.name],
                    dash: DASHES[i],
                    points: s.regret_series.map((r) => ({ x: r.as_of, y: r.regret })),
                  }))}
                />
                <div className="grid grid-4" style={{ marginTop: 20 }}>
                  {sum.data.strategies.map((s) => (
                    <div key={s.name} className="tile">
                      <div className="tile-label">{LABEL[s.name]}</div>
                      <Sparkline
                        values={s.regret_series.map((r) => r.regret)}
                        color={SERIES[s.name]}
                        width={150}
                        height={34}
                      />
                      <div className="tile-sub">
                        peak {fmt(Math.max(...s.regret_series.map((r) => r.regret)))} ·{" "}
                        {s.regret_series.filter((r) => r.regret > 0).length} of {s.n_steps} steps
                        with non-zero regret
                      </div>
                    </div>
                  ))}
                </div>
              </>
            )}
          </Card>

          <div className="card-note" style={{ paddingLeft: 4 }}>
            Backtest generated {new Date(cmp.data.generated_at).toLocaleString()} — served from
            the artifact <code>BACKTEST_REPORT.json</code> that the harness itself wrote, so no
            number here can differ from the one in the Markdown report.
          </div>
        </>
      )}
    </>
  );
}
