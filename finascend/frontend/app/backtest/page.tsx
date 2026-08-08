"use client";

/**
 * 05 — Backtest report. Regret over time and calibration over time, plus the
 * before/after on the interval fix and the generated Markdown itself.
 */

import { api, BacktestSummary } from "@/lib/api";
import { Card, Chip, ErrorState, Legend, Loading, PageHead, Tile, useApi } from "@/components/ui";
import { BarChart, LineChart, SERIES, Sparkline, fmt, pct } from "@/components/charts";

const LABEL: Record<string, string> = {
  rules_baseline: "Rules baseline",
  lp_optimizer: "LP / MILP",
  dp_knapsack: "Knapsack DP",
  chance_constrained: "Chance-constrained",
};

export default function BacktestPage() {
  const sum = useApi(() => api.get<BacktestSummary>("/backtest/summary"));
  const md = useApi(() => api.get<{ markdown: string; source: string }>("/backtest/report"));

  const s = sum.data;
  const cal = s?.calibration;
  const covered = cal ? Math.abs(cal.empirical - cal.nominal) <= 0.02 : false;

  return (
    <>
      <PageHead
        title="Backtest report"
        sub={
          <>
            A walk-forward replay with a structural no-look-ahead boundary: every step routes
            through <code>build_as_of_view</code>, which slices the world once and returns an
            object containing nothing dated after the decision date. The step function never
            receives the full dataset, so it cannot leak what it was not given — a leaky
            backtest is invisible in its own output, because it simply looks like a good model.
          </>
        }
      />

      {sum.error ? (
        <ErrorState error={sum.error} />
      ) : sum.loading || !s || !cal ? (
        <Loading rows={6} />
      ) : (
        <>
          <div className="grid grid-4">
            <Tile label="Decision points" value={s.config.n_steps as number}
                  sub={`${s.config.step_days}-day steps, ${s.config.horizon_days}-day horizon`} />
            <Tile label="Interval coverage" value={pct(cal.empirical)}
                  tone={covered ? "good" : "critical"}
                  sub={`against a ${pct(cal.nominal, 0)} nominal level, over ${cal.n_observations.toLocaleString()} forecast/outcome pairs`} />
            <Tile label="Was" value={pct(cal.previous_build.pooled)} tone="critical"
                  sub={`${cal.previous_build.note} — ${cal.previous_build.pooled_config}`} />
            <Tile label="Monte Carlo iterations" value={(s.config.n_iterations as number).toLocaleString()}
                  sub={`mean SE ${(s.steps.reduce((a, x) => a + x.mc_standard_error, 0) / s.steps.length).toFixed(3)} days on the RaR estimate`} />
          </div>

          <Card
            title="Interval calibration — before and after the conformal fix"
            note="The shortfall was never spread across the model. SARIMAX's analytic intervals were already honest; the 26% of steps that selected Holt-Winters ran a different construction — per-horizon quantiles of five walk-forward residuals — whose coverage ceiling is (n−1)/(n+1) = 66.7% and which therefore could not have expressed a 95% level however good the forecast was."
            right={<Chip tone={covered ? "good" : "critical"}>{covered ? "calibrated" : "off nominal"}</Chip>}
          >
            <BarChart
              bars={[
                { label: "pooled — before", value: cal.previous_build.pooled, color: "var(--series-4)" },
                { label: "pooled — after", value: cal.empirical, color: "var(--series-2)" },
                { label: "sarimax branch — before", value: cal.previous_build.sarimax_branch, color: "var(--series-4)" },
                { label: "holt_winters — before", value: cal.previous_build.holt_winters_branch, color: "var(--series-4)" },
              ]}
              valueFormat={(v) => pct(v)}
              note="empirical coverage, before and after"
            />
            <div className="card-note" style={{ marginTop: 16 }}>
              <strong>Not a blanket widening.</strong> The intervals genuinely are wider now;
              they had to be. Three things separate that from inflating them until the number
              looked right. The multiplier is <em>measured</em> from held-out error and never
              tuned against the coverage target. It lands within a few percent of the value a
              correctly specified model needs anyway — see q̂/z below. And coverage is now flat
              across the horizon, every band within about a point of nominal, where a blanket
              widening would over-cover short horizons to rescue long ones.
            </div>
          </Card>

          <Card
            title="Calibration over time — the conformal multiplier at each step"
            note="q̂ multiplies a standard deviation, so the reference line is z = 1.96 — the multiple of sigma a correctly specified model needs to cover 95% — not 1.0. Distance from that line is how far each step's model was from telling the truth about its own uncertainty."
          >
            <LineChart
              height={220}
              yLabel="q-hat"
              yFormat={(v) => v.toFixed(2)}
              markers={[{ y: 1.96, label: "z = 1.96 · a correctly scaled model", color: "var(--status-good)" }]}
              series={[
                {
                  name: "conformal q̂",
                  color: "var(--series-2)",
                  points: s.steps.map((st) => ({ x: st.as_of, y: st.conformal_q_hat })),
                },
              ]}
            />
            <div className="grid grid-3" style={{ marginTop: 20 }}>
              {Object.entries(
                s.steps.reduce<Record<string, number[]>>((acc, st) => {
                  (acc[st.forecast_model] ??= []).push(st.conformal_q_hat);
                  return acc;
                }, {})
              ).map(([model, qs], i) => (
                <div key={model} className="tile">
                  <div className="tile-label">{model}</div>
                  <div className="tile-value num" style={{ fontSize: "var(--fs-lg)" }}>
                    {((qs.reduce((a, b) => a + b, 0) / qs.length) / 1.959963984540054).toFixed(3)}
                    <span className="tile-unit">q̂/z</span>
                  </div>
                  <div className="tile-sub">
                    mean q̂ {(qs.reduce((a, b) => a + b, 0) / qs.length).toFixed(3)} over{" "}
                    {qs.length} steps · range {Math.min(...qs).toFixed(2)}–{Math.max(...qs).toFixed(2)}.
                    q̂/z is how many times too narrow this model&apos;s own scale was; 1.0 is right.
                  </div>
                </div>
              ))}
            </div>
          </Card>

          <Card
            title="Coverage by forecast horizon band"
            note="Reported as bands rather than single horizons. The replay advances as_of by a whole number of weeks, so a fixed horizon h always lands on the same weekday — and coverage genuinely varies by weekday, because weekend flows are near zero. The old six-checkpoint table averaged 89.5% while true pooled coverage was 85.9%: it was aliased against the calendar and flattered the result."
          >
            <LineChart
              height={220}
              yLabel="coverage"
              yFormat={(v) => pct(v, 0)}
              markers={[{ y: cal.nominal, label: `nominal ${pct(cal.nominal, 0)}` }]}
              series={[
                {
                  name: "empirical coverage",
                  color: "var(--series-2)",
                  points: cal.by_horizon.map((h) => ({ x: `d${h.label}`, y: h.coverage })),
                },
              ]}
            />
            <table className="data" style={{ marginTop: 16 }}>
              <thead>
                <tr>
                  <th>Days ahead</th><th className="n">n</th>
                  <th className="n">Coverage</th><th className="n">Mean width</th>
                </tr>
              </thead>
              <tbody>
                {cal.by_horizon.map((h) => (
                  <tr key={h.label}>
                    <td className="num">{h.label}</td>
                    <td className="n">{h.n.toLocaleString()}</td>
                    <td className="n" style={{
                      color: Math.abs(h.coverage - cal.nominal) <= 0.03 ? "var(--status-good)" : "var(--status-warn)",
                    }}>{pct(h.coverage)}</td>
                    <td className="n">{fmt(h.mean_width)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card
            title="Regret over time, by strategy"
            note="Regret is the extra penalty versus a planner who knew exactly what cash would arrive. It isolates the cost of uncertainty from the cost of bad method: even perfect foresight cannot avoid all penalty when obligations exceed available cash."
          >
            <Legend items={s.strategies.map((x) => ({ label: LABEL[x.name], color: SERIES[x.name] }))} />
            <LineChart
              height={280}
              yLabel="regret"
              series={s.strategies.map((x, i) => ({
                name: LABEL[x.name],
                color: SERIES[x.name],
                dash: ["", "6 3", "2 3", "9 3 2 3"][i],
                points: x.regret_series.map((r) => ({ x: r.as_of, y: r.regret })),
              }))}
            />
          </Card>

          <Card
            title="Runway-at-Risk across the replay"
            note="RaR and CRaR at every decision point. CRaR sits at or below RaR at every step, as it must — it is the mean of the tail beyond the RaR threshold. Were it ever above, the tail direction would be inverted and the metric would report comfort where it should report danger."
          >
            <Legend items={[
              { label: "RaR (95%)", color: "var(--series-2)" },
              { label: "CRaR (95%)", color: "var(--series-3)" },
            ]} />
            <LineChart
              height={240}
              yLabel="days"
              yFormat={(v) => v.toFixed(0)}
              series={[
                { name: "RaR", color: "var(--series-2)",
                  points: s.steps.map((st) => ({ x: st.as_of, y: st.runway_at_risk_days })) },
                { name: "CRaR", color: "var(--series-3)", dash: "6 3",
                  points: s.steps.map((st) => ({ x: st.as_of, y: st.conditional_runway_at_risk_days })) },
              ]}
            />
            <div className="grid grid-3" style={{ marginTop: 20 }}>
              <div className="tile">
                <div className="tile-label">Monte Carlo SE across steps</div>
                <Sparkline values={s.steps.map((st) => st.mc_standard_error)} width={170} height={34} />
                <div className="tile-sub">
                  mean {(s.steps.reduce((a, x) => a + x.mc_standard_error, 0) / s.steps.length).toFixed(3)} days
                </div>
              </div>
              <div className="tile">
                <div className="tile-label">P(shortfall) across steps</div>
                <Sparkline values={s.steps.map((st) => st.probability_of_shortfall)}
                           color="var(--series-4)" width={170} height={34} />
                <div className="tile-sub">
                  {pct(Math.min(...s.steps.map((x) => x.probability_of_shortfall)))} –{" "}
                  {pct(Math.max(...s.steps.map((x) => x.probability_of_shortfall)))}
                </div>
              </div>
              <div className="tile">
                <div className="tile-label">Opening balance across steps</div>
                <Sparkline values={s.steps.map((st) => st.opening_balance)}
                           color="var(--series-1)" width={170} height={34} />
                <div className="tile-sub">the structural break is visible as the downturn</div>
              </div>
            </div>
          </Card>

          <Card
            title="What this backtest does not establish"
            note="The limitations belong on the same page as the results, not in an appendix."
          >
            <ul style={{ fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.8, paddingLeft: 18, margin: 0 }}>
              <li><strong>Synthetic data cannot validate real-world counterparty behaviour.</strong> It validates that the estimators work against a known generating process — genuine, but strictly narrower.</li>
              <li><strong>The generating process is one the models suit.</strong> Delays are Gamma and the fitted candidates include Gamma, so the marginal fit is graded on a question it was told the answer to.</li>
              <li><strong>Regret is measured against an unachievable benchmark.</strong> It bounds efficiency loss under known dynamics; it does not predict live performance.</li>
              <li><strong>One seed, one world.</strong> Every figure describes a single realization. A production evaluation would repeat across seeds and report the distribution.</li>
              <li><strong>Interval width is still constant across weekdays</strong> while realized volatility is not — visible as over-coverage at weekends. Second-order next to what was fixed, and it errs safe, but a day-of-week variance model is the honest next step.</li>
            </ul>
          </Card>

          <Card title="Generated report" note={md.data ? `Served from ${md.data.source} — the file the harness itself wrote.` : undefined}>
            {md.error ? (
              <ErrorState error={md.error} />
            ) : md.loading || !md.data ? (
              <Loading rows={3} />
            ) : (
              <pre className="raw" style={{ maxHeight: 520 }}>{md.data.markdown}</pre>
            )}
          </Card>

          <div className="card-note" style={{ paddingLeft: 4 }}>
            Run generated {new Date(s.generated_at).toLocaleString()} · regime{" "}
            {String(s.config.regime)} · seed {String(s.config.seed)} ·{" "}
            {String(s.config.start_date)} to {String(s.config.end_date)}
          </div>
        </>
      )}
    </>
  );
}
