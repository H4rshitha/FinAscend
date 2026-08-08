"use client";

/**
 * 01 — Cash health.
 *
 * The default view answers one question in one sentence: how long will the
 * money last. Everything statistical is real and present, but folded into
 * method panels so the owner is not made to parse a confidence interval to
 * learn that they have about six weeks.
 *
 * The headline is Runway-at-Risk, NOT the point estimate — `days_to_zero`
 * ignores uncertainty and reads more optimistic than the evidence supports,
 * which is the exact failure this product exists to avoid. The point estimate
 * still appears, inside the method panel, labelled as the weaker number.
 */

import {
  api,
  BacktestSummary,
  FinancialSummary,
  ForecastResponse,
  RarResponse,
  DecisionPlan,
} from "@/lib/api";
import {
  Card,
  CardHead,
  HeroSkeleton,
  Icon,
  Loaded,
  Method,
  RowsSkeleton,
  Status,
  Tile,
  TileSkeleton,
  useApi,
  Tone,
} from "@/components/ui";
import { FanChart, LineChart } from "@/components/charts";
import {
  compact,
  dueIn,
  inr,
  inrShort,
  modelLabel,
  oneIn,
  pct,
  plainDays,
  shortDate,
} from "@/lib/format";

/** Runway bands. Stated here so the same thresholds drive colour AND words. */
function runwayTone(days: number): { tone: Tone; word: string } {
  if (days >= 60) return { tone: "good", word: "Comfortable" };
  if (days >= 30) return { tone: "warning", word: "Worth watching" };
  if (days >= 14) return { tone: "serious", word: "Getting tight" };
  return { tone: "critical", word: "Urgent" };
}

export default function CashHealth() {
  const rar = useApi(() => api.get<RarResponse>("/simulation/runway-at-risk"));
  const sum = useApi(() => api.get<FinancialSummary>("/financial-state/summary"));
  const fc = useApi(() => api.get<ForecastResponse>("/risk/forecast?horizon_days=90"));
  const plan = useApi(() => api.get<DecisionPlan>("/decisions/plan"));
  const bt = useApi(() => api.get<BacktestSummary>("/backtest/summary"));

  return (
    <div className="stack">
      {/* ---------------------------------------------------------------- */}
      {/* headline                                                          */}
      {/* ---------------------------------------------------------------- */}
      <Card className="hero">
        <div className="card-body">
          <Loaded q={rar} skeleton={<HeroSkeleton />}>
            {(r) => {
              const band = runwayTone(r.runway_at_risk_days);
              return (
                <>
                  <div className="row" style={{ marginBottom: "var(--s-3)" }}>
                    <Status tone={band.tone}>{band.word}</Status>
                    {sum.data ? (
                      <span className="small muted">as of {shortDate(sum.data.as_of)}</span>
                    ) : null}
                  </div>

                  <p className="tile-label" style={{ marginBottom: "var(--s-2)" }}>
                    Your cash should last
                  </p>
                  <div className="hero-figure num">
                    {plainDays(r.runway_at_risk_days)}
                    <span className="hero-unit">
                      ({r.runway_at_risk_days} days)
                    </span>
                  </div>

                  <p className="hero-sub">
                    We are {pct(r.confidence_level, 0)} confident it will last at least
                    this long. Put another way: there is {oneIn(1 - r.confidence_level)}{" "}
                    chance of running out sooner than {r.runway_at_risk_days} days.
                    {r.conditional_runway_at_risk_days < r.runway_at_risk_days ? (
                      <>
                        {" "}
                        If things do go badly, the money typically runs out around day{" "}
                        <strong>{Math.round(r.conditional_runway_at_risk_days)}</strong>.
                      </>
                    ) : null}
                  </p>
                </>
              );
            }}
          </Loaded>
        </div>

        <Method
          id="method-runway"
          label="How we worked this out"
          hint="Monte Carlo · Runway-at-Risk"
        >
          <p className="method-lede">
            We simulate the next 90 days ten thousand times. Each run draws a different
            plausible future: when each customer actually pays, and how far daily takings
            land from the forecast. We then look at how many of those runs hit zero, and
            when. The headline is the <strong>5th-percentile</strong> outcome, not the
            average — the average would describe a good day, and you cannot pay staff with
            an average.
          </p>

          <Loaded q={rar} skeleton={<TileSkeleton n={4} />}>
            {(r) => (
              <>
                <div className="grid grid-3">
                  <Tile
                    label="Runway-at-Risk"
                    value={`${r.runway_at_risk_days} d`}
                    note={`the ${pct(1 - r.confidence_level, 0)} worst case`}
                  />
                  <Tile
                    label="Conditional RaR"
                    value={`${r.conditional_runway_at_risk_days.toFixed(1)} d`}
                    note="average depth of that bad tail"
                  />
                  <Tile
                    label="Simulation error"
                    value={`± ${r.mc_standard_error.toFixed(3)} d`}
                    note={`${r.n_iterations.toLocaleString()} runs, seed ${r.random_seed}`}
                  />
                  <Tile
                    label="Chance of shortfall"
                    value={pct(r.probability_of_shortfall, 1)}
                    note="reaching zero within 90 days"
                  />
                </div>

                <div className="callout">
                  <strong>Why two numbers.</strong> Runway-at-Risk locates the cliff edge;
                  Conditional RaR measures the drop beyond it. Two businesses can share a
                  RaR of 11 days while one&rsquo;s bad tail averages 10 days and the
                  other&rsquo;s averages 3. Reporting only the first would hide that
                  difference. The simulation error is quoted because a runway figure
                  without its precision invites a decision it cannot support.
                </div>

                <p className="tiny muted">{r.interpretation}</p>
              </>
            )}
          </Loaded>

          <Loaded q={sum} skeleton={<TileSkeleton n={2} />}>
            {(s) => (
              <div className="callout callout-warning">
                <strong>The simpler number, and why we don&rsquo;t lead with it.</strong>{" "}
                A straight-line projection puts you at zero in{" "}
                {s.days_to_zero_point_estimate === null
                  ? "no fixed day within the horizon"
                  : `${s.days_to_zero_point_estimate} days`}
                . That ignores uncertainty entirely, so it reads more comfortable than the
                evidence supports. We show it here for completeness and headline the
                simulated figure instead.
              </div>
            )}
          </Loaded>
        </Method>
      </Card>

      {/* ---------------------------------------------------------------- */}
      {/* position                                                          */}
      {/* ---------------------------------------------------------------- */}
      <Card>
        <div className="card-body">
          <Loaded q={sum} skeleton={<TileSkeleton n={3} />}>
            {(s) => (
              <div className="grid grid-3">
                <Tile label="Money in the bank" value={inrShort(s.cash_balance)} note={inr(s.cash_balance)} />
                <Tile
                  label="Owed to you"
                  value={inrShort(s.outstanding_receivable_value)}
                  note={`across ${s.outstanding_receivables} unpaid invoices`}
                />
                <Loaded q={plan} skeleton={<div />}>
                  {(p) => (
                    <Tile
                      label="Bills due next 30 days"
                      value={inrShort(p.total_obligations_amount)}
                      note={`${p.n_obligations} payments`}
                      tone={p.shortfall > 0 ? "warning" : undefined}
                    />
                  )}
                </Loaded>
              </div>
            )}
          </Loaded>
        </div>
      </Card>

      {/* ---------------------------------------------------------------- */}
      {/* alerts + upcoming                                                 */}
      {/* ---------------------------------------------------------------- */}
      <Card>
        <CardHead
          title="What needs your attention"
          note="Bills falling due in the next 30 days, soonest first."
        />
        <div className="card-body">
          <Loaded q={plan} skeleton={<RowsSkeleton n={4} />}>
            {(p) => (
              <div className="stack-sm">
                {p.shortfall > 0 ? (
                  <div className="callout callout-warning">
                    <div className="row" style={{ marginBottom: 6 }}>
                      <Status tone="warning">Short by {inrShort(p.shortfall)}</Status>
                    </div>
                    You have {inrShort(p.available_cash)} available and{" "}
                    {inrShort(p.total_obligations_amount)} of bills due. That gap is the
                    reason there is a plan on the next page — it decides which bills to
                    pay first so the cheapest ones to delay are the ones that wait.
                  </div>
                ) : (
                  <div className="callout">
                    <Status tone="good">Everything due is covered</Status>
                  </div>
                )}

                {p.actions.map((a) => {
                  const tone: Tone =
                    a.action_type === "pay_now"
                      ? "good"
                      : a.action_type === "pay_partial"
                        ? "warning"
                        : "serious";
                  return (
                    <div
                      key={a.obligation_id}
                      className="row-between"
                      style={{
                        padding: "var(--s-3) 0",
                        borderBottom: "1px solid var(--border)",
                        gap: "var(--s-3)",
                      }}
                    >
                      <div style={{ minWidth: 0 }}>
                        <div style={{ fontWeight: 600 }}>{a.label}</div>
                        <div className="small muted">
                          {dueIn(a.days_until_due)}
                          {a.is_rigid ? " · cannot be part-paid" : ""}
                        </div>
                      </div>
                      <div className="right" style={{ flexShrink: 0 }}>
                        <div className="num" style={{ fontWeight: 640 }}>
                          {inrShort(a.amount_due)}
                        </div>
                        <Status tone={tone}>
                          {a.action_type === "pay_now"
                            ? "Pay in full"
                            : a.action_type === "pay_partial"
                              ? "Part-pay"
                              : "Delay"}
                        </Status>
                      </div>
                    </div>
                  );
                })}

                <a className="btn btn-primary" href="/plan" style={{ alignSelf: "flex-start", marginTop: "var(--s-2)" }}>
                  See the full plan
                </a>
              </div>
            )}
          </Loaded>
        </div>
      </Card>

      {/* ---------------------------------------------------------------- */}
      {/* forecast                                                          */}
      {/* ---------------------------------------------------------------- */}
      <Card>
        <CardHead
          title="The next 90 days"
          note="The shaded band is the range we actually expect, not a decoration — it is measured against past mistakes."
        />
        <div className="card-body">
          <Loaded q={fc} skeleton={<div className="skel" style={{ height: 240 }} />}>
            {(f) => (
              <>
                <p className="secondary small" style={{ maxWidth: "64ch" }}>
                  This is your <strong>day-to-day trading</strong> — takings in, costs out
                  — and it <strong>excludes money customers owe you</strong>. That is why
                  the line sits below break-even: your costs land every day, while invoice
                  payments arrive in lumps. Those payments are added separately in the
                  runway simulation, and keeping them out here is what stops them being
                  counted twice.
                </p>
                <FanChart
                  points={f.path}
                  confidence={f.interval_confidence}
                  note="Day-to-day trading only. Being below the break-even line is expected for this business, not an alarm."
                />
              </>
            )}
          </Loaded>
        </div>

        <Method id="method-forecast" label="How we forecast, and how honest the range is" hint="conformal intervals">
          <Loaded q={fc}>
            {(f) => (
              <>
                <p className="method-lede">
                  Three forecasting methods compete, and the winner is chosen by{" "}
                  <strong>walk-forward testing</strong> — repeatedly training on the past
                  and scoring on a future it has not seen. Selected here:{" "}
                  <strong>{modelLabel(f.selected_model)}</strong>.
                </p>
                <p className="tiny muted">{f.selection_rationale}</p>

                <div className="table-wrap">
                  <table className="data">
                    <thead>
                      <tr>
                        <th>Method</th>
                        <th>Out-of-sample error</th>
                        <th>AIC</th>
                        <th>Folds</th>
                      </tr>
                    </thead>
                    <tbody>
                      {f.candidates.map((c) => (
                        <tr key={c.model_name}>
                          <td>
                            {modelLabel(c.model_name)}
                            {c.model_name === f.selected_model ? (
                              <>
                                {" "}
                                <Status tone="good">chosen</Status>
                              </>
                            ) : null}
                          </td>
                          <td>{compact(c.rmse)}</td>
                          <td>{c.aic === null ? "—" : c.aic.toFixed(0)}</td>
                          <td>{c.n_folds}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                <div className="callout">
                  <strong>Why not pick on AIC.</strong> AIC measures fit to data the model
                  has already seen. Here the seasonal-naive baseline has no likelihood at
                  all, so ranking all three on AIC compares incompatible quantities — and
                  in this dataset the model with the <em>worse</em> AIC won out-of-sample.
                  That disagreement is exactly what walk-forward testing exists to catch.
                </div>

                {f.interval_calibration.q_hat !== null ? (
                  <>
                    <div className="grid grid-3">
                      <Tile
                        label="Conformal multiplier q̂"
                        value={f.interval_calibration.q_hat.toFixed(3)}
                        note={`reference z = ${f.interval_calibration.z_reference?.toFixed(3)}`}
                      />
                      <Tile
                        label="Scale ratio q̂/z"
                        value={f.interval_calibration.scale_ratio?.toFixed(3) ?? "—"}
                        note="1.0 = the model's own scale was right"
                      />
                      <Tile
                        label="Calibration scores"
                        value={String(f.interval_calibration.n_calibration_scores ?? "—")}
                        note="held-out residuals behind the width"
                      />
                    </div>
                    <p className="tiny muted">{f.interval_calibration.reading}</p>
                  </>
                ) : null}
              </>
            )}
          </Loaded>

          <Loaded q={bt}>
            {(b) => (
              <>
                <hr className="divider" />
                <h4>Were the ranges right in the past?</h4>
                <p className="method-lede">
                  A 95% range should contain what actually happened about 95 times in 100.
                  Measured over {b.calibration.n_observations.toLocaleString()} forecast/outcome
                  pairs from a replay of real history:{" "}
                  <strong>{pct(b.calibration.empirical)}</strong>.
                </p>
                <div className="row">
                  <Status tone={Math.abs(b.calibration.empirical - b.calibration.nominal) < 0.02 ? "good" : "warning"}>
                    {pct(b.calibration.empirical)} vs {pct(b.calibration.nominal, 0)} claimed
                  </Status>
                </div>

                <div className="callout callout-warning">
                  <strong>This used to be wrong, and we are not hiding it.</strong> An
                  earlier build of this system published{" "}
                  {pct(b.calibration.previous_build.pooled)} coverage against the same 95%
                  claim — ranges that were too narrow, which makes runway look safer than
                  it is. The cause was traced to one branch of the interval construction
                  (a quantile taken from only five residuals, which could never express
                  95%), and fixed by recalibrating against held-out error. It now measures{" "}
                  {pct(b.calibration.empirical)}.{" "}
                  <a href="/transparency#method-coverage">See the full before/after →</a>
                </div>

                <LineChart
                  title="Coverage by how far ahead we forecast"
                  note="Flat across the horizon is what a real fix looks like; simply widening every range would over-cover the near term to rescue the far term."
                  height={190}
                  series={[
                    {
                      name: "Measured coverage",
                      color: "var(--series-1)",
                      points: b.calibration.by_horizon.map((h) => ({ x: h.label, y: h.coverage })),
                    },
                  ]}
                  xLabels={b.calibration.by_horizon.map((h) => h.label)}
                  yDomain={[0.9, 1.0]}
                  yFmt={(v) => `${(v * 100).toFixed(0)}%`}
                  reference={{ y: b.calibration.nominal, label: "95% claimed" }}
                  table={
                    <table className="data">
                      <thead>
                        <tr><th>Days ahead</th><th>Pairs</th><th>Coverage</th><th>Mean width</th></tr>
                      </thead>
                      <tbody>
                        {b.calibration.by_horizon.map((h) => (
                          <tr key={h.label}>
                            <td>{h.label}</td>
                            <td>{h.n}</td>
                            <td>{pct(h.coverage)}</td>
                            <td>{compact(h.mean_width)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  }
                />
              </>
            )}
          </Loaded>
        </Method>
      </Card>
    </div>
  );
}
