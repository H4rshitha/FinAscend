"use client";

/**
 * 03 — Risk explorer.
 *
 * The toggle is the point of the page: the same engine, asked the same
 * question at three different confidence levels, gives three different
 * answers. Making that switchable turns "trust our number" into "here is how
 * the number moves when you change how cautious you want to be".
 *
 * The three scenarios are REAL re-queries at different confidence levels, not
 * a client-side rescaling of one result. Baseline is the 95% figure the rest of
 * the product headlines.
 */

import { useState } from "react";
import { api, RarResponse, UncertaintyModel } from "@/lib/api";
import {
  Card,
  CardHead,
  Icon,
  Loaded,
  Method,
  Status,
  Tile,
  TileSkeleton,
  Tone,
  useApi,
} from "@/components/ui";
import { ConvergencePlot } from "@/components/charts";
import { oneIn, pct, plainDays } from "@/lib/format";

type Scenario = "best" | "baseline" | "worst";

/**
 * Confidence levels behind the three words.
 *  best     — 50%: the middle outcome, what happens on a normal run of luck
 *  baseline — 95%: the product's standard planning figure
 *  worst    — 99%: plan-for-the-bad-quarter
 */
const SCENARIOS: Record<
  Scenario,
  { label: string; confidence: number; blurb: string; tone: Tone }
> = {
  best: {
    label: "If things go well",
    confidence: 0.5,
    blurb:
      "The middle of the range — customers pay roughly on their usual schedule and takings land near the forecast. Useful as an upper bound on optimism, not as a plan.",
    tone: "good",
  },
  baseline: {
    label: "Planning figure",
    confidence: 0.95,
    blurb:
      "What we recommend planning against. Nineteen times in twenty, your cash lasts at least this long. This is the number the rest of the app headlines.",
    tone: "warning",
  },
  worst: {
    label: "If things go badly",
    confidence: 0.99,
    blurb:
      "A bad quarter: several customers slow down at once and takings run below forecast. Only 1 run in 100 is worse than this.",
    tone: "serious",
  },
};

export default function RiskExplorer() {
  const [scenario, setScenario] = useState<Scenario>("baseline");
  const cfg = SCENARIOS[scenario];

  const rar = useApi(
    () => api.get<RarResponse>(`/simulation/runway-at-risk?confidence=${cfg.confidence}`),
    [cfg.confidence]
  );
  const model = useApi(() => api.get<UncertaintyModel>("/simulation/uncertainty-model"));

  return (
    <div className="stack">
      <div className="page-head">
        <h1>What if things go differently?</h1>
        <p className="lede">
          Same business, same simulation — asked how cautious you want to be. Switch
          between them to see how much the answer actually moves.
        </p>
      </div>

      <Card>
        <div className="card-body">
          <div className="segmented" role="group" aria-label="Scenario">
            {(Object.keys(SCENARIOS) as Scenario[]).map((k) => (
              <button
                key={k}
                type="button"
                aria-pressed={scenario === k}
                onClick={() => setScenario(k)}
              >
                {scenario === k ? Icon.check(13) : null}
                {SCENARIOS[k].label}
              </button>
            ))}
          </div>

          <p className="secondary small" style={{ marginTop: "var(--s-3)", maxWidth: "62ch" }}>
            {cfg.blurb}
          </p>

          <div style={{ marginTop: "var(--s-5)" }}>
            <Loaded q={rar} skeleton={<TileSkeleton n={4} />}>
              {(r) => (
                <>
                  <div className="row" style={{ marginBottom: "var(--s-3)" }}>
                    <Status tone={cfg.tone}>{cfg.label}</Status>
                    <span className="small muted">
                      confidence {pct(r.confidence_level, 0)}
                    </span>
                  </div>

                  <div className="hero-figure num" style={{ color: `var(--${cfg.tone})` }}>
                    {plainDays(r.runway_at_risk_days)}
                    <span className="hero-unit">({r.runway_at_risk_days} days)</span>
                  </div>

                  <p className="hero-sub">
                    There is {oneIn(1 - r.confidence_level)} chance of running out sooner
                    than this. In the runs that do go wrong, cash typically hits zero
                    around day {Math.round(r.conditional_runway_at_risk_days)}.
                  </p>

                  <div className="grid grid-3" style={{ marginTop: "var(--s-5)" }}>
                    <Tile label="Runway-at-Risk" value={`${r.runway_at_risk_days} d`} />
                    <Tile
                      label="Conditional RaR"
                      value={`${r.conditional_runway_at_risk_days.toFixed(1)} d`}
                      note="average of the bad tail"
                    />
                    <Tile
                      label="Chance of shortfall"
                      value={pct(r.probability_of_shortfall, 1)}
                      note="within 90 days"
                    />
                    <Tile
                      label="Simulation error"
                      value={`± ${r.mc_standard_error.toFixed(3)} d`}
                      note={`${r.n_iterations.toLocaleString()} runs`}
                    />
                  </div>
                </>
              )}
            </Loaded>
          </div>
        </div>

        <Method id="method-convergence" label="How many simulations, and why that many" hint="Monte Carlo convergence">
          <Loaded q={rar}>
            {(r) => (
              <>
                <p className="method-lede">
                  Every extra simulated run makes the estimate steadier, with diminishing
                  returns. The error falls roughly as one over the square root of the run
                  count, so the count is chosen from the precision the decision needs
                  rather than picked as a round number. At{" "}
                  {r.n_iterations.toLocaleString()} runs the error on the headline is{" "}
                  <strong>± {r.mc_standard_error.toFixed(3)} days</strong>, and it travels
                  with every result on this page.
                </p>

                <div className="callout">
                  <strong>An honest wrinkle.</strong> That square-root rate is
                  demonstrated on a continuous quantity. Runway is measured in{" "}
                  <em>whole days</em> and piles up at the 90-day horizon, so its error
                  decays in steps and eventually floors out rather than shrinking
                  smoothly — past a point, more runs stop buying precision. The measured
                  slope on the real simulator is −0.32 against a theoretical −0.5 for
                  exactly this reason, and it is documented rather than rounded off.
                </div>

                <p className="tiny muted">
                  The error is computed by bootstrap rather than the textbook
                  quantile-variance formula, which needs a density estimate at the
                  quantile — noisy, and badly behaved when outcomes pile up at the
                  horizon.
                </p>
              </>
            )}
          </Loaded>
        </Method>
      </Card>

      {/* ------------------------------------------------------------------ */}
      {/* what the simulation assumes                                        */}
      {/* ------------------------------------------------------------------ */}
      <Card>
        <CardHead
          title="What the simulation assumes about your customers"
          note="These assumptions drive the numbers above. They are fitted from payment history, not chosen by hand."
        />
        <div className="card-body">
          <Loaded q={model} skeleton={<TileSkeleton n={3} />}>
            {(m) => (
              <>
                <p className="secondary" style={{ maxWidth: "64ch" }}>
                  We fit each customer&rsquo;s payment delay separately, then link them
                  together so that a bad month can hit several at once — because in real
                  life it does.
                </p>
                <div className="grid grid-3" style={{ marginTop: "var(--s-4)" }}>
                  <Tile label="Customers modelled" value={m.fits.length} />
                  {/* The raw family name (`student_t`) is jargon and belongs in
                      the method panel, not the plain view. What the reader
                      needs is what the choice MEANS for them. */}
                  <Tile
                    label="How they're linked"
                    value={
                      m.copula.family === "student_t"
                        ? "Together in bad months"
                        : "Loosely linked"
                    }
                    note={
                      m.copula.family === "student_t"
                        ? "several can go slow at the same time"
                        : "delays move largely independently"
                    }
                  />
                  <Tile
                    label="Typical delay across all"
                    value={`${Math.round(
                      m.fits.reduce((a, f) => a + f.median_delay_days, 0) / m.fits.length
                    )} days`}
                    note="middle of the range, customer by customer"
                  />
                </div>
              </>
            )}
          </Loaded>
        </div>

        <Method id="method-copula" label="The dependence assumption, in full" hint="copula · marginals">
          <Loaded q={model}>
            {(m) => (
              <>
                <p className="method-lede">
                  Treating customers as independent is the intuitive default and it is
                  <strong> wrong in the direction that flatters you</strong>. If delays
                  were independent, all customers being slow at once would be
                  astronomically unlikely, the simulated worst case would be shallow, and
                  runway would look comfortable. In reality a downturn, a credit squeeze
                  or a festival period hits everyone together — and that is precisely the
                  event that empties the account, because it removes every inflow at once.
                </p>

                <div className="callout">
                  <strong>Measured:</strong> on the stressed test world, sampling
                  independently reported 28 days of runway where the linked model
                  reported 19. Independence overstated runway by 47%.
                </div>

                <p className="method-lede">
                  We use a Student-t link rather than a Gaussian one because the Gaussian
                  has <em>no tail dependence</em>: however high you set its correlation,
                  extreme joint events still decouple in the far tail — the wrong
                  behaviour for the only scenario that matters here.
                </p>

                <h4>Each customer&rsquo;s fitted delay</h4>
                <div className="table-wrap">
                  <table className="data">
                    <thead>
                      <tr>
                        <th>Customer</th>
                        <th>Payments seen</th>
                        <th>Best-fit shape</th>
                        <th>Typical delay</th>
                        <th>Slow case (90th pct)</th>
                        <th>Fit quality (KS p)</th>
                      </tr>
                    </thead>
                    <tbody>
                      {m.fits.map((f) => {
                        // Read the `selected` flag rather than matching on the
                        // family name — the winner is decided by AIC server-side
                        // and this keeps the two from ever disagreeing.
                        const win =
                          f.candidates.find((c) => c.selected) ??
                          f.candidates.find((c) => c.family === f.selected_family);
                        return (
                          <tr key={f.counterparty_id}>
                            <td>{f.counterparty_id}</td>
                            <td>{f.n_observations}</td>
                            <td>{f.selected_family}</td>
                            <td>{f.median_delay_days.toFixed(1)} d</td>
                            <td>{f.p90_delay_days.toFixed(1)} d</td>
                            <td>
                              {win && typeof win.ks_pvalue === "number"
                                ? win.ks_pvalue.toFixed(3)
                                : "—"}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>

                <div className="callout callout-warning">
                  <strong>What this cannot tell you.</strong> The shape is chosen from
                  three candidates by AIC, and the data here was generated from one of
                  those three — so the fit is being graded on a question it was told the
                  answer to. On real payment data, if the true pattern lies outside those
                  three families, this table would look just as convincing and mean much
                  less.
                </div>

                <p className="tiny muted">
                  Correlation is estimated from a time-aligned panel rather than ragged
                  per-customer arrays: defaults leave gaps at different dates, and a
                  single dropped row would otherwise compare one customer&rsquo;s week 5
                  against another&rsquo;s week 6.
                  {m.copula.correlation_source ? ` Source: ${m.copula.correlation_source}.` : ""}
                  {m.copula.df ? ` Student-t with ${m.copula.df} degrees of freedom.` : ""}
                </p>
              </>
            )}
          </Loaded>
        </Method>
      </Card>
    </div>
  );
}
