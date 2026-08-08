"use client";

/**
 * 03 — Credit risk. Calibration, ranking, and per-invoice attribution.
 *
 * The ordering is deliberate: calibration first, ROC-AUC second. The output is
 * consumed as an actual probability by the Monte Carlo engine, so a well-ranked
 * but badly scaled probability silently corrupts Runway-at-Risk.
 */

import { useState } from "react";
import { api, RiskExplain, RiskModels } from "@/lib/api";
import { Card, Chip, ErrorState, Loading, PageHead, Tile, useApi } from "@/components/ui";
import { BarChart, CalibrationCurve, ContributionChart, SLOTS, fmt, pct } from "@/components/charts";

const MODEL_LABEL: Record<string, string> = {
  rule_baseline: "Rules baseline",
  logistic_l2: "Logistic (L2)",
  gbm: "Gradient boosting",
};

export default function CreditPage() {
  const [model, setModel] = useState("gbm");
  const [row, setRow] = useState(0);

  const models = useApi(() => api.get<RiskModels>("/risk/models"));
  const explain = useApi(
    () => api.get<RiskExplain>(`/risk/${row}/explain?model=${model}`),
    [row, model]
  );

  const err = models.error;
  const perf = models.data?.performance ?? {};
  const names = Object.keys(perf);

  return (
    <>
      <PageHead
        title="Credit risk"
        sub={
          <>
            The deck answered the “black-box trust barrier” by avoiding ML and declaring the
            scorer rule-based and therefore auditable. That buys explainability by giving up
            predictive power, and never establishes the rules are any good. The position here is
            the harder one: fit a real model, then <em>earn</em> the trust with calibration,
            attribution and a measured comparison against that baseline.
          </>
        }
      />

      {err ? (
        <ErrorState error={err} />
      ) : models.loading || !models.data ? (
        <Loading rows={5} />
      ) : (
        <>
          <div className="grid grid-3">
            {names.map((n, i) => (
              <Tile
                key={n}
                label={MODEL_LABEL[n] ?? n}
                value={perf[n].roc_auc.toFixed(3)}
                unit="AUC"
                tone={perf[n].roc_auc > 0.6 ? "good" : perf[n].roc_auc > 0.55 ? "warn" : "critical"}
                sub={
                  <>
                    Brier <span className="num">{perf[n].brier_score.toFixed(3)}</span> · n_test{" "}
                    <span className="num">{perf[n].n_test}</span> · positive rate{" "}
                    <span className="num">{pct(perf[n].positive_rate)}</span>
                  </>
                }
              />
            ))}
          </div>

          <Card
            title="Ranking: ROC-AUC against the rules baseline"
            note="Accuracy is deliberately absent from this page. With a ~15% default rate, predicting “never defaults” scores 85% accuracy while being useless — quoting it would be the most misleading thing this module could do. AUC 0.65 is a credible credit-model number; real-world scorecards commonly land in 0.65–0.75."
          >
            <BarChart
              bars={names
                .sort((a, b) => perf[b].roc_auc - perf[a].roc_auc)
                .map((n, i) => ({
                  label: MODEL_LABEL[n] ?? n,
                  value: perf[n].roc_auc,
                  color: SLOTS[i % SLOTS.length],
                  sub: `Brier ${perf[n].brier_score.toFixed(3)}`,
                }))}
              valueFormat={(v) => v.toFixed(3)}
              note="ROC-AUC by model"
            />
            <div className="card-note" style={{ marginTop: 16 }}>
              The GBM beating the logistic model is explicable rather than mysterious: the
              generator&apos;s true stress→default link is a sigmoid of a latent variable, so
              there is genuine nonlinearity a linear model cannot capture. Both numbers are
              capped by the generator&apos;s own <code>DEFAULT_STRESS_BETA</code> and{" "}
              <code>DELAY_PERSISTENCE</code>, and should be read against them.
            </div>
            {models.data.lift_vs_baseline && (
              <table className="data" style={{ marginTop: 16 }}>
                <thead>
                  <tr>
                    <th>Candidate</th>
                    <th className="n">Candidate AUC</th>
                    <th className="n">Baseline AUC</th>
                    <th className="n">Lift</th>
                    <th>Verdict</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(models.data.lift_vs_baseline).map(([k, v]) => (
                    <tr key={k}>
                      <td>{MODEL_LABEL[k] ?? k}</td>
                      <td className="n">{v.candidate_auc.toFixed(3)}</td>
                      <td className="n">{v.baseline_auc.toFixed(3)}</td>
                      <td className="n" style={{ color: v.auc_lift > 0 ? "var(--status-good)" : "var(--status-critical)" }}>
                        {v.auc_lift > 0 ? "+" : ""}{v.auc_lift.toFixed(3)}
                      </td>
                      <td style={{ fontSize: 12, color: "var(--text-secondary)" }}>{v.verdict}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </Card>

          <div style={{ display: "flex", gap: 12, marginBottom: 16, alignItems: "center", flexWrap: "wrap" }}>
            <span className="tile-label">Model</span>
            <select value={model} onChange={(e) => { setModel(e.target.value); }}>
              {names.map((n) => (
                <option key={n} value={n}>{MODEL_LABEL[n] ?? n}</option>
              ))}
            </select>
            <span className="tile-label" style={{ marginLeft: 12 }}>Invoice row</span>
            <select value={row} onChange={(e) => setRow(Number(e.target.value))}>
              {Array.from({ length: 24 }, (_, i) => i * 7).map((i) => (
                <option key={i} value={i}>#{i}</option>
              ))}
            </select>
            <span style={{ fontSize: 11, color: "var(--text-muted)" }}>
              each change refetches <code>/risk/{row}/explain</code>
            </span>
          </div>

          {explain.error ? (
            <ErrorState error={explain.error} />
          ) : explain.loading || !explain.data ? (
            <Loading rows={4} />
          ) : (
            <div className="grid grid-2">
              <Card
                title="Calibration — predicted versus observed"
                note="This matters more than ranking here, because the output is consumed as an actual probability: it feeds the Monte Carlo engine and through it the optimizer. class_weight=&quot;balanced&quot; was tried and rejected — it improves ranking but reweights the likelihood, shifts the intercept and destroys the probability scale, more than doubling the Brier score."
              >
                <CalibrationCurve bins={explain.data.calibration} />
                <table className="data" style={{ marginTop: 16 }}>
                  <thead>
                    <tr>
                      <th className="n">Predicted</th>
                      <th className="n">Observed</th>
                      <th className="n">n</th>
                    </tr>
                  </thead>
                  <tbody>
                    {explain.data.calibration.map((b, i) => (
                      <tr key={i}>
                        <td className="n">{b.mean_predicted.toFixed(3)}</td>
                        <td className="n">{b.observed_rate.toFixed(3)}</td>
                        <td className="n">{b.n}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </Card>

              <Card
                title={`Why invoice #${row} scored ${pct(explain.data.default_probability, 1)}`}
                note={
                  model === "gbm"
                    ? "SHAP values via TreeExplainer, chosen over impurity-based importance because importance is a global model-level statistic and cannot answer “why is THIS invoice scored high”, which is the question a user reviewing one decision actually asks."
                    : "Coefficient × standardized value — the exact additive effect on the log-odds. The Wald intervals behind these are conservative and stated as such: they are unpenalized intervals around L2-penalized coefficients, so they indicate which features are precisely estimated rather than giving exact frequentist coverage."
                }
                right={
                  <Chip tone={explain.data.default_probability > 0.25 ? "critical" : explain.data.default_probability > 0.12 ? "warn" : "good"}>
                    p = {explain.data.default_probability.toFixed(4)}
                  </Chip>
                }
              >
                <ContributionChart
                  items={[...explain.data.feature_contributions]
                    .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
                    .slice(0, 9)}
                />
                <div className="card-note" style={{ marginTop: 16 }}>{explain.data.rationale}</div>
              </Card>
            </div>
          )}

          <div className="card-note" style={{ paddingLeft: 4 }}>
            <strong>Read against the ceiling.</strong> Default labels here are generated, not
            observed, so every AUC on this page is an upper bound on what the same features
            would achieve against real defaults. Features are computed strictly before the
            invoice being predicted — an “historical on-time rate” over the full dataset would
            include the invoice being predicted and produce an AUC near 1.0 that means nothing.
          </div>
        </>
      )}
    </>
  );
}
