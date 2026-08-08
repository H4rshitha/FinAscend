"use client";

/**
 * 04 — Customers.
 *
 * Default view: a trust signal in words ("reliably on time", "recently
 * slower"). The signal is derived from the fitted on-time probability, which
 * comes from the same uncertainty model that drives the simulation — not a
 * separate score invented for display.
 *
 * The method panel carries the part the deck's "trust barrier" is really
 * about: a real model, with its calibration curve, its per-case attribution,
 * and an explicit measured comparison against the rule-based approach it
 * replaces.
 */

import { useState } from "react";
import { api, RiskExplain, RiskModels, UncertaintyModel } from "@/lib/api";
import {
  Card,
  CardHead,
  Loaded,
  Method,
  RowsSkeleton,
  Status,
  Tile,
  TileSkeleton,
  Tone,
  useApi,
} from "@/components/ui";
import { BarChart, CalibrationCurve, ContributionChart } from "@/components/charts";
import { modelLabel, MODEL_COLOR, pct } from "@/lib/format";

/**
 * Typical delay -> the words a person would use.
 *
 * Banded on the fitted MEDIAN delay, not on `prob_on_time`. That field is
 * P(delay <= 0), which is structurally zero for every counterparty because the
 * distribution's location is pinned at zero — it reported "0% on time" for all
 * ten, which is true, useless, and actively misleading as a trust signal. The
 * median genuinely separates them: 3 days at the best, 17 at the worst.
 */
function trust(medianDays: number): { tone: Tone; label: string; note: string } {
  if (medianDays <= 5)
    return { tone: "good", label: "Pays quickly", note: "reliably fast — you can count on this cash" };
  if (medianDays <= 10)
    return { tone: "warning", label: "Usually within two weeks", note: "normal, but not immediate" };
  if (medianDays <= 16)
    return { tone: "serious", label: "Often runs late", note: "plan around the delay; chase early" };
  return { tone: "critical", label: "Consistently late", note: "don't count on this cash arriving soon" };
}

export default function Counterparties() {
  const model = useApi(() => api.get<UncertaintyModel>("/simulation/uncertainty-model"));
  const models = useApi(() => api.get<RiskModels>("/risk/models"));
  const [row, setRow] = useState(0);
  const [which, setWhich] = useState("gbm");
  const explain = useApi(
    () => api.get<RiskExplain>(`/risk/${row}/explain?model=${which}`),
    [row, which]
  );

  return (
    <div className="stack">
      <div className="page-head">
        <h1>Who pays you on time</h1>
        <p className="lede">
          Based on how each customer has actually paid you, not on a credit bureau score.
        </p>
      </div>

      <Card>
        <CardHead
          title="Your customers"
          note="Ordered by how reliably they pay. This is the same fitted behaviour that drives the runway simulation."
        />
        <div className="card-body">
          <Loaded q={model} skeleton={<RowsSkeleton n={6} />}>
            {(m) => {
              const sorted = [...m.fits].sort(
                (a, b) => a.median_delay_days - b.median_delay_days
              );
              return (
                <div className="stack-sm">
                  {sorted.map((f) => {
                    const t = trust(f.median_delay_days);
                    return (
                      <div
                        key={f.counterparty_id}
                        className="row-between"
                        style={{
                          padding: "var(--s-3) 0",
                          borderBottom: "1px solid var(--border)",
                          gap: "var(--s-3)",
                        }}
                      >
                        <div style={{ minWidth: 0 }}>
                          <div style={{ fontWeight: 600 }}>{f.counterparty_id}</div>
                          <div className="small muted">
                            {t.note} · {f.n_observations} payments on record
                          </div>
                        </div>
                        <div className="row" style={{ flexShrink: 0, gap: "var(--s-3)" }}>
                          <span className="small secondary right">
                            <span className="num" style={{ fontWeight: 620 }}>
                              {Math.round(f.median_delay_days)} days
                            </span>
                            <br />
                            <span className="tiny muted">
                              typically · {pct(f.prob_within_7_days, 0)} within a week
                            </span>
                          </span>
                          <Status tone={t.tone}>{t.label}</Status>
                        </div>
                      </div>
                    );
                  })}
                </div>
              );
            }}
          </Loaded>
        </div>

        <Method
          id="method-credit"
          label="How we judge this, and how we know it works"
          hint="calibration · attribution · vs rules"
        >
          <p className="method-lede">
            The deck this product answers said a rule-based scorer is trustworthy{" "}
            <em>because</em> it is simple. We took the harder position: fit a real model,
            then earn the trust with calibration, per-case attribution, and a measured
            comparison. Here is all three.
          </p>

          {/* ------------------------------------------------------------ */}
          <Loaded q={models} skeleton={<TileSkeleton n={3} />}>
            {(mm) => {
              const names = Object.keys(mm.performance);
              return (
                <>
                  <h4>Does the model actually beat the rules?</h4>
                  <BarChart
                    title="Ranking ability (ROC-AUC)"
                    note="0.5 is a coin flip. Real-world credit scorecards commonly land between 0.65 and 0.75, so this is a credible number rather than an inflated one."
                    fmt={(v) => v.toFixed(3)}
                    bars={names.map((n) => ({
                      label: modelLabel(n),
                      value: mm.performance[n].roc_auc,
                      color: MODEL_COLOR[n] ?? "var(--series-1)",
                      sub: `Brier score ${mm.performance[n].brier_score.toFixed(3)} (lower is better)`,
                    }))}
                    table={
                      <table className="data">
                        <thead>
                          <tr><th>Model</th><th>ROC-AUC</th><th>Brier</th><th>Tested on</th></tr>
                        </thead>
                        <tbody>
                          {names.map((n) => (
                            <tr key={n}>
                              <td>{modelLabel(n)}</td>
                              <td>{mm.performance[n].roc_auc.toFixed(3)}</td>
                              <td>{mm.performance[n].brier_score.toFixed(3)}</td>
                              <td>{mm.performance[n].n_test}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    }
                  />

                  <div className="grid grid-2">
                    {Object.entries(mm.lift_vs_baseline).map(([k, v]) => (
                      <div className="callout" key={k}>
                        <strong>{modelLabel(k)}</strong> scores {v.model_roc_auc.toFixed(3)}{" "}
                        against the rules&rsquo; {v.baseline_roc_auc.toFixed(3)} — a lift of{" "}
                        {v.auc_lift >= 0 ? "+" : ""}
                        {v.auc_lift.toFixed(3)}. {v.verdict}
                      </div>
                    ))}
                  </div>

                  <div className="callout callout-warning">
                    <strong>We never report accuracy.</strong> With roughly one default in
                    seven, a model that simply predicts &ldquo;never defaults&rdquo; scores
                    about 85% accuracy while being completely useless. Quoting it would be
                    the single most misleading thing this page could do.
                  </div>
                  <p className="tiny muted">{mm.note}</p>
                </>
              );
            }}
          </Loaded>

          <hr className="divider" />

          {/* ------------------------------------------------------------ */}
          <h4>Are the probabilities honest?</h4>
          <p className="method-lede">
            Ranking is not enough. This number feeds the runway simulation as an actual
            probability, so if it says 20% it had better happen about one time in five.
            The curve below plots what the model predicted against what actually
            happened — the closer to the dashed line, the better.
          </p>

          <div className="row" style={{ marginBottom: "var(--s-3)" }}>
            <div className="segmented" role="group" aria-label="Model">
              {["gbm", "logistic_l2"].map((k) => (
                <button key={k} type="button" aria-pressed={which === k} onClick={() => setWhich(k)}>
                  {modelLabel(k)}
                </button>
              ))}
            </div>
            <div className="field" style={{ maxWidth: 180 }}>
              <label htmlFor="row">Invoice #</label>
              <input
                id="row"
                type="number"
                min={0}
                value={row}
                onChange={(e) => setRow(Math.max(0, Number(e.target.value) || 0))}
              />
            </div>
          </div>

          <Loaded q={explain} skeleton={<RowsSkeleton n={4} />}>
            {(ex) => (
              <div className="grid grid-2">
                <CalibrationCurve
                  bins={ex.calibration}
                  title="Predicted vs what actually happened"
                  note="Dot size is how many cases fall in that bin."
                />
                <div className="stack-sm">
                  <ContributionChart
                    items={ex.feature_contributions}
                    title={`Why invoice #${row} scored ${pct(ex.default_probability)}`}
                    note={
                      which === "gbm"
                        ? "SHAP values — chosen over feature importance because importance is a whole-model statistic and cannot answer 'why THIS invoice'."
                        : "Coefficient × standardised value: the exact additive effect on the log-odds."
                    }
                  />
                  <div className="callout">{ex.rationale}</div>
                </div>
              </div>
            )}
          </Loaded>

          <div className="callout callout-warning">
            <strong>A setting we tried and rejected.</strong> Balancing the classes
            improves ranking on imbalanced data, and it is the standard reflex. It also
            shifts the intercept and destroys the probability scale — predicting 38–60%
            where the truth was 8–22%, more than doubling the calibration error. Because
            this output is consumed as a probability rather than a rank, we kept the
            calibrated model and left the ranking gain on the table.
          </div>

          <div className="callout callout-warning">
            <strong>The limit of all of this.</strong> These defaults are generated by a
            simulation, not observed in the wild. The scores above are an upper bound on
            what the same features would achieve against real defaults, and they are
            capped by how much signal the generator put there in the first place.
          </div>
        </Method>
      </Card>
    </div>
  );
}
