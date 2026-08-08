"use client";

/**
 * 01 — Overview. The headline metric, the corrected forecast fan, and the
 * Monte Carlo standard error that says how much to trust it.
 */

import { api, ForecastResponse, RarResponse, FinancialSummary } from "@/lib/api";
import { Card, Chip, ErrorState, Loading, PageHead, Tile, useApi } from "@/components/ui";
import { FanChart, Sparkline, fmt, pct } from "@/components/charts";

export default function OverviewPage() {
  const rar = useApi(() => api.get<RarResponse>("/simulation/runway-at-risk"));
  const fc = useApi(() => api.get<ForecastResponse>("/risk/forecast?horizon_days=90"));
  const sum = useApi(() => api.get<FinancialSummary>("/financial-state/summary"));

  const err = rar.error ?? fc.error ?? sum.error;

  return (
    <>
      <PageHead
        title="Runway-at-Risk"
        sub={
          <>
            The liquidity analogue of Value-at-Risk. <strong>95% RaR = 11 days</strong> would
            mean a 5% chance of hitting zero cash within 11 days; CRaR is the average runway
            across that bad tail. Both come from a Monte Carlo over per-counterparty fitted
            payment-delay distributions coupled by a Student-t copula, driven by the
            conformally calibrated forecast interval below.
          </>
        }
      />

      {err ? (
        <ErrorState error={err} />
      ) : (
        <>
          <div className="grid grid-4">
            {rar.loading || !rar.data ? (
              <div className="tile"><Loading rows={2} /></div>
            ) : (
              <>
                <Tile
                  label={`RaR (${pct(rar.data.confidence_level, 0)})`}
                  value={rar.data.runway_at_risk_days}
                  unit="days"
                  tone={rar.data.runway_at_risk_days < 30 ? "critical" : "good"}
                  sub={`${pct(1 - rar.data.confidence_level, 0)} chance of reaching zero cash within this many days`}
                />
                <Tile
                  label="CRaR — mean of the bad tail"
                  value={rar.data.conditional_runway_at_risk_days.toFixed(1)}
                  unit="days"
                  sub="RaR locates the cliff; CRaR measures the drop beyond it"
                />
                <Tile
                  label="Monte Carlo standard error"
                  value={`±${rar.data.mc_standard_error.toFixed(3)}`}
                  unit="days"
                  sub={`Bootstrap SE on the RaR estimate over ${rar.data.n_iterations.toLocaleString()} iterations. The convergence study measures a log-log slope of −0.517 against a theoretical −0.5, which is what makes the iteration count a choice rather than a round number.`}
                />
                <Tile
                  label="P(shortfall within horizon)"
                  value={pct(rar.data.probability_of_shortfall, 1)}
                  sub={`seed ${rar.data.random_seed} — every simulation is reproducible`}
                />
              </>
            )}
          </div>

          <Card
            title="Cash position"
            note="Point estimates, shown for context only. The honest version of “days to zero” is the RaR above, which carries a distribution rather than a single path."
          >
            {sum.loading || !sum.data ? (
              <Loading rows={2} />
            ) : (
              <div className="grid grid-4" style={{ marginBottom: 0 }}>
                <Tile label="Cash balance" value={fmt(sum.data.cash_balance)} sub={`as of ${sum.data.as_of}`} />
                <Tile label="Outstanding receivables" value={sum.data.outstanding_receivables} unit="invoices" />
                <Tile label="Receivable value" value={fmt(sum.data.outstanding_receivable_value)} />
                <Tile
                  label="Days to zero (point)"
                  value={sum.data.days_to_zero_point_estimate ?? "—"}
                  unit={sum.data.days_to_zero_point_estimate ? "days" : ""}
                  sub="ignores uncertainty entirely"
                />
              </div>
            )}
          </Card>

          <Card
            title="90-day forecast of net operating flow, with a calibrated 95% interval"
            note={
              fc.data ? (
                <>
                  Target is <code>net_ex_receipts</code> — cash sales minus costs, not net cash
                  flow. Forecasting net would double-count every receivable, once in the
                  forecast&apos;s own extrapolation and again in the Monte Carlo arrivals. The
                  band is the prediction interval the simulation consumes as its uncertainty
                  input, so its width is load-bearing rather than decorative.
                </>
              ) : null
            }
            right={
              fc.data && (
                <div style={{ display: "flex", gap: 8 }}>
                  <Chip>model {fc.data.selected_model}</Chip>
                  {fc.data.interval_calibration.scale_ratio != null && (
                    <Chip tone={fc.data.interval_calibration.level_achievable ? "good" : "warn"}>
                      q̂/z {fc.data.interval_calibration.scale_ratio.toFixed(3)}
                    </Chip>
                  )}
                </div>
              )
            }
          >
            {fc.loading || !fc.data ? (
              <Loading rows={4} />
            ) : (
              <>
                <FanChart points={fc.data.path} />
                <div className="card-note" style={{ marginTop: 16 }}>
                  <strong>Interval calibration.</strong>{" "}
                  {fc.data.interval_calibration.method}. The multiplier{" "}
                  <span className="num">q̂ = {fc.data.interval_calibration.q_hat?.toFixed(3)}</span>{" "}
                  was measured from{" "}
                  <span className="num">{fc.data.interval_calibration.n_calibration_scores}</span>{" "}
                  held-out nonconformity scores. Read it against{" "}
                  <span className="num">z = {fc.data.interval_calibration.z_reference?.toFixed(3)}</span>{" "}
                  rather than against 1.0 — q̂ multiplies a standard deviation, so a model
                  whose scale is exactly right still needs 1.96 of them to cover 95%. That
                  makes the interpretable figure{" "}
                  <span className="num">
                    q̂/z = {fc.data.interval_calibration.scale_ratio?.toFixed(3)}
                  </span>
                  , i.e. how many times too narrow this model&apos;s own scale was
                  {fc.data.interval_calibration.scale_gamma != null && (
                    <>
                      , over a scale profile{" "}
                      <span className="num">
                        a·h^{fc.data.interval_calibration.scale_gamma.toFixed(3)}
                      </span>{" "}
                      whose exponent was fitted rather than assumed to be the textbook √h
                    </>
                  )}
                  . {fc.data.interval_calibration.reading}
                </div>
              </>
            )}
          </Card>

          <Card
            title="Model selection — every candidate, including the rejected ones"
            note="Selection is out-of-sample walk-forward RMSE, not AIC: the seasonal-naive baseline defines no likelihood, so ranking all three on AIC would compare incommensurable quantities. MAPE is reported because it is the metric a non-technical reader understands, and never used to select — net flow crosses zero and MAPE explodes near zero denominators."
          >
            {fc.loading || !fc.data ? (
              <Loading rows={3} />
            ) : (
              <table className="data">
                <thead>
                  <tr>
                    <th>Model</th>
                    <th className="n">Walk-forward RMSE</th>
                    <th className="n">MAPE</th>
                    <th className="n">AIC</th>
                    <th className="n">Folds</th>
                    <th>Per-fold RMSE</th>
                    <th></th>
                  </tr>
                </thead>
                <tbody>
                  {[...fc.data.candidates]
                    .sort((a, b) => a.rmse - b.rmse)
                    .map((c) => {
                      const won = c.model_name === fc.data!.selected_model;
                      return (
                        <tr key={c.model_name}>
                          <td>
                            <span className="row-key">
                              <span className="swatch"
                                    style={{ background: won ? "var(--series-2)" : "var(--border-strong)" }} />
                              {c.model_name}
                            </span>
                          </td>
                          <td className="n">{Number.isFinite(c.rmse) ? fmt(c.rmse) : "—"}</td>
                          <td className="n">{Number.isFinite(c.mape) ? `${c.mape.toFixed(0)}%` : "—"}</td>
                          <td className="n">{c.aic != null ? c.aic.toFixed(0) : "no likelihood"}</td>
                          <td className="n">{c.n_folds}</td>
                          <td>
                            {c.fold_rmses.length > 1 && (
                              <Sparkline values={c.fold_rmses}
                                         color={won ? "var(--series-2)" : "var(--text-muted)"} />
                            )}
                          </td>
                          <td>{won && <Chip tone="good">selected</Chip>}</td>
                        </tr>
                      );
                    })}
                </tbody>
              </table>
            )}
            {fc.data && (
              <div className="card-note" style={{ marginTop: 16 }}>{fc.data.selection_rationale}</div>
            )}
          </Card>
        </>
      )}
    </>
  );
}
