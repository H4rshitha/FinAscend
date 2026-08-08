"use client";

/**
 * 02 — Action plan.
 *
 * The default view is a list of "pay this, delay that", each carrying the
 * backend's own `justification` string. That text is written server-side in
 * `/decisions/plan` from the solver's actual allocation — the frontend never
 * composes a reason, because a justification invented in the presentation
 * layer is exactly the black box this product argues against.
 *
 * The method panel carries the uncomfortable part: over a 49-step replay the
 * exact optimiser did NOT beat the simple rule. That result is one click away,
 * never deleted, and stated in the panel's own summary rather than buried at
 * the bottom.
 */

import { useState } from "react";
import {
  api,
  ApproveResponse,
  DecisionPlan,
  PriorityRanking,
  SolverComparison,
  SolverInstanceComparison,
} from "@/lib/api";
import {
  Card,
  CardHead,
  Icon,
  Loaded,
  Method,
  RowsSkeleton,
  Status,
  Tile,
  TileSkeleton,
  Tone,
  useApi,
} from "@/components/ui";
import { BarChart } from "@/components/charts";
import { compact, dueIn, inr, inrShort, modelLabel, pct, shortDate, STRATEGY_COLOR } from "@/lib/format";

const ACTION_UI: Record<
  string,
  { tone: Tone; verb: string; icon: () => React.ReactNode }
> = {
  pay_now: { tone: "good", verb: "Pay in full", icon: () => Icon.check() },
  pay_partial: { tone: "warning", verb: "Pay part now", icon: () => Icon.clock() },
  defer: { tone: "serious", verb: "Delay this one", icon: () => Icon.pause() },
};

export default function ActionPlan() {
  const plan = useApi(() => api.get<DecisionPlan>("/decisions/plan"));
  const instance = useApi(() => api.get<SolverInstanceComparison>("/decisions/solvers"));
  const replay = useApi(() => api.get<SolverComparison>("/backtest/solvers"));
  const priority = useApi(() => api.get<PriorityRanking>("/decisions/priority"));

  const [approved, setApproved] = useState<ApproveResponse | null>(null);
  const [approving, setApproving] = useState(false);
  const [approveError, setApproveError] = useState<string | null>(null);

  async function approve(id: string) {
    setApproving(true);
    setApproveError(null);
    try {
      setApproved(await api.post<ApproveResponse>(`/decisions/${id}/approve`));
    } catch (e) {
      setApproveError(e instanceof Error ? e.message : String(e));
    } finally {
      setApproving(false);
    }
  }

  return (
    <div className="stack">
      <div className="page-head">
        <h1>What to pay this month</h1>
        <p className="lede">
          There isn&rsquo;t enough to cover everything, so the order matters. Here is what
          we&rsquo;d do and why.
        </p>
      </div>

      <Loaded q={plan} skeleton={<TileSkeleton n={4} />}>
        {(p) => (
          <>
            <Card>
              <div className="card-body">
                <div className="grid grid-3">
                  <Tile label="Cash available" value={inrShort(p.available_cash)} note={inr(p.available_cash)} />
                  <Tile label="Bills due" value={inrShort(p.total_obligations_amount)} note={`${p.n_obligations} payments`} />
                  <Tile
                    label="Shortfall"
                    value={p.shortfall > 0 ? inrShort(p.shortfall) : "None"}
                    tone={p.shortfall > 0 ? "warning" : "good"}
                    note={p.shortfall > 0 ? "the gap this plan has to absorb" : "everything is covered"}
                  />
                  <Tile
                    label="Late fees if you follow this plan"
                    value={inr(p.expected_late_fees)}
                    note={`${p.n_paid_in_full} of ${p.n_obligations} paid in full`}
                  />
                </div>
              </div>
            </Card>

            {/* ------------------------------------------------------------ */}
            {/* the plan itself                                              */}
            {/* ------------------------------------------------------------ */}
            <Card>
              <CardHead
                title="The plan"
                note={`Prepared for ${shortDate(p.as_of)}. Nothing here is sent anywhere or paid automatically — it is a recommendation for you to approve.`}
              />
              <div className="card-body stack-sm">
                {p.actions.map((a) => {
                  const ui = ACTION_UI[a.action_type] ?? ACTION_UI.defer;
                  return (
                    <div
                      key={a.obligation_id}
                      style={{
                        border: "1px solid var(--border)",
                        borderRadius: "var(--radius-sm)",
                        padding: "var(--s-4)",
                        background: "var(--surface)",
                      }}
                    >
                      <div className="row-between" style={{ marginBottom: "var(--s-2)" }}>
                        <div>
                          <h3 style={{ fontSize: "var(--t-16)" }}>{a.label}</h3>
                          <span className="small muted">{dueIn(a.days_until_due)}</span>
                        </div>
                        <Status tone={ui.tone} icon={ui.icon()}>
                          {ui.verb}
                        </Status>
                      </div>

                      <div className="row" style={{ gap: "var(--s-5)", marginBottom: "var(--s-3)" }}>
                        <div>
                          <div className="tile-label">Amount due</div>
                          <div className="num" style={{ fontWeight: 620 }}>{inr(a.amount_due)}</div>
                        </div>
                        <div>
                          <div className="tile-label">Pay now</div>
                          <div className="num" style={{ fontWeight: 620, color: "var(--brand-700)" }}>
                            {inr(a.allocated_amount)}
                          </div>
                        </div>
                        {a.shortfall > 0 ? (
                          <div>
                            <div className="tile-label">Left over</div>
                            <div className="num" style={{ fontWeight: 620, color: "var(--serious)" }}>
                              {inr(a.shortfall)}
                            </div>
                          </div>
                        ) : null}
                      </div>

                      {/* The backend's own justification text, verbatim. */}
                      <p className="small secondary" style={{ margin: 0 }}>
                        {a.justification}
                      </p>
                    </div>
                  );
                })}

                <div className="row" style={{ marginTop: "var(--s-3)" }}>
                  {approved ? (
                    <Status tone="good">
                      Approved &middot; recorded in the audit log as entry #{approved.audit_sequence}
                    </Status>
                  ) : (
                    <button
                      type="button"
                      className="btn btn-primary"
                      disabled={approving}
                      onClick={() => approve(`plan-${p.as_of}`)}
                    >
                      {approving ? "Recording…" : "Approve this plan"}
                    </button>
                  )}
                  <a className="btn btn-ghost" href="/transparency">
                    See the audit trail
                  </a>
                </div>
                {approveError ? (
                  <p className="small" style={{ color: "var(--critical)" }}>{approveError}</p>
                ) : null}
                {approved ? (
                  <p className="tiny muted hash">chain hash {approved.audit_hash}</p>
                ) : null}
              </div>

              {/* ---------------------------------------------------------- */}
              {/* method                                                     */}
              {/* ---------------------------------------------------------- */}
              <Method
                id="method-solvers"
                label="How this order was decided — and where our clever version lost"
                hint="4 methods compared"
              >
                <p className="method-lede">
                  Every bill has a late fee per rupee per day. The plan maximises the fees
                  you avoid for the cash you have. Four different methods were run against
                  the same problem, and they disagree in ways worth seeing.
                </p>

                <Loaded q={instance} skeleton={<TileSkeleton n={3} />}>
                  {(s) => (
                    <>
                      <h4>On today&rsquo;s decision</h4>
                      <div className="table-wrap">
                        <table className="data">
                          <thead>
                            <tr>
                              <th>Method</th>
                              <th>Planned late fees</th>
                              <th>Status</th>
                              <th>Time</th>
                            </tr>
                          </thead>
                          <tbody>
                            <tr>
                              <td><span className="swatch" style={{ background: "var(--series-1)" }} />Simple rules</td>
                              <td>{inr(s.rules_baseline.objective_value)}</td>
                              <td>{s.rules_baseline.status}</td>
                              <td>{s.rules_baseline.solve_seconds.toFixed(3)}s</td>
                            </tr>
                            <tr>
                              <td><span className="swatch" style={{ background: "var(--series-2)" }} />Exact optimiser (LP)</td>
                              <td>{inr(s.lp.objective_value)}</td>
                              <td>{s.lp.status}</td>
                              <td>{s.lp.solve_seconds.toFixed(3)}s</td>
                            </tr>
                            <tr>
                              <td><span className="swatch" style={{ background: "var(--series-3)" }} />Exact optimiser (DP)</td>
                              <td>{inr(s.dp.objective_value)}</td>
                              <td>{s.dp.status}</td>
                              <td>{s.dp.solve_seconds.toFixed(3)}s</td>
                            </tr>
                            <tr>
                              <td><span className="swatch" style={{ background: "var(--series-4)" }} />Cautious optimiser</td>
                              <td>{inr(s.chance_constrained.solution.objective_value)}</td>
                              <td>
                                {s.chance_constrained.solution.status.startsWith("Infeasible")
                                  ? "Infeasible"
                                  : s.chance_constrained.solution.status}
                              </td>
                              <td>{s.chance_constrained.solution.solve_seconds.toFixed(3)}s</td>
                            </tr>
                          </tbody>
                        </table>
                      </div>

                      <div className="callout">
                        <div className="row" style={{ marginBottom: 6 }}>
                          <Status tone={s.solver_agreement.agree ? "good" : "critical"}>
                            {s.solver_agreement.agree ? "Two solvers agree" : "Solvers disagree"}
                          </Status>
                        </div>
                        <strong>Two independent solvers agreeing is the check.</strong> The
                        LP and the DP are separate implementations of the same problem, and
                        they differ by {inr(Math.abs(s.solver_agreement.absolute_delta))}{" "}
                        against a tolerance of {inr(s.solver_agreement.tolerance)}. The DP
                        searches a grid, so it can only ever match or slightly trail the LP
                        — if it ever came out <em>better</em>, that would mean a bug, most
                        likely spending cash that isn&rsquo;t there.
                        <div className="tiny muted" style={{ marginTop: 6 }}>
                          {s.solver_agreement.explanation}
                        </div>
                      </div>

                      {/* Infeasibility is a real outcome on a stressed book, and the
                          solver states it explicitly rather than returning a bland
                          "spend 0" that reads like cautious advice. Surface that. */}
                      {s.chance_constrained.solution.status.startsWith("Infeasible") ? (
                        <div className="callout callout-warning">
                          <div className="row" style={{ marginBottom: 6 }}>
                            <Status tone="serious">The cautious option cannot be satisfied</Status>
                          </div>
                          There is no payment schedule that gets the risk of running short
                          below {pct(s.chance_constrained.epsilon, 0)} — the chance is{" "}
                          {pct(s.chance_constrained.achieved_shortfall_probability, 0)} even
                          if you pay <em>nothing at all</em>. That means the shortfall is
                          driven by your cash position and when customers pay, not by which
                          bills you choose. No cleverer schedule fixes it; financing,
                          chasing payments faster, or renegotiating terms would.
                          <div className="tiny muted" style={{ marginTop: 6 }}>
                            The solver says this outright instead of returning &ldquo;spend
                            0&rdquo;, which would read as a cautious recommendation when it
                            actually means the target is unreachable.
                          </div>
                        </div>
                      ) : (
                        <div className="callout">
                          <strong>The cautious option.</strong> It caps spending to keep the
                          chance of running short below{" "}
                          {pct(s.chance_constrained.epsilon, 0)}, achieving{" "}
                          {pct(s.chance_constrained.achieved_shortfall_probability, 1)}.
                          Safer, and more expensive in fees — that trade is yours to make,
                          not ours to make silently.
                        </div>
                      )}
                    </>
                  )}
                </Loaded>

                <hr className="divider" />

                <Loaded q={replay} skeleton={<RowsSkeleton n={4} />}>
                  {(r) => (
                    <>
                      <h4>Over 49 real decisions, replayed</h4>
                      <div className="callout callout-warning">
                        <div className="row" style={{ marginBottom: 6 }}>
                          <Status tone="warning">The exact optimiser did not win</Status>
                        </div>
                        {r.finding}
                      </div>

                      <BarChart
                        title="Late fees actually incurred over the replay"
                        note="Lower is better. The over-commitment count beside each bar is the risk taken to get there — reading the fee column alone picks the wrong method."
                        fmt={(v) => compact(v)}
                        bars={r.strategies.map((s) => ({
                          label: modelLabel(s.name),
                          value: s.total_realized_penalty,
                          color: STRATEGY_COLOR[s.name] ?? "var(--series-1)",
                          sub: `over-committed at ${s.over_commitment_steps} of ${s.n_steps} steps`,
                        }))}
                        table={
                          <table className="data">
                            <thead>
                              <tr>
                                <th>Method</th>
                                <th>Fees incurred</th>
                                <th>vs rules</th>
                                <th>Over-commitment</th>
                                <th>p95 regret</th>
                              </tr>
                            </thead>
                            <tbody>
                              {r.strategies.map((s) => (
                                <tr key={s.name}>
                                  <td>
                                    <span className="swatch" style={{ background: STRATEGY_COLOR[s.name] }} />
                                    {modelLabel(s.name)}
                                  </td>
                                  <td>{inr(s.total_realized_penalty)}</td>
                                  <td>{s.vs_rules_baseline === 0 ? "—" : pct(s.vs_rules_baseline)}</td>
                                  <td>{s.over_commitment_steps}/{s.n_steps}</td>
                                  <td>{compact(s.p95_regret)}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        }
                      />

                      <div className="callout">
                        <strong>Why the sophisticated method lost.</strong> The optimiser
                        solves the allocation exactly — but against a <em>forecast</em> of
                        the cash that will arrive. When that forecast runs optimistic, it
                        commits money that never turns up, and the unfunded bills take
                        their full late fee. The simple rule is short-sighted, but its
                        short-sightedness happens to leave slack that absorbs the forecast
                        error. Solving the wrong problem precisely lost to solving roughly
                        the right problem approximately.
                      </div>

                      <div className="callout">
                        <strong>So which should you use?</strong> None of them dominates.
                        The cautious optimiser eliminated over-commitment completely and
                        paid for it in fees. That is a question about whether your business
                        can absorb a missed payment — a risk-appetite question, not a maths
                        one. An engine that quietly picked the most sophisticated option
                        would be answering it on your behalf without telling you.
                      </div>
                    </>
                  )}
                </Loaded>

                <hr className="divider" />

                <Loaded q={priority} skeleton={<RowsSkeleton n={3} />}>
                  {(pr) => (
                    <>
                      <h4>The simple rule, in full</h4>
                      <p className="method-lede">
                        Kept visible as the comparison point. It ranks by late fee per
                        rupee, breaking ties by what is due soonest and then by size.
                      </p>
                      <div className="table-wrap">
                        <table className="data">
                          <thead>
                            <tr><th>#</th><th>Bill</th><th>Score</th><th>Reason</th></tr>
                          </thead>
                          <tbody>
                            {pr.ranking.map((row) => (
                              <tr key={row.obligation_id}>
                                <td>{row.rank}</td>
                                <td>{row.obligation_id}</td>
                                <td>{row.score.toFixed(5)}</td>
                                <td style={{ textAlign: "left", whiteSpace: "normal" }} className="tiny muted">
                                  {row.reason}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </>
                  )}
                </Loaded>

                <p className="tiny muted">{p.baseline_comparison.caveat}</p>
              </Method>
            </Card>
          </>
        )}
      </Loaded>
    </div>
  );
}
