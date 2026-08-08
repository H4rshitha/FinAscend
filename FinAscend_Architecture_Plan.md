# FinAscend — Architecture Plan
Semi-autonomous liquidity management engine · SNUC Hacks '26, Track 3 FinTech (team Gradient Ascent)

This document serves two audiences, and the second one sets the bar.

**Audience 1 — the hackathon.** Every bullet on the PPT's two Innovation slides and the Technical Architecture diagram maps to a concrete, buildable module. Sections 0–3 hold that contract.

**Audience 2 — a quantitative research review.** This build is also a work sample for a quant research role, judged on genuine Python/statistics proficiency (pandas, numpy, scipy, statsmodels), probability and time-series foundations, comfort with structured *and* unstructured data, an optimization/competitive-programming background, and real curiosity about financial markets. That audience changes the standard for every module:

> A rule-based system that looks clean is a **weaker** deliverable than a real statistical model that is honestly validated and explained. Nothing here may make numbers look plausible without the underlying method being real. A reader of this codebase should conclude the author can do research — not that the author can call libraries.

Sections A–D exist to meet that second bar. Where the two audiences conflict, the second one wins.

---

## Status — what actually exists

Stated plainly, because a document whose whole premise is not overstating rigour cannot itself overstate its progress:

| Item | Status |
|---|---|
| This architecture document | Exists |
| Python runtime | **Installed and working** — Python 3.12.10 with a pinned `.venv` at `finascend/.venv`. |
| `finascend/` source tree | **Built.** Sections A, B and C are implemented; see the tier table below. |
| Section 2 Pydantic schemas | **Implemented and exercised** (`backend/app/schemas/`), with the §2.1 amendments applied. |
| Test suite | **204 tests, all passing** (`backend/tests/`). |

| Build tier | Status |
|---|---|
| 1 — Quant Core (Section A) | **Complete.** A.0–A.6 implemented with real fitted logic, plus **A.7 bankruptcy risk** (first-passage ruin probability validated against a closed form, hazard curve, Altman Z''), validated by the test suite. |
| 2 — Deterministic backbone (Section B) | **Complete for the decision path** — rules baseline, credit scorer, solver callers, hash-chain audit. **Both ingestion channels now built:** receipt OCR, and bank-statement parsing over upload *and* a real HTTP API client. Vendor-branded Decentro/Plaid connectors remain unbuilt (no live credentials); `HttpStatementProvider` is the integration point. |
| 3 — Backtesting harness (Section C) | **Complete.** Generates `BACKTEST_REPORT.md` from a real run. |
| 4 — API / RBAC / audit | **Complete at working-endpoint level.** FastAPI with JWT + RBAC; no Postgres/Celery (in-memory). Next.js frontend exists for the five quant pages; the statement and bankruptcy endpoints are API-only so far. |
| 5 — Graph RAG / voice / multilingual | **Honest 501 stubs**, exactly as specified. |

**A.7 was not in the original plan** and is recorded here as an addition rather
than back-dated into the traceability table: `RunwayAtRisk` answers "how long
until the cash runs out", which is a censored quantile and cannot be scored
against any single outcome. "P(the business fails within 90 days)" can be, and
is — Brier skill +0.49, ROC-AUC 0.954 against realized ruin. See
`QUANT_METHODOLOGY.md` §7.

Deliverables on disk: `QUANT_METHODOLOGY.md`, `FORECASTING_METHODOLOGY.md`,
`BACKTEST_REPORT.md`, `README.md`, and three executed notebooks in
`finascend/notebooks/`.

Any earlier revision of this file claiming a scaffold of "90+ files already created on disk" was wrong; that claim was removed rather than quietly softened, and the table above now describes what was actually built.

**Two results are reported against interest** rather than buried — the LP optimizer did not beat the rules baseline in the backtest, and the forecast prediction intervals are measurably overconfident (~85% coverage against a 95% nominal level). Both are written up in `BACKTEST_REPORT.md` and `QUANT_METHODOLOGY.md`.

---

## 0. Traceability — PPT → System Module

| PPT bullet (Innovation slides + architecture diagram) | Module in this plan |
|---|---|
| Adversarial Reasoning Agent (Devil's Advocate) | `services/adversarial/devils_advocate_agent.py` + `schemas/adversarial.py` |
| Monte Carlo Simulations (receivable uncertainty) | `services/quant_core/monte_carlo_engine.py` (fitted delay distributions + copula dependence); `services/simulation/` is a thin caller |
| Scenario Toggle (Best/Worst case) | `services/simulation/scenario_toggle.py`, `ScenarioType` enum — scenarios defined as percentile cuts of the Monte Carlo distribution, not hand-set multipliers |
| DBSCAN-based Ambiguity Resolution | `services/data_intelligence/dbscan_resolver.py` + `schemas/ingestion.py::DuplicateResolutionResult` |
| AI-based PII & Compliance Checks | `services/data_intelligence/pii_masker.py`, `compliance_checker.py` |
| Hashed-Chain Logging (PostgreSQL) | `services/audit/hash_chain_logger.py` + `schemas/audit.py::AuditLogEntry` |
| Credit Risk Scoring Engine | `services/quant_core/risk_scoring.py` (regularized logistic + GBM, calibrated and explained); `rule_based_prioritizer` retained as the **named baseline** it is measured against |
| Decentro API / multi-source ingestion (OCR + APIs) | `services/ingestion/decentro_client.py`, `ocr_service.py` (Google Cloud Vision), `plaid_client.py` |
| Structured JSON Normalization | `services/ingestion/normalizer.py` |
| Dual Framework decision engine (rule-based + personalized) | `services/decision_engine/rule_based_prioritizer.py`, `personalized_weighting.py` |
| Linear Programming Solver | `services/quant_core/optimization/lp_solver.py` (PuLP / `scipy.optimize.milp`) |
| Gantt Chart Scheduling | `services/action_generator/gantt_scheduler.py`, `frontend/components/actions/GanttChart.tsx` |
| Graph RAG (Neo4j) | `services/graph_rag/*`, `db/neo4j/session.py` |
| Human-in-the-loop review | `ReviewStatus` enum, `/decisions/{id}/approve`, `frontend/components/actions/ReviewQueue.tsx` |
| Market & seasonal trend awareness | `services/market_intelligence/*`, grounded in STL decomposition (A.6) |
| Multilingual chatbot + Voice interface | `services/nlp/*`, `/ws/voice/*`, `frontend/components/voice/` |

### Quant Core modules (no PPT bullet — these are the research contribution)

| Capability | Module |
|---|---|
| Synthetic ground-truth data generator | `services/quant_core/synthetic_data.py` |
| Cash-flow forecasting (seasonal naive / Holt-Winters / SARIMAX) | `services/quant_core/forecasting.py` |
| Runway-at-Risk (RaR / CRaR) engine | `services/quant_core/monte_carlo_engine.py` |
| Dual-solver allocation + cross-check | `services/quant_core/optimization/{lp_solver,dp_solver,cross_validation}.py` |
| Chance-constrained allocation (SAA) | `services/quant_core/optimization/chance_constrained.py` |
| Credit / default risk model + explainability | `services/quant_core/risk_scoring.py` |
| Statistical anomaly detection (robust z / Isolation Forest) | `services/quant_core/anomaly_detection.py` |
| Unstructured text → vendor/category via embeddings | `services/quant_core/unstructured.py` |
| Walk-forward backtesting & calibration harness | `services/backtesting/*` |

Architecture-diagram data flow preserved exactly: **Data Sources → Ingestion Pipeline (Vision OCR → DBSCAN → Normalizer) → Data Storage (Postgres + Financial Knowledge Graph + Audit Chain) → AI Engine (Graph RAG, Credit Risk Scorer, Monte Carlo, LP Optimizer) → Decision Framework (Baseline/Personalized Prioritizer, Devil's Advocate) → Action Layer (Gantt, Email, Scenario Analyzer) → Review Queue**, with Market Intelligence feeding contextual signals into the AI Engine and user preferences feeding back from the UI. The Quant Core sits **inside** the AI Engine box — it is the implementation of it, not a new stage.

### Gaps filled beyond the PPT (flagged, not hallucinated as "in the deck")
- **RBAC** (owner/accountant/viewer) — the deck shows a single review queue with no access tiers; small businesses commonly have a bookkeeper who shouldn't approve payments.
- **Idempotency + rate-cap enforcement** on Decentro/Plaid sync, matching the deck's own feasibility claim of "~30 calls/user/month" — this needs an actual counter, not just a plan.
- **WebSocket layer** — the deck's architecture diagram is drawn as a static pipeline; a semi-autonomous system reviewing live decisions needs streaming, not just polling.
- **Explicit error/response contract** — required for a non-technical-user review queue to fail gracefully.
- **A substantive answer to the deck's own "black-box trust barrier."** The deck raises the trust problem and answers it by *avoiding* ML — declaring the scorer rule-based and therefore auditable. That is a cop-out: it buys explainability by giving up predictive power, and it never proves the rules are any good. The stronger answer, and the one built here, is a fitted model shipped **with** a calibration curve, per-feature attribution surfaced in the API response, and an explicit measured lift over the rules baseline. Explainability is earned by measurement, not by refusing to model.

---

## 0.5 Build priority order — **do not reorder**

1. **Quant Core (Section A).** The differentiator. Real logic, not stubs, before anything else is touched.
2. **Ingestion → Normalization → Decision Engine deterministic backbone (Section B),** wired to real Quant Core outputs instead of placeholders.
3. **Backtesting & validation harness (Section C).** Non-negotiable, not an afterthought.
4. **Everything else in this architecture plan** — API layer, frontend, audit chain, RBAC, WebSockets. Implement it, but it is polish, and it ranks *below* 1–3.
5. **Graph RAG, voice interface, multilingual chatbot.** Lowest priority. Stub cleanly (working endpoints returning honest "not yet implemented" responses) unless everything above is finished with time to spare. They demonstrate product breadth, not the target skills.

This inverts the ordering an earlier revision of this document proposed (ingestion first, simulation and optimization layered on afterwards). Tiers 4 and 5 are explicitly *lower* priority than the statistical work, not higher.

### Minimum defensible slice — decided up front, not improvised at the deadline

Tiers 1–4 are a lot of surface even with Tier 5 stubbed: seven `quant_core` modules, a four-file optimization submodule, a four-file backtesting harness, three notebooks, the deterministic backbone, and the API/RBAC/audit layer. Deciding the cut line *now* is what stops a deadline from making the choice badly. If time runs out **inside Tier 1**, the slice that still stands on its own is:

| Keep | Why it survives the cut |
|---|---|
| **A.0** `synthetic_data.py` | Nothing downstream is validatable without a known generating process. Cutting it makes everything else assertion rather than evidence. |
| **A.1** `forecasting.py` | Feeds A.2's uncertainty propagation; the seasonal-naive baseline alone demonstrates model selection discipline. |
| **A.2** `monte_carlo_engine.py` | **RaR/CRaR is the headline artifact.** If exactly one thing ships, this is it. |
| **`notebooks/02_...ipynb`** | Carries the parameter-recovery check and the convergence study — the two things that prove the method is real. |
| **`tests/unit/quant_core/` for A.0–A.2** | Non-negotiable *within* the slice. A.0→A.2 without the recovery test proves nothing; the test is the differentiator, not the module. |
| **`QUANT_METHODOLOGY.md`**, scoped to A.0–A.2 | Cheap to write, disproportionately high signal. |

**A.3, A.4, A.5, A.6 are individually strong but more cuttable than A.0–A.2.** Each is a self-contained addition; none is load-bearing for RaR.

**Knock-on effects of taking this cut** — recorded so the fallback isn't discovered to be broken mid-deadline:
- Section B's wiring shrinks: `credit_risk_scorer` has no A.4 to call and stays on the rules table, which is then described honestly as the baseline *without* a measured comparison rather than as a validated model.
- Section C loses A.3 as its plan generator. **The harness does not die** — `rule_based_prioritizer` (deterministic, already required by Section B, cheap) stands in as the plan generator, so regret against a hindsight-optimal plan remains computable and interval calibration tracking is unaffected. Section C survives the cut in reduced form, which is why it stays ahead of Tier 4.

---

## 1. Directory Structure

```
finascend/
├── backend/                          # FastAPI
│   ├── app/
│   │   ├── main.py                   # app factory, startup hooks
│   │   ├── core/                     # config, security (JWT/RBAC), logging
│   │   ├── api/v1/
│   │   │   ├── endpoints/            # ingestion, data_intelligence, financial_state,
│   │   │   │                         # risk, decisions, simulation, graph, actions,
│   │   │   │                         # audit, chat, voice, market, backtest
│   │   │   ├── websockets/           # decision_stream, simulation, ingestion, voice
│   │   │   └── router.py
│   │   ├── schemas/                  # Pydantic — see Sections 2 and 2.1
│   │   ├── models/                   # SQLAlchemy ORM + Neo4j node/edge defs
│   │   ├── services/
│   │   │   ├── quant_core/           # ── SECTION A: the research core ──
│   │   │   │   ├── synthetic_data.py         # A.0 ground-truth generator
│   │   │   │   ├── forecasting.py            # A.1 seasonal naive / Holt-Winters / SARIMAX
│   │   │   │   ├── monte_carlo_engine.py     # A.2 fitted delays + copula → RaR / CRaR
│   │   │   │   ├── risk_scoring.py           # A.4 logistic + GBM + explainability
│   │   │   │   ├── anomaly_detection.py      # A.5 robust z-score + Isolation Forest
│   │   │   │   ├── unstructured.py           # A.6 embedding-based vendor/category
│   │   │   │   └── optimization/
│   │   │   │       ├── lp_solver.py          # A.3 LP / MILP
│   │   │   │       ├── dp_solver.py          # A.3 knapsack DP (independent solution)
│   │   │   │       ├── cross_validation.py   # A.3 solver agreement harness
│   │   │   │       └── chance_constrained.py # A.3 SAA over Monte Carlo draws
│   │   │   ├── backtesting/          # ── SECTION C ──
│   │   │   │   ├── replay_harness.py         # day-by-day, no look-ahead
│   │   │   │   ├── hindsight_optimal.py      # full-knowledge benchmark plan
│   │   │   │   ├── regret.py                 # efficiency loss vs. hindsight
│   │   │   │   └── calibration.py            # are the 95% intervals honest?
│   │   │   ├── ingestion/            # bank_statement_parser, invoice_parser, ocr_service,
│   │   │   │                         # decentro_client, plaid_client, normalizer
│   │   │   ├── data_intelligence/    # dbscan_resolver, pii_masker, compliance_checker
│   │   │   ├── decision_engine/      # lp_solver (thin caller), rule_based_prioritizer,
│   │   │   │                         # personalized_weighting, credit_risk_scorer
│   │   │   ├── simulation/           # monte_carlo_engine (thin caller), scenario_toggle
│   │   │   ├── adversarial/          # devils_advocate_agent (LangChain)
│   │   │   ├── graph_rag/            # neo4j_client, relationship_reasoner, cypher_queries
│   │   │   ├── action_generator/     # email_drafter, payment_rescheduler, gantt_scheduler
│   │   │   ├── audit/                # hash_chain_logger, integrity_verifier
│   │   │   ├── market_intelligence/  # macro_signals, seasonal_trends (STL-grounded)
│   │   │   └── nlp/                  # multilingual_chatbot, voice_interface, translation
│   │   ├── db/postgres/ , db/neo4j/  # session/engine factories
│   │   ├── workers/                  # Celery: ingestion, simulation, decentro_sync tasks
│   │   └── utils/                    # dates (days-to-zero math), hashing, seeding
│   ├── tests/
│   │   ├── unit/quant_core/          # statistical-correctness tests — see Section D
│   │   ├── unit/ , integration/
│   ├── alembic/                      # migrations
│   ├── requirements.txt / Dockerfile / .env.example
│
├── notebooks/                        # research narrative — see Section D
│   ├── 01_forecasting_model_selection.ipynb
│   ├── 02_monte_carlo_runway_at_risk.ipynb
│   └── 03_credit_risk_model_comparison.ipynb
│
├── frontend/                         # Next.js (App Router)
│   ├── app/
│   │   ├── (dashboard)/page.tsx      # RunwayAtRiskCard, LiquidityGauge, cashflow fan chart
│   │   ├── obligations/page.tsx
│   │   ├── simulation/page.tsx       # Monte Carlo + scenario toggle
│   │   ├── decisions/[id]/page.tsx   # Devil's Advocate panel, approve/reject
│   │   ├── actions/page.tsx          # Gantt, email drafts, review queue
│   │   ├── audit/page.tsx            # hash-chain viewer
│   │   ├── chat/page.tsx             # multilingual + voice
│   │   └── settings/page.tsx         # personalized weighting, connected accounts
│   ├── components/{dashboard,obligations,simulation,decision,actions,graph,audit,voice,common}/
│   ├── lib/                          # api-client.ts, websocket-client.ts, types.ts, i18n.ts
│   ├── hooks/                        # useDecisionStream, useSimulation, useVoiceInput
│   └── store/                        # financialStateStore, decisionStore (Zustand)
│
├── QUANT_METHODOLOGY.md              # Section D — every statistical choice, in prose
├── FORECASTING_METHODOLOGY.md        # Section A.1 — research memo
├── BACKTEST_REPORT.md                # Section C output
└── docs/API_SPEC.md                  # Section 3, full detail
```

---

## 2. Shared Data Models (Pydantic v2)

All schemas live in `backend/app/schemas/` and share a common base (`BaseSchema`: ORM-friendly, enums serialize as values, unknown fields rejected). The six core models as originally specified:

```python
# schemas/inflow.py
class Inflow(BaseSchema):
    id: UUID
    business_id: UUID
    amount: Money                       # Decimal, 14 digits / 2 decimal places
    currency: Currency = Currency.INR
    expected_date: date
    received_date: Optional[date] = None
    counterparty_id: Optional[UUID] = None
    counterparty_name: str
    source_type: SourceType             # bank_statement | digital_invoice | receipt_ocr | decentro_api | plaid_api | manual_entry
    source_reference: Optional[str] = None
    is_receivable: bool = True
    certainty: UnitInterval             # see 2.1 — now a fitted output, not a default
    is_duplicate_of: Optional[UUID] = None   # set by DBSCAN resolver
    created_at: datetime

# schemas/outflow.py
class Outflow(BaseSchema):
    id: UUID
    business_id: UUID
    amount: Money
    currency: Currency = Currency.INR
    due_date: date
    paid_date: Optional[date] = None
    counterparty_id: Optional[UUID] = None
    counterparty_name: str
    category: str                        # rent | payroll | vendor_payment | tax | loan_emi ...
    source_type: SourceType
    source_reference: Optional[str] = None
    is_recurring: bool = False
    is_duplicate_of: Optional[UUID] = None
    created_at: datetime

# schemas/obligation.py
class Obligation(BaseSchema):
    id: UUID
    business_id: UUID
    direction: ObligationDirection       # payable | receivable
    linked_inflow_id: Optional[UUID] = None
    linked_outflow_id: Optional[UUID] = None
    counterparty_id: UUID
    amount: Money
    due_date: date
    status: ObligationStatus = ObligationStatus.PENDING
    urgency: UnitInterval                # see 2.1 — computed, not hand-assigned
    penalty_severity: UnitInterval        # see 2.1 — derived from contract terms
    flexibility: FlexibilityLevel = FlexibilityLevel.MODERATE   # rigid | moderate | flexible
    late_fee_rate_per_day: Optional[float] = None
    priority_rank: Optional[int] = None    # assigned by rule-based/personalized prioritizer
    priority_score: Optional[float] = None
    created_at: datetime
    updated_at: datetime
    # validator: a payable can't link an Inflow, a receivable can't link an Outflow

# schemas/risk_score.py
class RiskScore(BaseSchema):
    id: UUID
    business_id: UUID
    obligation_id: Optional[UUID] = None
    counterparty_id: Optional[UUID] = None
    urgency_score: UnitInterval
    impact_score: UnitInterval
    default_probability: UnitInterval
    composite_score: UnitInterval
    weighting_scheme_version: str          # see 2.1 — now model + training-set version
    rationale: str                          # human-readable, no opaque internals
    computed_at: datetime

# schemas/simulation_result.py
class CashflowPercentileBand(BaseSchema):
    as_of_date: date
    p10: Money
    p50: Money
    p90: Money

class SimulationResult(BaseSchema):
    id: UUID
    business_id: UUID
    scenario_type: ScenarioType = ScenarioType.BASELINE   # baseline|best_case|worst_case|custom
    num_iterations: int                     # see 2.1 — derived from the convergence study
    random_seed: int                        # required, not Optional — reproducibility is mandatory
    horizon_days: int = 90
    projected_days_to_zero: Optional[int] = None
    probability_of_shortfall: float          # 0-1
    percentile_bands: list[CashflowPercentileBand]
    generated_at: datetime

# schemas/decision_plan.py
class DecisionActionItem(BaseSchema):
    obligation_id: UUID
    action_type: DecisionActionType         # pay_now|pay_partial|reschedule|negotiate|escalate_collection|defer
    allocated_amount: Money
    proposed_date: Optional[str] = None
    justification: str                       # chain-of-thought, safe for non-technical review

class DecisionPlan(BaseSchema):
    id: UUID
    business_id: UUID
    scenario_type: ScenarioType = ScenarioType.BASELINE
    simulation_result_id: Optional[UUID] = None
    available_cash: Money
    total_obligations_amount: Money
    solver_name: str                          # pulp_cbc | scipy_milp | dp_knapsack | saa_chance_constrained
    solver_status: str                        # solver's own status, e.g. "Optimal"
    objective_value: Optional[float] = None
    actions: list[DecisionActionItem]
    baseline_plan_id: Optional[UUID] = None    # personalized plan's rule-based origin
    devils_advocate_challenge_id: Optional[UUID] = None
    review_status: ReviewStatus = ReviewStatus.PENDING_REVIEW
    reviewed_by: Optional[UUID] = None
    audit_log_entry_id: Optional[UUID] = None
    created_at: datetime
```

Supporting schemas: `CounterpartyProfile` (relationship tier, on-time rate, Graph-RAG node id), `IngestionRecord` / `DuplicateResolutionResult` / `PIIMaskingResult`, `DevilsAdvocateChallenge`, `AuditLogEntry` (hash-chain), `EmailDraft` / `ReschedulePlan` / `GanttScheduleItem` / `ActionItem`.

### 2.1 Amendments required by the Quant Core

These are the *only* schema changes. Each names the original field so the diff is unambiguous.

| Field | Change | Why |
|---|---|---|
| `SimulationResult.receivable_uncertainty_model: str = "beta_delay_distribution"` | **Removed as a string.** Replaced by `uncertainty_model: UncertaintyModelSpec` — per-counterparty fitted family (Gamma / Weibull / Log-normal), MLE parameters, KS statistic and p-value, and the selection rationale; plus `copula: CopulaSpec` (family, Student-t degrees of freedom, source of the correlation matrix). | A free-text label asserting a distribution nobody fitted is exactly the "plausible without being real" failure. A.2 requires the fit to be recorded, not named. |
| `SimulationResult` | **Add** `runway_at_risk_days`, `conditional_runway_at_risk_days`, `confidence_level`, `mc_standard_error`. | RaR/CRaR is the headline metric of the product, not a buried field. |
| `SimulationResult.num_iterations: int = 10_000` | Default **removed**; the value is passed in and justified. 10,000 may well be the answer, but only as the smallest N whose RaR standard error falls under the tolerance stated in notebook 02. | No magic numbers carried over from a slide deck. |
| `SimulationResult.random_seed: Optional[int] = None` | → `random_seed: int`, required. | Reproducibility is a hard requirement, so it cannot be optional. |
| `RiskScore` | **Add** `model_name` (`rule_baseline` \| `logistic_l2` \| `gbm`), `feature_contributions: list[FeatureContribution]` (SHAP value or coefficient×value, with confidence intervals for the logistic model), `baseline_comparison: BaselineLift` (ROC-AUC of the model vs. the rules baseline, plus Brier score). | A.4 requires the explanation to reach the API response, where a non-technical user can see *why* a score is high. |
| `RiskScore.composite_score` — described as a "deterministic weighted combination" | Reworded: fitted model output, with `rationale` **generated from** `feature_contributions` rather than written by a rule template. | The rules table is now the baseline being beaten, not the product. |
| `RiskScore.weighting_scheme_version: str = "v1"` | → model identifier + training-set hash + fit timestamp. | "Which config produced this" must now identify a *fitted artifact*, which a version string alone cannot. |
| `DecisionPlan` | **Add** `lp_objective_value`, `dp_objective_value`, `solver_agreement: SolverAgreement` (absolute delta, tolerance, and a written explanation whenever they diverge), `chance_constraint_epsilon`, `saa_num_scenarios`. | A.3 requires both solvers and the chance-constrained variant to be visible in the output, including where they disagree. |
| `DecisionPlan.solver_name: str = "pulp_cbc"` | Default removed; the field now records which of the four solvers produced *this* plan. | A plan carries one solver's result; the comparison lives in `solver_agreement`. |
| `Inflow.certainty: UnitInterval = 1.0` | Default **removed.** `certainty` becomes an output of the fitted per-counterparty delay model (A.2), i.e. P(received by `expected_date`). | Provenance rule: every number traces to the generator or a fit. A default of 1.0 asserts perfect certainty about a receivable, which is the single strongest claim in the schema and the least defensible. |
| `Obligation.urgency`, `Obligation.penalty_severity` | Computed, not hand-assigned. `penalty_severity` derives from `late_fee_rate_per_day` and contract terms; `urgency` from due-date proximity combined with the counterparty's fitted default probability. | Provenance rule. These two fields feed the prioritizer *and* the LP objective — hand-typed values would silently determine the optimizer's answer. |

**New schemas to specify:**
- `ForecastResult` — point path, prediction intervals per horizon step, chosen model, walk-forward selection metrics (AIC/BIC where applicable, out-of-sample MAPE/RMSE), and the rejected candidates with their scores.
- `UncertaintyModelSpec`, `CopulaSpec`, `FeatureContribution`, `BaselineLift`, `SolverAgreement` (referenced above).
- `BacktestReport`, `RegretMetrics`, `CalibrationResult` — Section C outputs.

---

## 3. API Endpoint Specification

Full request/response shapes: `docs/API_SPEC.md`. The 12 REST groups + 4 WS channels:

**REST:** `/ingestion/*` (bank statements, invoices, OCR receipts, Decentro sync) → `/data-intelligence/*` (DBSCAN dedupe, PII mask, compliance) → `/financial-state/*` (summary + days-to-zero, inflows, outflows, obligations CRUD) → `/risk/*` (credit risk scoring) → `/decisions/*` (generate, get, challenge, approve/reject) → `/simulation/*` (Monte Carlo run, scenario toggle) → `/graph/*` → `/actions/*` (email draft, reschedule plan, Gantt, review queue, send) → `/audit/*` (chain, verify, per-entity trail) → `/chat`, `/voice/*` → `/market/*`.

**Added for the Quant Core:**

| Endpoint | Returns |
|---|---|
| `GET /risk/forecast` | `ForecastResult` — point path **and** prediction intervals, plus which model won walk-forward selection |
| `GET /simulation/runway-at-risk` | RaR, CRaR, confidence level, Monte Carlo standard error. First-class, not nested inside a simulation blob. |
| `GET /risk/{id}/explain` | `feature_contributions` + `baseline_comparison` — the trust-barrier answer, served |
| `GET /decisions/{id}/solvers` | LP vs. DP vs. chance-constrained objective values and `solver_agreement` |
| `GET /backtest/report` | `BacktestReport` — regret and calibration summary |

**WebSocket:** `/ws/ingestion/{batch_id}` · `/ws/decisions/{session_id}` (live baseline→personalized→devils_advocate→final stream) · `/ws/simulation/{sim_id}` (Monte Carlo progress) · `/ws/voice/{session_id}`.

Auth: JWT bearer, RBAC roles `owner`/`accountant`/`viewer`. Error envelope: `{error_code, message, details}`.

**Tier-5 stub contract.** `/graph/*`, `/voice/*`, and `/chat` ship as *working endpoints that honestly decline*: HTTP `501` with the standard error envelope and a `message` naming what is not yet built. They route, authenticate, and validate correctly. They never return a fabricated answer, and they are never dressed up with canned responses that imply a working model.

---

# SECTION A — Quant Core

`backend/app/services/quant_core/`

A new top-level service package. **Everything in it must be real:** fitted from data, with visible assumptions, docstrings naming the method and citing why it was chosen over alternatives, and a test that checks the method *behaves correctly* — a fitted distribution passing a goodness-of-fit test, a Monte Carlo estimate converging as N grows, an error metric verified against a known synthetic ground truth. "Returns without erroring" is not a test.

## A.0 — Synthetic data generator (`synthetic_data.py`)

There is no real bank data, so the generator is the foundation everything else is validated against.

Produces multi-year daily inflow/outflow histories with:
- **Trend + weekly/monthly seasonality, composed** — a real signal structure, not noise with a label on it
- **Occasional regime shocks** — a bad quarter, a large one-off receivable
- **Per-counterparty payment-delay behaviour drawn from a known distribution**

That last point is the central validation trick and it is used explicitly: because the generator *knows* the true delay parameters, the fitting code in A.2 can be checked for whether it **recovers** them. This is what makes the statistics falsifiable rather than decorative, and it is demonstrated in notebook 02.

Parameterized to emit both **"easy"** and **"adversarial"** regimes — heavy-tailed, highly correlated — so every downstream model is tested where it is likely to fail, not only where it is likely to look good.

## A.1 — Time-series cash flow forecasting (`forecasting.py`)

Implement, behind one common interface:
1. **Seasonal naive baseline** — the honest floor. Any model that cannot beat it is not earning its complexity.
2. **Holt-Winters exponential smoothing** (`statsmodels`)
3. **SARIMAX** (`statsmodels`)

**Model selection: walk-forward (rolling-origin) cross-validation**, not a single train/test split — a single split on a seasonal series measures luck as much as skill. Select by AIC/BIC where applicable and confirm out-of-sample with MAPE/RMSE.

Output a `days_to_zero` point estimate **and propagate forecast uncertainty (prediction intervals) into A.2**, rather than handing the Monte Carlo a deterministic path. Treating a forecast as certain and then simulating "uncertainty" on top of it double-counts confidence and understates risk.

**`FORECASTING_METHODOLOGY.md`** — a short research memo: what was tried, what was rejected, and why. Written to be probed in an interview, so the rejections matter as much as the selection.

## A.2 — Monte Carlo liquidity risk engine (`monte_carlo_engine.py`)

- **Fit an actual delay/default distribution per counterparty** from historical payment data. Candidates: Gamma, Weibull, Log-normal, fit by MLE via `scipy.stats`, validated with a KS test, best fit selected and **the choice logged** — including the runners-up, so the selection is auditable.
- **Model dependence between counterparties.** Implement a Gaussian or Student-t copula (the latter for fat tails) so correlated slowdowns are captured. A comment must explain *why naive independent sampling understates tail risk*: independence makes simultaneous late payment across many counterparties astronomically unlikely, when in reality a downturn delays everyone at once — precisely the scenario that empties the account. Independent sampling produces a comfortable tail that does not exist.
- **Runway-at-Risk (RaR)** — explicitly analogous to VaR in market risk. "95% RaR = 11 days" means a 5% chance of hitting zero cash within 11 days. Report **CRaR** alongside it, the conditional/expected-shortfall version, because RaR alone says nothing about how bad the bad case is. This is the single most interview-worthy artifact in the project and is treated as the **headline metric of the app**.
- **Verify Monte Carlo convergence** — standard error vs. N, estimated and plotted in notebook 02. N is chosen because the error is small enough, not because 10,000 is a round number.

## A.3 — Optimization: two independent solvers, cross-checked (`optimization/`)

- **`lp_solver.py`** — linear/mixed-integer program (PuLP or `scipy.optimize.linprog` / `scipy.optimize.milp`) allocating available cash across obligations to minimize penalty, respecting due dates and forbidding partial payment where `flexibility == RIGID`.
- **`dp_solver.py`** — an **independent** solution to the same problem as a knapsack/DP: which obligations to fully or partially fund this period under a cash budget, minimizing weighted penalty. This is the competitive-programming signal. The docstring states the **state space, the transition, and the complexity** explicitly. Written for clarity over cleverness — it should be walkable line by line in a live interview.
- **`cross_validation.py`** — run both solvers on the same scenario and assert the objective values agree within solver tolerance on small/medium cases where the DP is tractable. Two independent implementations agreeing is real evidence of correctness; one implementation returning a plausible number is not. **Where they diverge, the divergence is explained and written up** — e.g. the DP handles integer lot constraints that the LP relaxation does not. Divergence is a finding, not something to hide.
- **`chance_constrained.py`** — extend to chance-constrained optimization: given the Monte Carlo scenarios from A.2, solve the allocation so that P(shortfall within horizon) ≤ ε, via **Sample Average Approximation** (optimize against a batch of Monte Carlo draws rather than one point estimate). This is the piece that actually connects the optimization layer to the risk layer, instead of treating `available_cash` as a known constant when A.2 has just finished demonstrating that it isn't.

  **Scenario cap — decided here, not left to default.** SAA cost scales with (scenario count × binary variables), and the rigid-obligation constraints from `lp_solver.py` are binary. Feeding all 10,000 A.2 draws into a MILP is not viable, and it would stall Section C's harness, which re-solves this problem on **every replay day**. The chance-constrained solver therefore takes a **subsample of 200–500 scenarios**, never the full draw set, with the cap as an explicit parameter rather than a hidden constant. Two consequences to handle rather than ignore:
  - The subsample is drawn with the run's seed and recorded in `DecisionPlan.saa_num_scenarios`, so the solve is reproducible.
  - A smaller sample makes the ε-constraint itself noisy — the estimated P(shortfall) has sampling error. The A.3 notebook/test should show the chosen cap is large enough that the SAA solution is stable across resamples, which is the same "why this N" argument A.2 makes for its iteration count. Picking 200–500 because it solves quickly, without that check, would be exactly the magic-number failure §2.1 struck elsewhere.

## A.4 — Credit / default risk model (`risk_scoring.py`)

Replaces "pure rule-based, therefore auditable" with something both statistically real and explainable — the harder and more convincing answer to the deck's own trust-barrier problem.

- **Feature engineering** from counterparty history: recency/frequency/monetary-style features, historical on-time rate, payment volatility, trend in payment behaviour.
- **Regularized logistic regression** as the primary model (interpretable coefficients), with a **gradient-boosted tree** (scikit-learn) as the comparison.
- **Proper train/test split or k-fold CV**, reporting **ROC-AUC and a calibration curve** (predicted vs. observed default rate by bucket). Accuracy alone is not reported — on imbalanced default data it is uninformative to the point of being misleading.
- **Explainability surfaced in the API response**: SHAP values, or at minimum logistic coefficients with confidence intervals, so a non-technical user sees *why* a score is high.
- **Explicit comparison against a simple rule-based baseline** (e.g. days overdue × amount), reporting the lift. Complexity has to be shown to be earned, not assumed. If the GBM does not beat the logistic model, or the logistic model does not beat the rules, that result is reported as-is.

## A.5 — Anomaly / duplicate detection upgrade (`anomaly_detection.py`)

Keeps DBSCAN from the original plan and adds a statistical layer beside it:
- **Robust z-scores (MAD-based, not mean/std)** for flagging outlier transactions — the mean and standard deviation are themselves distorted by the outliers being hunted.
- **Isolation Forest** as an alternative unsupervised detector, with a report of **where it agrees and disagrees with the DBSCAN clusters and why**. That comparison is itself a small piece of research and is written up as one.

## A.6 — Unstructured data (`unstructured.py`)

- **OCR'd receipt/invoice text** → vendor/category via **embedding similarity to a small reference set**. Not regex — the point is to demonstrate handling genuinely unstructured text.
- **`market_intelligence`, if kept, must be grounded**: "seasonal trends" comes from an actual **STL decomposition** (`statsmodels`) of the time series, not a vague macro-signal black box. If real news/text is pulled, claims are **quantified with a proper test** — e.g. a t-test comparing pre/post periods — rather than sentiment asserted casually.

---

# SECTION B — Wire the deterministic backbone to real Quant Core outputs

Everything in `services/ingestion`, `services/data_intelligence`, and `services/decision_engine` stays as originally specified, with these modules becoming thin callers rather than independent implementations:

| Module | Now calls | Note |
|---|---|---|
| `decision_engine/credit_risk_scorer.py` | **A.4** `risk_scoring.py` | Not a standalone rule table |
| `simulation/monte_carlo_engine.py` | **A.2** | |
| `simulation/scenario_toggle.py` | **A.2** | Best/worst case = percentile cuts of the simulated distribution |
| `decision_engine/lp_solver.py` | **A.3** | Exposes LP **and** DP results plus the chance-constrained variant through `DecisionPlan` |
| `decision_engine/rule_based_prioritizer.py` | *unchanged — deliberately* | See below |

**`rule_based_prioritizer.py` stays as a genuine baseline.** It is kept and used as the explicit comparison point for the personalized/ML-driven prioritizer, exactly as A.4 compares against its rules baseline. A baseline you can point to and say "here is what naive rules get wrong, measured" is worth more than a baseline quietly deleted once the fancy model works.

---

# SECTION C — Backtesting & validation harness

`backend/app/services/backtesting/` · **Not skippable.**

The harness:

1. **Replays a synthetic multi-month history day by day.**
2. At each step, generates a forecast (A.1), a risk estimate (A.2), and a recommended action plan (A.3), **using only information available up to that day.** No look-ahead. The replay boundary is enforced in code, not by convention — look-ahead leakage is the single easiest way to produce impressive backtest numbers that mean nothing.

   **Cost note:** this loop calls A.2 and A.3 once per replay day, so a multi-month replay re-solves the chance-constrained MILP tens to hundreds of times. The A.3 scenario cap exists primarily to keep this tractable; the harness must also be able to run against the plain LP/DP solver instead of the chance-constrained one, so a slow SAA solve degrades the backtest's richness rather than blocking it entirely.
3. **Compares the recommended plan's outcome against a hindsight-optimal plan** computed with full knowledge of what actually happened, and reports the **regret** (efficiency loss). Regret against a perfect-information benchmark is the honest measure: it says how much was lost to uncertainty rather than to a bad method.
4. **Tracks forecast calibration over time** — are the prediction intervals honest? Do ~95% of actual outcomes fall inside the 95% interval? A model with tight, wrong intervals is more dangerous than one with wide, honest ones, and only this check distinguishes them.
5. **Outputs `BACKTEST_REPORT.md`** summarizing all of the above in plain language, written the way a research review is written — not a hackathon README.

---

# SECTION D — Deliverables checklist

- [ ] **`notebooks/`** — at least 3 Jupyter notebooks:
  - `01_forecasting_model_selection.ipynb` — walk-forward selection, rejected candidates
  - `02_monte_carlo_runway_at_risk.ipynb` — RaR derivation, parameter-recovery check against A.0's known ground truth, convergence check
  - `03_credit_risk_model_comparison.ipynb` — rules vs. logistic vs. GBM, with calibration plots
- [ ] **`QUANT_METHODOLOGY.md`** — one top-level document explaining in prose every statistical and optimization choice in Section A, what alternatives were considered, and what the honest limitations are — including being upfront about what synthetic data cannot tell you.
- [ ] **`BACKTEST_REPORT.md`** — Section C output.
- [ ] **`backend/tests/unit/quant_core/`** — pytest suite testing **statistical correctness**: does the fitted distribution recover known parameters within tolerance on synthetic data with a known generating process? Does the Monte Carlo standard error shrink as √N? Is the error metric right against a hand-computed ground truth? Not "does the function return without erroring."
- [ ] **The rest of this architecture plan**, implemented at least to working-endpoint level (Section B, plus the API/audit/RBAC layer). This may be simpler than the full Next.js/Neo4j/Celery/voice stack if time is short: **a working FastAPI backend with real logic outranks a fully wired frontend.**

---

## Code quality bar

- **Type hints and docstrings on every public function in `quant_core/`**, with the docstring naming the method — e.g. *"MLE fit of a Weibull distribution via `scipy.stats.weibull_min.fit`"* — and why it was chosen over the alternatives.
- **All randomness seeded and reproducible.** Seeds are explicit parameters, never module-level globals.
- **No hardcoded "looks reasonable" numbers anywhere in the Quant Core.** Every number traces back to either the synthetic generator or a fitted/computed value. This is why `certainty = 1.0`, `num_iterations = 10_000`, and hand-set `urgency`/`penalty_severity` were struck from the schemas in §2.1.
- **Clarity over cleverness in the DP and optimization code.** It will come up in a live interview; it should be walkable line by line.

---

## Implementation roadmap

### Step 0 — Environment (blocking)

Nothing in Sections A–C can be *validated* without a working interpreter, and unvalidated statistical code is precisely what this document rules out. Current machine state:

- `python` / `python3` resolve only to the **Windows Store stub** at `%LOCALAPPDATA%\Microsoft\WindowsApps\python.exe` — not a usable interpreter
- `py`, `conda`, `uv` — all absent
- `node` and `git` — present

Required: install **Python 3.12** (winget or python.org), create `.venv`, and pin `requirements.txt` — numpy, pandas, scipy, statsmodels, scikit-learn, PuLP, shap, matplotlib, jupyter, fastapi, pydantic v2, pytest.

### Steps 1–5 — follow the Section 0.5 priority order

1. **Quant Core (A.0 → A.6),** each module landing with its statistical-correctness test. A.0 first: nothing downstream can be validated before the ground-truth generator exists.
2. **Deterministic backbone (Section B)** wired to real Quant Core outputs.
3. **Backtesting harness (Section C)** + `BACKTEST_REPORT.md`.
4. **API layer, audit chain, RBAC, WebSockets, frontend.**
5. **Graph RAG, voice, multilingual** — honest `501` stubs unless everything above is complete.

Notebooks and `QUANT_METHODOLOGY.md` are written *alongside* steps 1–3, not retrofitted afterwards. A methodology document reconstructed after the fact tends to describe the choices that worked rather than the choices that were made.

---

## Known limitations

Stated here, in the architecture document, for the same reason `QUANT_METHODOLOGY.md` will state them — a reviewer will find these anyway, and finding them unacknowledged is worse than finding them listed:

- **Synthetic data cannot validate real-world counterparty behaviour.** It validates that the *estimators work*, which is a genuine but strictly narrower claim.
- **Parameter recovery proves the estimator inverts its own generating process.** If the real world is not Gamma/Weibull/Log-normal, recovery on synthetic data says nothing about fit on real data. The KS test on real data would be the check that matters, and it cannot be run here.
- **Copula correlation is assumed, not observed.** The dependence structure is imposed by the generator, so the copula recovers a correlation that was put there by hand. On real data this parameter would be the hardest and most consequential thing to estimate.
- **Backtest regret is measured against a hindsight-optimal plan on synthetic history.** It bounds the method's efficiency loss under known dynamics; it does not predict live performance.
- **Default labels are generated, not observed.** The A.4 ROC-AUC is therefore an upper bound on what the same features would achieve against real defaults.
