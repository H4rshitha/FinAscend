/**
 * The only path to data in this application.
 *
 * There are no fixture files, no seeded constants and no fallback numbers
 * anywhere in the frontend. If the backend is down, pages render an error that
 * says so and tells you how to start it — they never degrade into showing
 * plausible-looking figures, because a dashboard that invents data is
 * indistinguishable from one that does not, which is exactly the failure the
 * rest of this project is built to avoid.
 *
 * Every type below mirrors a real FastAPI handler. If a field is optional here
 * it is because the backend can genuinely omit it, not to paper over a shape
 * mismatch.
 */

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000/api/v1";

export const TOKEN_STORAGE_KEY = "finascend.token";

/**
 * The bearer token for the signed-in user.
 *
 * Held in a module variable AND mirrored to localStorage. The module variable
 * is what requests read, so a signed-in session works without touching storage
 * on every call; localStorage is only how the session survives a refresh.
 *
 * STORAGE TRADE-OFF, STATED. localStorage is readable by any script running on
 * the page, so an XSS bug becomes a token theft. The more secure arrangement is
 * an httpOnly, SameSite cookie the browser attaches automatically and no script
 * can read — but that requires the API to set cookies and a CSRF defence to go
 * with it, since the browser would then attach the credential to cross-site
 * requests too. That is the right production change and it is a backend change,
 * not a frontend one. Recorded here rather than left implied.
 */
let authToken: string | null = null;

export function setAuthToken(token: string | null): void {
  authToken = token;
  if (typeof window === "undefined") return;
  if (token) window.localStorage.setItem(TOKEN_STORAGE_KEY, token);
  else window.localStorage.removeItem(TOKEN_STORAGE_KEY);
}

/** Restore a token saved by a previous session. Returns it, or null. */
export function loadStoredToken(): string | null {
  if (typeof window === "undefined") return null;
  authToken = window.localStorage.getItem(TOKEN_STORAGE_KEY);
  return authToken;
}

export function getAuthToken(): string | null {
  return authToken;
}

/** Raised when a request needs a session and there isn't one. */
export class NotAuthenticatedError extends Error {
  constructor() {
    super("Not signed in.");
    this.name = "NotAuthenticatedError";
  }
}

export class ApiError extends Error {
  status: number;
  detail: unknown;
  constructor(message: string, status = 0, detail: unknown = null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
  /** True when the backend could not be reached at all, vs. returned an error. */
  get isOffline() {
    return this.status === 0;
  }
}

/** Requests that are legitimately made while signed out. */
const PUBLIC_PATHS = ["/auth/login", "/auth/signup", "/auth/options"];

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const isPublic = PUBLIC_PATHS.some((p) => path.startsWith(p));
  const headers: Record<string, string> = { ...((init?.headers as Record<string, string>) ?? {}) };

  if (!isPublic) {
    const token = authToken ?? loadStoredToken();
    if (!token) throw new NotAuthenticatedError();
    headers.Authorization = `Bearer ${token}`;
  }

  const res = await fetch(`${API_BASE}${path}`, { ...init, headers }).catch(() => {
    throw new ApiError(`Cannot reach the FinAscend API at ${API_BASE}.`, 0, {
      hint: "start_backend",
    });
  });

  if (!res.ok) {
    let detail: unknown = null;
    try {
      detail = (await res.json()).detail ?? null;
    } catch {
      /* body was not JSON; the status alone is the message */
    }
    const d = detail as { message?: string } | null;
    throw new ApiError(d?.message ?? `${path} failed (HTTP ${res.status})`, res.status, detail);
  }
  return (await res.json()) as T;
}

export const api = {
  get: <T,>(path: string) => request<T>(path),
  post: <T,>(path: string, body?: FormData) =>
    request<T>(path, { method: "POST", body }),
  /** POST a JSON body. Used by the auth routes, which take JSON not FormData. */
  postJson: <T,>(path: string, body: unknown) =>
    request<T>(path, {
      method: "POST",
      body: JSON.stringify(body),
      headers: { "Content-Type": "application/json" },
    }),
  /**
   * Fetch an authenticated image and hand back an object URL.
   *
   * An `<img src>` cannot carry an Authorization header, so pointing it
   * straight at a protected endpoint just renders a broken image. Fetching the
   * bytes with the token and wrapping them in a blob URL keeps the auth model
   * intact. The caller MUST revoke the returned URL on unmount.
   */
  objectUrl: async (path: string): Promise<string> => {
    const token = authToken ?? loadStoredToken();
    if (!token) throw new NotAuthenticatedError();
    const res = await fetch(`${API_BASE}${path}`, {
      headers: { Authorization: `Bearer ${token}` },
    });
    if (!res.ok) throw new ApiError(`image failed (HTTP ${res.status})`, res.status);
    return URL.createObjectURL(await res.blob());
  },
};

// ===========================================================================
// financial state
// ===========================================================================

export interface FinancialSummary {
  as_of: string;
  cash_balance: number;
  outstanding_receivables: number;
  outstanding_receivable_value: number;
  days_to_zero_point_estimate: number | null;
  note: string;
}

// ===========================================================================
// forecasting
// ===========================================================================

export interface ForecastPoint {
  as_of_date: string;
  point: number;
  lower: number;
  upper: number;
}

export interface CandidateScore {
  model_name: string;
  rmse: number;
  mape: number;
  aic: number | null;
  bic: number | null;
  n_folds: number;
  fold_rmses: number[];
}

export interface ForecastResponse {
  selected_model: string;
  selection_rationale: string;
  interval_confidence: number;
  days_to_zero: number | null;
  candidates: CandidateScore[];
  path: ForecastPoint[];
  interval_calibration: {
    method: string;
    /** Multiplies a sigma — compare against `z_reference`, never against 1.0. */
    q_hat: number | null;
    z_reference: number | null;
    scale_ratio: number | null;
    n_calibration_scores: number | null;
    level_achievable: boolean | null;
    scale_gamma: number | null;
    reading: string;
  };
}

// ===========================================================================
// simulation
// ===========================================================================

export interface RarResponse {
  runway_at_risk_days: number;
  conditional_runway_at_risk_days: number;
  confidence_level: number;
  probability_of_shortfall: number;
  n_iterations: number;
  mc_standard_error: number;
  random_seed: number;
  interpretation: string;
}

export interface DistCandidate {
  family: string;
  params: Record<string, number>;
  ks_statistic: number;
  ks_pvalue: number;
  log_likelihood: number;
  aic: number;
  /** The winner, per AIC. Read this rather than matching on family name. */
  selected: boolean;
}

export interface UncertaintyModel {
  fitted_at: string;
  copula: {
    family: string;
    df?: number | null;
    correlation_matrix: number[][];
    counterparty_order: string[];
    correlation_source?: string;
  };
  fits: {
    counterparty_id: string;
    n_observations: number;
    selected_family: string;
    selected_params: Record<string, number>;
    /**
     * P(delay <= 0) — structurally ZERO for every counterparty, because `loc`
     * is pinned at 0 and the families are continuous. Do not render it as a
     * trust signal; it carries no information. Use the delay statistics below.
     */
    prob_on_time: number;
    prob_on_time_note?: string;
    mean_delay_days: number;
    median_delay_days: number;
    p90_delay_days: number;
    prob_within_7_days: number;
    prob_within_30_days: number;
    selection_rationale: string;
    candidates: DistCandidate[];
  }[];
}

// ===========================================================================
// credit risk
// ===========================================================================

export interface ModelPerformance {
  model_name: string;
  roc_auc: number;
  brier_score: number;
  log_loss: number;
  n_train: number;
  n_test: number;
}

/** Shape returned by `compare_to_baseline` — used in two places, so named once. */
export interface BaselineComparison {
  baseline_model: string;
  baseline_roc_auc: number;
  model_roc_auc: number;
  auc_lift: number;
  verdict: string;
}

export interface RiskModels {
  performance: Record<string, ModelPerformance>;
  lift_vs_baseline: Record<string, BaselineComparison>;
  note: string;
}

export interface CalibrationBin {
  bucket_lower: number;
  bucket_upper: number;
  mean_predicted: number;
  observed_rate: number;
  n: number;
}

export interface FeatureContribution {
  feature: string;
  value: number;
  contribution: number;
  direction?: string;
  ci_lower?: number | null;
  ci_upper?: number | null;
}

export interface RiskExplain {
  model: string;
  default_probability: number;
  rationale: string;
  feature_contributions: FeatureContribution[];
  calibration: CalibrationBin[];
  baseline_comparison: BaselineComparison | null;
}

// ===========================================================================
// decisions
// ===========================================================================

export interface PlanAction {
  obligation_id: string;
  label: string;
  category: string;
  action_type: "pay_now" | "pay_partial" | "defer";
  amount_due: number;
  allocated_amount: number;
  shortfall: number;
  days_until_due: number;
  is_rigid: boolean;
  late_fee_if_unpaid: number;
  /** Written by the backend for non-technical review. Never composed here. */
  justification: string;
}

export interface DecisionPlan {
  as_of: string;
  available_cash: number;
  total_obligations_amount: number;
  solver_name: string;
  solver_status: string;
  objective_value: number;
  review_status: string;
  n_obligations: number;
  n_paid_in_full: number;
  expected_late_fees: number;
  shortfall: number;
  actions: PlanAction[];
  baseline_comparison: {
    rules_objective_value: number;
    lp_objective_value: number;
    lp_better_on_this_instance: boolean;
    caveat: string;
  };
}

export interface AllocationItem {
  obligation_id: string;
  allocated_amount: number;
  fully_funded: boolean;
}

export interface SolverSolution {
  solver_name: string;
  status: string;
  objective_value: number;
  allocations: AllocationItem[];
  solve_seconds: number;
}

export interface SolverInstanceComparison {
  available_cash: number;
  total_obligations: number;
  lp: SolverSolution;
  dp: SolverSolution;
  solver_agreement: {
    lp_objective_value: number;
    dp_objective_value: number;
    absolute_delta: number;
    tolerance: number;
    agree: boolean;
    explanation: string;
  };
  /**
   * Note the nesting: the chance-constrained result wraps a normal
   * `SolverSolution` under `solution`, alongside the risk parameters. Its
   * `status` carries a full infeasibility explanation when even zero spend
   * breaches epsilon — which is a real outcome on a stressed book, not an
   * error, and the UI must say so rather than render a spend limit.
   */
  chance_constrained: {
    epsilon: number;
    saa_num_scenarios: number;
    achieved_shortfall_probability: number;
    solution: SolverSolution;
    stability_across_resamples: number;
  };
  rules_baseline: SolverSolution;
  optimizer_lift_vs_baseline: Record<string, number | string>;
}

export interface PriorityRanking {
  ranking: { obligation_id: string; rank: number; score: number; reason: string }[];
}

export interface ApproveResponse {
  decision_id: string;
  review_status: string;
  audit_sequence: number;
  audit_hash: string;
}

// ===========================================================================
// backtest
// ===========================================================================

export interface StrategyRow {
  name: string;
  total_realized_penalty: number;
  total_hindsight_penalty: number;
  relative_regret: number;
  mean_regret: number;
  p95_regret: number;
  over_commitment_steps: number;
  n_steps: number;
  vs_rules_baseline: number;
}

export interface SolverComparison {
  config: Record<string, string | number>;
  generated_at: string;
  strategies: StrategyRow[];
  finding: string;
}

export interface RegretPoint {
  as_of: string;
  regret: number;
  realized_penalty: number;
  hindsight_penalty: number;
  planned_spend: number;
  realized_cash: number;
  over_committed: boolean;
}

export interface BacktestSummary {
  generated_at: string;
  config: Record<string, string | number>;
  strategies: (StrategyRow & { regret_series: RegretPoint[] })[];
  calibration: {
    nominal: number;
    empirical: number;
    n_observations: number;
    mean_interval_width: number;
    verdict: string;
    previous_build: {
      pooled: number;
      pooled_config: string;
      diagnostic_before_pooled: number;
      diagnostic_after_pooled: number;
      sarimax_branch: number;
      holt_winters_branch: number;
      diagnostic_config: string;
      note: string;
    };
    by_horizon: {
      label: string;
      from: number;
      to: number;
      n: number;
      coverage: number;
      mean_width: number;
    }[];
  };
  steps: {
    as_of: string;
    opening_balance: number;
    forecast_model: string;
    conformal_q_hat: number;
    runway_at_risk_days: number;
    conditional_runway_at_risk_days: number;
    probability_of_shortfall: number;
    mc_standard_error: number;
    n_receivables: number;
  }[];
}

export interface BacktestCalibration {
  nominal: number;
  empirical: number;
  n_observations: number;
  verdict: string;
  by_horizon: {
    label: string;
    from: number;
    to: number;
    n: number;
    coverage: number;
    mean_width: number;
  }[];
  q_hat_by_step: { as_of: string; model: string; q_hat: number }[];
  [k: string]: unknown;
}

// ===========================================================================
// audit
// ===========================================================================

export interface AuditEntry {
  sequence: number;
  timestamp: string;
  actor_id: string;
  action: string;
  entity_type: string;
  entity_id: string;
  payload: Record<string, unknown>;
  previous_hash: string;
  entry_hash: string;
}

export interface AuditChain {
  head_hash: string | null;
  n_entries: number;
  entries: AuditEntry[];
}

export interface AuditVerify {
  valid: boolean;
  first_broken_sequence: number | null;
  message: string;
  caveat: string;
}

// ===========================================================================
// ingestion / OCR
// ===========================================================================

export interface ReceiptSample {
  id: string;
  difficulty: string;
  index: number;
  truth: {
    vendor_name: string;
    invoice_number: string;
    issue_date: string;
    total_amount: number;
    tax_amount: number;
    category: string;
  };
}

export interface OcrPipelineResult {
  ocr: {
    engine: string;
    n_regions: number;
    mean_confidence: number;
    elapsed_ms: number;
    text: string;
    lines: { text: string; confidence: number; bbox: number[] }[];
  };
  extraction: {
    vendor_name: string | null;
    invoice_number: string | null;
    issue_date: string | null;
    total_amount: number | null;
    tax_amount: number | null;
    field_confidence: Record<string, number>;
  };
  classification: {
    category: string;
    confidence: number;
    runner_up: string | null;
    runner_up_score: number;
    margin: number;
    is_uncertain: boolean;
    method: string;
  } | null;
  record: {
    id: string;
    counterparty_name: string;
    amount: number;
    currency: string;
    due_date: string;
    category: string;
    source_type: string;
    source_reference: string;
    needs_review: boolean;
    review_reasons: string[];
    extraction_confidence: number;
  } | null;
  duplicate_screen: {
    is_exact_duplicate: boolean;
    dbscan_label: number;
    robust_z: number;
    flagged_by_isolation_forest: boolean;
    reason: string;
    caveat: string;
  } | null;
  rejected: { reason: string; why_this_is_correct: string } | null;
  truth?: Record<string, string | number>;
  correct?: Record<string, boolean>;
}

// ===========================================================================
// auth / session
// ===========================================================================

export type CompanySize = "solo" | "small" | "medium" | "large";
export type PlanName = "essentials" | "professional" | "enterprise";

/**
 * Capability strings, mirroring `app/core/entitlements.Capability`.
 *
 * The UI checks these, never the plan name — so re-packaging the plans is a
 * backend change alone. The backend is the source of truth; this type only
 * gives the editor autocomplete over the same vocabulary.
 */
export type Capability =
  | "cash_health"
  | "action_plan"
  | "receipt_capture"
  | "customer_list"
  | "method_panels"
  | "scenario_explorer"
  | "solver_comparison"
  | "credit_explainability"
  | "backtest_history"
  | "audit_log"
  | "audit_export";

export interface SessionUser {
  id: string;
  email: string;
  full_name: string;
  role: string;
}

export interface SessionOrg {
  id: string;
  name: string;
  company_size: CompanySize;
  company_size_label: string;
  plan: PlanName;
  plan_label: string;
  plan_tagline: string;
  industry: string | null;
}

export interface Session {
  access_token: string;
  token_type: string;
  expires_in_minutes: number;
  user: SessionUser;
  organization: SessionOrg;
  capabilities: Capability[];
}

export interface SignupOptions {
  company_sizes: {
    value: CompanySize;
    label: string;
    headcount: string;
    hint: string;
    plan: PlanName;
    plan_label: string;
  }[];
  plans: {
    plan: PlanName;
    label: string;
    tagline: string;
    capabilities: Capability[];
  }[];
}

export interface SignupPayload {
  full_name: string;
  email: string;
  password: string;
  company_name: string;
  company_size: CompanySize;
  industry?: string | null;
}
