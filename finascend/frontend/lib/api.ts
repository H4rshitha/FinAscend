/**
 * The only path to data in this application.
 *
 * There are no fixture files, no seeded constants and no fallback numbers
 * anywhere in the frontend. If the backend is down, pages render an error that
 * says so and tells you how to start it — they never degrade into showing
 * plausible-looking figures, because a dashboard that invents data is
 * indistinguishable from one that does not, which is exactly the failure the
 * rest of this project is built to avoid.
 */

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000/api/v1";

let tokenPromise: Promise<string> | null = null;

/** Fetch and cache a demo bearer token for the session. */
async function getToken(): Promise<string> {
  if (!tokenPromise) {
    tokenPromise = fetch(`${API_BASE}/auth/token?role=owner`, {
      method: "POST",
    })
      .then((r) => {
        if (!r.ok) throw new ApiError(`auth failed (HTTP ${r.status})`, r.status);
        return r.json();
      })
      .then((j) => j.access_token as string)
      .catch((e) => {
        // Do not cache a failed handshake — the backend may simply not be up
        // yet, and every later call should retry rather than inherit the error.
        tokenPromise = null;
        throw e;
      });
  }
  return tokenPromise;
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
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let token: string;
  try {
    token = await getToken();
  } catch {
    throw new ApiError(
      `Cannot reach the FinAscend API at ${API_BASE}.`,
      0,
      { hint: "start_backend" }
    );
  }

  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      Authorization: `Bearer ${token}`,
      ...(init?.headers ?? {}),
    },
  }).catch(() => {
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
    const d = detail as { message?: string; details?: Record<string, string> } | null;
    throw new ApiError(
      d?.message ?? `${path} failed (HTTP ${res.status})`,
      res.status,
      detail
    );
  }
  return (await res.json()) as T;
}

export const api = {
  get: <T>(path: string) => request<T>(path),
  post: <T>(path: string, body?: FormData) =>
    request<T>(path, { method: "POST", body }),
  /** Absolute URL for an <img src>, which cannot carry an Authorization header. */
  imageUrl: (path: string) => `${API_BASE}${path}`,
  token: getToken,
};

// ---------------------------------------------------------------------------
// Response shapes — mirrored from the FastAPI handlers.
// ---------------------------------------------------------------------------

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
    /** q_hat / z. 1.0 = the model's own scale was right. */
    scale_ratio: number | null;
    n_calibration_scores: number | null;
    level_achievable: boolean | null;
    scale_gamma: number | null;
    reading: string;
  };
}

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

export interface FinancialSummary {
  as_of: string;
  cash_balance: number;
  outstanding_receivables: number;
  outstanding_receivable_value: number;
  days_to_zero_point_estimate: number | null;
  note: string;
}

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
      /** Directly comparable to `empirical` — same replay configuration. */
      pooled: number;
      pooled_config: string;
      /** From a denser 14-day diagnostic replay; NOT this configuration. */
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

export interface ModelPerformance {
  roc_auc: number;
  brier_score: number;
  n_train: number;
  n_test: number;
  positive_rate: number;
  [k: string]: number | string;
}

export interface RiskModels {
  performance: Record<string, ModelPerformance>;
  lift_vs_baseline: Record<
    string,
    { candidate_auc: number; baseline_auc: number; auc_lift: number; verdict: string;
      [k: string]: number | string }
  >;
  note: string;
}

export interface CalibrationBin {
  bin_lower: number;
  bin_upper: number;
  mean_predicted: number;
  observed_rate: number;
  n: number;
}

export interface FeatureContribution {
  feature: string;
  value: number;
  contribution: number;
  ci_lower?: number | null;
  ci_upper?: number | null;
}

export interface RiskExplain {
  model: string;
  default_probability: number;
  rationale: string;
  feature_contributions: FeatureContribution[];
  calibration: CalibrationBin[];
  baseline_comparison: { auc_lift: number; verdict: string } | null;
}

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
