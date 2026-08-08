/**
 * Formatting for the plain-language layer.
 *
 * The rule that governs this file: the DEFAULT view speaks the way a person
 * speaks. "about 4 weeks", "₹42 lakh", "1 in 20". The expanded method panels
 * are allowed to be exact, because the audience there has asked for exactness.
 * So most helpers come in a rounded pair and a precise pair, and the page picks
 * by which layer it is rendering.
 */

/** Exact rupees with Indian digit grouping: 1,58,29,640. */
export const inr = (n: number, digits = 0) =>
  "₹" +
  n.toLocaleString("en-IN", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });

/**
 * Approximate rupees in the units Indian businesses actually speak in.
 * 4209877 -> "₹42.1 lakh". Used in the default view only; tables use `inr`.
 */
export function inrShort(n: number): string {
  const a = Math.abs(n);
  const sign = n < 0 ? "−" : "";
  if (a >= 1e7) return `${sign}₹${(a / 1e7).toFixed(a / 1e7 >= 10 ? 0 : 2)} crore`;
  if (a >= 1e5) return `${sign}₹${(a / 1e5).toFixed(a / 1e5 >= 10 ? 0 : 1)} lakh`;
  if (a >= 1000) return `${sign}₹${(a / 1000).toFixed(0)},000`;
  return `${sign}₹${Math.round(a)}`;
}

/** Compact axis label: 1.6M, 250k. For chart ticks, where space is scarce. */
export function compact(n: number): string {
  const a = Math.abs(n);
  const sign = n < 0 ? "−" : "";
  if (a >= 1e7) return `${sign}${(a / 1e7).toFixed(1)}Cr`;
  if (a >= 1e5) return `${sign}${(a / 1e5).toFixed(1)}L`;
  if (a >= 1000) return `${sign}${Math.round(a / 1000)}k`;
  return `${sign}${Math.round(a)}`;
}

export const pct = (n: number, d = 1) => `${(n * 100).toFixed(d)}%`;

/** "1 in 20" reads more concretely than "5%" for a risk a person must weigh. */
export function oneIn(p: number): string {
  if (p <= 0) return "effectively never";
  if (p >= 1) return "certain";
  const n = Math.round(1 / p);
  return `about 1 in ${n}`;
}

/** Days as a person would say them. 41 -> "about 6 weeks". */
export function plainDays(d: number): string {
  const n = Math.round(d);
  if (n <= 0) return "no time left";
  if (n === 1) return "1 day";
  if (n < 14) return `${n} days`;
  if (n < 60) {
    const w = Math.round(n / 7);
    return `about ${w} week${w === 1 ? "" : "s"}`;
  }
  const m = Math.round(n / 30);
  return `about ${m} month${m === 1 ? "" : "s"}`;
}

/** "in 3 days" / "in about 2 weeks", for a due date. */
export function dueIn(days: number): string {
  const n = Math.round(days);
  if (n <= 0) return "due today";
  if (n === 1) return "due tomorrow";
  if (n < 14) return `due in ${n} days`;
  const w = Math.round(n / 7);
  return `due in about ${w} week${w === 1 ? "" : "s"}`;
}

export function shortDate(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleDateString("en-IN", { day: "numeric", month: "short", year: "numeric" });
}

export function dayMonth(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleDateString("en-IN", { day: "numeric", month: "short" });
}

/** Model identifiers -> the words a non-specialist would use. */
export const MODEL_LABELS: Record<string, string> = {
  sarimax: "Seasonal ARIMA",
  holt_winters: "Holt-Winters",
  seasonal_naive: "Repeat-last-week",
  rule_baseline: "Simple rules",
  rules_baseline: "Simple rules",
  logistic_l2: "Logistic regression",
  gbm: "Gradient boosting",
  lp_optimizer: "Exact optimiser (LP)",
  dp_knapsack: "Exact optimiser (DP)",
  chance_constrained: "Cautious optimiser",
  pulp_cbc: "Exact optimiser (LP)",
  dp_bounded_knapsack: "Exact optimiser (DP)",
  rule_based: "Simple rules",
};

export const modelLabel = (k: string) => MODEL_LABELS[k] ?? k.replace(/_/g, " ");

/** Fixed series color per entity — never per rank, so filtering never repaints. */
export const STRATEGY_COLOR: Record<string, string> = {
  rules_baseline: "var(--series-1)",
  rule_based: "var(--series-1)",
  lp_optimizer: "var(--series-2)",
  pulp_cbc: "var(--series-2)",
  dp_knapsack: "var(--series-3)",
  dp_bounded_knapsack: "var(--series-3)",
  chance_constrained: "var(--series-4)",
};

export const MODEL_COLOR: Record<string, string> = {
  rule_baseline: "var(--series-1)",
  logistic_l2: "var(--series-2)",
  gbm: "var(--series-3)",
};

/**
 * Amber (--series-4) measured 2.21:1 against the chart surface, below the 3:1
 * floor. The validator's relief rule permits it only with a visible direct
 * label, so charts consult this to decide where a label is mandatory rather
 * than optional.
 */
export const NEEDS_DIRECT_LABEL = new Set(["var(--series-4)"]);
