export const dash = "\u2013";

export function pct(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(value)) return dash;
  return `${(value * 100).toFixed(digits)}%`;
}
export function num(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) return dash;
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: 0 });
}
export function fixed(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(value)) return dash;
  return Number(value).toFixed(digits);
}

const toDate = (iso) => {
  if (!iso) return null;
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? null : d;
};

export function dateTime(iso) {
  const d = toDate(iso);
  if (!d) return dash;
  return d.toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}
export function timeOnly(iso) {
  const d = toDate(iso);
  return d ? d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit", second: "2-digit" }) : dash;
}
export function relTime(iso) {
  const d = toDate(iso);
  if (!d) return dash;
  const s = Math.round((Date.now() - d.getTime()) / 1000);
  if (s < 5) return "just now";
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  if (s < 86400 * 14) return `${Math.floor(s / 86400)} d ago`;
  return dateTime(iso);
}
export function duration(seconds) {
  if (seconds === null || seconds === undefined) return dash;
  if (seconds < 60) return `${seconds.toFixed(seconds < 10 ? 1 : 0)}s`;
  return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
}
export const shortId = (id) => (id ? String(id).slice(0, 8) : dash);
export const titleCase = (s) => String(s || "").replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());

export const METRIC_LABELS = {
  accuracy: "Accuracy", precision: "Precision", recall: "Recall", f1_score: "F1 score", roc_auc: "ROC-AUC", pr_auc: "PR-AUC",
};
export const ALGO_COLORS = {
  "Logistic Regression": "var(--s-lr)", "Random Forest": "var(--s-rf)", XGBoost: "var(--s-xgb)",
};
export const algoColor = (name) => ALGO_COLORS[name] || "var(--ink-3)";
