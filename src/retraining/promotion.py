"""
Promotion safety gate for automated retraining.

Pure functions (no MLflow / filesystem access) so the gate logic is trivial to unit-test.
"""
from typing import Any, Dict, Optional

_EPS = 1e-9


def evaluate_promotion(
    old_metrics: Optional[Dict[str, Any]],
    new_metrics: Dict[str, Any],
    criteria: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Decides whether a freshly trained candidate may replace the current production model.

    Checks (thresholds come from ``retraining.promotion_criteria`` in config.yaml):
      1. Absolute recall floor            -> candidate recall >= min_recall_threshold
      2. Recall vs. current production    -> candidate recall - production recall >= min_recall_improvement_pct / 100
      3. Precision regression guard       -> production precision - candidate precision <= max_precision_drop_pct / 100

    Checks 2 and 3 are skipped when there is no previous production model to compare against.

    Returns:
        {"promote": bool, "status": "PROMOTED" | "REJECTED_INFERIOR", "reason": str, "checks": [...]}
    """
    min_recall = float(criteria.get("min_recall_threshold", 0.75))
    min_improvement = float(criteria.get("min_recall_improvement_pct", 0.0)) / 100.0
    max_precision_drop = float(criteria.get("max_precision_drop_pct", 5.0)) / 100.0

    new_recall = float(new_metrics.get("recall", 0.0))
    new_precision = float(new_metrics.get("precision", 0.0))

    checks = [{
        "name": "Minimum recall",
        "passed": new_recall + _EPS >= min_recall,
        "detail": f"candidate recall {new_recall:.4f} vs required floor {min_recall:.2f}",
    }]

    if old_metrics and "recall" in old_metrics:
        old_recall = float(old_metrics["recall"])
        delta = new_recall - old_recall
        checks.append({
            "name": "Recall vs production",
            "passed": delta + _EPS >= min_improvement,
            "detail": f"candidate {new_recall:.4f} vs production {old_recall:.4f} (change {delta:+.4f}, required >= {min_improvement:+.4f})",
        })
        if "precision" in old_metrics:
            old_precision = float(old_metrics["precision"])
            drop = old_precision - new_precision
            checks.append({
                "name": "Precision regression",
                "passed": drop <= max_precision_drop + _EPS,
                "detail": f"candidate {new_precision:.4f} vs production {old_precision:.4f} (drop {drop:+.4f}, allowed <= {max_precision_drop:.4f})",
            })

    failed = [c for c in checks if not c["passed"]]
    promote = not failed
    if promote:
        reason = f"Candidate passed all {len(checks)} promotion checks (recall {new_recall:.4f})."
    else:
        reason = "Candidate rejected: " + "; ".join(f"{c['name']} - {c['detail']}" for c in failed)

    return {
        "promote": promote,
        "status": "PROMOTED" if promote else "REJECTED_INFERIOR",
        "reason": reason,
        "checks": checks,
    }
