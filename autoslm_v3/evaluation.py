"""Baseline-relative promotion gates; judge scores are never the sole gate."""
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class EvaluationSummary:
    model_id: str
    task_metrics: Dict[str, float]
    general_metrics: Dict[str, float]
    safety_passed: bool
    latency_ms: Optional[float] = None
    peak_memory_mb: Optional[float] = None
    sample_count: int = 0


@dataclass(frozen=True)
class PromotionDecision:
    promoted: bool
    reasons: tuple


def compare_and_promote(
    baseline: EvaluationSummary,
    candidate: EvaluationSummary,
    primary_metric: str,
    minimum_improvement: float = 0.01,
    allowed_general_regression: float = 0.02,
) -> PromotionDecision:
    """Conservative gate. Metrics must be normalized higher-is-better scores."""
    if baseline.sample_count <= 0 or candidate.sample_count <= 0:
        raise ValueError("both baseline and candidate need non-zero evaluation samples")
    if primary_metric not in baseline.task_metrics or primary_metric not in candidate.task_metrics:
        raise ValueError("primary metric must exist for both baseline and candidate")
    reasons = []
    delta = candidate.task_metrics[primary_metric] - baseline.task_metrics[primary_metric]
    if delta < minimum_improvement:
        reasons.append(f"primary metric improvement {delta:.4f} is below required {minimum_improvement:.4f}")
    if not candidate.safety_passed:
        reasons.append("candidate failed the safety gate")
    for metric, base_value in baseline.general_metrics.items():
        if metric in candidate.general_metrics:
            regression = base_value - candidate.general_metrics[metric]
            if regression > allowed_general_regression:
                reasons.append(f"general capability regression on {metric}: {regression:.4f}")
    if (baseline.latency_ms is not None and candidate.latency_ms is not None
            and candidate.latency_ms > baseline.latency_ms * 1.5):
        reasons.append("candidate latency is more than 1.5x baseline")
    return PromotionDecision(not reasons, tuple(reasons))
