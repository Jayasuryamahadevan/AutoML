"""Baseline-relative promotion gates; judge scores are never the sole gate."""
from dataclasses import dataclass
from math import isfinite
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


def _validate_summary(summary: EvaluationSummary) -> None:
    if summary.sample_count <= 0:
        raise ValueError("evaluation summaries need non-zero sample counts")
    for group_name, metrics in (
        ("task_metrics", summary.task_metrics),
        ("general_metrics", summary.general_metrics),
    ):
        for name, value in metrics.items():
            if not isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{group_name}.{name} must be a finite normalized score in [0, 1]"
                )
    if summary.latency_ms is not None and (
        not isfinite(summary.latency_ms) or summary.latency_ms <= 0
    ):
        raise ValueError("latency_ms must be positive and finite")
    if summary.peak_memory_mb is not None and (
        not isfinite(summary.peak_memory_mb) or summary.peak_memory_mb <= 0
    ):
        raise ValueError("peak_memory_mb must be positive and finite")


def compare_and_promote(
    baseline: EvaluationSummary,
    candidate: EvaluationSummary,
    primary_metric: str,
    minimum_improvement: float = 0.01,
    allowed_general_regression: float = 0.02,
) -> PromotionDecision:
    """Conservative gate. Metrics must be normalized higher-is-better scores."""
    _validate_summary(baseline)
    _validate_summary(candidate)
    if not 0.0 <= minimum_improvement <= 1.0:
        raise ValueError("minimum_improvement must be in [0, 1]")
    if not 0.0 <= allowed_general_regression <= 1.0:
        raise ValueError("allowed_general_regression must be in [0, 1]")
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
