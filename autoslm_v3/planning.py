"""Deterministic, budget-aware training-stage planning."""
from dataclasses import asdict, dataclass
from enum import Enum
from typing import List, Optional


class Stage(str, Enum):
    PRETRAIN = "pretrain"
    MIDTRAIN = "midtrain"
    SFT = "sft"
    PREFERENCE = "preference_optimization"
    RL = "reinforcement_learning"
    EVALUATE = "evaluate"


@dataclass(frozen=True)
class ProjectSpec:
    task: str
    base_model: Optional[str] = None
    base_model_is_instruct: bool = True
    train_from_scratch: bool = False
    raw_domain_tokens: int = 0
    supervised_examples: int = 0
    preference_pairs: int = 0
    has_verifiable_reward: bool = False
    licensed_corpus: bool = False
    compute_budget_gpu_hours: float = 0.0
    target_domain_shift: str = "unknown"


@dataclass(frozen=True)
class StageDecision:
    stage: Stage
    enabled: bool
    rationale: str
    prerequisites: List[str]


@dataclass(frozen=True)
class TrainingPlan:
    task: str
    stages: List[StageDecision]
    warnings: List[str]

    def to_dict(self):
        return {
            "task": self.task,
            "stages": [
                {**asdict(item), "stage": item.stage.value}
                for item in self.stages
            ],
            "warnings": list(self.warnings),
        }


def build_training_plan(spec: ProjectSpec) -> TrainingPlan:
    """Choose stages from explicit evidence; never infer data or compute."""
    if not spec.task.strip():
        raise ValueError("task must be a non-empty description")
    if min(spec.raw_domain_tokens, spec.supervised_examples,
           spec.preference_pairs) < 0:
        raise ValueError("dataset sizes cannot be negative")
    if spec.compute_budget_gpu_hours < 0:
        raise ValueError("compute budget cannot be negative")

    stages: List[StageDecision] = []
    warnings: List[str] = []

    if spec.train_from_scratch:
        prereqs = ["licensed_corpus", "tokenizer_and_corpus_plan",
                   "compute_and_scaling_plan", "locked_evaluation_suite"]
        enabled = spec.licensed_corpus and spec.compute_budget_gpu_hours >= 100
        stages.append(StageDecision(
            Stage.PRETRAIN, enabled,
            "Scratch pretraining is explicitly requested and licensed data plus a minimally credible compute budget are present."
            if enabled else "Not enabled: from-scratch pretraining requires licensed data and a realistic compute plan.",
            prereqs,
        ))
        if not enabled:
            warnings.append("Do not launch scratch pretraining until corpus rights, tokenizer, token budget and compute estimates are validated.")

    midtrain_justified = (
        not spec.train_from_scratch
        and spec.raw_domain_tokens >= 10_000_000
        and spec.target_domain_shift in {"medium", "high"}
    )
    stages.append(StageDecision(
        Stage.MIDTRAIN, midtrain_justified,
        "Large raw-domain corpus and material domain shift justify a controlled continued-pretraining experiment."
        if midtrain_justified else "Skipped: insufficient raw-domain corpus or no demonstrated domain-distribution gap.",
        ["raw_text_corpus", "deduplication_and_provenance",
         "general_replay_mixture", "forgetting_regression_suite"],
    ))
    if spec.raw_domain_tokens and spec.raw_domain_tokens < 10_000_000:
        warnings.append("Raw domain text is below the conservative mid-training threshold; prefer retrieval or SFT until evidence supports CPT.")

    sft_enabled = spec.supervised_examples >= 32
    stages.append(StageDecision(
        Stage.SFT, sft_enabled,
        "Supervised examples are available for instruction/task adaptation."
        if sft_enabled else "Skipped: provide a reviewed supervised dataset before SFT.",
        ["leakage_safe_train_validation_test_split", "chat_template_validation"],
    ))
    if spec.supervised_examples and spec.supervised_examples < 32:
        warnings.append("Small SFT dataset: treat as an experiment; add human review and strong regularization.")

    preference_enabled = spec.preference_pairs >= 100
    stages.append(StageDecision(
        Stage.PREFERENCE, preference_enabled,
        "Preference pairs are available for preference optimization."
        if preference_enabled else "Skipped: no sufficiently sized preference dataset.",
        ["chosen_rejected_pairs", "preference_quality_audit", "reference_policy"],
    ))
    if spec.preference_pairs and spec.preference_pairs < 100:
        warnings.append("Preference dataset is small; do not assume DPO will improve quality.")

    stages.append(StageDecision(
        Stage.RL, spec.has_verifiable_reward,
        "A verifiable reward signal exists; a bounded RL experiment may be justified."
        if spec.has_verifiable_reward else "Skipped: no verifiable reward function; an LLM judge alone is insufficient.",
        ["reward_unit_tests", "reward_hacking_tests", "rollout_budget"],
    ))

    stages.append(StageDecision(
        Stage.EVALUATE, True,
        "Always evaluate base and candidate models on locked, independent tests.",
        ["base_model_baseline", "locked_test_set", "regression_and_safety_suite"],
    ))

    if not spec.base_model:
        warnings.append("No base model specified: model selection and compatibility checks remain upstream.")
    if not any(d.enabled for d in stages if d.stage != Stage.EVALUATE):
        warnings.append("No training stage is currently justified. Improve evidence or use the base model with retrieval.")

    return TrainingPlan(spec.task, stages, warnings)
