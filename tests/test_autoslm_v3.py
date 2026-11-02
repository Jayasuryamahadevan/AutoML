import pytest

from autoslm_v3.data import split_dataset
from autoslm_v3.evaluation import EvaluationSummary, compare_and_promote
from autoslm_v3.planning import PlannerPolicy, ProjectSpec, Stage, build_training_plan


def examples(n=30):
    return [{"instruction": f"question {i}", "response": f"answer {i}",
             "group_id": f"document-{i // 3}"} for i in range(n)]


def test_split_is_deterministic_and_disjoint():
    first = split_dataset(examples(), seed=7)
    second = split_dataset(examples(), seed=7)
    assert first.fingerprint == second.fingerprint
    sets = [{x["_autoslm_content_hash"] for x in p}
            for p in (first.train, first.validation, first.test)]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])


def test_duplicate_rows_are_removed():
    rows = examples()
    rows.append(dict(rows[0]))
    split = split_dataset(rows)
    assert split.duplicates_removed == 1


def test_repeated_prompt_with_different_answers_stays_in_one_partition():
    rows = examples()
    rows.extend([
        {"instruction": "same question", "response": "answer A"},
        {"instruction": "same question", "response": "answer B"},
    ])
    split = split_dataset(rows)
    owners = set()
    for name, part in (("train", split.train), ("validation", split.validation), ("test", split.test)):
        if any(row["instruction"] == "same question" for row in part):
            owners.add(name)
    assert len(owners) == 1


def test_group_ids_do_not_cross_partitions():
    split = split_dataset(examples())
    owners = {}
    for name, part in (("train", split.train), ("validation", split.validation), ("test", split.test)):
        for row in part:
            owners.setdefault(row["group_id"], set()).add(name)
    assert all(len(value) == 1 for value in owners.values())


def test_plan_skips_expensive_stages_without_evidence():
    plan = build_training_plan(ProjectSpec(task="domain assistant", supervised_examples=100))
    enabled = {d.stage for d in plan.stages if d.enabled}
    assert Stage.SFT in enabled
    assert Stage.PRETRAIN not in enabled
    assert Stage.MIDTRAIN not in enabled
    assert Stage.PREFERENCE not in enabled
    assert Stage.EVALUATE in enabled


def test_planner_thresholds_are_configurable():
    plan = build_training_plan(
        ProjectSpec(task="domain model", raw_domain_tokens=500, target_domain_shift="high"),
        PlannerPolicy(midtraining_min_raw_tokens=500),
    )
    assert next(d for d in plan.stages if d.stage == Stage.MIDTRAIN).enabled


def test_midtraining_requires_corpus_and_domain_shift():
    plan = build_training_plan(ProjectSpec(
        task="legal domain model", raw_domain_tokens=20_000_000,
        target_domain_shift="high", supervised_examples=100,
    ))
    assert next(d for d in plan.stages if d.stage == Stage.MIDTRAIN).enabled


def test_promotion_requires_improvement_and_safety():
    base = EvaluationSummary("base", {"accuracy": .70}, {"general": .8}, True, sample_count=100)
    good = EvaluationSummary("candidate", {"accuracy": .74}, {"general": .79}, True, sample_count=100)
    bad = EvaluationSummary("unsafe", {"accuracy": .80}, {"general": .79}, False, sample_count=100)
    assert compare_and_promote(base, good, "accuracy").promoted
    assert not compare_and_promote(base, bad, "accuracy").promoted


def test_small_evaluation_set_is_rejected():
    base = EvaluationSummary("base", {"accuracy": .7}, {}, True, sample_count=0)
    with pytest.raises(ValueError):
        compare_and_promote(base, base, "accuracy")


def test_non_normalized_metrics_are_rejected():
    invalid = EvaluationSummary("bad", {"accuracy": 87.0}, {}, True, sample_count=10)
    with pytest.raises(ValueError):
        compare_and_promote(invalid, invalid, "accuracy")
