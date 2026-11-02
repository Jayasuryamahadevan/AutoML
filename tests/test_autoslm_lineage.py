import tempfile
from pathlib import Path

import pytest

from autoslm_v3.lineage import LineageStore
from autoslm_v3.provenance import Provenance, training_eligibility
from autoslm_v3.synthetic import generate_synthetic


def test_lineage_hash_chain_and_run_filter():
    with tempfile.TemporaryDirectory() as tmp:
        store = LineageStore(str(Path(tmp) / "events.sqlite"))
        store.append("run-1", "run_started", {"seed": 42})
        store.append("run-1", "checkpoint_saved", {"step": 10})
        store.append("run-2", "run_started", {})
        assert len(list(store.list_events("run-1"))) == 2
        assert store.verify_chain()


def test_provenance_fails_closed_for_unknown_license():
    p = Provenance("dataset", None, "unknown", "manual_upload")
    result = training_eligibility(p, ["cc0-1.0"])
    assert not result["eligible"]


def test_synthetic_generation_tracks_teacher_and_filters_duplicates():
    seeds = [{"category": "math"}, {"category": "math"}]
    def generator(_):
        return {"instruction": "What is 2+2?", "response": "4", "category": "math"}
    result = generate_synthetic(seeds, generator, "teacher-test", "v1")
    assert result["stats"]["accepted"] == 1
    assert result["stats"]["rejected"] == 1
    assert result["accepted"][0]["provenance"]["teacher_model"] == "teacher-test"


def test_synthetic_requires_teacher_metadata():
    with pytest.raises(ValueError):
        Provenance("synth", None, "synthetic-internal", "teacher_generation", synthetic=True).validate()


def test_verifier_rejects_invalid_solution():
    result = generate_synthetic(
        [{"category": "code"}],
        lambda _: {"instruction": "Write code", "response": "bad"},
        "teacher-test", "v1",
        verifier=lambda _: {"verified": False},
    )
    assert result["stats"]["accepted"] == 0
    assert result["rejected"][0]["reason"] == "verification_failed"
