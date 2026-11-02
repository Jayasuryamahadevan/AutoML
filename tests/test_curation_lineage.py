import json

import pytest

from autoslm_v3.curation import (
    CurationPolicy, apply_curation, approve_profile, profile_corpus,
)
from autoslm_v3.lineage import LineageStore


def rows():
    return [
        {"text": "A sufficiently long training document about algebra.", "source_id": "book-1", "license": "CC-BY"},
        {"text": "A sufficiently long training document about algebra.", "source_id": "book-2", "license": "CC-BY"},
        {"text": "A sufficiently long held out evaluation example.", "source_id": "eval", "license": "CC-BY"},
        {"text": "Tiny", "source_id": "book-3", "license": "CC-BY"},
        {"text": "A sufficiently long document without license metadata.", "source_id": "book-4"},
    ]


def test_profile_is_non_destructive_and_apply_requires_approval():
    data = rows()
    before = json.dumps(data, sort_keys=True)
    policy = CurationPolicy(min_chars=20)
    profile = profile_corpus(data, policy)
    assert json.dumps(data, sort_keys=True) == before
    with pytest.raises(PermissionError):
        apply_curation(data, profile, policy, "not-approved")


def test_apply_deduplicates_decontaminates_and_audits():
    data = rows()
    policy = CurationPolicy(min_chars=20)
    profile = profile_corpus(data, policy)
    result = apply_curation(
        data, profile, policy, approve_profile(profile, policy),
        evaluation_texts=["A sufficiently long held out evaluation example."],
    )
    assert result.audit_passed
    assert len(result.rows) == 1
    assert result.removed["duplicate"] == 1
    assert result.removed["evaluation_overlap"] == 1
    assert result.removed["too_short"] == 1
    assert result.removed["missing_license"] == 1


def test_stale_profile_is_rejected():
    data = rows()
    policy = CurationPolicy()
    profile = profile_corpus(data, policy)
    token = approve_profile(profile, policy)
    data[0]["text"] = "The corpus changed after profiling."
    with pytest.raises(ValueError, match="changed"):
        apply_curation(data, profile, policy, token)


def test_lineage_hash_chain_detects_tampering(tmp_path):
    store = LineageStore(str(tmp_path / "lineage.sqlite3"))
    store.append("run-1", "dataset_profiled", {"fingerprint": "abc"})
    store.append("run-1", "policy_approved", {"policy": "v1"})
    assert store.verify_chain()
    with store._connect() as db:
        db.execute("UPDATE events SET payload_json=? WHERE seq=1", ('{"fingerprint":"tampered"}',))
    assert not store.verify_chain()


def test_lineage_can_filter_events_by_run(tmp_path):
    store = LineageStore(str(tmp_path / "lineage.sqlite3"))
    store.append("run-1", "started", {})
    store.append("run-2", "started", {})
    assert len(list(store.list_events("run-1"))) == 1
    assert len(list(store.list_events())) == 2
    assert store.verify_chain()
