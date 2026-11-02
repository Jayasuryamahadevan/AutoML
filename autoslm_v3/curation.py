"""Two-phase dataset curation inspired by transparent frontier-lab data flows.

Profiling is non-destructive. Applying filters requires an approval token bound
to the exact corpus fingerprint and policy fingerprint.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Optional

from autoslm_v3.lineage import _canonical


def fingerprint(rows: Iterable[Mapping[str, Any]]) -> str:
    return hashlib.sha256(_canonical([dict(r) for r in rows]).encode("utf-8")).hexdigest()


def _normalise(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _hash(text: str) -> str:
    return hashlib.sha256(text.casefold().encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CurationPolicy:
    min_chars: int = 20
    max_chars: int = 200_000
    require_source_id: bool = True
    require_license: bool = True
    reject_pii_markers: bool = True


@dataclass(frozen=True)
class CurationProfile:
    input_fingerprint: str
    row_count: int
    missing_text: int
    missing_source: int
    missing_license: int
    duplicate_content: int
    too_short: int
    too_long: int
    pii_marker_count: int
    policy: CurationPolicy

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class CurationResult:
    rows: list[dict[str, Any]]
    input_fingerprint: str
    output_fingerprint: str
    policy_fingerprint: str
    removed: dict[str, int]
    audit_passed: bool


def _policy_hash(policy: CurationPolicy) -> str:
    return hashlib.sha256(_canonical(asdict(policy)).encode("utf-8")).hexdigest()


def profile_corpus(rows: Iterable[Mapping[str, Any]], policy: Optional[CurationPolicy] = None) -> CurationProfile:
    data = [dict(r) for r in rows]
    policy = policy or CurationPolicy()
    seen = set()
    missing_text = missing_source = missing_license = duplicates = short = long = pii = 0
    for row in data:
        text = _normalise(row.get("text", row.get("content", "")))
        missing_text += int(not text)
        missing_source += int(not row.get("source_id"))
        missing_license += int(not row.get("license"))
        digest = _hash(text) if text else None
        if digest and digest in seen:
            duplicates += 1
        if digest:
            seen.add(digest)
        short += int(bool(text) and len(text) < policy.min_chars)
        long += int(len(text) > policy.max_chars)
        pii += int(bool(re.search(r"\b(?:\d{3}-\d{2}-\d{4}|\d{10})\b", text)))
    return CurationProfile(
        fingerprint(data), len(data), missing_text, missing_source, missing_license,
        duplicates, short, long, pii, policy,
    )


def approve_profile(profile: CurationProfile, policy: CurationPolicy) -> str:
    """Return a review token; production UI must only issue this after human approval."""
    payload = {
        "input_fingerprint": profile.input_fingerprint,
        "profile": profile.to_dict(),
        "policy_fingerprint": _policy_hash(policy),
        "approval": "human-reviewed",
    }
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def apply_curation(
    rows: Iterable[Mapping[str, Any]],
    profile: CurationProfile,
    policy: CurationPolicy,
    approval_token: str,
    evaluation_texts: Iterable[str] = (),
) -> CurationResult:
    data = [dict(r) for r in rows]
    if fingerprint(data) != profile.input_fingerprint:
        raise ValueError("input corpus changed after profiling; profile is stale")
    if profile.policy != policy or approval_token != approve_profile(profile, policy):
        raise PermissionError("policy/profile has not been approved for this exact corpus")
    eval_hashes = {_hash(_normalise(t)) for t in evaluation_texts}
    removed = {k: 0 for k in (
        "missing_text", "missing_source", "missing_license", "too_short",
        "too_long", "duplicate", "evaluation_overlap", "pii_marker",
    )}
    seen = set()
    output = []
    for original in data:
        row = dict(original)
        text = _normalise(row.get("text", row.get("content", "")))
        digest = _hash(text) if text else ""
        if not text:
            removed["missing_text"] += 1
        elif policy.require_source_id and not row.get("source_id"):
            removed["missing_source"] += 1
        elif policy.require_license and not row.get("license"):
            removed["missing_license"] += 1
        elif len(text) < policy.min_chars:
            removed["too_short"] += 1
        elif len(text) > policy.max_chars:
            removed["too_long"] += 1
        elif digest in seen:
            removed["duplicate"] += 1
        elif digest in eval_hashes:
            removed["evaluation_overlap"] += 1
        elif policy.reject_pii_markers and re.search(r"\b(?:\d{3}-\d{2}-\d{4}|\d{10})\b", text):
            removed["pii_marker"] += 1
        else:
            seen.add(digest)
            row["text"] = text
            row["content_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
            output.append(row)
    audit = (
        len(output) + sum(removed.values()) == len(data)
        and len({r["content_sha256"] for r in output}) == len(output)
        and not any(_hash(r["text"]) in eval_hashes for r in output)
    )
    if not audit:
        raise RuntimeError("curation audit failed; refusing to emit dataset")
    return CurationResult(
        output, profile.input_fingerprint, fingerprint(output),
        _policy_hash(policy), removed, True,
    )
