"""Auditable synthetic data generation with verification and deduplication gates.

The generator is injected by the caller (local model, approved API, or test
stub). This module deliberately does not hide teacher calls or costs.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional

from .provenance import Provenance


@dataclass
class SyntheticRecord:
    instruction: str
    response: str
    category: str
    provenance: Dict[str, Any]
    quality_score: Optional[float] = None
    verification: Optional[Dict[str, Any]] = None

    @property
    def fingerprint(self) -> str:
        canonical = json.dumps(
            [self.instruction.strip().casefold(), self.response.strip().casefold()],
            ensure_ascii=False, separators=(",", ":")
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _valid_shape(item: Dict[str, Any]) -> bool:
    return bool(str(item.get("instruction", "")).strip() and str(item.get("response", "")).strip())


def _repetition_ratio(text: str) -> float:
    words = re.findall(r"\w+", text.casefold())
    if len(words) < 8:
        return 0.0
    trigrams = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
    return 1.0 - (len(set(trigrams)) / max(1, len(trigrams)))


def generate_synthetic(
    seeds: Iterable[Dict[str, Any]],
    generator: Callable[[Dict[str, Any]], Dict[str, Any]],
    teacher_model: str,
    prompt_template_version: str,
    verifier: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    quality_scorer: Optional[Callable[[Dict[str, Any]], float]] = None,
    minimum_quality: float = 0.7,
    max_repetition_ratio: float = 0.25,
) -> Dict[str, Any]:
    """Generate one candidate per seed; reject malformed, repetitive, failed or duplicate samples."""
    accepted: List[SyntheticRecord] = []
    rejected: List[Dict[str, Any]] = []
    seen = set()
    for index, seed in enumerate(seeds):
        prompt = {
            "template_version": prompt_template_version,
            "seed": seed,
            "instruction": "Generate one diverse, correct training example grounded in the supplied seed.",
        }
        prompt_hash = hashlib.sha256(
            json.dumps(prompt, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
        ).hexdigest()
        try:
            candidate = generator(prompt)
            if not isinstance(candidate, dict) or not _valid_shape(candidate):
                raise ValueError("malformed_sample")
            response = str(candidate["response"])
            if _repetition_ratio(response) > max_repetition_ratio:
                raise ValueError("pathological_repetition")
            verification = verifier(candidate) if verifier else {"verified": None, "method": "not_configured"}
            if verifier and not verification.get("verified", False):
                raise ValueError("verification_failed")
            score = quality_scorer(candidate) if quality_scorer else None
            if score is not None and score < minimum_quality:
                raise ValueError("quality_below_threshold")
            record = SyntheticRecord(
                instruction=str(candidate["instruction"]).strip(),
                response=response.strip(),
                category=str(candidate.get("category", seed.get("category", "general"))),
                provenance=asdict(Provenance(
                    source_name="autoslm_synthetic_generation",
                    source_uri=None,
                    license_id="synthetic-internal",
                    acquisition_method="teacher_generation",
                    source_document_id=str(seed.get("source_document_id")) if seed.get("source_document_id") else None,
                    transformation_chain=["seed_conditioning", "teacher_generation", "structural_filter"],
                    synthetic=True,
                    teacher_model=teacher_model,
                    generation_prompt_hash=prompt_hash,
                )),
                quality_score=score,
                verification=verification,
            )
            if record.fingerprint in seen:
                raise ValueError("duplicate_sample")
            seen.add(record.fingerprint)
            accepted.append(record)
        except Exception as exc:
            rejected.append({"seed_index": index, "reason": str(exc), "prompt_hash": prompt_hash})
    return {
        "accepted": [asdict(item) for item in accepted],
        "rejected": rejected,
        "stats": {
            "seeds": len(accepted) + len(rejected),
            "accepted": len(accepted),
            "rejected": len(rejected),
            "acceptance_rate": len(accepted) / max(1, len(accepted) + len(rejected)),
        },
    }
