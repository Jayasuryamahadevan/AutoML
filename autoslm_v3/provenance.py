"""Dataset provenance contracts and conservative training eligibility checks."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import hashlib
import json


@dataclass(frozen=True)
class Provenance:
    source_name: str
    source_uri: Optional[str]
    license_id: str
    acquisition_method: str
    acquired_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source_document_id: Optional[str] = None
    source_revision: Optional[str] = None
    transformation_chain: List[str] = field(default_factory=list)
    synthetic: bool = False
    teacher_model: Optional[str] = None
    generation_prompt_hash: Optional[str] = None
    consent_or_permission_ref: Optional[str] = None

    def validate(self) -> None:
        if not self.source_name.strip():
            raise ValueError("source_name is required")
        if not self.license_id.strip():
            raise ValueError("license_id is required; use 'unknown' to quarantine")
        if self.synthetic and not self.teacher_model:
            raise ValueError("synthetic records require teacher_model provenance")
        if self.synthetic and not self.generation_prompt_hash:
            raise ValueError("synthetic records require generation_prompt_hash")

    @property
    def fingerprint(self) -> str:
        raw = json.dumps(asdict(self), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def training_eligibility(provenance: Provenance, allowed_licenses: List[str]) -> Dict[str, Any]:
    """Fail closed: unknown/unapproved licenses are quarantined, not silently accepted."""
    provenance.validate()
    allowed = {item.casefold() for item in allowed_licenses}
    license_ok = provenance.license_id.casefold() in allowed
    permission_ok = provenance.consent_or_permission_ref is not None or provenance.license_id.casefold() in {
        "cc0-1.0", "public-domain", "mit", "apache-2.0", "bsd-3-clause", "bsd-2-clause"
    }
    return {
        "eligible": bool(license_ok and permission_ok),
        "license_approved": license_ok,
        "permission_evidence_present": permission_ok,
        "reason": "approved" if license_ok and permission_ok else "quarantine: license or permission evidence missing",
        "provenance_fingerprint": provenance.fingerprint,
    }
