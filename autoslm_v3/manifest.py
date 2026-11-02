"""Immutable experiment metadata and content fingerprints."""
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build_experiment_manifest(
    experiment_id: str,
    base_model: str,
    stage: str,
    config: Dict[str, Any],
    dataset_fingerprints: Dict[str, str],
    seed: int,
) -> Dict[str, Any]:
    if not experiment_id or not base_model or not stage:
        raise ValueError("experiment_id, base_model and stage are required")
    return {
        "schema_version": "1.0",
        "experiment_id": experiment_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_model": base_model,
        "stage": stage,
        "config": config,
        "config_sha256": sha256_json(config),
        "dataset_fingerprints": dataset_fingerprints,
        "seed": seed,
        "code_revision": git_revision(),
        "runtime": {"python": platform.python_version(), "platform": platform.platform()},
    }
