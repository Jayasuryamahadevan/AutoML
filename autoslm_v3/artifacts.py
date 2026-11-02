"""Adapter artifact integrity, inventory and reload smoke-test helpers."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Any, Dict

def _sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""): digest.update(chunk)
    return digest.hexdigest()

def write_adapter_manifest(adapter_dir: str, metadata: Dict[str, Any] | None = None) -> Dict[str, Any]:
    root=Path(adapter_dir)
    if not root.is_dir(): raise FileNotFoundError(root)
    if any(p.is_symlink() for p in root.rglob("*")):
        raise ValueError("adapter artifact must not contain symlinks")
    files=sorted(p for p in root.rglob("*") if p.is_file() and p.name!="artifact_manifest.json")
    if not files: raise ValueError("adapter directory contains no files")
    manifest={"schema_version":"1.0","artifact_type":"peft_adapter",
        "files":{str(p.relative_to(root)):_sha256(p) for p in files},
        "metadata":metadata or {}}
    (root/"artifact_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True),encoding="utf-8")
    return manifest

def verify_adapter_manifest(adapter_dir: str) -> Dict[str, Any]:
    root=Path(adapter_dir); path=root/"artifact_manifest.json"
    if not path.is_file(): raise FileNotFoundError("artifact_manifest.json missing")
    manifest=json.loads(path.read_text(encoding="utf-8"))
    expected=manifest.get("files",{})
    if not isinstance(expected, dict):
        raise ValueError("adapter manifest files field must be an object")
    for relative in expected:
        candidate=Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError(f"unsafe adapter manifest path: {relative}")
    if any(p.is_symlink() for p in root.rglob("*")):
        raise ValueError("adapter artifact must not contain symlinks")
    actual={str(p.relative_to(root)):_sha256(p) for p in root.rglob("*")
            if p.is_file() and p.name!="artifact_manifest.json"}
    missing=sorted(set(expected)-set(actual)); extra=sorted(set(actual)-set(expected))
    changed=sorted(k for k in expected.keys() & actual.keys() if expected[k]!=actual[k])
    return {"valid":not(missing or extra or changed),"missing":missing,"extra":extra,"changed":changed}
