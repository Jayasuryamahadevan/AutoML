"""Local SQLite model registry with evidence-gated alias promotion."""
from __future__ import annotations
import hashlib,json,sqlite3,time,uuid
from pathlib import Path
from typing import Any,Dict,List
from .artifacts import verify_adapter_manifest

class ModelRegistry:
    def __init__(self,path:str="outputs/registry.sqlite3"):
        self.path=str(path); Path(self.path).parent.mkdir(parents=True,exist_ok=True)
        with self._connect() as db:
            db.executescript("""
            CREATE TABLE IF NOT EXISTS model_versions(
              version_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, base_model TEXT NOT NULL,
              stage TEXT NOT NULL, adapter_path TEXT NOT NULL, dataset_fingerprint TEXT NOT NULL,
              artifact_fingerprint TEXT NOT NULL, status TEXT NOT NULL,
              metrics_json TEXT NOT NULL, created_at REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS aliases(
              alias TEXT PRIMARY KEY, version_id TEXT NOT NULL REFERENCES model_versions(version_id),
              updated_at REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS audit_events(
              seq INTEGER PRIMARY KEY AUTOINCREMENT, event_type TEXT NOT NULL,
              version_id TEXT NOT NULL, payload_json TEXT NOT NULL, created_at REAL NOT NULL);
            """)
    def _connect(self):
        db=sqlite3.connect(self.path,timeout=30)
        db.row_factory=sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        return db
    def register(self,run_id:str,base_model:str,stage:str,adapter_path:str,
                 dataset_fingerprint:str,metrics:Dict[str,Any]|None=None)->Dict[str,Any]:
        if not all(str(x).strip() for x in (run_id,base_model,stage,dataset_fingerprint)):
            raise ValueError("run_id, base_model, stage and dataset_fingerprint are required")
        path=Path(adapter_path).resolve()
        verification=verify_adapter_manifest(str(path))
        if not verification["valid"]: raise ValueError(f"adapter integrity verification failed: {verification}")
        manifest=json.loads((path/"artifact_manifest.json").read_text(encoding="utf-8"))
        artifact_fingerprint=hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(",",":")).encode()).hexdigest()
        version_id=str(uuid.uuid4())
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("INSERT INTO model_versions VALUES(?,?,?,?,?,?,?,?,?,?)",
                (version_id,run_id,base_model,stage,str(path),dataset_fingerprint,
                 artifact_fingerprint,"candidate",json.dumps(metrics or {},sort_keys=True),time.time()))
            db.execute("INSERT INTO audit_events(event_type,version_id,payload_json,created_at) VALUES(?,?,?,?)",
                ("registered",version_id,json.dumps({"stage":stage,"base_model":base_model}),time.time()))
        return self.get(version_id)
    def promote(self,version_id:str,alias:str,evidence:Dict[str,Any])->Dict[str,Any]:
        if not alias.strip(): raise ValueError("alias is required")
        required=("benchmark_fingerprint","safety_passed","human_review_passed","task_regressions")
        missing=[k for k in required if k not in evidence]
        if missing: raise ValueError(f"promotion evidence missing: {', '.join(missing)}")
        if not str(evidence["benchmark_fingerprint"]).strip(): raise ValueError("benchmark_fingerprint is required")
        if evidence["safety_passed"] is not True: raise ValueError("promotion blocked: safety checks did not pass")
        if evidence["human_review_passed"] is not True: raise ValueError("promotion blocked: human review did not pass")
        if evidence["task_regressions"] != []: raise ValueError("promotion blocked: task regressions must be empty")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row=db.execute("SELECT status FROM model_versions WHERE version_id=?",(version_id,)).fetchone()
            if row is None: raise KeyError(version_id)
            if row["status"]=="rejected": raise ValueError("rejected versions cannot be promoted")
            db.execute("UPDATE model_versions SET status='promoted' WHERE version_id=?",(version_id,))
            db.execute("INSERT INTO aliases(alias,version_id,updated_at) VALUES(?,?,?) ON CONFLICT(alias) DO UPDATE SET version_id=excluded.version_id,updated_at=excluded.updated_at",(alias,version_id,time.time()))
            db.execute("INSERT INTO audit_events(event_type,version_id,payload_json,created_at) VALUES(?,?,?,?)",
                ("promoted",version_id,json.dumps({"alias":alias,"evidence":evidence},sort_keys=True),time.time()))
        return self.get(version_id)
    def reject(self,version_id:str,reason:str)->Dict[str,Any]:
        if not reason.strip(): raise ValueError("rejection reason is required")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row=db.execute("SELECT status FROM model_versions WHERE version_id=?",(version_id,)).fetchone()
            if row is None: raise KeyError(version_id)
            if db.execute("SELECT 1 FROM aliases WHERE version_id=?",(version_id,)).fetchone():
                raise ValueError("active aliased version cannot be rejected; move the alias first")
            db.execute("UPDATE model_versions SET status='rejected' WHERE version_id=?",(version_id,))
            db.execute("INSERT INTO audit_events(event_type,version_id,payload_json,created_at) VALUES(?,?,?,?)",
                ("rejected",version_id,json.dumps({"reason":reason}),time.time()))
        return self.get(version_id)
    def get(self,version_id:str)->Dict[str,Any]:
        with self._connect() as db:
            row=db.execute("SELECT * FROM model_versions WHERE version_id=?",(version_id,)).fetchone()
        if row is None: raise KeyError(version_id)
        result=dict(row); result["metrics"]=json.loads(result.pop("metrics_json")); return result
    def resolve(self,alias:str)->Dict[str,Any]:
        with self._connect() as db:
            row=db.execute("SELECT version_id FROM aliases WHERE alias=?",(alias,)).fetchone()
        if row is None: raise KeyError(alias)
        return self.get(row["version_id"])
    def list_versions(self)->List[Dict[str,Any]]:
        with self._connect() as db:
            ids=[r["version_id"] for r in db.execute("SELECT version_id FROM model_versions ORDER BY created_at DESC")]
        return [self.get(x) for x in ids]
