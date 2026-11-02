"""Append-only, hash-chained experiment event log.

This is a local-first provenance primitive, not a substitute for a secured
central tracking service. SQLite transactions make each event durable; the
hash chain makes accidental edits/deletions detectable.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


class LineageStore:
    def __init__(self, path: str = "runs/autoslm_lineage.sqlite3"):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT UNIQUE NOT NULL,
                    run_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    payload_json TEXT NOT NULL,
                    previous_hash TEXT NOT NULL,
                    event_hash TEXT UNIQUE NOT NULL
                )
            """)
            db.execute("CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id, seq)")

    def _connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        return db

    def append(self, run_id: str, event_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not run_id or not event_type:
            raise ValueError("run_id and event_type are required")
        created_at = time.time()
        event_id = str(uuid.uuid4())
        payload_json = _canonical(payload)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            last = db.execute("SELECT event_hash FROM events ORDER BY seq DESC LIMIT 1").fetchone()
            previous_hash = last["event_hash"] if last else "GENESIS"
            body = {
                "event_id": event_id, "run_id": run_id, "event_type": event_type,
                "created_at": created_at, "payload_json": payload_json,
                "previous_hash": previous_hash,
            }
            event_hash = hashlib.sha256(_canonical(body).encode("utf-8")).hexdigest()
            db.execute(
                """INSERT INTO events
                   (event_id, run_id, event_type, created_at, payload_json, previous_hash, event_hash)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (event_id, run_id, event_type, created_at, payload_json, previous_hash, event_hash),
            )
        return {**body, "payload": json.loads(payload_json), "event_hash": event_hash}

    def list_events(self, run_id: Optional[str] = None) -> Iterable[Dict[str, Any]]:
        with self._connect() as db:
            if run_id:
                rows = db.execute("SELECT * FROM events WHERE run_id=? ORDER BY seq", (run_id,)).fetchall()
            else:
                rows = db.execute("SELECT * FROM events ORDER BY seq").fetchall()
        for row in rows:
            item = dict(row)
            item["payload"] = json.loads(item.pop("payload_json"))
            yield item

    def verify_chain(self) -> bool:
        previous = "GENESIS"
        for event in self.list_events():
            body = {
                "event_id": event["event_id"], "run_id": event["run_id"],
                "event_type": event["event_type"], "created_at": event["created_at"],
                "payload_json": _canonical(event["payload"]),
                "previous_hash": previous,
            }
            expected = hashlib.sha256(_canonical(body).encode("utf-8")).hexdigest()
            if event["previous_hash"] != previous or event["event_hash"] != expected:
                return False
            previous = event["event_hash"]
        return True
