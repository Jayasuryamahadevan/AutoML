"""Independent benchmark contracts and auditable base/candidate comparison.

Benchmark files are JSONL and must be curated independently of training data.
This module deliberately requires an explicit provenance record per item.
"""
from __future__ import annotations
import hashlib, json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List

REQUIRED = ("id", "task", "prompt", "reference", "source", "license", "split")

@dataclass(frozen=True)
class BenchmarkReport:
    benchmark_id: str
    fingerprint: str
    sample_count: int
    exact_match: float
    token_f1: float
    per_task: Dict[str, Dict[str, float]]
    predictions_path: str

def load_locked_benchmark(path: str) -> List[Dict[str, Any]]:
    rows=[]
    with Path(path).open(encoding="utf-8") as stream:
        for n,line in enumerate(stream,1):
            if not line.strip(): continue
            try: row=json.loads(line)
            except json.JSONDecodeError as exc: raise ValueError(f"invalid benchmark JSON on line {n}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"benchmark line {n} must be a JSON object")
            missing=[key for key in REQUIRED if not isinstance(row.get(key), str) or not row[key].strip()]
            if missing: raise ValueError(f"benchmark line {n} missing: {', '.join(missing)}")
            if row["split"] != "test": raise ValueError(f"benchmark line {n} must be marked split=test")
            rows.append(row)
    if len(rows)<10: raise ValueError("locked benchmark requires at least 10 reviewed examples")
    ids=[str(x["id"]) for x in rows]
    if len(ids)!=len(set(ids)): raise ValueError("benchmark IDs must be unique")
    return rows

def _norm(text: str) -> str:
    return " ".join(str(text).casefold().split())

def _f1(pred: str, ref: str) -> float:
    from collections import Counter
    p,r=_norm(pred).split(),_norm(ref).split()
    if not p or not r: return float(p==r)
    overlap=sum((Counter(p)&Counter(r)).values())
    if not overlap: return 0.0
    precision,recall=overlap/len(p),overlap/len(r)
    return 2*precision*recall/(precision+recall)

def evaluate_callable(model_id: str, predictor: Callable[[str], str],
                      benchmark_path: str, output_path: str) -> BenchmarkReport:
    if Path(benchmark_path).resolve() == Path(output_path).resolve():
        raise ValueError("predictions output must not overwrite the locked benchmark")
    rows=load_locked_benchmark(benchmark_path)
    canonical="".join(json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False) for x in rows)
    fingerprint=hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    results=[]; tasks={}
    for row in rows:
        prediction=str(predictor(str(row["prompt"])))
        em=float(_norm(prediction)==_norm(str(row["reference"])))
        f1=_f1(prediction,str(row["reference"]))
        item={"id":row["id"],"task":row["task"],"prediction":prediction,
              "reference":row["reference"],"exact_match":em,"token_f1":f1}
        results.append(item)
        bucket=tasks.setdefault(str(row["task"]),{"count":0,"exact_match_sum":0.0,"token_f1_sum":0.0})
        bucket["count"]+=1; bucket["exact_match_sum"]+=em; bucket["token_f1_sum"]+=f1
    per_task={k:{"count":v["count"],"exact_match":v["exact_match_sum"]/v["count"],
                 "token_f1":v["token_f1_sum"]/v["count"]} for k,v in tasks.items()}
    Path(output_path).parent.mkdir(parents=True,exist_ok=True)
    Path(output_path).write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in results),encoding="utf-8")
    return BenchmarkReport(model_id,fingerprint,len(results),
        sum(x["exact_match"] for x in results)/len(results),
        sum(x["token_f1"] for x in results)/len(results),per_task,str(output_path))

def compare_reports(baseline: BenchmarkReport, candidate: BenchmarkReport) -> Dict[str, Any]:
    if baseline.fingerprint != candidate.fingerprint:
        raise ValueError("base and candidate must use the identical locked benchmark")
    if baseline.sample_count != candidate.sample_count:
        raise ValueError("base and candidate sample counts differ")
    if baseline.per_task.keys() != candidate.per_task.keys():
        raise ValueError("base and candidate task coverage differs")
    return {"benchmark_fingerprint":baseline.fingerprint,
            "baseline_model":baseline.benchmark_id,"candidate_model":candidate.benchmark_id,
            "exact_match_delta":candidate.exact_match-baseline.exact_match,
            "token_f1_delta":candidate.token_f1-baseline.token_f1,
            "per_task_delta":{task:{
                "exact_match":candidate.per_task[task]["exact_match"]-baseline.per_task[task]["exact_match"],
                "token_f1":candidate.per_task[task]["token_f1"]-baseline.per_task[task]["token_f1"]}
                for task in baseline.per_task.keys() & candidate.per_task.keys()}}
