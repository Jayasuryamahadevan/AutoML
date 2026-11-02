"""Register, promote, reject and inspect local AutoSLM adapter versions."""
import argparse,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from autoslm_v3.registry import ModelRegistry

def main():
    p=argparse.ArgumentParser(); p.add_argument("--db",default="outputs/registry.sqlite3")
    sub=p.add_subparsers(dest="command",required=True)
    reg=sub.add_parser("register"); reg.add_argument("--run-id",required=True); reg.add_argument("--base-model",required=True); reg.add_argument("--stage",required=True); reg.add_argument("--adapter",required=True); reg.add_argument("--dataset-fingerprint",required=True); reg.add_argument("--metrics")
    prom=sub.add_parser("promote"); prom.add_argument("--version-id",required=True); prom.add_argument("--alias",required=True); prom.add_argument("--evidence",required=True,help="JSON file with benchmark_fingerprint, safety_passed, human_review_passed and task_regressions")
    rej=sub.add_parser("reject"); rej.add_argument("--version-id",required=True); rej.add_argument("--reason",required=True)
    sub.add_parser("list")
    resolve=sub.add_parser("resolve"); resolve.add_argument("--alias",required=True)
    a=p.parse_args(); registry=ModelRegistry(a.db)
    if a.command=="register":
        metrics=json.loads(Path(a.metrics).read_text()) if a.metrics else {}
        result=registry.register(a.run_id,a.base_model,a.stage,a.adapter,a.dataset_fingerprint,metrics)
    elif a.command=="promote":
        result=registry.promote(a.version_id,a.alias,json.loads(Path(a.evidence).read_text(encoding="utf-8")))
    elif a.command=="reject": result=registry.reject(a.version_id,a.reason)
    elif a.command=="list": result=registry.list_versions()
    else: result=registry.resolve(a.alias)
    print(json.dumps(result,indent=2,default=str))
if __name__=="__main__": main()
