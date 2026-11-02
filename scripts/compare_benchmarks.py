"""Compare saved base and candidate reports from the same locked benchmark."""
import argparse,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from autoslm_v3.benchmark import BenchmarkReport,compare_reports

def load(path):
    data=json.loads(Path(path).read_text(encoding="utf-8"))
    return BenchmarkReport(**data)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--baseline",required=True)
    parser.add_argument("--candidate",required=True)
    parser.add_argument("--output",default="outputs/benchmark-comparison.json")
    args=parser.parse_args()
    report=compare_reports(load(args.baseline),load(args.candidate))
    path=Path(args.output); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))
if __name__=="__main__": main()
