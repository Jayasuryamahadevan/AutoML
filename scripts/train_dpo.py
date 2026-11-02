"""CLI for AutoSLM v3 Direct Preference Optimization."""
from pathlib import Path
import argparse, json, sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from autoslm_v3.preference import run_dpo
from autoslm_v3.sft import config_from_json

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",required=True); p.add_argument("--config",required=True)
    p.add_argument("--allowed-licenses",required=True); p.add_argument("--beta",type=float,default=.1)
    p.add_argument("--run-id",default="dpo-experiment")
    a=p.parse_args()
    result=run_dpo(a.dataset,config_from_json(a.config),
                   [x.strip() for x in a.allowed_licenses.split(",") if x.strip()],a.beta,a.run_id)
    print(json.dumps(result,indent=2,default=str))
if __name__=="__main__": main()
