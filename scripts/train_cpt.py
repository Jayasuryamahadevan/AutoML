"""CLI for AutoSLM v3 continued pretraining."""
from pathlib import Path
import argparse, json, sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from autoslm_v3.cpt import run_cpt
from autoslm_v3.sft import config_from_json

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--domain-corpus",required=True); p.add_argument("--replay-corpus",required=True)
    p.add_argument("--config",required=True); p.add_argument("--allowed-licenses",required=True)
    p.add_argument("--replay-fraction",type=float,default=.2); p.add_argument("--block-size",type=int,default=256)
    p.add_argument("--run-id",default="cpt-experiment")
    a=p.parse_args()
    result=run_cpt(a.domain_corpus,a.replay_corpus,config_from_json(a.config),
                   [x.strip() for x in a.allowed_licenses.split(",") if x.strip()],
                   a.replay_fraction,a.block_size,a.run_id)
    print(json.dumps(result,indent=2,default=str))
if __name__=="__main__": main()
