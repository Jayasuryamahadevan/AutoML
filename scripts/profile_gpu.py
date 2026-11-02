"""Preflight and optionally smoke-test CUDA + bitsandbytes model loading.

The default mode is read-only and does not download models. Pass --load-model
explicitly to download/load a small model and execute a single inference pass.
"""
from __future__ import annotations
import argparse, json, platform, time
from pathlib import Path
import sys

def main(argv=None):
    parser=argparse.ArgumentParser(description="AutoSLM v3 GPU preflight")
    parser.add_argument("--load-model",action="store_true",help="explicitly load the selected model in NF4 and run one generation")
    parser.add_argument("--model",default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--max-new-tokens",type=int,default=8)
    parser.add_argument("--output",default="outputs/gpu_preflight.json")
    args=parser.parse_args(argv)
    import torch
    report={"python":platform.python_version(),"platform":platform.platform(),
            "torch":torch.__version__,"cuda_runtime":torch.version.cuda,
            "cuda_available":torch.cuda.is_available(),"model":args.model,
            "model_load_attempted":False}
    if not torch.cuda.is_available():
        report["status"]="cuda_unavailable"
        print(json.dumps(report,indent=2))
        return 2
    report["gpu_name"]=torch.cuda.get_device_name(0)
    props=torch.cuda.get_device_properties(0)
    report["total_vram_mb"]=round(props.total_memory/1024**2,2)
    free,total=torch.cuda.mem_get_info(0)
    report["free_vram_mb"]=round(free/1024**2,2)
    report["compute_capability"]=f"{props.major}.{props.minor}"
    try:
        import bitsandbytes as bnb
        report["bitsandbytes"]=getattr(bnb,"__version__","unknown")
        report["bitsandbytes_import"]="ok"
    except Exception as exc:
        report["bitsandbytes_import"]="failed"
        report["bitsandbytes_error"]=f"{type(exc).__name__}: {exc}"
    try:
        import transformers
        report["transformers"]=transformers.__version__
    except Exception as exc:
        report["transformers_error"]=f"{type(exc).__name__}: {exc}"
    if args.load_model:
        if args.max_new_tokens<1 or args.max_new_tokens>64:
            raise ValueError("--max-new-tokens must be between 1 and 64 for preflight")
        if report.get("bitsandbytes_import")!="ok":
            report["status"]="bitsandbytes_unavailable"
        else:
            from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
            report["model_load_attempted"]=True
            torch.cuda.reset_peak_memory_stats()
            started=time.time()
            quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
            tokenizer=AutoTokenizer.from_pretrained(args.model)
            model=AutoModelForCausalLM.from_pretrained(args.model,quantization_config=quant,
                device_map={"":0},torch_dtype=torch.float16)
            tokens=tokenizer("Reply with the word READY.",return_tensors="pt")
            tokens={k:v.to("cuda") for k,v in tokens.items()}
            with torch.inference_mode():
                model.generate(**tokens,max_new_tokens=args.max_new_tokens,do_sample=False,
                               pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
            report["load_and_generate_seconds"]=round(time.time()-started,2)
            report["peak_allocated_mb"]=round(torch.cuda.max_memory_allocated()/1024**2,2)
            report["peak_reserved_mb"]=round(torch.cuda.max_memory_reserved()/1024**2,2)
            report["status"]="model_load_and_generation_passed"
            del model,tokenizer
            torch.cuda.empty_cache()
    else:
        report["status"]="cuda_and_package_preflight_completed"
    path=Path(args.output); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))
    return 0 if report["status"] in ("cuda_and_package_preflight_completed","model_load_and_generation_passed") else 2

if __name__=="__main__":
    raise SystemExit(main())
