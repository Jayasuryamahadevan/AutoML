"""Merge a PEFT adapter and convert the resulting HF model to GGUF via llama.cpp."""
from __future__ import annotations
import argparse,shutil,subprocess,sys
from pathlib import Path

def main(argv=None):
    p=argparse.ArgumentParser(description="Merge AutoSLM adapter and export GGUF")
    p.add_argument("--base-model",required=True); p.add_argument("--adapter",required=True)
    p.add_argument("--output-dir",required=True); p.add_argument("--llama-cpp-dir",required=True)
    p.add_argument("--quantization",default="Q4_K_M",help="llama.cpp quantization type, e.g. Q4_K_M")
    p.add_argument("--skip-quantize",action="store_true"); p.add_argument("--minimum-free-gb",type=float,default=5.0)
    a=p.parse_args(argv)
    if a.minimum_free_gb<1: raise ValueError("--minimum-free-gb must be >= 1")
    adapter=Path(a.adapter).resolve(); out=Path(a.output_dir).resolve(); llama=Path(a.llama_cpp_dir).resolve()
    converter=llama/"convert_hf_to_gguf.py"
    if not converter.is_file(): raise FileNotFoundError(f"llama.cpp converter not found: {converter}")
    quantizer=shutil.which("llama-quantize")
    if not a.skip_quantize and not quantizer: raise FileNotFoundError("llama-quantize not found on PATH; build llama.cpp or pass --skip-quantize")
    out.mkdir(parents=True,exist_ok=True)
    free_gb=shutil.disk_usage(out).free/1024**3
    if free_gb<a.minimum_free_gb: raise RuntimeError(f"need at least {a.minimum_free_gb:.1f} GB free disk; found {free_gb:.1f} GB")
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from peft import PeftModel
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    tokenizer=AutoTokenizer.from_pretrained(a.base_model)
    model=AutoModelForCausalLM.from_pretrained(a.base_model,torch_dtype=dtype,device_map="auto" if torch.cuda.is_available() else None)
    model=PeftModel.from_pretrained(model,str(adapter))
    merged=model.merge_and_unload()
    hf_dir=out/"merged-hf"
    merged.save_pretrained(hf_dir,safe_serialization=True,max_shard_size="2GB")
    tokenizer.save_pretrained(hf_dir)
    f16=out/"model-f16.gguf"
    subprocess.run([sys.executable,str(converter),str(hf_dir),"--outfile",str(f16),"--outtype","f16"],check=True)
    final=f16
    if not a.skip_quantize:
        final=out/f"model-{a.quantization.lower()}.gguf"
        subprocess.run([quantizer,str(f16),str(final),a.quantization],check=True)
    print(f"GGUF export completed: {final}")
if __name__=="__main__": main()
