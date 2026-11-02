"""Create a local Ollama model from an existing GGUF artifact."""
import argparse,re,shutil,subprocess
from pathlib import Path

def main(argv=None):
    p=argparse.ArgumentParser(description="Import a local GGUF into Ollama")
    p.add_argument("--gguf",required=True); p.add_argument("--name",required=True)
    p.add_argument("--modelfile",default="outputs/ollama/Modelfile")
    p.add_argument("--context-length",type=int,default=2048)
    a=p.parse_args(argv)
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._:-]{0,63}",a.name):
        raise ValueError("invalid Ollama model name")
    if a.context_length<256: raise ValueError("--context-length must be at least 256")
    gguf=Path(a.gguf).resolve()
    if not gguf.is_file() or gguf.suffix.lower()!=".gguf": raise FileNotFoundError("a valid .gguf file is required")
    ollama=shutil.which("ollama")
    if not ollama: raise FileNotFoundError("ollama executable not found on PATH")
    modelfile=Path(a.modelfile).resolve(); modelfile.parent.mkdir(parents=True,exist_ok=True)
    modelfile.write_text(f"FROM {gguf}\nPARAMETER num_ctx {a.context_length}\n",encoding="utf-8")
    subprocess.run([ollama,"create",a.name,"-f",str(modelfile)],check=True)
    print(f"Local Ollama model created: {a.name}")
if __name__=="__main__": main()
