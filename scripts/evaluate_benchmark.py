"""Evaluate a base model or PEFT adapter on an independent locked benchmark."""
from pathlib import Path
import argparse, json, sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from autoslm_v3.benchmark import evaluate_callable

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--benchmark",required=True); p.add_argument("--model",required=True)
    p.add_argument("--adapter"); p.add_argument("--predictions",required=True)
    p.add_argument("--report",required=True); p.add_argument("--max-new-tokens",type=int,default=128)
    p.add_argument("--load-in-4bit",action="store_true")
    a=p.parse_args()
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
    base=a.model
    tok=AutoTokenizer.from_pretrained(base)
    if a.load_in_4bit:
        q=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
        model=AutoModelForCausalLM.from_pretrained(base,quantization_config=q,device_map="auto")
    else:
        model=AutoModelForCausalLM.from_pretrained(base,torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,device_map="auto")
    if a.adapter:
        from peft import PeftModel
        model=PeftModel.from_pretrained(model,a.adapter)
    model.eval()
    def predict(prompt):
        messages=[{"role":"user","content":prompt}]
        text=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True) if tok.chat_template else prompt
        inputs=tok(text,return_tensors="pt",truncation=True,max_length=512)
        device=next(model.parameters()).device
        inputs={k:v.to(device) for k,v in inputs.items()}
        with torch.inference_mode():
            ids=model.generate(**inputs,max_new_tokens=a.max_new_tokens,do_sample=False,
                               pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
        return tok.decode(ids[0,inputs["input_ids"].shape[1]:],skip_special_tokens=True).strip()
    report=evaluate_callable(a.adapter or base,predict,a.benchmark,a.predictions)
    Path(a.report).parent.mkdir(parents=True,exist_ok=True)
    Path(a.report).write_text(json.dumps(report.__dict__,indent=2),encoding="utf-8")
    print(json.dumps(report.__dict__,indent=2))
if __name__=="__main__": main()
