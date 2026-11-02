"""Preference-pair validation and QLoRA DPO training.

The custom Trainer disables the PEFT adapter to obtain a frozen reference policy,
avoiding a second 4-bit model copy. Treat this as an experimental single-process
implementation until verified against the pinned Transformers/PEFT runtime.
"""
from __future__ import annotations
import json, random, time
from pathlib import Path
from typing import Any, Dict, Iterable, List
from .sft import SFTConfig, _require_ml_dependencies
from .artifacts import write_adapter_manifest, verify_adapter_manifest
from .lineage import LineageStore

def validate_preference_pairs(rows: Iterable[Dict[str,Any]]) -> List[Dict[str,str]]:
    result=[]; seen=set()
    for n,row in enumerate(rows,1):
        prompt=str(row.get("prompt","")).strip(); chosen=str(row.get("chosen","")).strip(); rejected=str(row.get("rejected","")).strip()
        if not prompt or not chosen or not rejected: raise ValueError(f"preference row {n} requires prompt, chosen and rejected")
        if chosen==rejected: raise ValueError(f"preference row {n} has identical chosen/rejected")
        if not row.get("source") or not row.get("license"): raise ValueError(f"preference row {n} requires source and license provenance")
        key=(prompt,chosen,rejected)
        if key not in seen: seen.add(key); result.append({"prompt":prompt,"chosen":chosen,"rejected":rejected,"source":str(row["source"]),"license":str(row["license"])})
    if not result: raise ValueError("no valid preference pairs")
    return result

def dpo_loss(policy_chosen_logps,policy_rejected_logps,reference_chosen_logps,reference_rejected_logps,beta:float=0.1):
    if beta<=0: raise ValueError("beta must be positive")
    import torch.nn.functional as F
    return -F.logsigmoid(beta*((policy_chosen_logps-policy_rejected_logps)-(reference_chosen_logps-reference_rejected_logps))).mean()

def _sequence_logps(model,ids,mask,labels):
    logits=model(input_ids=ids,attention_mask=mask).logits[:,:-1,:]
    targets=labels[:,1:]; valid=targets.ne(-100)
    token_logps=logits.log_softmax(-1).gather(-1,targets.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    return (token_logps*valid).sum(-1)

def run_dpo(dataset_path:str,config:SFTConfig,allowed_licenses:Iterable[str],beta:float=0.1,run_id:str="dpo-experiment")->Dict:
    torch,Dataset,LoraConfig,get_peft_model,prepare_model_for_kbit_training,AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig,Trainer,TrainingArguments,TrainerCallback=_require_ml_dependencies()
    if not torch.cuda.is_available() or not config.use_4bit: raise RuntimeError("DPO runner currently requires CUDA QLoRA")
    if beta<=0: raise ValueError("beta must be positive")
    allowed={x.casefold() for x in allowed_licenses}; raw=[]
    with Path(dataset_path).open(encoding="utf-8") as stream:
        for n,line in enumerate(stream,1):
            if not line.strip(): continue
            row=json.loads(line)
            if str(row.get("license","")).casefold() not in allowed: raise ValueError(f"line {n}: unapproved preference license")
            raw.append(row)
    rows=validate_preference_pairs(raw)
    if len(rows)<10: raise ValueError("DPO requires at least 10 reviewed preference pairs")
    groups={}
    for row in rows: groups.setdefault(row["prompt"],[]).append(row)
    keys=list(groups)
    if len(keys)<3:
        raise ValueError("DPO requires at least three distinct prompts to create train, validation and test partitions")
    random.Random(config.seed).shuffle(keys)
    n_test=max(1,int(len(keys)*config.test_fraction)); n_val=max(1,int(len(keys)*config.validation_fraction))
    test_keys=set(keys[:n_test]); val_keys=set(keys[n_test:n_test+n_val]); train_keys=set(keys[n_test+n_val:])
    if not train_keys: raise ValueError("DPO prompt split leaves no training examples")
    tokenizer=AutoTokenizer.from_pretrained(config.model_name_or_path,trust_remote_code=config.trust_remote_code)
    if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
    def encode(part):
        output=[]
        for row in part:
            prompt_ids=tokenizer(row["prompt"],add_special_tokens=False)["input_ids"]; item={}
            for name,text in (("chosen",row["chosen"]),("rejected",row["rejected"])):
                answer=tokenizer(text+(tokenizer.eos_token or ""),add_special_tokens=False)["input_ids"][:config.max_seq_length]
                p=prompt_ids[-max(0,config.max_seq_length-len(answer)):] if len(answer)<config.max_seq_length else []
                ids=p+answer; if not answer: raise ValueError(f"preference row has empty tokenized {name} response")
                item[name+"_ids"]=ids; item[name+"_labels"]=[-100]*len(p)+answer
            output.append(item)
        return output
    def collate(items):
        batch={}
        for name in ("chosen","rejected"):
            maximum=max(len(x[name+"_ids"]) for x in items)
            batch[name+"_ids"]=torch.tensor([x[name+"_ids"]+[tokenizer.pad_token_id]*(maximum-len(x[name+"_ids"])) for x in items],dtype=torch.long)
            batch[name+"_mask"]=torch.tensor([[1]*len(x[name+"_ids"])+[0]*(maximum-len(x[name+"_ids"])) for x in items],dtype=torch.long)
            batch[name+"_labels"]=torch.tensor([x[name+"_labels"]+[-100]*(maximum-len(x[name+"_labels"])) for x in items],dtype=torch.long)
        return batch
    train=[x for k in train_keys for x in groups[k]]; val=[x for k in val_keys for x in groups[k]]; test=[x for k in test_keys for x in groups[k]]
    if not train or not val or not test:
        raise ValueError("DPO split must produce non-empty train, validation and test pairs")
    train_ds=Dataset.from_list(encode(train)); val_ds=Dataset.from_list(encode(val)); test_ds=Dataset.from_list(encode(test))
    q=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
    model=AutoModelForCausalLM.from_pretrained(config.model_name_or_path,quantization_config=q,device_map={"":0},torch_dtype=torch.float16,trust_remote_code=config.trust_remote_code)
    model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True); model.config.use_cache=False
    model=get_peft_model(model,LoraConfig(r=config.lora_rank,lora_alpha=config.lora_alpha,lora_dropout=config.lora_dropout,bias="none",task_type="CAUSAL_LM",target_modules=list(config.target_modules)))
    class DPOTrainer(Trainer):
        def compute_loss(self,model,inputs,return_outputs=False,**kwargs):
            chosen=_sequence_logps(model,inputs["chosen_ids"],inputs["chosen_mask"],inputs["chosen_labels"])
            rejected=_sequence_logps(model,inputs["rejected_ids"],inputs["rejected_mask"],inputs["rejected_labels"])
            with torch.no_grad():
                with model.disable_adapter():
                    ref_chosen=_sequence_logps(model,inputs["chosen_ids"],inputs["chosen_mask"],inputs["chosen_labels"])
                    ref_rejected=_sequence_logps(model,inputs["rejected_ids"],inputs["rejected_mask"],inputs["rejected_labels"])
            loss=dpo_loss(chosen,rejected,ref_chosen,ref_rejected,beta)
            if return_outputs:
                return loss, {"loss": loss.detach()}
            return loss
    out=Path(config.output_dir)/run_id; out.mkdir(parents=True,exist_ok=False)
    lineage=LineageStore(str(out/"lineage.sqlite3")); lineage.append(run_id,"dpo_started",{"pairs":len(rows),"beta":beta,"train":len(train),"validation":len(val),"test":len(test)})
    args=TrainingArguments(output_dir=str(out/"checkpoints"),per_device_train_batch_size=1,per_device_eval_batch_size=1,gradient_accumulation_steps=config.gradient_accumulation_steps,learning_rate=config.learning_rate,num_train_epochs=config.num_train_epochs,logging_steps=config.logging_steps,save_strategy="epoch",evaluation_strategy="epoch",save_total_limit=2,fp16=True,gradient_checkpointing=True,optim="paged_adamw_8bit",report_to="none",remove_unused_columns=False,seed=config.seed)
    trainer=DPOTrainer(model=model,args=args,train_dataset=train_ds,eval_dataset=val_ds,data_collator=collate)
    started=time.time(); trained=trainer.train(); validation=trainer.evaluate(val_ds,metric_key_prefix="validation"); locked_test=trainer.evaluate(test_ds,metric_key_prefix="locked_test")
    adapter=out/"adapter"; trainer.model.save_pretrained(adapter); tokenizer.save_pretrained(adapter)
    artifact=write_adapter_manifest(str(adapter),{"run_id":run_id,"stage":"dpo","base_model":config.model_name_or_path,"beta":beta}); integrity=verify_adapter_manifest(str(adapter))
    if not integrity["valid"]: raise RuntimeError("DPO adapter artifact integrity failure")
    result={"run_id":run_id,"status":"completed","stage":"dpo","train_metrics":trained.metrics,"validation_metrics":validation,"locked_test_metrics":locked_test,"peak_allocated_mb":round(torch.cuda.max_memory_allocated()/1024**2,2),"duration_seconds":round(time.time()-started,2),"adapter_path":str(adapter),"artifact_manifest":artifact,"artifact_verification":integrity,"note":"DPO eval_loss is not a quality score; run independent preference and task benchmarks."}
    (out/"result.json").write_text(json.dumps(result,indent=2,default=str),encoding="utf-8"); lineage.append(run_id,"dpo_completed",result)
    return result
