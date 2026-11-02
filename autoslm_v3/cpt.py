"""Continued-pretraining (CPT) QLoRA stage with general replay.

Requires licensed domain and replay corpora. This experimental Trainer path
must be profiled on the target GPU; it is not a 4-GB fit guarantee.
"""
from __future__ import annotations
import hashlib, json, random, time
from pathlib import Path
from typing import Dict, Iterable, List
from .artifacts import write_adapter_manifest, verify_adapter_manifest
from .sft import SFTConfig, _CausalCollator, _require_ml_dependencies
from .lineage import LineageStore

def load_cpt_corpus(path: str, allowed_licenses: Iterable[str]) -> List[Dict[str,str]]:
    allowed={x.casefold() for x in allowed_licenses}; rows=[]; seen=set()
    with Path(path).open(encoding="utf-8") as stream:
        for n,line in enumerate(stream,1):
            if not line.strip(): continue
            row=json.loads(line); text=str(row.get("text","")).strip()
            license_id=str(row.get("license","")).casefold(); source=str(row.get("source","")).strip()
            if not text or not source or license_id not in allowed:
                raise ValueError(f"line {n}: text, source and explicitly allowed license are required")
            digest=hashlib.sha256(text.casefold().encode()).hexdigest()
            if digest in seen: continue
            seen.add(digest); rows.append({"text":text,"source":source,"license":license_id})
    if len(rows)<3: raise ValueError("CPT requires at least three unique eligible documents")
    return rows

def mix_domain_and_replay(domain, replay, replay_fraction: float=0.2, seed: int=42):
    if not 0<=replay_fraction<1: raise ValueError("replay_fraction must be in [0,1)")
    if replay_fraction and not replay: raise ValueError("general replay corpus is required")
    if not domain: raise ValueError("domain corpus cannot be empty")
    rng=random.Random(seed)
    target=round(len(domain)*replay_fraction/max(1e-9,1-replay_fraction)) if replay_fraction else 0
    mixed=[dict(x,corpus_role="domain") for x in domain]
    if target: mixed.extend(dict(x,corpus_role="general_replay") for x in (replay*(target//len(replay)+1))[:target])
    rng.shuffle(mixed); return mixed

def chunk_texts(rows, tokenizer, block_size: int=512):
    if block_size<32: raise ValueError("block_size must be at least 32")
    chunks=[]
    for row in rows:
        ids=tokenizer(str(row["text"]),add_special_tokens=True)["input_ids"]
        for start in range(0,len(ids)-block_size+1,block_size):
            chunk=ids[start:start+block_size]
            chunks.append({"input_ids":chunk,"attention_mask":[1]*len(chunk),"labels":chunk.copy()})
    if not chunks: raise ValueError("corpus produced no full token blocks")
    return chunks

def run_cpt(domain_path: str, replay_path: str, config: SFTConfig,
            allowed_licenses: Iterable[str], replay_fraction: float=0.2,
            block_size: int=256, run_id: str="cpt-experiment") -> Dict:
    torch, Dataset, LoraConfig, get_peft_model, prepare_model_for_kbit_training, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, TrainerCallback = _require_ml_dependencies()
    if not torch.cuda.is_available() or not config.use_4bit:
        raise RuntimeError("CPT runner currently requires CUDA QLoRA; CPU mode is not supported")
    domain=load_cpt_corpus(domain_path,allowed_licenses); replay=load_cpt_corpus(replay_path,allowed_licenses)
    mixed=mix_domain_and_replay(domain,replay,replay_fraction,config.seed)
    groups={}
    for row in mixed: groups.setdefault(row["source"],[]).append(row)
    keys=list(groups); random.Random(config.seed).shuffle(keys)
    if len(keys)<3: raise ValueError("CPT needs at least three distinct source documents after mixing")
    n_test=max(1,int(len(keys)*config.test_fraction)); n_val=max(1,int(len(keys)*config.validation_fraction))
    test_keys=set(keys[:n_test]); val_keys=set(keys[n_test:n_test+n_val]); train_keys=set(keys[n_test+n_val:])
    if not train_keys: raise ValueError("source split leaves no CPT training documents")
    train=[x for x in mixed if x["source"] in train_keys]; val=[x for x in mixed if x["source"] in val_keys]; test=[x for x in mixed if x["source"] in test_keys]
    tokenizer=AutoTokenizer.from_pretrained(config.model_name_or_path,trust_remote_code=config.trust_remote_code)
    if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
    q=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
    model=AutoModelForCausalLM.from_pretrained(config.model_name_or_path,quantization_config=q,device_map={"":0},torch_dtype=torch.float16,trust_remote_code=config.trust_remote_code)
    model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True); model.config.use_cache=False
    model=get_peft_model(model,LoraConfig(r=config.lora_rank,lora_alpha=config.lora_alpha,lora_dropout=config.lora_dropout,bias="none",task_type="CAUSAL_LM",target_modules=list(config.target_modules)))
    train_ds=Dataset.from_list(chunk_texts(train,tokenizer,block_size)); val_ds=Dataset.from_list(chunk_texts(val,tokenizer,block_size)); test_ds=Dataset.from_list(chunk_texts(test,tokenizer,block_size))
    out=Path(config.output_dir)/run_id; out.mkdir(parents=True,exist_ok=False)
    lineage=LineageStore(str(out/"lineage.sqlite3")); lineage.append(run_id,"cpt_started",{"domain_documents":len(domain),"replay_documents":len(replay),"replay_fraction":replay_fraction,"train_blocks":len(train_ds),"validation_blocks":len(val_ds),"test_blocks":len(test_ds)})
    args=TrainingArguments(output_dir=str(out/"checkpoints"),per_device_train_batch_size=1,per_device_eval_batch_size=1,gradient_accumulation_steps=config.gradient_accumulation_steps,learning_rate=config.learning_rate,num_train_epochs=config.num_train_epochs,logging_steps=config.logging_steps,evaluation_strategy="epoch",save_strategy="epoch",load_best_model_at_end=True,metric_for_best_model="eval_loss",greater_is_better=False,save_total_limit=2,fp16=True,gradient_checkpointing=True,optim="paged_adamw_8bit",report_to="none",remove_unused_columns=False,seed=config.seed)
    trainer=Trainer(model=model,args=args,train_dataset=train_ds,eval_dataset=val_ds,data_collator=_CausalCollator(tokenizer))
    started=time.time(); trained=trainer.train(); validation=trainer.evaluate(val_ds,metric_key_prefix="validation"); locked_test=trainer.evaluate(test_ds,metric_key_prefix="locked_test")
    adapter=out/"adapter"; trainer.model.save_pretrained(adapter); tokenizer.save_pretrained(adapter)
    artifact=write_adapter_manifest(str(adapter),{"run_id":run_id,"stage":"cpt","base_model":config.model_name_or_path}); integrity=verify_adapter_manifest(str(adapter))
    if not integrity["valid"]: raise RuntimeError("CPT adapter artifact integrity failure")
    result={"run_id":run_id,"status":"completed","stage":"cpt","train_metrics":trained.metrics,"validation_metrics":validation,"locked_test_metrics":locked_test,"gpu":torch.cuda.get_device_name(0),"peak_allocated_mb":round(torch.cuda.max_memory_allocated()/1024**2,2),"duration_seconds":round(time.time()-started,2),"adapter_path":str(adapter),"artifact_manifest":artifact,"artifact_verification":integrity,"replay_fraction":replay_fraction,"note":"CPT loss does not establish downstream task quality; run independent regression and benchmark suites."}
    (out/"result.json").write_text(json.dumps(result,indent=2,default=str),encoding="utf-8"); lineage.append(run_id,"cpt_completed",result)
    return result
