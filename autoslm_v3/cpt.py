"""Controlled continued-pretraining data preparation and objective helpers.

This is a data/objective foundation. It does not claim a validated CPT run on
4 GB VRAM; callers must provide licensed domain text and general replay text.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Any, Dict, Iterable, List

def load_cpt_corpus(path: str, allowed_licenses: Iterable[str]) -> List[Dict[str,str]]:
    allowed={x.casefold() for x in allowed_licenses}; rows=[]; seen=set()
    with Path(path).open(encoding="utf-8") as stream:
        for n,line in enumerate(stream,1):
            if not line.strip(): continue
            row=json.loads(line)
            text=str(row.get("text","")).strip()
            license_id=str(row.get("license","")).casefold()
            source=str(row.get("source","")).strip()
            if not text or not source or license_id not in allowed:
                raise ValueError(f"line {n}: text, source and explicitly allowed license are required")
            digest=hashlib.sha256(text.casefold().encode()).hexdigest()
            if digest in seen: continue
            seen.add(digest); rows.append({"text":text,"source":source,"license":license_id})
    if not rows: raise ValueError("no eligible corpus records")
    return rows

def mix_domain_and_replay(domain: List[Dict[str,str]], replay: List[Dict[str,str]],
                          replay_fraction: float=0.2, seed: int=42) -> List[Dict[str,str]]:
    if not 0.0 <= replay_fraction < 1.0: raise ValueError("replay_fraction must be in [0,1)")
    if replay_fraction and not replay: raise ValueError("replay corpus required when replay_fraction > 0")
    if not domain: raise ValueError("domain corpus cannot be empty")
    import random
    rng=random.Random(seed)
    target=max(1,round(len(domain)*replay_fraction/max(1e-9,1-replay_fraction))) if replay_fraction else 0
    selected=[dict(x,corpus_role="domain") for x in domain]
    if target:
        selected.extend(dict(x,corpus_role="general_replay") for x in (replay* (target//len(replay)+1))[:target])
    rng.shuffle(selected)
    return selected

def chunk_texts(rows: List[Dict[str,str]], tokenizer, block_size: int=512):
    if block_size<32: raise ValueError("block_size must be at least 32")
    chunks=[]
    for row in rows:
        ids=tokenizer(str(row["text"]),add_special_tokens=True)["input_ids"]
        for start in range(0,len(ids)-block_size+1,block_size):
            chunk=ids[start:start+block_size]
            chunks.append({"input_ids":chunk,"attention_mask":[1]*len(chunk),"labels":chunk.copy()})
    if not chunks: raise ValueError("corpus produced no full token blocks")
    return chunks
