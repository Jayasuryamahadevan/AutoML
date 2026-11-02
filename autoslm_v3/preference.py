"""Preference-pair validation and numerically stable DPO objective.

The loss primitive is framework-independent. Full distributed/TRL training,
reference-policy caching and reward-model evaluation are separate runtime work.
"""
from __future__ import annotations
import math
from typing import Any, Dict, Iterable, List

def validate_preference_pairs(rows: Iterable[Dict[str,Any]]) -> List[Dict[str,str]]:
    result=[]; seen=set()
    for n,row in enumerate(rows,1):
        prompt=str(row.get("prompt","")).strip()
        chosen=str(row.get("chosen","")).strip()
        rejected=str(row.get("rejected","")).strip()
        if not prompt or not chosen or not rejected:
            raise ValueError(f"preference row {n} requires prompt, chosen and rejected")
        if chosen==rejected: raise ValueError(f"preference row {n} has identical chosen/rejected")
        if not row.get("source") or not row.get("license"):
            raise ValueError(f"preference row {n} requires source and license provenance")
        key=(prompt,chosen,rejected)
        if key in seen: continue
        seen.add(key); result.append({"prompt":prompt,"chosen":chosen,"rejected":rejected,
                                     "source":str(row["source"]),"license":str(row["license"])})
    if not result: raise ValueError("no valid preference pairs")
    return result

def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             reference_chosen_logps, reference_rejected_logps,
             beta: float=0.1):
    """Return per-example DPO logistic loss; inputs are torch tensors."""
    if beta<=0: raise ValueError("beta must be positive")
    import torch.nn.functional as F
    policy_ratio=policy_chosen_logps-policy_rejected_logps
    reference_ratio=reference_chosen_logps-reference_rejected_logps
    return -F.logsigmoid(beta*(policy_ratio-reference_ratio)).mean()

def preference_pair_margin(policy_chosen_logps: float, policy_rejected_logps: float) -> float:
    if not math.isfinite(policy_chosen_logps) or not math.isfinite(policy_rejected_logps):
        raise ValueError("log probabilities must be finite")
    return policy_chosen_logps-policy_rejected_logps
