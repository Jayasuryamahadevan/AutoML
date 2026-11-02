"""Reproducible QLoRA SFT pipeline for low-VRAM CUDA GPUs.

The module keeps heavy ML dependencies lazy so data/config tests run on CPU-only
CI. It uses Transformers Trainer + PEFT directly to avoid TRL API churn.
"""
from __future__ import annotations

import argparse
import json
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .data import split_dataset
from .lineage import LineageStore
from .manifest import build_experiment_manifest


@dataclass(frozen=True)
class SFTConfig:
    model_name_or_path: str = "Qwen/Qwen2.5-0.5B-Instruct"
    output_dir: str = "outputs/autoslm_v3"
    max_seq_length: int = 512
    test_fraction: float = 0.1
    validation_fraction: float = 0.1
    seed: int = 42
    learning_rate: float = 2e-4
    num_train_epochs: float = 2.0
    max_steps: int = -1
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    logging_steps: int = 5
    eval_steps: int = 25
    save_steps: int = 25
    warmup_ratio: float = 0.03
    max_grad_norm: float = 0.3
    target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj")
    trust_remote_code: bool = False
    use_4bit: bool = True
    gradient_checkpointing: bool = True
    allow_cpu: bool = False
    max_new_tokens: int = 128
    max_eval_samples: int = 100

    def validate(self) -> None:
        if not self.model_name_or_path.strip():
            raise ValueError("model_name_or_path is required")
        if self.max_seq_length < 64:
            raise ValueError("max_seq_length must be at least 64")
        if not 0 < self.test_fraction < 1 or not 0 < self.validation_fraction < 1:
            raise ValueError("split fractions must be between zero and one")
        if self.test_fraction + self.validation_fraction >= 1:
            raise ValueError("test_fraction + validation_fraction must be below one")
        if self.per_device_train_batch_size != 1 and self.use_4bit:
            raise ValueError("4-bit low-VRAM preset requires per_device_train_batch_size=1")
        if self.gradient_accumulation_steps < 1 or self.lora_rank < 1:
            raise ValueError("gradient_accumulation_steps and lora_rank must be positive")
        if self.learning_rate <= 0 or self.num_train_epochs <= 0:
            raise ValueError("learning_rate and num_train_epochs must be positive")
        if self.eval_steps < 1 or self.save_steps < 1:
            raise ValueError("eval_steps and save_steps must be positive")
        if self.max_new_tokens < 1 or self.max_eval_samples < 1:
            raise ValueError("max_new_tokens and max_eval_samples must be positive")
        if not self.target_modules:
            raise ValueError("at least one LoRA target module is required")


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number} must contain a JSON object")
            prompt, answer = _extract_prompt_answer(row)
            if not prompt.strip() or not answer.strip():
                raise ValueError(f"line {line_number} has an empty prompt or answer")
            normalized = dict(row)
            normalized["instruction"] = prompt
            normalized["response"] = answer
            rows.append(normalized)
    if len(rows) < 3:
        raise ValueError("at least three examples are required before train/validation/test splitting")
    return rows


def _extract_prompt_answer(row: Dict[str, Any]):
    if isinstance(row.get("messages"), list):
        messages = row["messages"]
        assistant = [m for m in messages if isinstance(m, dict) and m.get("role") == "assistant"]
        non_assistant = [m for m in messages if isinstance(m, dict) and m.get("role") != "assistant"]
        if not assistant or not non_assistant:
            raise ValueError("messages format requires at least one user/system and assistant message")
        prompt = "\n".join(f"{m['role']}: {m.get('content', '')}" for m in non_assistant)
        answer = "\n".join(str(m.get("content", "")) for m in assistant)
        return prompt, answer
    prompt = row.get("instruction", row.get("prompt", ""))
    extra = row.get("input", "")
    answer = row.get("response", row.get("output", row.get("completion", "")))
    if extra:
        prompt = f"{prompt}\n\n{extra}"
    return str(prompt), str(answer)


def prepare_splits(rows: List[Dict[str, Any]], config: SFTConfig):
    config.validate()
    return split_dataset(
        rows, test_fraction=config.test_fraction,
        validation_fraction=config.validation_fraction, seed=config.seed,
    )


def _require_ml_dependencies():
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
            Trainer, TrainingArguments, TrainerCallback,
        )
    except ImportError as exc:
        raise RuntimeError(
            "SFT dependencies are missing. Install torch, transformers, datasets, "
            "accelerate, peft and bitsandbytes from the CUDA-compatible lock set."
        ) from exc
    return (torch, Dataset, LoraConfig, get_peft_model,
            prepare_model_for_kbit_training, AutoModelForCausalLM,
            AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, TrainerCallback)


def _format_prompt(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return f"### User:\n{prompt}\n\n### Assistant:\n"


def _tokenize_rows(rows, tokenizer, max_length):
    encoded = []
    for row in rows:
        prompt = _format_prompt(tokenizer, str(row["instruction"]))
        answer = str(row["response"])
        eos = tokenizer.eos_token or ""
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(answer + eos, add_special_tokens=False)["input_ids"]
        # Keep the response intact where possible; truncate prompt first.
        if len(answer_ids) >= max_length:
            answer_ids = answer_ids[:max_length]
            prompt_ids = []
        else:
            prompt_ids = prompt_ids[-(max_length - len(answer_ids)):]
        input_ids = prompt_ids + answer_ids
        labels = ([-100] * len(prompt_ids)) + answer_ids.copy()
        if not answer_ids:
            raise ValueError("tokenization produced an empty assistant response")
        encoded.append({"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels})
    return encoded


class _CausalCollator:
    def __init__(self, tokenizer, pad_to_multiple_of: int = 8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        import torch
        max_len = max(len(x["input_ids"]) for x in features)
        if self.pad_to_multiple_of:
            max_len = ((max_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of
        pad_id = self.tokenizer.pad_token_id
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for item in features:
            pad = max_len - len(item["input_ids"])
            batch["input_ids"].append(item["input_ids"] + [pad_id] * pad)
            batch["attention_mask"].append(item["attention_mask"] + [0] * pad)
            batch["labels"].append(item["labels"] + [-100] * pad)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}



def _normalized_answer(text: str) -> str:
    import re
    return re.sub(r"\s+", " ", text.casefold()).strip()


def _token_f1(prediction: str, reference: str) -> float:
    from collections import Counter
    pred = _normalized_answer(prediction).split()
    ref = _normalized_answer(reference).split()
    if not pred or not ref:
        return float(pred == ref)
    overlap = sum((Counter(pred) & Counter(ref)).values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(pred), overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def evaluate_generation(model, tokenizer, rows, max_input_length: int,
                        max_new_tokens: int, max_samples: int):
    """Deterministic greedy generation metrics on the locked holdout."""
    import torch
    model.eval()
    selected = rows[:max_samples]
    exact, f1, predictions = [], [], []
    for row in selected:
        prompt = _format_prompt(tokenizer, str(row["instruction"]))
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True,
                            max_length=max_input_length, add_special_tokens=False)
        device = next(model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            generated = model.generate(
                **encoded, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        continuation = generated[0, encoded["input_ids"].shape[1]:]
        prediction = tokenizer.decode(continuation, skip_special_tokens=True).strip()
        reference = str(row["response"]).strip()
        exact_score = float(_normalized_answer(prediction) == _normalized_answer(reference))
        f1_score = _token_f1(prediction, reference)
        exact.append(exact_score)
        f1.append(f1_score)
        predictions.append({
            "instruction": str(row["instruction"]),
            "reference": reference,
            "prediction": prediction,
            "exact_match": exact_score,
            "token_f1": f1_score,
        })
    return {
        "sample_count": len(selected),
        "exact_match": sum(exact) / max(1, len(exact)),
        "token_f1": sum(f1) / max(1, len(f1)),
        "metric_note": "Greedy-generation lexical metrics; not a semantic or safety certification.",
    }, predictions


def run_sft(dataset_path: str, config: SFTConfig, run_id: Optional[str] = None, resume_from_checkpoint: Optional[str] = None) -> Dict[str, Any]:
    """Train LoRA adapter, select by validation loss, then evaluate locked test once."""
    config.validate()
    torch, Dataset, LoraConfig, get_peft_model, prepare_model_for_kbit_training, \
        AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, TrainerCallback = _require_ml_dependencies()
    if not torch.cuda.is_available() and not config.allow_cpu:
        raise RuntimeError("CUDA is unavailable. Set allow_cpu only for smoke tests; RTX 3050 training requires CUDA.")
    if not torch.cuda.is_available() and config.use_4bit:
        raise RuntimeError("bitsandbytes 4-bit QLoRA requires a supported CUDA environment.")
    if torch.cuda.is_available():
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        if total_bytes < 3.5 * 1024**3:
            raise RuntimeError("Detected less than 3.5 GiB total VRAM; this preset requires a 4 GB-class GPU.")
    if resume_from_checkpoint and not run_id:
        raise ValueError("run_id is required when resuming an existing run")
    run_id = run_id or str(uuid.uuid4())
    output_dir = Path(config.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=bool(resume_from_checkpoint))
    lineage = LineageStore(str(output_dir / "lineage.sqlite3"))
    started = time.time()
    lineage.append(run_id, "run_started", {"dataset_path": str(Path(dataset_path).resolve()), "config": asdict(config)})

    try:
        rows = load_jsonl(dataset_path)
        split = prepare_splits(rows, config)
        split_dir = output_dir / "splits"
        split.write_jsonl(str(split_dir))
        lineage.append(run_id, "dataset_split", {
            "fingerprint": split.fingerprint, "train": len(split.train),
            "validation": len(split.validation), "test": len(split.test),
            "duplicates_removed": split.duplicates_removed,
        })

        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name_or_path, trust_remote_code=config.trust_remote_code
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        if config.use_4bit:
            quantization = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                config.model_name_or_path, quantization_config=quantization,
                device_map={"": 0}, torch_dtype=torch.float16,
                trust_remote_code=config.trust_remote_code,
            )
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=config.gradient_checkpointing
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                config.model_name_or_path, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=config.trust_remote_code,
            )
        model.config.use_cache = False
        lora = LoraConfig(
            r=config.lora_rank, lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout, bias="none",
            task_type="CAUSAL_LM", target_modules=list(config.target_modules),
        )
        model = get_peft_model(model, lora)
        model.print_trainable_parameters()

        train_rows = _tokenize_rows(split.train, tokenizer, config.max_seq_length)
        validation_rows = _tokenize_rows(split.validation, tokenizer, config.max_seq_length)
        test_rows = _tokenize_rows(split.test, tokenizer, config.max_seq_length)
        train_ds = Dataset.from_list(train_rows)
        validation_ds = Dataset.from_list(validation_rows)
        test_ds = Dataset.from_list(test_rows)
        manifest = build_experiment_manifest(
            run_id, config.model_name_or_path, "sft",
            asdict(config), {"all_examples": split.fingerprint}, config.seed,
        )
        manifest["split_sizes"] = {"train": len(train_rows), "validation": len(validation_rows), "test": len(test_rows)}
        manifest["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        manifest["torch_version"] = torch.__version__
        manifest["cuda_version"] = torch.version.cuda
        manifest["transformers_version"] = __import__("transformers").__version__
        manifest["peft_version"] = __import__("peft").__version__
        try:
            manifest["bitsandbytes_version"] = __import__("bitsandbytes").__version__
        except ImportError:
            manifest["bitsandbytes_version"] = None
        manifest["tokenizer_name"] = config.model_name_or_path
        manifest["tokenizer_vocab_size"] = len(tokenizer)
        (output_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        args = TrainingArguments(
            output_dir=str(output_dir / "checkpoints"),
            per_device_train_batch_size=config.per_device_train_batch_size,
            per_device_eval_batch_size=config.per_device_eval_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate, num_train_epochs=config.num_train_epochs,
            max_steps=config.max_steps, warmup_ratio=config.warmup_ratio,
            logging_steps=config.logging_steps, eval_steps=config.eval_steps,
            save_steps=config.save_steps, evaluation_strategy="epoch",
            save_strategy="epoch", load_best_model_at_end=True,
            metric_for_best_model="eval_loss", greater_is_better=False,
            save_total_limit=2, fp16=torch.cuda.is_available(), bf16=False,
            gradient_checkpointing=config.gradient_checkpointing,
            optim="paged_adamw_8bit" if config.use_4bit else "adamw_torch",
            max_grad_norm=config.max_grad_norm, report_to="none",
            remove_unused_columns=False, dataloader_pin_memory=torch.cuda.is_available(),
            group_by_length=False, seed=config.seed,
        )
        class _LineageCallback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                lineage.append(run_id, "training_log", {
                    "step": state.global_step, "epoch": state.epoch,
                    "metrics": logs or {},
                })
                return control

            def on_save(self, args, state, control, **kwargs):
                lineage.append(run_id, "checkpoint_saved", {
                    "step": state.global_step, "checkpoint_dir": args.output_dir,
                    "best_model_checkpoint": state.best_model_checkpoint,
                })
                return control

            def on_evaluate(self, args, state, control, metrics=None, **kwargs):
                lineage.append(run_id, "evaluation_completed", {
                    "step": state.global_step, "epoch": state.epoch,
                    "metrics": metrics or {},
                })
                return control

        trainer = Trainer(
            model=model, args=args, train_dataset=train_ds,
            eval_dataset=validation_ds, data_collator=_CausalCollator(tokenizer),
            callbacks=[_LineageCallback()],
        )
        lineage.append(run_id, "training_started", {"trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)})
        train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        validation_metrics = trainer.evaluate(eval_dataset=validation_ds, metric_key_prefix="validation")
        # The selected checkpoint is loaded by Trainer before this final test read.
        test_metrics = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="locked_test")
        generation_metrics, predictions = evaluate_generation(
            trainer.model, tokenizer, split.test, config.max_seq_length,
            config.max_new_tokens, config.max_eval_samples,
        )
        (output_dir / "locked_test_predictions.jsonl").write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\\n" for item in predictions),
            encoding="utf-8",
        )
        adapter_dir = output_dir / "adapter"
        trainer.model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        result = {
            "run_id": run_id, "status": "completed",
            "adapter_path": str(adapter_dir),
            "train_metrics": train_result.metrics,
            "validation_metrics": validation_metrics,
            "locked_test_metrics": test_metrics,
            "locked_test_generation_metrics": generation_metrics,
            "dataset_fingerprint": split.fingerprint,
            "duration_seconds": round(time.time() - started, 2),
            "manifest_path": str(output_dir / "experiment_manifest.json"),
        }
        (output_dir / "result.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        lineage.append(run_id, "run_completed", result)
        return result
    except Exception as exc:
        lineage.append(run_id, "run_failed", {"error_type": type(exc).__name__, "error": str(exc)})
        raise


def config_from_json(path: str) -> SFTConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if "target_modules" in raw:
        raw["target_modules"] = tuple(raw["target_modules"])
    config = SFTConfig(**raw)
    config.validate()
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description="AutoSLM v3 low-VRAM QLoRA SFT")
    parser.add_argument("--dataset", required=True, help="JSONL with instruction/response or prompt/completion")
    parser.add_argument("--config", required=True, help="JSON SFT config")
    parser.add_argument("--run-id")
    parser.add_argument("--resume-from-checkpoint")
    args = parser.parse_args(argv)
    result = run_sft(args.dataset, config_from_json(args.config), args.run_id, args.resume_from_checkpoint)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
