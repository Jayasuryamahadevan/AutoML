# AutoSLM v3: remaining training and evaluation stages

This document describes the experimental code paths added after the first QLoRA SFT milestone. A path being implemented does not mean it has been validated on the RTX 3050. Run the stage-specific smoke test and record device telemetry before production claims.

## 1. SFT: base-versus-adapter and artifacts

SFT now:
- evaluates the unmodified base model on the same locked test examples before adapter training;
- trains using validation loss and restores the best validation checkpoint;
- evaluates the locked test after training;
- writes base and candidate greedy-generation predictions as valid JSONL;
- records peak CUDA allocated/reserved memory and total VRAM;
- writes an adapter file inventory with SHA-256 checksums and verifies it before marking the run complete.

The lexical exact-match/token-F1 delta is diagnostic only. It is not a semantic quality, factuality or safety score.

## 2. Independent benchmark protocol

Create a benchmark JSONL with at least 10 human-reviewed items, one per line:

```json
{"id":"qa-0001","task":"domain_qa","prompt":"Question","reference":"Expected answer","source":"human-review-2026-11","license":"internal-consent","split":"test"}
```

Required fields: `id`, `task`, `prompt`, `reference`, `source`, `license`, `split=test`. IDs must be unique. Keep this benchmark out of training and synthetic-data generation. Use the exact same file for base and adapter:

```bash
python scripts/evaluate_benchmark.py --benchmark eval/locked_test.jsonl --model Qwen/Qwen2.5-0.5B-Instruct --load-in-4bit --predictions outputs/base.jsonl --report outputs/base-report.json
python scripts/evaluate_benchmark.py --benchmark eval/locked_test.jsonl --model Qwen/Qwen2.5-0.5B-Instruct --adapter outputs/autoslm_v3/RUN_ID/adapter --load-in-4bit --predictions outputs/candidate.jsonl --report outputs/candidate-report.json
```

Compare reports only when their benchmark fingerprints match. The benchmark runner emits aggregate and per-task lexical metrics. Add task-specific executable checks (e.g. exact numeric tolerance, JSON schema validity, code tests) where applicable.

## 3. Continued pretraining (CPT)

Both domain and general replay corpora must be JSONL. Each row needs `text`, `source`, and `license`. Licenses must be explicitly allowed on the command line. The source field is the document grouping key.

```bash
python scripts/train_cpt.py --domain-corpus data/domain.jsonl --replay-corpus data/general-replay.jsonl --config configs/autoslm_v3_rtx3050.json --allowed-licenses CC-BY-4.0,CC0-1.0 --replay-fraction 0.2 --block-size 256 --run-id cpt-001
```

The CPT runner deduplicates, partitions by source, mixes general replay, trains a causal-LM objective, selects by validation loss and evaluates a held-out corpus partition. This is not a downstream forgetting evaluation. You must run the independent benchmark before and after CPT and reject a candidate that regresses general capability beyond the policy threshold. Corpus source/license declarations are user-supplied metadata, not legal advice or automated license verification.

## 4. DPO

Preference JSONL requires `prompt`, `chosen`, `rejected`, `source`, and `license`. The CLI rejects unapproved licenses and requires at least 10 pairs. Repeated prompts remain in one partition.

```bash
python scripts/train_dpo.py --dataset data/preferences.jsonl --config configs/autoslm_v3_rtx3050.json --allowed-licenses internal-consent --beta 0.1 --run-id dpo-001
```

The experimental trainer uses the PEFT adapter as the policy and disables it to calculate reference-policy log probabilities, avoiding a second model copy. This is a memory-saving single-process design; it still requires runtime validation for the exact Transformers/PEFT/quantization versions. DPO training loss is not a measure of preference win rate. Evaluate with a separate pairwise human-preference set and independent task benchmark.

## 5. Artifact verification

`autoslm_v3.artifacts.verify_adapter_manifest(path)` checks missing, extra and modified adapter files. To validate actual loading, run the independent benchmark script with `--adapter`; this loads the base model and adapter through PEFT.

## 6. What remains unverified

- No actual RTX 3050 run has been performed by CI. CUDA compatibility, peak VRAM, throughput and OOM behavior remain unverified.
- CPT and DPO are experimental single-GPU training paths, not distributed production training.
- The SFT split is made after the assembled examples are loaded. For strong generalization claims, split raw source documents before synthetic augmentation and keep a human-curated locked benchmark independent.
- Benchmark exact-match/token-F1 are lexical proxies. Implement task-specific validators, factuality checks, safety evaluations and human preference reviews.
- No reward-model training, GRPO/PPO, scratch pretraining, model registry, signed artifact publishing, GGUF conversion or Ollama deployment is included.
- Dataset provenance fields do not prove that a source license is valid; retain original permission evidence and audit it.

## 7. Suggested verification order

1. Run `python -m pytest tests -q` on CPU.
2. Install a PyTorch CUDA build matching the NVIDIA driver; verify `torch.cuda.is_available()` and `bitsandbytes` import.
3. Run a tiny SFT job on the RTX 3050 and record result.json, peak VRAM and throughput.
4. Reload the adapter through `scripts/evaluate_benchmark.py`.
5. Compare base/candidate on the independent locked benchmark.
6. Only then run CPT or DPO. Start with a small dataset and compare against the same baseline benchmark.


## Compare two saved benchmark reports

After evaluating the base and adapter with the exact same locked benchmark:

```bash
python scripts/compare_benchmarks.py --baseline outputs/base-report.json --candidate outputs/candidate-report.json --output outputs/comparison.json
```

The command rejects reports with different benchmark fingerprints or task coverage. The resulting metric deltas remain lexical diagnostics; promotion still requires task-specific tests, safety checks and human review.


## Local model registry and promotion gate

Register an adapter only after its artifact manifest verifies:

```bash
python scripts/model_registry.py register --run-id RUN_ID --base-model Qwen/Qwen2.5-0.5B-Instruct --stage sft --adapter outputs/autoslm_v3/RUN_ID/adapter --dataset-fingerprint DATASET_FINGERPRINT
```

The command returns a `version_id`. Promotion requires a JSON evidence file with all fields below:

```json
{"benchmark_fingerprint":"sha256-of-locked-benchmark","safety_passed":true,"human_review_passed":true,"task_regressions":[]}
```

```bash
python scripts/model_registry.py promote --version-id VERSION_ID --alias production --evidence eval/promotion-evidence.json
python scripts/model_registry.py resolve --alias production
python scripts/model_registry.py list
```

Promotion is intentionally fail-closed when safety or human review is absent or task regressions exist. The registry stores the evidence submitted by the operator; it does not independently prove that a reviewer performed the review or that the benchmark is valid. Protect the SQLite database and evidence files accordingly.


## GGUF export and local Ollama import

Install/build llama.cpp separately and make sure its Python converter and `llama-quantize` binary exist. The export explicitly merges the adapter into the base model in full precision, then converts and quantizes. This needs substantial temporary disk space and system RAM; it is not a 4-GB-VRAM guarantee.

```bash
python scripts/export_gguf.py --base-model Qwen/Qwen2.5-0.5B-Instruct --adapter outputs/autoslm_v3/RUN_ID/adapter --output-dir outputs/gguf/RUN_ID --llama-cpp-dir ~/llama.cpp --quantization Q4_K_M
```

For an already-created GGUF file, import it into the local Ollama installation:

```bash
python scripts/deploy_ollama.py --gguf outputs/gguf/RUN_ID/model-q4_k_m.gguf --name autoslm-domain:latest --context-length 2048
ollama run autoslm-domain:latest
```

These commands are opt-in and local-only. They were not executed in CPU CI. Verify the exported model against the same locked benchmark after import; successful conversion/import is not a quality or safety result.
