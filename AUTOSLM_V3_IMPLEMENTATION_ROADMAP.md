# AutoSLM v3 implementation roadmap

This roadmap is intentionally gated by measurable acceptance criteria. A stage is not complete because an agent or trainer class exists; it is complete when its data contract, objective, checkpoint semantics, evaluation and failure recovery are tested.

## Milestone 0 — Scientific foundation (this branch)

- [x] Separate v3 package without breaking v2 imports.
- [x] Explicit pretrain / midtrain / SFT / preference / RL planning decisions.
- [x] Deterministic group-aware train/validation/test split with content hashes.
- [x] Baseline-relative promotion gate.
- [x] Experiment manifest with config and dataset fingerprints.
- [x] Initial unit tests.
- [ ] CI result green on GitHub Actions.
- [x] Integrate group-aware train/validation/holdout partitioning into the legacy workflow; retries now use validation and the held-out set is evaluated once.
- [ ] Replace synthetic-only holdout with an externally curated, human-reviewed locked benchmark.
- [ ] Split source documents/tasks before synthetic augmentation when source corpora are available.
- [ ] Add source/license/provenance schema and persistent immutable dataset registry.

## Milestone 1 — Data engine

Deliverables:
- Source adapters for local files, Hugging Face datasets and approved APIs.
- Schema validation, language detection, PII policy hooks, license allowlist and provenance.
- Exact and near-duplicate detection; document/entity-level splitting.
- Quality scoring calibrated against human-reviewed examples.
- Synthetic generation only inside train partition; generated data must retain teacher/model/prompt provenance.
- Dataset cards, fingerprints and immutable versions.

Acceptance:
- No normalized exact duplicate crosses partitions.
- Source/document groups never cross partitions.
- Every training row traces to source or a declared synthetic-generation record.
- Test data is inaccessible to generation and tuning jobs by API contract.

## Milestone 2 — Baseline and model selection

Deliverables:
- Model compatibility catalogue (architecture, tokenizer, context length, license, chat template, supported target modules).
- Hardware profiler (VRAM, RAM, disk, compute capability).
- Candidate evaluation before training.
- Task-specific benchmark specification and cost estimator.

Acceptance:
- Planner rejects incompatible model/trainer combinations before downloading weights.
- Baseline report is stored before any training starts.
- Estimated VRAM and disk are checked against available resources.

## Milestone 3 — Post-training engine (first production training path)

Deliverables:
- SFT backend using current Transformers + TRL SFTTrainer / PEFT.
- Correct chat-template application and label masking.
- Gradient accumulation, precision selection, checkpoint/resume and early stopping.
- Adapter metadata and merge/export validation.
- Optional DPO backend only for audited preference pairs.

Acceptance:
- Smoke train on tiny public fixture dataset on CPU where supported or a small GPU fixture.
- Training can resume from a checkpoint.
- Adapter reload reproduces inference.
- Validation is used for selection; locked test is untouched until final evaluation.

## Milestone 4 — Mid-training engine

Deliverables:
- Causal-LM packing and tokenization pipeline.
- Domain/general replay mixture with documented sampling ratios.
- Continued-pretraining trainer backend.
- Perplexity and downstream evaluation before/after.
- Forgetting detection and stop/rollback policy.

Acceptance:
- Domain performance improves against base on locked domain tests.
- General capability regression remains within a predeclared tolerance.
- Token count, tokens/sec, GPU hours and data mixture are recorded.

## Milestone 5 — Pretraining research track

Deliverables:
- Tokenizer training and vocabulary analysis.
- Corpus governance, deduplication and data mixture design.
- Distributed training backend, optimizer/scheduler and checkpoint resumption.
- Scaling experiments and compute budget report.
- Full reproducibility bundle.

Acceptance:
- Small-scale reproduction of a training run from manifest.
- Scaling curves and held-out loss are reported.
- Legal provenance and compute feasibility are reviewed before launch.

## Milestone 6 — Post-training extensions

- DPO/KTO only with audited preference data.
- GRPO/RL only with verifiable rewards, unit tests and reward-hacking evaluations.
- Tool-call SFT and execution-grounded tool-use evaluation.
- Safety and refusal training with dedicated regression tests.

## Milestone 7 — Release and operations

- Model registry with immutable artifacts and model cards.
- Adapter/full-model/GGUF export as separate, validated formats.
- Inference serving, versioned endpoints, canary rollout and rollback.
- Monitoring for latency, throughput, error rate, drift and safety.
- Persistent job state, worker isolation and resource quotas.

## First end-to-end target

The first complete v3 release should support: select an instruct base model -> baseline evaluation -> leakage-safe dataset preparation -> QLoRA SFT -> validation-based checkpoint selection -> locked test comparison -> adapter export -> reproducible model card.

Do not implement pretraining first. A reliable SFT path and trustworthy evaluation are the highest-value foundation for the current project's scale and hardware.
