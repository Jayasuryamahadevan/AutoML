# AutoSLM: Frontier-lab operating model

## What the public evidence supports

This design synthesizes public workflows; it does not claim access to proprietary internal processes.

- NVIDIA Nemotron: explicit data recipes for ingestion, profiling, approved curation, audit, decontamination and token-budget subsets; separate data-production and model-family training recipes; lineage and versioned configs.
- AllenAI OLMo: publishes the full model flow—pretraining, mid-training, post-training data, training code, evaluation and decontamination.
- Liquid AI LFM: reports hardware-in-the-loop architecture search, distillation, vocabulary/context adaptation, staged SFT, on-policy distillation and agentic RL with separately operated training, rollout and sandbox components.
- Anthropic: publicly reports broad proprietary/public/synthetic data sources, filtering and deduplication, post-training, constitutional alignment and capability/safety evaluations. Internal datasets, recipes and many implementation details remain undisclosed.

## Operating system AutoSLM should implement

1. Research specification: target task distribution, capability gap, success metric, baseline, constraints, budget and falsifiable hypothesis.
2. Data acquisition: source registry, license/provenance, collection timestamp, source/document IDs, raw immutable snapshots and content hashes.
3. Profiling: measure distributions and quality signals without changing source data.
4. Policy approval: human approval bound to exact corpus and policy fingerprints.
5. Curation: normalization, language/length/quality filters, exact and fuzzy deduplication, PII handling, benchmark decontamination and audit.
6. Dataset construction: immutable versions, source/document-aware split before augmentation, training-only synthetic generation, data mixture manifests and token accounting.
7. Model design: choose architecture/tokenizer/context from measured constraints; architecture search is an experiment, not an LLM guess.
8. Pretraining: causal LM objective, distributed backend, checkpointing, token/compute accounting and scaling experiments.
9. Mid-training: controlled continued pretraining on domain data with general replay and forgetting tests.
10. Post-training: SFT; preference optimization only with valid preference labels; RL only with executable/verifiable rewards and reward-hacking tests; optional distillation.
11. Evaluation: base baseline, validation for selection, locked external test once per candidate, task metrics, safety, robustness, latency, memory, confidence intervals and human review.
12. Release: model card, dataset cards, exact lineage, artifact hashes, compatibility tests, canary deployment and rollback.

## Non-negotiable invariants

- No test-set examples in training, synthetic prompts, judge calibration or hyperparameter search.
- Every transformation creates a new dataset version and records parent fingerprints.
- Every model artifact records code commit, config, tokenizer/base-model revision, dataset fingerprints, seed, runtime versions and hardware.
- LLM judges are diagnostic unless calibrated against human labels; they are not sole release authorities.
- Every stage has explicit entry criteria, objective, exit criteria and rollback path.
- A pipeline may conclude that no training is justified. Retrieval, prompting or a smaller adaptation may be better.
- A run is not a deployment. Registry acceptance and production rollout are separate state transitions.

## Implemented in this tranche

- Two-phase corpus profiling and explicit approval.
- Exact duplicate removal and held-out evaluation-text decontamination.
- Per-row content hashes and dataset fingerprints.
- Post-curation audit that blocks output on invariant violations.
- SQLite append-only hash-chained event ledger.

## Remaining work

Fuzzy near-duplicate detection, language identification, licensed source connectors, source-level split orchestration, object-store immutability, external benchmark management, actual pretraining/CPT backends, preference/RL trainers, distributed execution, artifact registry and serving remain separate milestones.

The current synthetic-data workflow still creates examples before splitting. It must be changed so source documents/tasks are split first, and synthetic augmentation is restricted to the training partition. The locked benchmark should be independently sourced and human-reviewed.
