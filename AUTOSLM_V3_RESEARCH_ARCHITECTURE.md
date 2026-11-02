# AutoSLM v3: Research-grade architecture

## Product thesis

AutoSLM is an experiment-driven system that chooses and executes the least expensive valid training recipe that measurably improves a base model on a predeclared task distribution. It is not an agent swarm that assumes every model needs every training stage.

## Scientific principles

1. Hypothesis before training: record capability gap, target distribution, baseline, success criteria, compute budget and stage rationale before GPU time.
2. Data provenance: every example carries source, license/provenance, collection time, transformation history and content hash where available.
3. Split before synthesis: create immutable train/validation/test partitions before augmentation, paraphrasing or synthetic generation. Never use test data for training, prompt construction, judge calibration or hyperparameter selection.
4. Baseline first: evaluate the unmodified base model on the exact locked test suite before adapting it.
5. Stage-specific objectives: pretraining optimizes next-token prediction; mid-training continues language modelling on a controlled domain mixture; post-training optimizes task behaviour using SFT, preference data or verifiable rewards.
6. No judge-only promotion: LLM-as-judge is diagnostic. Promotion requires objective task metrics, independent test data and human review where appropriate.
7. Reproducibility: record immutable data versions, code revision, model revision, tokenizer, config, seed, environment, checkpoints and metrics.
8. Budget-aware decisions: skip stages without enough data, compute or measurable justification.

## Training stage policy

### Pretraining (from scratch)

Use only when the product explicitly requires a new foundation model, has a tokenizer/corpus strategy, substantial licensed data, a compute plan and a scaling/evaluation plan. Objective: causal language modelling over a broad, deduplicated corpus. This is not a sensible default for single-GPU SLM fine-tuning.

### Mid-training / continued pretraining

Use when the base model has a material domain-distribution gap and there is a sufficiently large, high-quality raw-text corpus. Continue the causal-LM objective on a measured mixture of domain and general replay data. Track domain perplexity and general-capability regression. Use replay/mixing, conservative learning rates and early stopping to mitigate catastrophic forgetting.

### Post-training

- SFT: high-quality instruction/response or conversational trajectories.
- Preference optimization (DPO/KTO): only with valid preference data and chosen/rejected or preference labels.
- RL / GRPO: only with verifiable rewards (exact answer, unit tests, constrained schema); an LLM judge is not an unverified reward oracle.
- Tool use: train on tool-call trajectories with explicit schemas and execution-grounded outcomes.

### Parameter-efficient adaptation

LoRA/QLoRA is an implementation strategy, not a training stage. It can be used for SFT and supported preference optimization. Full fine-tuning or continued pretraining may be required for some distribution shifts; choose based on experiments.

## Reference pipeline

1. Task specification and capability-gap analysis.
2. Base-model selection and baseline evaluation.
3. Source acquisition, licensing, filtering, deduplication and provenance.
4. Leakage-safe train/validation/test split.
5. Stage planner chooses: no adaptation, mid-training, SFT, preference optimization, or a justified combination.
6. Training runs in isolated, resumable workers with resource checks and immutable configs.
7. Evaluation compares base and candidate on locked tests, plus regression suites.
8. Promotion gate checks target improvement, general-capability non-inferiority, safety, latency, memory and artifact integrity.
9. Model registry stores adapter/full model, tokenizer, config, data lineage, evaluation report and model card.

## Evaluation protocol

- Keep train, validation and test disjoint by normalized content hash and preferably source/document/entity.
- Validation is for early stopping and model selection. Test is read only for final candidate comparisons.
- Report per-task metrics, macro averages, confidence intervals (paired bootstrap where appropriate), sample counts and failure examples.
- Use task-specific exact/semantic metrics, schema validity, executable tests or expert review. LLM-as-judge is secondary and must be calibrated against human labels.
- Include base-model and simple non-training baselines. A model that does not beat its baseline does not ship.
- Track forgetting: general capability suite, safety suite and domain suite before and after every stage.

## Compute-aware operating modes

- Laptop / 4 GB VRAM: data curation, planning, evaluation, small-model LoRA experiments; do not promise full pretraining.
- Single 12 GB GPU: small-model QLoRA SFT and constrained experiments; mid-training depends on model size, sequence length and throughput.
- Multi-GPU: continued pretraining and larger sweeps after profiling.
- From-scratch pretraining: separate research programme with token/compute scaling estimates and multi-GPU infrastructure.

## Current implementation boundary

This branch adds the stage planner, deterministic leakage-resistant dataset splitting, evaluation gates and experiment manifests as a foundation. It does not claim that full pretraining, continued pretraining, DPO, GRPO, a model registry or production deployment are implemented. Those require dedicated trainer backends, dataset adapters, hardware integration and end-to-end tests on actual accelerators.

## Research references

- AllenAI OLMo / OLMo 2: open full-stack model flow, including pretraining, mid-training, post-training and evaluation artifacts: https://allenai.org/olmo and https://allenai.org/olmo2
- Stanford CRFM HELM: holistic, standardized and multi-metric evaluation: https://crfm.stanford.edu/helm/
- EvoLM (NeurIPS 2025): systematic study of pretraining, continued pretraining, SFT and RL dynamics: https://papers.nips.cc/paper_files/paper/2025/hash/778cc31583604ebf246726055e9f737f-Abstract-Conference.html
- Meta Llama 3.1: iterative SFT/DPO post-training and data quality practices: https://ai.meta.com/blog/meta-llama-3-1/
- Hugging Face TRL: SFT, DPO, GRPO and PEFT trainer ecosystem: https://github.com/huggingface/trl
