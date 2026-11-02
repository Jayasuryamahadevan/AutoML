# Frontier-lab methods: evidence review and AutoSLM implementation map

Research snapshot: 2026-11-02. This document distinguishes publicly documented practices from implementation choices. It does not claim access to private training recipes or infrastructure.

## 1. What the public evidence says

### NVIDIA / Nemotron

Public Nemotron reports and documentation describe a lifecycle rather than a single fine-tuning step:

- Pretraining corpora combine natural and synthetic data. Public reports describe source-specific quality tiers, multilingual/code/math/science subsets, filtering, deduplication and deliberate data-mixture schedules.
- Continued pretraining / mid-training is task- or domain-directed. Nemotron 3.5 Super reports Indic continued pretraining, long-context extension, and a software-engineering stage built from repository context, issue/PR pairs, commit histories, build logs and agent traces. The team evaluates pre/post checkpoints using a smaller experimental post-training recipe before committing to the full post-training stack.
- Post-training is a mixture of SFT, preference optimization, RL from verifiable rewards, human/AI feedback, and distillation. The exact mix varies by model and objective.
- Synthetic examples are not accepted merely because a teacher generated them. Public descriptions include structural checks, repetition filtering, execution/compiler/numerical verification, quality filtering and benchmark decontamination.
- Evaluation includes domain capability, general regression, safety, agent environments and hardware/performance trade-offs.

Primary public sources:
- Nemotron 3.5 Super/Lightning report: https://research.nvidia.com/labs/nemotron/Nemotron-3.5-Super-and-Lightning/
- Nemotron training FAQ: https://docs.nvidia.com/nemotron/nightly/train-models/faq/
- NeMo Curator: https://github.com/NVIDIA-NeMo/Curator
- NeMo Curator text curation docs: https://docs.nvidia.com/nemo/curator/curate-text

### Anthropic / Claude

Anthropic publishes less of its foundation-training implementation than NVIDIA or Ai2. Its system cards and research papers still reveal useful process patterns:

- Pretraining data is a proprietary mixture that can include public web data, third-party datasets, human/contractor data, opt-in user data and internally generated synthetic data. Public system cards describe crawler policy, cleaning, deduplication and classification.
- Post-training includes supervised demonstrations, human preference feedback, AI feedback / Constitutional AI, and reinforcement learning. Constitutional AI uses written principles to critique/revise responses and can train preference models from AI-generated comparisons; human researchers still inspect the effects.
- Synthetic data is used for controlled capabilities and research. Anthropic's public hidden-objectives paper documents a staged synthetic-document generation pipeline, mid-training, SFT and RL phases, with explicit train/test bias separation.
- System cards describe evaluation snapshots throughout training, capability and safety evaluations, red teaming, external testing, deployment risk assessment and ongoing monitoring.
- Interpretability research (e.g. sparse autoencoders and attribution graphs) is used as an additional diagnostic, not a replacement for behavioral evaluation.

Primary public sources:
- Anthropic Transparency Hub: https://www.anthropic.com/transparency
- Claude Opus 4.6 System Card: https://www-cdn.anthropic.com/14e4fb01875d2a69f646fa5e574dea2b1c0ff7b5.pdf
- Constitutional character training: https://www.anthropic.com/research/claude-character
- Auditing Language Models for Hidden Objectives: https://assets.anthropic.com/m/317564659027fb33/original/Auditing-Language-Models-for-Hidden-Objectives.pdf

### Liquid AI / LFM

LFM2 public technical reporting emphasizes that a small model is a hardware-constrained systems problem, not just a parameter-count problem:

- Architecture search is hardware-in-the-loop: latency, memory and deployment target are part of model design.
- The LFM2 report describes a large-scale pretraining run, tempered/decoupled Top-K knowledge distillation, difficulty-ordered curriculum, and a multi-stage post-training recipe involving SFT, length-normalized preference optimization and model merging.
- Model families and deployment artifacts target CPU/edge runtimes as well as accelerators.
- Small-model quality depends on teacher quality, data selection, distillation objective, curriculum, architecture, quantization and measured target-hardware latency.

Primary public sources:
- LFM2 technical report: https://arxiv.org/abs/2511.23404
- Liquid AI report page: https://www.liquid.ai/research/lfm2-technical-report

### Open-science reference: Ai2 OLMo

OLMo is particularly valuable as a reproducibility model: it publishes artifacts across pretraining, mid-training and post-training, along with training code, data tools and evaluation code. AutoSLM should adopt this artifact-level transparency even if its compute scale is much smaller.

- https://allenai.org/olmo
- https://allenai.org/olmo2

## 2. Cross-lab synthesis: the recurring engineering pattern

1. Define target capability and deployment constraints before choosing the model or training stage.
2. Acquire data with legal/permission evidence and immutable source metadata.
3. Normalize, classify, filter, deduplicate and decontaminate data.
4. Split by source/document/task before augmentation. Freeze the test set.
5. Generate synthetic data iteratively; keep teacher, prompt, version, seed and verifier provenance.
6. Validate synthetic examples with deterministic tools whenever possible (tests, compilers, schema validators, math checks); use calibrated model/human review for subjective quality.
7. Run base-model baselines and small ablations before expensive training.
8. Choose a justified sequence: continued pretraining only for raw-text distribution shift; SFT for demonstrations; preference optimization for comparative preferences; RL only when rewards are reliable and verifiable.
9. Evaluate every checkpoint on validation and regression suites; run locked test only for the selected candidate.
10. Record full lineage: code/model/tokenizer/data revisions, configs, seeds, hardware, environment, metrics, artifacts and failures.
11. Promote only with evidence: target gain, non-inferior general capability, safety, latency, memory, license and artifact integrity.
12. Release model card, dataset cards, evaluation recipe and deployable artifact; monitor and rollback.

## 3. What AutoSLM now implements on this branch

- Existing v3 foundation: stage planning, deterministic group-aware split, validation-vs-test separation, baseline-relative promotion primitive and experiment manifest.
- New provenance contract: license and permission evidence; fail-closed eligibility result.
- New synthetic generation pipeline: injected teacher backend, prompt hash, teacher identity, structural checks, optional deterministic verifier, optional quality scorer, repetition rejection and duplicate rejection.
- New append-only SQLite lineage log: per-event payload, timestamp, run ID, previous hash and event hash; chain verification detects accidental mutation.
- Tests for these primitives and CI execution.

These are production-oriented primitives, not yet a complete training platform. They do not implement distributed pretraining, a full NeMo-style crawler/curator, a calibrated reward model, Anthropic-scale human feedback, LFM hardware-in-the-loop architecture search, or a production model registry/serving fleet.

## 4. Implementation sequence

### P0 — Data governance and observability
- Add provenance to every source row and synthetic row.
- Quarantine unknown/unapproved licenses.
- Persist every pipeline transition, metric, artifact hash and exception.
- Make test-set access a separate capability, not a path passed to training code.

### P1 — Reproducible SFT
- Use Transformers + TRL + PEFT behind a versioned trainer adapter.
- Support chat-template correctness, response-only loss masking, QLoRA, gradient accumulation, checkpoint resume and validation-based checkpoint selection.
- Persist environment lockfile, model/tokenizer revision, seed, CUDA/GPU metadata and exact trainer config.

### P2 — Evaluation as a product
- Task-specific exact match, structured-output validity, executable tests, latency and memory.
- Paired confidence intervals and per-slice breakdowns.
- Human-calibrated LLM judge only as a secondary metric.
- Benchmark decontamination before release.

### P3 — Continued pretraining
- Raw text only; packed causal-LM sequences.
- Domain/general replay mixture, measured token ratios, conservative LR and forgetting gates.
- Compare pre/post checkpoints with the same downstream evaluation recipe.

### P4 — Preference and RL
- DPO/KTO only after validating preference-pair provenance and annotator/judge agreement.
- GRPO/RL only with deterministic reward checks, sandboxing, reward-hacking tests and policy constraints.
- Keep reward-model evaluation separate from the policy's own training judge.

### P5 — Hardware-aware SLM research
- Benchmark candidate architectures on actual target hardware: prefill/decode tokens/s, peak memory, energy where measurable, context length and quantized quality.
- Architecture search and from-scratch pretraining are separate research tracks requiring compute budgets and controlled ablations.

## 5. Non-negotiable scientific rules

- No training on test data, including synthetic descendants of test prompts.
- No silent acceptance of unknown licenses.
- No LLM judge as the sole release metric.
- No claiming deployment when only a local adapter exists.
- No claiming pretraining, RL, or model improvements until a real run and independent evaluation exist.
- A failed run is still an experiment: preserve logs, config, metrics and failure reason.
