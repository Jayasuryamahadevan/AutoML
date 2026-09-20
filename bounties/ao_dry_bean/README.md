# AO Labs Dry Bean benchmark

Prototype for AO Labs' open **$300 “Benchmark our weightless NNs on a new dataset”** bounty (`aolabsai/ao_pyth#10`).

This branch intentionally does **not** claim an AO accuracy result yet. AO's reference app depends on the private-beta `ao_core` package, so the AO run must be executed only after AO Labs confirms the dataset/scope and grants current runtime access.

## Dataset

UCI Dry Bean (UCI Machine Learning Repository dataset 602):

- 13,611 rows
- 16 numerical input features
- 7 bean classes
- multiclass classification

The loader uses `ucimlrepo` rather than committing the dataset into this repository.

## Encoding

AO agents consume binary inputs. The preprocessing pipeline:

1. creates a fixed stratified train/test split (`random_state=42`);
2. fits per-feature min/max values on the **training split only**;
3. clips test values to the learned training range;
4. quantizes each feature to an unsigned 8-bit value;
5. expands each feature into 8 binary neurons (16 x 8 = 128 input neurons);
6. encodes the 7 target classes into a 3-bit output channel.

The AO architecture uses `forward_full_conn`, following AO Labs' published tabular NetBox architecture pattern.

## Setup

```bash
cd bounties/ao_dry_bean
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

AO Labs must grant access to its private-beta `ao_core` package before AO inference can run. Install it using AO Labs' current instructions after access is approved.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests do not require `ao_core` or network access.

## Run

```bash
streamlit run app.py
```

The Random Forest baseline can run with the public dependencies. The AO button fails closed with a clear message when `ao_core` is unavailable.

## Acceptance-oriented output

The app reports held-out accuracy and can export row-level expected/predicted labels to CSV for both the AO model and the baseline.

Before a bounty submission, record:

- AO Labs confirmation that Dry Bean is eligible and the $300 reward remains available;
- the exact `ao_core` version/commit or installation source used;
- train/test row counts and inference-step setting;
- AO held-out accuracy;
- baseline held-out accuracy;
- any invalid 3-bit AO outputs and limitations;
- the exact commands used to reproduce the result.
