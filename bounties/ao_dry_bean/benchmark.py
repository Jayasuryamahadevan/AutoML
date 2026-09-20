from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from ao_model import AODryBeanClassifier
from data_pipeline import DryBeanSplit


@dataclass(frozen=True)
class BenchmarkResult:
    model: str
    accuracy: float
    predictions: np.ndarray


def _subset(array: np.ndarray, limit: int | None) -> np.ndarray:
    if limit is None:
        return array
    return array[: min(limit, len(array))]


def run_ao(
    split: DryBeanSplit,
    *,
    train_limit: int | None = None,
    test_limit: int | None = None,
    inference_steps: int = 1,
) -> BenchmarkResult:
    x_train = _subset(split.x_train, train_limit)
    y_train = _subset(split.y_train, train_limit)
    x_test = _subset(split.x_test, test_limit)
    y_test_raw = _subset(split.y_test_raw, test_limit)

    model = AODryBeanClassifier(inference_steps=inference_steps)
    model.fit(x_train, y_train)
    prediction_bits = model.predict_bits(x_test)
    predictions = split.encoder.decode_labels(prediction_bits)

    return BenchmarkResult(
        model="AO weightless neural network",
        accuracy=float(accuracy_score(y_test_raw, predictions)),
        predictions=predictions,
    )


def run_baseline(
    split: DryBeanSplit,
    *,
    train_limit: int | None = None,
    test_limit: int | None = None,
) -> BenchmarkResult:
    x_train = _subset(split.x_train, train_limit)
    y_train_raw = _subset(split.y_train_raw, train_limit)
    x_test = _subset(split.x_test, test_limit)
    y_test_raw = _subset(split.y_test_raw, test_limit)

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    model.fit(x_train, y_train_raw)
    predictions = model.predict(x_test)

    return BenchmarkResult(
        model="Random Forest baseline",
        accuracy=float(accuracy_score(y_test_raw, predictions)),
        predictions=predictions,
    )


def predictions_frame(
    split: DryBeanSplit,
    result: BenchmarkResult,
    *,
    test_limit: int | None = None,
) -> pd.DataFrame:
    truth = _subset(split.y_test_raw, test_limit)
    return pd.DataFrame(
        {
            "expected": truth,
            "predicted": result.predictions,
            "correct": truth == result.predictions,
        }
    )
