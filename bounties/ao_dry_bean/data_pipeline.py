from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATASET_ID = 602
BITS_PER_FEATURE = 8
OUTPUT_BITS = 3
RANDOM_STATE = 42


@dataclass(frozen=True)
class FeatureEncoder:
    minimum: np.ndarray
    maximum: np.ndarray
    classes: tuple[str, ...]

    @classmethod
    def fit(cls, features: pd.DataFrame, labels: pd.Series) -> "FeatureEncoder":
        values = features.to_numpy(dtype=np.float64)
        return cls(
            minimum=np.nanmin(values, axis=0),
            maximum=np.nanmax(values, axis=0),
            classes=tuple(sorted(labels.astype(str).unique().tolist())),
        )

    def transform_features(self, features: pd.DataFrame) -> np.ndarray:
        values = features.to_numpy(dtype=np.float64)
        span = self.maximum - self.minimum
        safe_span = np.where(span == 0, 1.0, span)
        scaled = np.clip((values - self.minimum) / safe_span, 0.0, 1.0)
        quantized = np.rint(scaled * 255.0).astype(np.uint8)
        return np.unpackbits(quantized[..., None], axis=2).reshape(len(features), -1)

    def transform_labels(self, labels: pd.Series) -> np.ndarray:
        class_to_index = {name: index for index, name in enumerate(self.classes)}
        indices = np.array([class_to_index[str(value)] for value in labels], dtype=np.uint8)
        shifts = np.arange(OUTPUT_BITS - 1, -1, -1, dtype=np.uint8)
        return ((indices[:, None] >> shifts) & 1).astype(np.uint8)

    def decode_labels(self, encoded: np.ndarray) -> np.ndarray:
        encoded = np.asarray(encoded, dtype=np.uint8)
        weights = 1 << np.arange(OUTPUT_BITS - 1, -1, -1)
        indices = encoded @ weights
        return np.array(
            [self.classes[index] if index < len(self.classes) else "INVALID" for index in indices],
            dtype=object,
        )


@dataclass(frozen=True)
class DryBeanSplit:
    encoder: FeatureEncoder
    feature_names: tuple[str, ...]
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    y_train_raw: np.ndarray
    y_test_raw: np.ndarray


def prepare_split(
    features: pd.DataFrame,
    labels: pd.Series,
    *,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> DryBeanSplit:
    x_train_raw, x_test_raw, y_train_raw, y_test_raw = train_test_split(
        features,
        labels.astype(str),
        test_size=test_size,
        random_state=random_state,
        stratify=labels.astype(str),
    )
    encoder = FeatureEncoder.fit(x_train_raw, y_train_raw)
    return DryBeanSplit(
        encoder=encoder,
        feature_names=tuple(features.columns.astype(str)),
        x_train=encoder.transform_features(x_train_raw),
        x_test=encoder.transform_features(x_test_raw),
        y_train=encoder.transform_labels(y_train_raw),
        y_test=encoder.transform_labels(y_test_raw),
        y_train_raw=y_train_raw.to_numpy(dtype=object),
        y_test_raw=y_test_raw.to_numpy(dtype=object),
    )


def load_dry_bean() -> DryBeanSplit:
    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=DATASET_ID)
    features = dataset.data.features.copy()
    targets = dataset.data.targets.copy()
    labels = targets.iloc[:, 0] if isinstance(targets, pd.DataFrame) else targets
    return prepare_split(features, labels)
