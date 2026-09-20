from __future__ import annotations

import numpy as np

from arch__dry_bean import Arch


class AODryBeanClassifier:
    def __init__(self, *, inference_steps: int = 1) -> None:
        if inference_steps < 1:
            raise ValueError("inference_steps must be >= 1")

        try:
            import ao_core as ao
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "ao_core is required for AO inference. Obtain AO Labs private-beta access first."
            ) from exc

        self._agent = ao.Agent(Arch, notes="UCI Dry Bean benchmark", save_meta=False)
        self._inference_steps = inference_steps

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "AODryBeanClassifier":
        self._agent.next_state_batch(
            np.asarray(features, dtype=np.uint8),
            np.asarray(labels, dtype=np.uint8),
            unsequenced=True,
        )
        return self

    def predict_bits(self, features: np.ndarray) -> np.ndarray:
        predictions: list[np.ndarray] = []
        z_index = self._agent.arch.Z__flat

        for row in np.asarray(features, dtype=np.uint8):
            self._agent.reset_state()
            for _ in range(self._inference_steps):
                self._agent.next_state(row, DD=False, unsequenced=True)
            state = self._agent.state
            predictions.append(
                np.asarray(self._agent.story[state - 1, z_index], dtype=np.uint8).copy()
            )

        return np.stack(predictions)
