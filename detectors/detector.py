from __future__ import annotations

import numpy as np


class AttackDetector:
    def __init__(self, z_thresh: float = 6.0) -> None:
        self.z_thresh = float(z_thresh)
        self._mean = None
        self._var = None
        self._count = 0

    def _update_stats(self, x: np.ndarray) -> None:
        self._count += 1
        if self._mean is None:
            self._mean = x.astype(float)
            self._var = np.zeros_like(self._mean)
            return
        delta = x - self._mean
        self._mean += delta / self._count
        delta2 = x - self._mean
        self._var += delta * delta2

    def evaluate(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        info: dict,
        step: int,
    ) -> bool:
        if step < 1000:
            self._update_stats(obs)
            return False
        if self._mean is None or self._var is None:
            return False
        variance = self._var / max(self._count - 1, 1)
        std = np.sqrt(np.maximum(variance, 1e-8))
        z = np.abs((obs - self._mean) / std)
        detected = bool(np.any(z > self.z_thresh))
        return detected
