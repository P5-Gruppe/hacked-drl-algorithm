from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from attacks.noise import GaussianObservationNoise


class AdversarialObsWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, std: float = 0.01, prob: float = 0.0) -> None:
        super().__init__(env)
        self._attack = GaussianObservationNoise(std=std, prob=prob)
        self._step_idx = 0

    def observation(self, observation: np.ndarray) -> np.ndarray:
        attacked = self._attack.apply(observation, self._step_idx)
        return attacked

    def step(self, action: np.ndarray):
        self._step_idx += 1
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self.observation(obs)
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs: Any):
        self._step_idx = 0
        obs, info = self.env.reset(**kwargs)
        obs = self.observation(obs)
        return obs, info
