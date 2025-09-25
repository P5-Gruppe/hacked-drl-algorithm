from __future__ import annotations

# PRIMARY IMPLEMENTATION TARGET
# Implement the custom PPO-based algorithm here (backbone, detection head,
# correction module, policy head, PPO storage and updates, combined loss).
# `algo/learner.py` is just a basic random-action starter for quick runs.

from typing import Any

import numpy as np
from gymnasium.spaces import Box


# -----------------------------------------------------------------------------
# Backbone (Shared feature extractor)
# - Diagram: BACKBONE (17 → 128 → 128 MLP)
# - Input: observation vector
# - Output: feature vector (dim=128)
# - Implementation hint: 2-layer MLP with ReLU
# class SharedBackbone(nn.Module): ...
# def build_backbone(obs_dim: int) -> nn.Module: ...
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Detection Head
# - Diagram: DETECTION (128 → 64 → Sigmoid)
# - Input: backbone features
# - Output: hack probability in [0,1]
# - Loss: BCE against attack label
# class DetectionHead(nn.Module): ...
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Policy Head (Gaussian)
# - Diagram: POLICY (128 → 64 → Gaussian)
# - Input: corrected features (from correction module)
# - Output: mean and log_std for Gaussian policy over actions
# - Action: tanh-squash and scale to env action bounds
# class PolicyHead(nn.Module): ...
# def sample_action(mean, log_std, action_space): ...
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Correction Module
# - Diagram: CORRECTION MODULE (insertion point: modify policy input)
# - Input: features, hack_prob
# - Output: corrected features
# - Baseline: identity pass-through; later try gating/residual blending
# class CorrectionModule(nn.Module): ...
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# PPO Learner (Storage + Update)
# - Diagram: PPO Loss → Combined Loss (with Detection Loss)
# - Responsibilities:
#   * rollout buffer (obs, actions, logprobs, rewards, dones, values)
#   * compute GAE / returns
#   * optimize combined loss: PPO + lambda * BCE
# - class PPOLearner: add, update(), etc.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Attack Interfaces
# - Diagram: ATTACK_DECISION → FGSM/STAR → perturbed state
# - Interface: ObservationAttack.apply(obs, step) -> obs'
# - Implementations: FGSM, STAR (later); Wire decision policy here
# - class FGSMAttack(ObservationAttack): ...
# - class STARAttack(ObservationAttack): ...
# -----------------------------------------------------------------------------


class Learner:
    """Comment-only scaffold; acts randomly for now.

    Wiring plan:
      obs → BACKBONE → features → DETECTION → hack_prob → CORRECTION → POLICY → action
      Losses: PPO loss + detection BCE → combined → backprop into BACKBONE
    """

    def __init__(self, observation_space: Box, action_space: Box) -> None:
        self.observation_space = observation_space
        self.action_space = action_space

        # TODO: build modules
        # self.backbone = build_backbone(obs_dim)
        # self.detector = DetectionHead(...)
        # self.correction = CorrectionModule(...)
        # self.policy = PolicyHead(...)
        # self.ppo = PPOLearner(...)

    def act(self, observation: np.ndarray) -> np.ndarray:
        # TODO (future): forward through backbone → detection → correction → policy
        # For now, return random action as baseline behavior.
        return self.action_space.sample()

    def observe(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        info: dict,
    ) -> None:
        # TODO (future): push transition into PPO buffer, trigger updates when ready
        # TODO (future): compute detection label from attack decision, BCE loss
        return None
