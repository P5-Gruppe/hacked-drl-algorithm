from __future__ import annotations

"""Basic PPO starter implementation in the Learner.

This remains minimal and self-contained so training shows learning curves.
Your advanced algorithm should still be implemented in `algo/ppo_agent.py`.
"""

from typing import Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gymnasium.spaces import Box


def _to_tensor(x: np.ndarray) -> torch.Tensor:
    t = torch.as_tensor(x, dtype=torch.float32)
    if t.ndim > 1:
        t = t.view(-1)
    return t


class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.net(obs)
        mu = self.mu(x)
        log_std = self.log_std.expand_as(mu)
        return mu, log_std


class Critic(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.v = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.v(obs).squeeze(-1)


class Learner:
    """Minimal PPO learner with per-episode updates."""

    def __init__(self, observation_space: Box, action_space: Box) -> None:
        self.observation_space = observation_space
        self.action_space = action_space

        obs_dim = int(np.prod(self.observation_space.shape))
        act_dim = int(np.prod(self.action_space.shape))

        # Hyperparameters
        self.gamma = 0.99
        self.lam = 0.95
        self.clip_ratio = 0.2
        self.pi_lr = 3e-4
        self.vf_lr = 1e-3
        self.train_iters = 10

        # Exploration and stability
        self.ent_coef = 0.01
        self.max_grad_norm = 0.5
        # Early stopping target KL for PPO
        self.target_kl = 0.02

        # Models/optimizers
        self.actor = Actor(obs_dim, act_dim)
        self.critic = Critic(obs_dim)
        self.pi_opt = optim.Adam(self.actor.parameters(), lr=self.pi_lr)
        self.vf_opt = optim.Adam(self.critic.parameters(), lr=self.vf_lr)

        # Buffers
        self.buf_obs: List[np.ndarray] = []
        self.buf_act: List[np.ndarray] = []  # clipped actions (used by env)
        self.buf_act_raw: List[np.ndarray] = (
            []
        )  # pre-clipped actions (used for PPO logp)
        self.buf_logp: List[float] = []
        self.buf_rew: List[float] = []
        self.buf_done: List[bool] = []
        # Old policy parameters for exact KL tracking
        self.buf_mu: List[np.ndarray] = []
        self.buf_log_std: List[np.ndarray] = []

    def act(self, observation: np.ndarray) -> np.ndarray:
        obs_t = _to_tensor(observation)
        with torch.no_grad():
            mu, log_std = self.actor(obs_t)
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(mu, std)
            a_raw = dist.sample()
            logp = dist.log_prob(a_raw).sum()
        # clip to action bounds
        low = torch.as_tensor(self.action_space.low, dtype=torch.float32)
        high = torch.as_tensor(self.action_space.high, dtype=torch.float32)
        a_clip = torch.max(torch.min(a_raw, high), low)

        # store
        self.buf_obs.append(observation.copy())
        self.buf_act_raw.append(a_raw.detach().cpu().numpy())
        self.buf_act.append(a_clip.detach().cpu().numpy())
        self.buf_logp.append(float(logp.item()))
        # cache old policy params for exact KL
        self.buf_mu.append(mu.detach().cpu().numpy())
        self.buf_log_std.append(log_std.detach().cpu().numpy())
        return a_clip.detach().cpu().numpy()

    def observe(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        info: dict,
    ) -> None:
        self.buf_rew.append(float(reward))
        self.buf_done.append(bool(done))
        if done:
            self._update()
            self._reset_buffer()
        return None

    # ---------------- PPO helpers ----------------
    def _compute_gae(
        self, rewards: np.ndarray, values: np.ndarray, dones: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        T = len(rewards)
        adv = np.zeros(T, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(T)):
            nonterminal = 1.0 - float(dones[t])
            next_value = values[t + 1] if t < T - 1 else 0.0
            delta = rewards[t] + self.gamma * next_value * nonterminal - values[t]
            last_gae = delta + self.gamma * self.lam * nonterminal * last_gae
            adv[t] = last_gae
        ret = adv + values
        return adv, ret

    def _update(self) -> None:
        if len(self.buf_rew) == 0:
            return
        obs = torch.as_tensor(np.stack(self.buf_obs), dtype=torch.float32)
        acts_raw = torch.as_tensor(np.stack(self.buf_act_raw), dtype=torch.float32)
        old_logp = torch.as_tensor(np.array(self.buf_logp, dtype=np.float32))
        old_mu = torch.as_tensor(np.stack(self.buf_mu), dtype=torch.float32)
        old_log_std = torch.as_tensor(np.stack(self.buf_log_std), dtype=torch.float32)
        with torch.no_grad():
            vals = self.critic(obs).cpu().numpy().astype(np.float32)
        rews = np.array(self.buf_rew, dtype=np.float32)
        dones = np.array(self.buf_done, dtype=np.bool_)

        adv, ret = self._compute_gae(rews, vals, dones)
        adv_t = torch.as_tensor(adv, dtype=torch.float32)
        ret_t = torch.as_tensor(ret, dtype=torch.float32)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        for _ in range(self.train_iters):
            mu, log_std = self.actor(obs)
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(mu, std)
            # Exact KL between old and new policy on collected states
            with torch.no_grad():
                dist_old = torch.distributions.Normal(old_mu, torch.exp(old_log_std))
                mean_kl = (
                    torch.distributions.kl_divergence(dist_old, dist).sum(dim=-1).mean()
                )
            # compute logp on the same (pre-clipped) samples used to compute old_logp
            logp = dist.log_prob(acts_raw).sum(dim=-1)
            ratio = torch.exp(logp - old_logp)
            clipped = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
            pg_loss = -(torch.min(ratio * adv_t, clipped * adv_t)).mean()

            # entropy bonus
            entropy = dist.entropy().sum(dim=-1).mean()
            pi_loss = pg_loss - self.ent_coef * entropy

            v_pred = self.critic(obs)
            v_loss = F.mse_loss(v_pred, ret_t)

            self.pi_opt.zero_grad()
            pi_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.pi_opt.step()

            self.vf_opt.zero_grad()
            v_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.vf_opt.step()

            # KL-based early stopping
            if float(mean_kl.item()) > 1.5 * self.target_kl:
                break

    def _reset_buffer(self) -> None:
        self.buf_obs.clear()
        self.buf_act.clear()
        self.buf_act_raw.clear()
        self.buf_logp.clear()
        self.buf_rew.clear()
        self.buf_done.clear()
        self.buf_mu.clear()
        self.buf_log_std.clear()
