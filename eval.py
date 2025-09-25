from __future__ import annotations

import argparse
import os
import random
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch

from train import make_env
from algo.learner import Actor  # reuse the same network definition


def load_actor(run_dir: str, obs_dim: int, act_dim: int) -> Actor:
    actor_path = os.path.join(run_dir, "actor.pt")
    if not os.path.isfile(actor_path):
        raise FileNotFoundError(f"actor.pt not found in {run_dir}")
    actor = Actor(obs_dim, act_dim)
    state = torch.load(actor_path, map_location="cpu")
    actor.load_state_dict(state)
    actor.eval()
    return actor


def evaluate(
    run_dir: str,
    num_episodes: int,
    render: bool,
    deterministic: bool = True,
    seed: Optional[int] = None,
) -> List[float]:
    env = make_env(render=render)
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    actor = load_actor(run_dir, obs_dim, act_dim)

    # Global RNG seeding
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    returns: List[float] = []
    for ep in range(num_episodes):
        reset_kwargs = {}
        if seed is not None:
            reset_kwargs["seed"] = seed + ep
        obs, info = env.reset(**reset_kwargs)
        ep_ret = 0.0
        while True:
            obs_t = torch.as_tensor(obs, dtype=torch.float32)
            if obs_t.ndim > 1:
                obs_t = obs_t.view(-1)
            with torch.no_grad():
                mu, log_std = actor(obs_t)
                if deterministic:
                    act_t = mu
                else:
                    std = torch.exp(log_std)
                    dist = torch.distributions.Normal(mu, std)
                    act_t = dist.sample()
            low = torch.as_tensor(env.action_space.low, dtype=torch.float32)
            high = torch.as_tensor(env.action_space.high, dtype=torch.float32)
            act_t = torch.max(torch.min(act_t, high), low)
            action = act_t.cpu().numpy()

            obs, reward, terminated, truncated, info = env.step(action)
            ep_ret += float(reward)
            if terminated or truncated:
                break
        returns.append(ep_ret)

    env.close()
    return returns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True, help="Path to run directory containing actor.pt")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--stochastic", action="store_true", help="Sample actions instead of using mean")
    parser.add_argument("--seed", type=int, default=None, help="Evaluation seed for reproducible rollouts")
    args = parser.parse_args()

    returns = evaluate(
        args.run_dir,
        args.episodes,
        args.render,
        deterministic=not args.stochastic,
        seed=args.seed,
    )
    if len(returns) == 0:
        print("No episodes evaluated.")
        return

    mean_ret = float(np.mean(returns))
    std_ret = float(np.std(returns))
    print(f"Evaluated {len(returns)} episodes | mean: {mean_ret:.2f} ± {std_ret:.2f}")

    # Save plot next to the run dir
    plt.figure(figsize=(8, 4.5))
    plt.plot(range(1, len(returns) + 1), returns, marker="o", label="Episode return")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Evaluation Returns")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(args.run_dir, "eval_returns.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved evaluation plot to {out_path}")


if __name__ == "__main__":
    main()

