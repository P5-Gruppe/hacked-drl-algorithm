from __future__ import annotations

import argparse
import os
import json
from datetime import datetime

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from algo.learner import Learner
from detectors.detector import AttackDetector


def make_env(render: bool) -> gym.Env:
    render_mode = "human" if render else None
    env = gym.make("HalfCheetah-v5", render_mode=render_mode)
    return env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--run-dir", type=str, default="", help="Directory to store logs; default creates runs/<timestamp>")
    args = parser.parse_args()

    env = make_env(render=args.render)
    detector = AttackDetector()
    learner = Learner(env.observation_space, env.action_space)

    obs, info = env.reset()
    episode_returns = []
    current_episode_return = 0.0
    # Prepare run directory
    run_dir = args.run_dir.strip()
    if not run_dir:
        run_dir = os.path.join("runs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    # Save minimal config
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"steps": args.steps, "render": bool(args.render)}, f, indent=2)
    # Open CSV for streaming episode returns
    returns_csv_path = os.path.join(run_dir, "returns.csv")
    returns_csv = open(returns_csv_path, "w", encoding="utf-8")
    returns_csv.write("episode,return\n")
    returns_csv.flush()
    # Progress reporting at 10% intervals
    total_steps = int(args.steps)
    progress_marks = {max(1, int(total_steps * k / 10)) for k in range(1, 11)}
    reported = set()
    for step in range(args.steps):
        action = learner.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        current_episode_return += float(reward)
        detected = detector.evaluate(obs, action, reward, info, step)
        info = dict(info)
        info["attack_detected"] = detected
        learner.observe(obs, action, reward, terminated or truncated, info)
        # Print progress when crossing each 10% mark
        step1 = step + 1
        if step1 in progress_marks and step1 not in reported:
            pct = int(round(step1 / total_steps * 100))
            print(f"Progress: {pct}% ({step1}/{total_steps} steps)")
            reported.add(step1)
        if terminated or truncated:
            episode_returns.append(current_episode_return)
            # Stream to CSV for persistence
            returns_csv.write(f"{len(episode_returns)},{current_episode_return}\n")
            returns_csv.flush()
            current_episode_return = 0.0
            obs, info = env.reset()

    env.close()
    print("Finished", args.steps, "steps.")

    # If training ended mid-episode, include the partial return
    if current_episode_return != 0.0:
        episode_returns.append(current_episode_return)
        returns_csv.write(f"{len(episode_returns)},{current_episode_return}\n")
        returns_csv.flush()
    returns_csv.close()

    if len(episode_returns) > 0:
        plt.figure(figsize=(8, 4.5))
        plt.plot(range(1, len(episode_returns) + 1), episode_returns, label="Episode return")
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.title("HalfCheetah Training Performance")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        output_path = os.path.join(run_dir, "training_curve.png")
        plt.savefig(output_path, dpi=150)
        # Save JSON summary
        with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump({"num_episodes": len(episode_returns), "returns": episode_returns}, f, indent=2)
        print(f"Saved training artifacts in {run_dir}")

    # Save model weights if available
    try:
        import torch  # local import
        if hasattr(learner, "actor"):
            torch.save(learner.actor.state_dict(), os.path.join(run_dir, "actor.pt"))
        if hasattr(learner, "critic"):
            torch.save(learner.critic.state_dict(), os.path.join(run_dir, "critic.pt"))
    except Exception as e:
        # Best-effort save; ignore if unsupported
        print(f"Model save skipped: {e}")


if __name__ == "__main__":
    main()
