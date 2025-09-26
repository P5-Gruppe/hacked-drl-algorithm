# cheetah-adversarial

Minimal project to train and evaluate a HalfCheetah-v5 agent, with scaffolding for adversarial attacks and detection. The starter `Learner` includes a basic PPO so learning curves improve out of the box. A more advanced algorithm can be implemented in `algo/ppo_agent.py` later.

## Prerequisites

- Python 3.11+ recommended
- macOS/Linux desktop session for rendering (optional)

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Quick start

Train for a small number of steps to see if the installation worked (with windowed render):

```bash
python train.py --steps 2000 --render
```

(Render will show the cheetah training.)

Results:

- A learning curve `training_curve.png` is saved under `runs/{timestamp}/`.
- Returns should trend upward with larger `--steps` (e.g., 200k–1M).

For longer training, do not use `--render`, as it will take longer to finish:

```bash
python train.py --steps 200000
# Artifacts saved under runs/<timestamp>/
```

Each run directory contains:

- `config.json` — run configuration
- `returns.csv` — per-episode returns (streamed during training)
- `training_curve.png` and `summary.json`
- `actor.pt`, `critic.pt` — saved model weights (if available)

Evaluate a saved run:

```bash
python eval.py --run-dir runs/<timestamp> --episodes 5
# Add --render to watch; add --stochastic to sample actions
# Reproducible evaluation:
python eval.py --run-dir runs/<timestamp> --episodes 10 --seed 42 --render
```

The episodes parameter specifies how many times you will see the trained cheetah run.

- To see the logs and statistics of the training, look under `runs/{timestamp}/`

The evaluation plot is saved to `runs/<timestamp>/eval_returns.png`.

## Project structure

- `train.py` — main training entry point and environment factory (`make_env`).
- `eval.py` — evaluation script that loads `actor.pt` from a run directory.
- `algo/learner.py` — basic PPO starter (actor-critic, per-episode GAE + PPO updates).
- `algo/ppo_agent.py` — placeholder for your custom PPO-based algorithm (backbone, detection, correction, combined loss). Not used by default.
- `env_wrappers/` — environment wrappers. The starter training currently uses a clean env (no noise).
- `attacks/` — attack scaffolding (FGSM/STAR stubs to be implemented later).
- `detectors/` — simple statistical detector example.