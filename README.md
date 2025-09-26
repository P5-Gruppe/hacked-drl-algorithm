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

## Training

- Train for a small number of steps to see if the installation worked:
- Give a second and then the window will appear, if using render.
- (Render will show the cheetah training)

```bash
python train.py --steps 2000 --render
```

Results:

- A learning curve `training_curve.png` is saved under `runs/{timestamp}/`.
- Returns should trend upwards with larger `--steps` (e.g., 200k–1M).

- For longer training, do not use `--render`, since it makes the training process significantly slower:

```bash
python train.py --steps 200000
```

Each run directory contains:

- `config.json` — run configuration
- `returns.csv` — per-episode returns (streamed during training)
- `training_curve.png` and `summary.json`
- `actor.pt`, `critic.pt` — saved model weights (if available)

## Evaluation

- To evaluate a trained cheetah use the command:

```bash
python eval.py --run-dir runs/<timestamp> --episodes 3 --seed 42 --render
```

- The episodes parameter specifies how many times you will see the trained cheetah run.
- It is important to use the same seed `--seed 42` when testing 2 different cheetah's 
  so they will have the same evaluation environment.
- The evaluation plot is saved to `runs/<timestamp>/eval_returns.png`.

## Project structure

- `train.py` — main training entry point and environment factory (`make_env`).
- `eval.py` — evaluation script that loads `actor.pt` from a run directory.
- `algo/learner.py` — basic PPO starter (actor-critic, per-episode GAE + PPO updates).
- `algo/ppo_agent.py` — placeholder for our custom PPO-based algorithm (backbone, detection, correction, combined loss). Not developed yet.
- `env_wrappers/` — environment wrappers. The starter training currently uses a clean env (no noise).
- `attacks/` — attack scaffolding (FGSM/STAR stubs to be implemented later).
- `detectors/` — simple statistical detector example.