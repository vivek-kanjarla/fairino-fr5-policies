# fairino-fr5-policies

Imitation-learning training and inference pipeline for the Fairino FR5 cobot,
using data collected from the SO-101 → FR5 teleoperation setup.

A **shared** data + training + deploy harness lives in `common/`; each policy
type (ACT, and future ones like Diffusion Policy, VQ-BeT, …) is a self-contained
folder under `policies/<name>/`. Training and deploy pick the policy by name, so
adding a model never touches the shared code.

---

## What this does

Takes the LeRobot v3.0 dataset recorded by [so101-fr5-teleop](../so101-fr5-teleop)
(or built locally from raw episodes, see below) and trains a policy on it. The
trained policy runs closed-loop on the real FR5 at 30 Hz.

---

## Setup

`lerobot==0.5.1` pins `torch`/`torchvision`/`numpy` below the newest releases, so
install into a fresh venv rather than a bleeding-edge system Python:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

All commands below are run **from the repo root**.

---

## Usage

**1. Convert raw episodes → LeRobot dataset**

Raw recordings live in `episodes/episode_*/` (`data.csv` at ~100 Hz + `wrist_cam.mp4`
at ~30 Hz + `meta.json`). The converter resamples the CSV onto the camera timestamps
(one row per video frame), maps columns to `observation.state` (6 actual joints),
`action` (6 cmd joints + gripper), and `observation.eef_pose`, and writes a LeRobot
v3.0 dataset. `--extract-frames` dumps aligned JPEGs so training I/O is fast.

```bash
python common/convert_episodes.py --episodes episodes --out lerobot_dataset --extract-frames
```

Point `dataset.root` in the policy's config at the output dir.

**2. EDA** — `eda/eda.ipynb` explores the converted dataset before training.

**3. Train**

```bash
python common/train.py --policy act --config policies/act/config.local.yaml   # laptop-friendly
# policies/act/config.yaml is the larger paper-scale setup
```

Checkpoints land in `policies/<policy>/checkpoints/`; `best.pt` is saved whenever
val L1 improves. The checkpoint records which policy produced it.

**Smoke test** — verify the whole pipeline (convert → load → train step → checkpoint
→ predict) in a few seconds on CPU:

```bash
python common/smoke_test.py          # defaults to act
```

**4. Deploy on the robot**

```bash
python common/deploy.py --checkpoint policies/act/checkpoints/best.pt
```

`deploy.py` reads the policy name from the checkpoint and auto-loads the matching
`policies/<policy>/model.py`. Runs 150 steps (5 s) by default; change with `--steps N`.
If the model was trained with images but the camera isn't plugged in, use `--no-image`.

---

## Repo layout

```
common/                 shared across all policies
  convert_episodes.py   raw episodes → LeRobot v3.0 dataset (100→30 Hz resample)
  dataset.py            LeRobot parquet → PyTorch DataLoader
  train.py              policy-agnostic training loop (--policy <name>)
  deploy.py             loads any checkpoint, runs it on the FR5
  smoke_test.py         end-to-end pipeline check
policies/
  act/                  ACT (Action Chunking Transformer), lerobot ACTPolicy wrapper
    model.py            defines build_model(cfg, stats, device)
    config.yaml         paper-scale hyperparameters
    config.local.yaml   small laptop run
    config.smoke.yaml   tiny CPU smoke config
eda/
  eda.ipynb             dataset exploration
```

### Adding a new policy

1. `mkdir policies/<name>` with a `model.py` that defines
   `build_model(cfg, stats, device) -> nn.Module`, where the returned model
   implements `forward(obs_state, actions, action_is_pad, obs_image) -> (loss, l1, kl)`,
   `reset()`, and `predict(obs_state, obs_image)`.
2. Add a `config.yaml` (same `dataset` / `model` / `training` sections).
3. Train with `python common/train.py --policy <name>`. No shared code changes.

---

## Key numbers (ACT)

| | |
|---|---|
| Action dim | 7 (joints 1–6 in deg + gripper 0–1) |
| Obs dim | 6 (actual joint positions in deg) |
| Control frequency | 30 Hz |
| Action chunk size | 100 steps (~3.3 s) |
| Image | 640×480 wrist cam, resized to 224×224 |
