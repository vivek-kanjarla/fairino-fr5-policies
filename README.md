# fairino-fr5-act-pipeline

ACT (Action Chunking with Transformers) training and inference pipeline for the Fairino FR5 cobot, using data collected from the SO-101 → FR5 teleoperation setup.

---

## What this does

Takes the LeRobot v3.0 dataset recorded by [so101-fr5-teleop](../so101-fr5-teleop) and trains an ACT policy on it. The trained policy runs closed-loop on the real FR5 at 30 Hz.

---

## Setup

```bash
pip install -r requirements.txt
```

Dataset should already be at `../so101-fr5-teleop/lerobot_dataset/` from the teleop repo. If it's somewhere else, update `dataset.root` in `training/config.yaml`.

---

## Usage

**1. EDA** — EDA of the dataset before training


**2. Train**

```bash
cd training
python train.py
# or with a different config
python train.py --config config.yaml
```

Checkpoints land in `training/checkpoints/`. `best.pt` is saved whenever val L1 improves.

**3. Deploy on the robot**

```bash
cd inference
python deploy.py --checkpoint ../training/checkpoints/best.pt
```

Runs 150 steps (5s) by default. Change with `--steps N`. If the model was trained with images but you don't have the camera plugged in, use `--no-image`.

---

## Repo layout

```
eda/
  eda.ipynb           dataset exploration
training/
  config.yaml         all hyperparameters
  dataset.py          LeRobot parquet → PyTorch DataLoader
  model.py            lerobot ACTPolicy wrapper
  train.py            training loop
inference/
  deploy.py           loads checkpoint, runs policy on FR5
```

---

## Key numbers

| | |
|---|---|
| Action dim | 7 (joints 1–6 in deg + gripper 0–1) |
| Obs dim | 6 (actual joint positions in deg) |
| Control frequency | 30 Hz |
| Action chunk size | 100 steps (~3.3 s) |
| Image | 640×480 wrist cam, resized to 224×224 |
