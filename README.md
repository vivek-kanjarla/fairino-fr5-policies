# fairino-fr5-act-pipeline

ACT (Action Chunking with Transformers) training and inference pipeline for the Fairino FR5 cobot, using data collected from the SO-101 → FR5 teleoperation setup.

---

## What this does

Takes the LeRobot v3.0 dataset recorded by [so101-fr5-teleop](../so101-fr5-teleop) and trains an ACT policy on it. The trained policy runs closed-loop on the real FR5 at 30 Hz.

---

## Setup

`lerobot==0.5.1` pins `torch`/`torchvision`/`numpy` below the newest releases, so
install into a fresh venv rather than a bleeding-edge system Python:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Usage

**1. Convert raw episodes → LeRobot dataset**

Raw recordings live in `episodes/episode_*/` (`data.csv` at ~100 Hz + `wrist_cam.mp4`
at ~30 Hz + `meta.json`). The converter resamples the CSV onto the camera timestamps
(one row per video frame), maps columns to `observation.state` (6 actual joints),
`action` (6 cmd joints + gripper), and `observation.eef_pose`, and writes a LeRobot
v3.0 dataset. `--extract-frames` dumps aligned JPEGs so training I/O is fast.

```bash
python tools/convert_episodes.py --episodes episodes --out lerobot_dataset --extract-frames
```

Point `dataset.root` in your training config at the output dir.

**2. EDA** — `eda/eda.ipynb` explores the converted dataset before training.

**3. Train**

```bash
cd training
python train.py --config config.local.yaml   # small, laptop-friendly run
# config.yaml is the larger paper-scale setup
```

Checkpoints land in `training/checkpoints/`. `best.pt` is saved whenever val L1 improves.

**Smoke test** — verify the whole pipeline (convert → load → train step → checkpoint
→ predict) in a few seconds on CPU:

```bash
python tools/smoke_test.py
```

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
