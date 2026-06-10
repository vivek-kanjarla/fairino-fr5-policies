# Proprioception Modes — Benchmarking State Dependence

A single, policy-agnostic switch for benchmarking how much a policy leans on
proprioceptive state vs vision. This is the practical lever for attacking the
proprioceptive shortcut documented in [`il_failure_modes.md`](il_failure_modes.md).

Implemented as a shared wrapper in [`common/proprio.py`](../common/proprio.py),
wired into **all six policies** — ACT, Diffusion Policy, DiT-Flow, π0, π0.5, and
π0-FAST — so the same `--proprio {full,dropout,none}` benchmark switch works
across the whole repo.

---

## 1. The three modes

| Mode | Training | Eval / Deploy | Purpose |
|---|---|---|---|
| `full` | state used as-is | state used as-is | Baseline — current behaviour |
| `dropout` | state zeroed per-sample with prob `rate` | full state | Force vision to contribute without giving up state at deploy |
| `none` | state always zeroed | state always zeroed | State-free / vision-only — the most aggressive shortcut fix |

All three are the **same masking operation** applied to the normalised state
vector just before it enters the underlying lerobot policy:

```python
none:    state → 0                          (always)
dropout: state → 0 for a random p-fraction  (training only; per-sample)
full:    state → state                      (unchanged)
```

`dropout` is `none` only for a random subset of training samples; at eval it
reverts to `full`. `none` is `dropout` with rate 1.0 that also stays on at eval.
One function covers all three.

---

## 2. Why this is a wrapper, not per-policy code

The masking lives in the **model wrapper** (`policies/<p>/model.py`), in the
`_make_batch` step that both `forward()` (training) and `predict()` (deploy) go
through. Consequences:

- **`train.py` and `deploy.py` stay policy-agnostic.** They always pass the real
  state; the model decides what to do with it based on its config.
- **The mode is saved in the checkpoint config.** `deploy.py` rebuilds the model
  from that config, so deployment reproduces the exact training-time proprio
  handling automatically — no deploy flag needed, no chance of mismatch.
- **Adding a new policy is two lines:** build a `ProprioConfig` in `__init__`,
  call `mask_state(...)` after normalising the state. That's it.

```python
# in any policy wrapper's __init__:
self.proprio = ProprioConfig(cfg.proprio_mode, cfg.proprio_dropout_rate)

# in _make_batch, right after normalising:
state = mask_state(state, self.proprio, self.training)
```

The `self.training` flag (set by `model.train()` / `model.eval()`) is what makes
`dropout` train-only and full-at-eval automatically.

---

## 3. How to run the benchmark

Each variant is one flag. Checkpoints are isolated automatically under
`<checkpoint_dir>/proprio_<mode>/` so runs never clobber each other.

```bash
# ACT — three variants
python common/train.py --policy act --proprio full
python common/train.py --policy act --proprio dropout --proprio-rate 0.3
python common/train.py --policy act --proprio none

# Diffusion — three variants
python common/train.py --policy diffusion --proprio full
python common/train.py --policy diffusion --proprio dropout --proprio-rate 0.3
python common/train.py --policy diffusion --proprio none
```

Resulting checkpoint layout:

```
policies/act/checkpoints/proprio_full/best.pt
policies/act/checkpoints/proprio_dropout/best.pt
policies/act/checkpoints/proprio_none/best.pt
policies/diffusion/checkpoints/proprio_full/best.pt
...
```

You can also set `proprio_mode` directly in the config yaml (default is `full`),
but the CLI flag is the benchmark path because it isolates the checkpoint dir.

Deploy reads the mode from the checkpoint — no extra flag:

```bash
python common/deploy.py --checkpoint policies/act/checkpoints/proprio_none/best.pt
```

---

## 4. The full benchmark matrix

Two policies × three proprio modes = six runs:

| | ACT | Diffusion |
|---|---|---|
| `full` | baseline | baseline |
| `dropout` | force-vision | force-vision |
| `none` | state-free | state-free |

If you later switch the **action space** to delta EEF (see
[`action_spaces.md`](action_spaces.md)), this becomes a 2×3×2 grid. Run the
proprio axis first on the current action space to isolate its effect, then
re-run the winner with delta EEF.

---

## 5. What to measure for each run

The point of the benchmark is to see which mode breaks the shortcut. For each
trained variant, collect:

### 5.1 Conflict-swap probe (cheap, no robot)

Run `experiments/shortcut_probe.py` (the causal-intervention probe). For each
mode record:

| Metric | `full` (expected) | `dropout` (expected) | `none` (expected) |
|---|---|---|---|
| Image-ablation Δ | small | larger | largest |
| State-ablation Δ | large | smaller | ~0 (no state) |
| Ablation ratio (state/image) | ~5× | ~2× | <1× |
| Conflict-swap → image corr | ~−0.1 | higher | highest |
| Conflict-swap → state corr | ~+0.85 | lower | ~0 |

A successful shortcut fix pushes image correlation **up** and state correlation
**down**. `none` should show near-zero state correlation by construction (it is a
sanity check that the masking works); the interesting comparison is `full` vs
`dropout` and whether `dropout` recovers most of `none`'s vision-reliance while
keeping state available at deploy.

### 5.2 Spatial generalization (the real test, needs robot)

Place the object at workspace grid cells — both **in-domain** (where you
trained) and **out-of-domain** (outside training range). 3 rollouts per cell.

| | In-domain success | OOD success |
|---|---|---|
| `full` | high | low (shortcut → fixed reach) |
| `dropout` | high | medium |
| `none` | medium-high | highest (if vision coverage is sufficient) |

`none` only wins OOD if the cameras actually see the object throughout the
trajectory — otherwise the policy is flying blind. Check camera coverage first
(see the overhead-camera caveat in `il_failure_modes.md` §7.4).

### 5.3 Smoothness

You previously observed jerky motion. Log max joint velocity / trajectory
straightness per rollout to see whether any mode changes it.

---

## 6. Important caveats

- **`none` needs vision.** The wrapper raises an error if `proprio_mode='none'`
  and `use_image=False` — with no state and no camera there is nothing to
  condition on. State-free is vision-only by definition.

- **`none` is functionally, not architecturally, state-free.** The state encoder
  still exists but receives a constant (zero), so it carries zero per-sample
  information — the model learns to ignore it. This is the cleanest way to keep
  the architecture identical across modes (an apples-to-apples benchmark) without
  rebuilding each lerobot backbone's `input_features`. The few wasted parameters
  in the state projection are negligible.

- **Fix the data first.** Running this benchmark on the current ~54-demo dataset
  with a biased start pose will show `none` collapsing to low success (vision was
  never trained well enough to take over). Collect the fixed-home + workspace-grid
  dataset first (Phase 1 in `il_failure_modes.md` §11), *then* benchmark proprio
  modes on the good data. Otherwise you are benchmarking on top of a broken
  training distribution.

- **`dropout` is per-sample, whole-vector.** Each sample in a batch independently
  has its *entire* state zeroed or kept. This matches how "proprioception dropout"
  is used in the shortcut literature (force vision). Per-dimension dropout or
  additive state noise are possible variants — not implemented here to keep the
  benchmark clean.

---

## 7. Files touched

| File | Change |
|---|---|
| `common/proprio.py` | `ProprioConfig`, `mask_state`, `describe` |
| `policies/{act,diffusion,dit_flow,pi0,pi05,pi0_fast}/model.py` | Build `ProprioConfig`, call `mask_state` in `_make_batch`; `predict()` forces `training=False` |
| `policies/{act,diffusion,dit_flow,pi0,pi05,pi0_fast}/config.yaml` | `proprio_mode`, `proprio_dropout_rate` keys |
| `common/train.py` | `--proprio` / `--proprio-rate` CLI overrides + checkpoint isolation |
| `common/deploy.py` | Feeds 7-dim state (joints + last gripper cmd); mode applied automatically by the model |

All six policies share the identical wiring (`ProprioConfig` in `__init__` +
`mask_state` after normalizing the state). For the **π0 family** (`pi0`, `pi05`,
`pi0_fast`), `none` mode is valid even without a camera because they always have
language conditioning — they are vision-**and**-language models. The masking was
runtime-verified on ACT, Diffusion, and DiT-Flow; the π0 family uses the same
code path but can't be loaded here (it needs PaliGemma weights), so it was
compile-verified only.
