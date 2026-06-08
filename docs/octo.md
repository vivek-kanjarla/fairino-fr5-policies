# Octo — generalist policy, zero-shot + finetuned on the FR5

Octo is a **generalist robot policy** from the Octo team (RAIL, Berkeley): a transformer
with a diffusion action head, pretrained on **800k trajectories** from the Open-X-Embodiment
dataset. Unlike ACT / Diffusion Policy (which we train from scratch on ~54 FR5 episodes),
Octo brings a **pretrained visuomotor prior** that you adapt to the FR5 by finetuning.

This doc explains what Octo is, how it differs from the from-scratch policies, and how the
three FR5 workflows (zero-shot, finetune, finetuned inference) work. The runnable code and
setup live in [`policies/octo/`](../policies/octo/README.md).

---

## 1. Where Octo sits relative to the other policies

| | ACT / Diffusion / DiT | π0 / π0.5 (lerobot) | **Octo** |
|---|---|---|---|
| Framework | PyTorch (lerobot) | PyTorch (lerobot) | **JAX / Flax** |
| Pretraining | none (scratch on FR5) | VLM-pretrained | **800k Open-X trajectories** |
| Action head | direct / DDPM / flow | flow matching | **diffusion head** |
| Language | DiT/π only | yes | **yes (T5 tokens)** |
| In this repo | `common/train.py` loop | lerobot wrapper | **own scripts + venv** |

Octo is the only policy here that is **JAX**, so it cannot share `common/train.py` /
`common/deploy.py`. It runs in an isolated `.venv-octo` with its own three scripts. This is
a deliberate separation, not a hack — mixing JAX and a PyTorch/CUDA stack in one environment
is fragile.

---

## 2. Architecture (what the pretrained model actually is)

```
   wrist image (256×256)          language ("pick up the block…")
        │                               │
   image tokenizer                 T5 tokenizer → tokens
        │                               │
        └──────────────┬────────────────┘
                       ▼
        ┌─────────────────────────────────┐
        │   Octo transformer (block-wise   │   + learned "readout" tokens
        │   causal over a 2-frame window)  │     that read out the action info
        └─────────────────────────────────┘
                       │  readout_action embeddings
                       ▼
              Diffusion action head  →  action chunk (horizon=4, dim=7)
```

Verified shapes for `octo-{small,base}-1.5`:

- **window_size = 2** — the transformer attends over the current + previous frame.
- **action_horizon = 4** — each step predicts the next 4 actions.
- **action_dim = 7** — which, for the FR5, lines up exactly with **6 joints + gripper**, so
  no head surgery is needed to finetune.
- Observation the model reads: `image_primary` (B, 2, 256, 256, 3) uint8 +
  `timestep_pad_mask` (B, 2). Octo also has an `image_wrist` (128×128) slot; the FR5 has one
  camera, so we map it to `image_primary` and leave `image_wrist` zeroed + masked.

The action head is a **diffusion head**: at inference it denoises a noise sample into an
action chunk, conditioned on the readout embeddings (same family as Diffusion Policy, but
the conditioning comes from a much larger pretrained transformer).

---

## 3. Workflow 1 — zero-shot inference (no finetuning)

Load the Octo team's weights and run them on an FR5 wrist image + instruction:

```python
model = OctoModel.load_pretrained("hf://rail-berkeley/octo-base-1.5")
task  = model.create_tasks(texts=["pick up the block and place it in the bin"])
obs   = {"image_primary": imgs(1,2,256,256,3), "timestep_pad_mask": ones(1,2)}
actions = model.sample_actions(obs, task,
              unnormalization_statistics=model.dataset_statistics["bridge_dataset"]["action"],
              rng=jax.random.PRNGKey(0))      # → (1, 4, 7)
```

**What zero-shot does and doesn't give you.** The model runs and produces a 7-D action
chunk — but those numbers live in **Octo's pretrained action space** (normalized delta
end-effector poses borrowed from a source dataset's statistics), **not FR5 joint commands**.
The FR5's action space (absolute joint angles + gripper) was never in Octo's training mix,
so zero-shot will not drive the arm correctly. This workflow is the **baseline**: it proves
the model loads and runs on FR5-shaped inputs, and gives a reference to compare the
finetuned model against.

---

## 4. Workflow 2 — finetuning on the FR5

Because the FR5 action (7-D) matches Octo's head, finetuning is the **direct** path — load
the pretrained model and keep training it on FR5 batches:

```python
state = TrainState.create(rng, model, optax.adamw(lr))     # lr warmup→cosine
# per step:
batch = process_text(fr5_adapter.sample_batch(B), model.text_processor)
embeddings = bound.octo_transformer(batch["observation"], batch["task"], pad_mask, train=True)
loss, _    = bound.heads["action"].loss(embeddings, batch["action"], pad_mask, action_pad_mask)
grads      = jax.grad(loss)        # → apply_gradients
```

### Finetuning modes — Octo does NOT use LoRA

A frequent misconception is that Octo is finetuned with LoRA. It isn't. The official
recipe (`octo-models/octo` `scripts/configs/finetune_config.py`) freezes parameters **by
key pattern** via `frozen_keys`, in three modes:

| Mode | `frozen_keys` | Trains | Use |
|---|---|---|---|
| `head_only` *(our default)* | `octo_transformer.*` | heads only | tiny datasets (FR5) — keep the 800k-trajectory visual prior, adapt only the action head |
| `head_mlp_only` | `octo_transformer.*` + head map-head attn/probe | head MLP only | most conservative |
| `full` | `None` | everything | Octo paper's default; more data + compute |

`finetune.py` uses Octo's official `create_optimizer` — cosine LR (peak 3e-4, warmup),
gradient clipping (global-norm 1.0), AdamW with **no weight decay on biases/LayerNorm** —
plus the `frozen_keys` for the chosen mode. This is the exact pretrained recipe (no
custom optimizer, no LoRA). Verified: `head_only`/`head_mlp_only` leave every transformer
parameter unchanged; `full` updates them.

Design choices for the FR5 (in `policies/octo/finetune.py`):

- **Default `head_only`.** The FR5 dataset is tiny (~54 episodes); freezing the
  800k-trajectory transformer and adapting only the action head is the most robust choice
  (strong frozen feature extractor + a head re-mapped to FR5 actions). Switch to `full`
  with `--mode full` once you have more data.
- **Vision + language only, no proprioceptive state.** Octo's default obs is image +
  language — there is no joint-state input. This is *exactly* the state-free recipe that
  [`il_failure_modes.md`](il_failure_modes.md) and [`proprioception_modes.md`](proprioception_modes.md)
  recommend for breaking the proprioceptive shortcut. Octo gets it for free.
- **Wrist camera → `image_primary`** (256), `image_wrist` zeroed + masked.
- **Action normalization** with the dataset's own mean/std (saved next to the checkpoint;
  reused to unnormalize at inference).

The data adapter ([`fr5_octo_data.py`](../policies/octo/fr5_octo_data.py)) reads the LeRobot
parquet + extracted frames directly (no torch/lerobot), windows them into Octo's
`(B, window, …)` batch structure, and tokenizes language via the model's T5 processor.

---

## 5. Workflow 3 — inference with the finetuned model

Identical inference call to zero-shot, but loading the finetuned checkpoint and **our** FR5
action stats, so `sample_actions` unnormalizes back into FR5 joint space:

```python
model   = OctoModel.load_pretrained("policies/octo/checkpoints", step=5000)
stats   = load("policies/octo/checkpoints/fr5_action_stats.npz")
actions = model.sample_actions(obs, task, unnormalization_statistics=stats, rng=…)
# actions[0,0] → [j1..j6, gripper] in FR5 units
```

For deployment, wrap this the way `common/deploy.py` wraps the torch policies: 30 Hz loop,
maintain a 2-frame history window, call `sample_actions`, send the first action of the chunk
(`action[:6]` joints via `servo_j`, `action[6]` gripper). Octo predicts a 4-step chunk, so
you can also execute it open-loop or with temporal ensembling
(see [`inference.md`](inference.md)).

---

## 6. Why finetune Octo instead of training ACT from scratch?

The whole reason Octo exists: the FR5 dataset is small (~54 episodes), which is exactly the
regime where from-scratch policies fall into the proprioceptive shortcut and fail to
generalize (see [`il_failure_modes.md`](il_failure_modes.md) §4). Octo brings a visual prior
learned from **800k trajectories across many robots** — its image encoder already "knows how
to see" manipulation scenes before it sees a single FR5 frame. Finetuning adapts that prior
to the FR5 instead of learning vision from 54 episodes. Combined with its **state-free**
(vision+language) design, Octo is structurally better positioned to generalize spatially than
a from-scratch state-based ACT — at the cost of a heavier (JAX) stack and a larger model.

---

## 7. Practical notes

- **Model size:** `octo-base-1.5` (93M) is the **largest** official Octo checkpoint — there
  is no `octo-large` (the team released only `octo-small` and `octo-base`). It is the
  **Linux / GPU default** (`config.yaml` and the scripts use it). `octo-small-1.5` (27M) is
  the lighter option for local/CPU testing (`--model hf://rail-berkeley/octo-small-1.5`).
  Both are far smaller than the π0 VLMs.
- **Dependency pins are load-bearing:** numpy `<2`, scipy `1.11`, tensorflow `2.15`, jax
  `0.4.20` form one compatible set (see `setup_env.sh`). Don't upgrade them piecemeal.
- **All three workflows were verified** against `octo-small-1.5` on CPU: zero-shot output
  `(1,4,7)`, a finetune step (loss + gradient update through the bound module), and a
  save→reload→FR5-space `sample_actions` round trip. The Linux GPU box only needs
  `setup_env.sh` (CUDA build) and the real dataset path.
