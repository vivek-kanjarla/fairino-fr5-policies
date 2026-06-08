# Octo — generalist policy, zero-shot + finetuned on the FR5 (the complete guide)

Octo is a **generalist robot policy** from the Octo team (RAIL, Berkeley): a transformer
backbone with a **diffusion action head**, pretrained on **800k trajectories** from the
**Open-X-Embodiment (OXE)** dataset. Unlike ACT / Diffusion Policy / DiT (which this repo
trains *from scratch* on ~54 FR5 episodes), Octo brings a **pretrained visuomotor prior** that
you adapt to the FR5 by **finetuning**.

This is the **main, definitive** Octo document for the repo. It is meant to stand on its own: it
goes deep on *everything* — what Octo is, how it was pretrained, the architecture, the three FR5
workflows (zero-shot / finetune / finetuned inference), the FR5 data adapter, the action-space
and normalization mechanics, real-robot deployment at 30 Hz, the isolated dependency stack, and
troubleshooting. Two companion docs go even deeper on two sub-topics, and this doc points you to
them at the right moments:

- [`octo_model.md`](octo_model.md) — the **deepest architecture internals** (tokenizers,
  block-wise attention, readout tokens, the diffusion head's math).
- [`octo_finetuning.md`](octo_finetuning.md) — the **deepest finetuning treatment**
  (full / head_only / head_mlp_only, `frozen_keys`, the optimizer recipe, and why it is *not*
  LoRA).

Every technical term is defined the first time it appears, in the house style of
[`act.md`](act.md). The runnable code and setup live in
[`policies/octo/`](../policies/octo/README.md); every number below is verified against that code
and the installed Octo source (read-only — no weights were loaded to write this doc).

---

## Table of contents

1. Where Octo sits relative to the other policies
2. Pretraining & the Open-X-Embodiment background (why Octo generalizes)
3. Architecture — a substantial summary (deepest internals → `octo_model.md`)
4. Workflow 1 — zero-shot inference, and the action-space/embodiment mismatch
5. Workflow 2 — finetuning on the FR5 (summary; deepest treatment → `octo_finetuning.md`)
6. The FR5 data adapter — the exact batch structure, normalization, language tokenization
7. Workflow 3 — inference with the finetuned model (back to FR5 joint space)
8. Deployment on the real FR5 — the 30 Hz loop, history window, chunk execution
9. The isolated environment & dependency pins — what, and why
10. Practical guidance & troubleshooting
11. Summary + commands cheat-sheet

---

## 1. Where Octo sits relative to the other policies

This repo trains five PyTorch policies (ACT, Diffusion Policy, DiT-Flow, π0, π0-FAST) through a
shared training loop (`common/train.py`) and a shared deploy loop (`common/deploy.py`). **Octo
is the odd one out**, for one structural reason: it is written in **JAX/Flax**, not PyTorch.

- **JAX** is Google's array-computing library — like NumPy but with automatic differentiation
  and just-in-time (**JIT**) compilation onto GPU/TPU. **Flax** is the neural-network library
  built on JAX (its analogue of PyTorch's `nn.Module`).

A JAX program and a PyTorch program each bring their own copy of the entire numerical stack
(CUDA kernels, BLAS, a pinned NumPy, etc.). Putting both in one Python environment reliably
breaks — the two stacks fight over shared native libraries (see §9). So Octo lives in its **own
isolated virtual environment** (`.venv-octo`) with **its own three standalone scripts** and does
**not** touch `common/train.py` or `common/deploy.py`.

| | ACT / Diffusion / DiT | π0 / π0.5 (lerobot) | **Octo** |
|---|---|---|---|
| Framework | PyTorch (lerobot) | PyTorch (lerobot) | **JAX / Flax** |
| Pretraining | none (scratch on FR5) | VLM-pretrained | **800k Open-X trajectories** |
| Action head | direct / DDPM / flow | flow matching | **diffusion head (DDPM)** |
| Language conditioning | DiT/π only | yes | **yes (frozen T5 tokens)** |
| Proprioceptive state input | yes (`state` token) | yes | **no — vision + language only** |
| In this repo | `common/train.py` loop | lerobot wrapper | **own scripts + `.venv-octo`** |

Two consequences worth holding onto:

1. **Pretrained vs from-scratch.** ACT/Diffusion/DiT start from random weights and must learn
   *everything* — including how to see — from ~54 FR5 episodes. Octo starts from weights that
   already encode a rich visual prior from 800k trajectories. You only *adapt* it. This is the
   single biggest reason to reach for Octo on a tiny dataset (see §2 and §5).
2. **A separate subsystem, by design.** The isolation is not a hack to be cleaned up later — it
   is the correct way to run a JAX model next to a PyTorch pipeline. The cost is that Octo has
   its own setup, its own scripts, and its own deploy story (§8).

> The **architecture-internals** comparison (encoder cores, how inputs enter, multimodality
> handling) lives in [`octo_model.md`](octo_model.md) §8. This section is the
> framework/workflow view.

---

## 2. Pretraining & the Open-X-Embodiment background

The phrase "pretrained on 800k trajectories from Open-X-Embodiment" is the whole reason Octo is
worth using. This section unpacks exactly what it means and *why that diversity gives Octo a
transferable prior*.

### 2.1 What "pretraining" means here

**Pretraining** is the expensive, one-time phase where a model learns general structure from a
huge dataset. The Octo team did this once, on a cluster, and released the resulting weights.
**Finetuning** (§5) is the cheap second phase you run on your own small dataset. You are *not*
learning from zero — you are nudging an already-competent model.

A **trajectory** (a.k.a. an **episode**) is one complete recorded robot attempt at a task: a
time-ordered sequence of `(observation, action)` pairs, e.g. a few seconds of "reach, grasp,
lift, place" sampled at the robot's control rate. 800k of them is on the order of *tens of
millions* of individual `(observation, action)` training frames.

### 2.2 What Open-X-Embodiment is

**Open-X-Embodiment (OXE)** is a community effort that pooled robot-manipulation datasets from
many labs into one mixture with a common format. "X-Embodiment" = **across embodiments**: an
*embodiment* is a specific robot body (a WidowX arm, a Franka, a UR5, a Google robot, …). The
point of OXE is to train *one* policy on *many* robots at once.

Octo's released checkpoints carry **normalization statistics for ~25 source datasets** (a hint
of how broad its training mixture was). The names you will see in `model.dataset_statistics`
include:

| Source dataset (key) | Robot / origin (informally) |
|---|---|
| `bridge_dataset` | WidowX (UC Berkeley BridgeData) |
| `fractal20220817_data` | Google RT-1 robot |
| `kuka` | KUKA arm |
| `taco_play`, `jaco_play` | TACO / Kinova Jaco play data |
| `berkeley_autolab_ur5` | UR5 |
| `berkeley_cable_routing`, `berkeley_fanuc_manipulation` | Berkeley AutoLab / Fanuc |
| `roboturk`, `nyu_door_opening`, `viola`, `toto` | various |
| `language_table` | Google Language-Table |
| `stanford_hydra`, `austin_buds`, `austin_sailor`, `austin_sirius` | Stanford / UT-Austin |
| `cmu_stretch`, `ucsd_kitchen`, `utaustin_mutex` | CMU / UCSD / UT-Austin |
| `iamlab_cmu_pickup_insert`, `furniture_bench` | CMU IAM-Lab / FurnitureBench |
| `dlr_edan_shared_control`, `bc_z`, `nyu_franka_play` | DLR / Google BC-Z / NYU Franka |

That is **many different robots, many different cameras, many different action conventions**, in
one training mixture. The exact list of ~25 keys above is for `octo-small-1.5`; both released
sizes were trained on the same OXE mixture family.

### 2.3 Why diversity → a transferable visual prior

A model trained on a *single* robot in a *single* lab can quietly cheat: it can memorize that
lab's lighting, that table, even the correlation between the robot's starting joint angles and
where the object usually is. Such a model "works" on its own data and falls apart elsewhere.
That exact trap — leaning on spurious shortcuts instead of perception — is the
**proprioceptive shortcut** documented in [`il_failure_modes.md`](il_failure_modes.md) §1.

OXE's diversity is the antidote during pretraining:

- **No single visual shortcut survives.** Across dozens of labs, lighting, backgrounds, camera
  placements, and object sets all vary. The only feature that consistently lowers the
  pretraining loss across *all* of them is **actually perceiving the scene**. So Octo's image
  encoder is forced to learn genuine manipulation-relevant vision — "where is the object, where
  is the gripper, what is the spatial relation" — rather than dataset-specific cosmetics. That
  learned ability to *see* manipulation scenes is the **transferable visual prior**.
- **No single joint-state shortcut survives — and Octo has no joint-state input anyway.** The
  robots in OXE have different numbers of joints, different kinematics, and different action
  conventions, so "echo the proprioceptive state" cannot lower the loss across the mixture. And
  Octo's default observation is **image + language only** — there is *no* proprioceptive
  (joint-angle) input channel at all (see §3 and §5.3). The shortcut is structurally
  unavailable: the model *must* use vision.

This is the crux of the FR5 argument (developed in §5): the FR5 dataset is tiny (~54 episodes),
which is exactly the regime where a from-scratch state-based policy falls into the
proprioceptive shortcut and fails to generalize spatially. Octo's image encoder already "knows
how to see" manipulation before it ever sees an FR5 frame, and Octo never receives joint state —
so finetuning it *adapts a strong prior* instead of *learning vision from 54 episodes*.

### 2.4 The two released sizes

There are exactly two Octo checkpoints — there is **no** "octo-large":

| Spec | `octo-small-1.5` | `octo-base-1.5` | what it is |
|---|---|---|---|
| `token_embedding_size` (width `D`) | **384** | **768** | width of every token vector |
| transformer layers | **12** | **12** | depth (attention+MLP blocks) |
| attention heads | **6** | **12** | parallel attention sub-computations per block |
| MLP (feed-forward) dim | **1536** | **3072** | hidden width of each block's MLP |
| ≈ ViT scale | ViT-Small | ViT-Base | comparable image-transformer size |
| parameters | **≈ 27M** | **≈ 93M** | total weights |
| download size | ~150 MB | **~547 MB** | HuggingFace checkpoint |

`octo-base-1.5` (93M) is the **largest** official checkpoint and the **Linux/GPU default**
(`config.yaml` and the scripts use it). `octo-small-1.5` (27M) is the lighter **local/CPU test**
option (`--model hf://rail-berkeley/octo-small-1.5`). Both are far smaller than the π0-family
VLMs (which are ~2B+). The widths/depths above are confirmed in
[`octo_model.md`](octo_model.md) §5.

---

## 3. Architecture — a substantial summary

This section gives you a complete working mental model of the model. **The deepest internals —
the tokenizer math, the block-wise attention rules, the readout-token mechanism, and the DDPM
head derivation — are in [`octo_model.md`](octo_model.md).** Read that when you want every shape
and every attention rule; read this to understand the shape of the whole thing.

### 3.1 The pipeline, end to end

```
   wrist image (256×256×3)              language ("pick up the block …")
        │                                     │
   ImageTokenizer                        LanguageTokenizer
   SmallStem16 conv stem → patchify      T5 encoder (frozen)
   256→16×16 = 256 image tokens/frame    word-pieces → token vectors
        │                                     │
        └───────────────────┬─────────────────┘
                            ▼
        assemble token groups (window_size = 2):
        [ task prefix | t0: obs + readout | t1: obs + readout ]
                            │
              ┌─────────────▼──────────────────────────┐
              │   Octo transformer (12 layers)         │  + learned readout tokens
              │   block-causal over the 2-frame window │    (readouts={"action":1})
              └─────────────┬──────────────────────────┘
                            │  readout_action embedding (last frame) = c
                            ▼
              ┌────────────────────────────────────────┐
              │  Diffusion action head (MLP score net)  │
              │  noise ──denoise (conditioned on c)──►  │
              │  action chunk  (action_horizon=4, dim=7)│
              └────────────────────────────────────────┘
```

Walking the diagram:

1. **Image tokenizer.** Each camera frame is turned into a grid of **tokens** (a *token* is a
   fixed-size vector of length `D`). A small convolutional **stem** (`SmallStem16`) plus
   **patchify** downsamples a 256×256 image by 16× into a 16×16 grid → **256 image tokens** per
   frame. (The optional `image_wrist` slot is 128×128 → 8×8 = 64 tokens; on the FR5 it is zeroed
   and masked, see §6.) Pixels are passed in as raw `uint8`; the tokenizer normalizes internally.
2. **Language tokenizer.** The instruction string is split into sub-word pieces and run through a
   frozen **T5** text encoder (*Text-To-Text Transfer Transformer*), producing one token vector
   per word-piece. "Frozen" = those weights don't train; Octo treats the language vectors as a
   fixed high-quality representation.
3. **Readout tokens.** Octo's config declares `readouts = {"action": 1}` — one learned,
   input-free **readout token** per timestep. It is a CLS-like "scratchpad": it *reads* the
   scene into a single embedding, and **nothing attends back to it** (so it never disturbs the
   other tokens).
4. **Block-causal transformer.** Tokens are grouped — a shared **task PrefixGroup** plus
   per-timestep **TimestepGroups** (each frame's observation tokens and its readout) — and
   attention is declared by **rules between groups**: causal over a `window_size = 2` history
   window (a frame may attend to the current or earlier frame, never a future one). This is
   **block-causal attention**, the heart of Octo. A 12-layer transformer mixes the tokens.
5. **Diffusion action head.** The transformer's `readout_action` embedding for the **last** frame
   becomes a conditioning vector `c`. The `DiffusionActionHead` is a small **MLP score network**
   (3 residual blocks, hidden 256) that starts from Gaussian noise and **DDPM-denoises** it,
   conditioned on `c`, into a chunk of `action_horizon = 4` actions of `action_dim = 7` numbers
   each. Verified head defaults from the source (`action_heads.py`): `diffusion_steps = 20`
   reverse steps, `n_diffusion_samples = 1`, `max_action = 5.0`, `loss_type = "mse"`,
   mean-pooled readout (`use_map = False`).

### 3.2 The shapes that matter for the FR5

| Quantity | Value | Why it matters for the FR5 |
|---|---|---|
| `window_size` | **2** | the model attends over the current + previous frame; deploy must keep a 2-frame history (§8) |
| `action_horizon` | **4** | each query predicts the next 4 actions (a *chunk*) |
| `action_dim` | **7** | matches the FR5 exactly: **6 joints + gripper** → **no head surgery** to finetune (§5) |
| `image_primary` | (B, 2, 256, 256, 3) uint8 | the FR5 wrist frame, tiled across the window, goes here |
| `image_wrist` | (B, 2, 128, 128, 3) uint8 | unused on the FR5: zeros + masked |
| `timestep_pad_mask` | (B, 2) bool | which of the 2 history slots are real vs padding |

The single luckiest fact in the whole FR5↔Octo story: **`action_dim = 7` already equals
6 FR5 joints + 1 gripper.** Octo's pretrained head outputs exactly the right number of numbers,
so finetuning needs no architectural surgery (§5.1). The *meaning* of those 7 numbers still has
to be remapped from Octo's pretrained action space to FR5 joints — that is what finetuning +
normalization does (§4, §6, §7).

> Deepest dive: tokenizers (§3 of `octo_model.md`), the `AttentionRule` vocabulary and the
> who-attends-to-whom matrix (§4), the transformer block dims (§5), and the DDPM head math (§6).

---

## 4. Workflow 1 — zero-shot inference (no finetuning)

**Goal:** load the Octo team's pretrained weights and run them on an FR5 wrist image + an
instruction, *without any FR5 training*. Code: [`inference_pretrained.py`](../policies/octo/inference_pretrained.py).

### 4.1 The exact call

```python
model = load_octo("hf://rail-berkeley/octo-base-1.5")        # or octo-small-1.5 for CPU
window = model_window_size(model)                            # = 2

task = model.create_tasks(texts=["pick up the block and place it in the bin"])
obs  = {                                                     # one FR5 frame tiled across the window
    "image_primary":     imgs,                               # (1, 2, 256, 256, 3) uint8
    "timestep_pad_mask": np.ones((1, 2), bool),
}
stats = pick_pretrained_action_stats(model)                  # borrows bridge_dataset["action"]
actions = model.sample_actions(obs, task,
              unnormalization_statistics=stats,
              rng=jax.random.PRNGKey(0))                     # → (1, 4, 7)
```

`sample_actions` runs a full forward pass (tokenize → transformer → diffusion head's reverse
loop) and returns a **(1, 4, 7)** array: batch 1, `action_horizon = 4` future steps, `action_dim
= 7` numbers each. The repo script tiles a single 256×256 wrist frame across both history slots
(a "fresh-start" padding — at deploy you would instead keep two *real* consecutive frames).

### 4.2 What `unnormalization_statistics` is doing — and the catch

Octo predicts actions in a **normalized** space (roughly zero-mean, unit-variance per
dimension), because that is how it was trained. To turn the network's normalized output back into
real numbers, `sample_actions` multiplies by a **std** and adds a **mean** — the
`unnormalization_statistics`. Crucially:

> Octo has **no FR5 statistics** zero-shot (it never saw the FR5). The script must borrow a
> *source* dataset's stats — `pick_pretrained_action_stats(model)` returns
> `model.dataset_statistics["bridge_dataset"]["action"]` (falling back to the first available
> dataset). So the returned numbers live in **`bridge_dataset`'s action space**, which is
> **normalized delta end-effector poses** of a **WidowX** robot.

### 4.3 Why zero-shot will NOT drive the FR5 — the mismatch, thoroughly

Zero-shot runs and produces a clean (1, 4, 7) chunk — but those numbers are **not FR5 joint
commands**. There are *two independent mismatches*, and either one alone would break it:

**(a) Action-space mismatch.** Octo's pretrained action space (the OXE/DROID convention) is
**delta end-effector (EEF) pose**: `[Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper]` — *how far to
move the hand* this step, in Cartesian space. The FR5 policies in this repo use **absolute joint
angles**: `[q1_cmd, …, q6_cmd, gripper]` — *the target angle for each motor*. These are
different *kinds* of number with different units and different semantics. A "+0.02 m forward"
delta-EEF value sent to `servo_j` as if it were a joint angle in degrees is meaningless. See
[`action_spaces.md`](action_spaces.md) §1–§2 for the full taxonomy (absolute joint vs delta
joint vs absolute EEF vs delta EEF) and why only relative EEF generalizes spatially.

**(b) Embodiment mismatch.** Even *as* a delta-EEF command, the numbers were unnormalized with
**WidowX (`bridge_dataset`)** statistics — a different arm, different workspace scale, different
camera viewpoint. The FR5's geometry and the FR5 wrist camera were **never in Octo's training
mix**, so the pretrained model has no reason to emit numbers calibrated to *this* arm.

```
   Octo zero-shot output  =  normalized Δ-EEF   ×  bridge_dataset(std)  +  bridge_dataset(mean)
                             └── for a WidowX, delta end-effector, Berkeley camera ──┘
   FR5 needs              =  absolute joint angles (deg) + gripper, for THIS arm + THIS camera
                             └────────────────── never seen in pretraining ──────────────────┘
```

The two action spaces do not even agree on *what each number means*, and the calibration is for
the wrong robot. So **zero-shot will not move the FR5 correctly**. That is expected and fine —
this workflow is the **baseline**: it proves the pretrained model *loads* and *runs end-to-end on
FR5-shaped inputs* (the right input shapes, a real (1,4,7) output), and it gives a reference to
compare the finetuned model against. The mismatch is precisely the **motivation for finetuning**
(§5): finetuning teaches Octo's head to emit **FR5 absolute-joint** actions, and saves **FR5**
normalization stats so the output comes back in real FR5 units (§6, §7).

> The script prints exactly this caveat at the end of every run
> (`"these are Octo-pretrained-space actions, not FR5 joint commands"`).

---

## 5. Workflow 2 — finetuning on the FR5 (summary)

**Goal:** adapt the pretrained model to the FR5 so its head emits FR5 actions. Code:
[`finetune.py`](../policies/octo/finetune.py), configured by
[`config.yaml`](../policies/octo/config.yaml). This section is a solid, self-contained summary;
the **deepest treatment** — every freeze mode, the exact `frozen_keys` patterns, the gradient-
masking mechanism, the full optimizer recipe, and the "Octo is not LoRA" argument — is in
[`octo_finetuning.md`](octo_finetuning.md).

### 5.1 Why finetuning is *direct* (the dims already match)

Finetuning means: load the pretrained weights and keep training them on FR5 batches. Because the
FR5 action is 7-D and Octo's head is `action_dim = 7, action_horizon = 4`, **the shapes line up
exactly — no "head surgery."** `finetune.py` asserts this and stops with a clear message if your
config's `action_dim`/`action_horizon` ever diverge from the pretrained head (it would then tell
you to rebuild the head via `from_config` + `merge_params`). For the FR5, that branch is never
hit.

The per-step training math (verified to run on `octo-small` CPU):

```python
# loss_fn — bind the pretrained module to params, run the transformer, then the head's loss
bound      = model.module.bind({"params": params}, rngs={"dropout": dropout_rng})
embeddings = bound.octo_transformer(batch["observation"], batch["task"],
                                    batch["observation"]["timestep_pad_mask"], train=True)
loss, _    = bound.heads["action"].loss(embeddings, batch["action"],
                                        batch["observation"]["timestep_pad_mask"],
                                        batch["action_pad_mask"], train=True)
# train_step — value_and_grad → apply_gradients (frozen params get zero gradient, see below)
```

The head's `loss` is the **DDPM denoising objective**: flatten the ground-truth chunk to
`4 × 7 = 28` numbers, add noise at a random level, ask the score net (conditioned on the readout
embedding) to predict that noise, and take MSE against the true noise. Same denoising family as
[`diffusion_policy.md`](diffusion_policy.md); full math in [`octo_model.md`](octo_model.md) §6.

### 5.2 The three modes (summary) — Octo does NOT use LoRA

A frequent misconception is that Octo is finetuned with **LoRA** (a low-rank adapter trick). It
isn't. The official recipe freezes parameters **by name pattern** via `frozen_keys` and trains
the rest with ordinary full-rank gradients. The freeze is implemented as **gradient masking**
(an optax partition): matching parameters get a zero update, forever.

| Mode | `frozen_keys` (frozen) | What trains | Use |
|---|---|---|---|
| `head_only` *(our default)* | `octo_transformer.*` | the full head weights | tiny datasets (FR5) — keep the 800k-trajectory visual prior, adapt only the action head |
| `head_mlp_only` | `octo_transformer.*` + head `map_head` probe/attention | only the head's MLP | most conservative |
| `full` | `None` | everything (backbone + head) | Octo paper's default; needs more data + compute |

`finetune.py` uses Octo's **official `create_optimizer`**: cosine LR (peak `3e-4`, linear
warmup), gradient clipping at global-norm `1.0`, AdamW with **no weight decay on biases/LayerNorm**
(the ViT/timm convention), plus the `frozen_keys` for the chosen mode. This is the *exact*
pretrained recipe — no custom optimizer, no LoRA. **Verified:** `head_only`/`head_mlp_only` leave
every `octo_transformer.*` parameter *exactly* unchanged; `full` updates them.

> Deepest dive — the full LoRA-vs-freeze comparison, the optimizer derivation (warmup, cosine
> decay, the weight-decay mask, grad clipping), and per-mode "when to use": all in
> [`octo_finetuning.md`](octo_finetuning.md).

### 5.3 The FR5 design choices (and the proprioceptive-shortcut connection)

- **Default `head_only`.** With ~54 FR5 episodes, freezing the 800k-trajectory transformer and
  adapting only the action head is the most robust choice: a strong frozen feature extractor +
  a head re-mapped to FR5 actions, with the least overfitting. Switch to `--mode full` once you
  have a larger, more varied dataset (§10).
- **Vision + language only — no proprioceptive state.** Octo's default observation is image +
  language; there is *no* joint-state input. This is precisely the **state-free** recipe that
  [`il_failure_modes.md`](il_failure_modes.md) and [`proprioception_modes.md`](proprioception_modes.md)
  recommend to break the **proprioceptive shortcut** (when a state-based policy learns to echo
  its own joint state and ignore the camera, then fails when the scene changes). The PyTorch
  policies get state-free behavior via an opt-in `proprio_mode='none'` switch; **Octo gets it for
  free, by design** — it structurally *cannot* take the shortcut. See
  [`octo_finetuning.md`](octo_finetuning.md) §6.3.
- **Wrist camera → `image_primary`** (256×256); `image_wrist` zeroed + masked (§6).
- **Action normalization** with the dataset's own mean/std, saved as
  `checkpoints/fr5_action_stats.npz` and reused to unnormalize at inference (§6, §7).

### 5.4 Why finetune Octo instead of training ACT from scratch?

The FR5 dataset is small (~54 episodes) — exactly the regime where from-scratch state-based
policies fall into the proprioceptive shortcut and fail to generalize spatially
([`il_failure_modes.md`](il_failure_modes.md) §1, §11). Octo brings a visual prior learned from
**800k trajectories across many robots** — its encoder already "knows how to see" before the
first FR5 frame — *and* it is **state-free** by construction. Finetuning adapts that prior rather
than learning vision from 54 episodes. So finetuned Octo is structurally better positioned to
generalize than a from-scratch state-based ACT — at the cost of a heavier (JAX) stack and a
larger model.

---

## 6. The FR5 data adapter — the exact batch structure

Code: [`fr5_octo_data.py`](../policies/octo/fr5_octo_data.py). This is the bridge between the
repo's **LeRobot v3 dataset** (the same dataset the PyTorch policies train on) and the **Octo
batch structure** the transformer + head expect. It is written deliberately *without* torch or
lerobot, so the JAX venv stays lean (§9).

### 6.1 How it reads the dataset (no torch / lerobot)

The LeRobot v3 dataset is **parquet metadata + extracted JPG frames** (with an MP4 fallback). The
adapter reads it **directly** with `pyarrow` (parquet) and OpenCV/Pillow (images):

- `meta/info.json` — dataset feature schema.
- `data/chunk-000/file-000.parquet` — one row per frame; columns include `action`,
  `frame_index`, `task_index`. Loaded once into a pandas DataFrame (`self.df`).
- `meta/episodes/chunk-000/file-000.parquet` — per-episode bounds (`dataset_from_index`,
  `dataset_to_index`, `episode_index`), used to build the sampling index so windows never cross
  episode boundaries.
- `meta/tasks.parquet` — `task_index → instruction` map (or the config's `instruction` override).
- Frames: `frames/<image_key>/ep-<NNN>/<frame_index:06d>.jpg`, resized to 256×256 for
  `image_primary`. If a JPG is missing it falls back to decoding the episode MP4
  (`videos/.../file-<NNN>.mp4`); if *that* fails too it returns a zeros frame rather than
  crashing — robust missing-frame handling.

### 6.2 Windowing — frames → Octo's `(B, window, …)` structure

Octo reads a **window** of `window_size = 2` frames, and for each frame a chunk of
`action_horizon = 4` future actions. The adapter builds one *example* by:

1. Picking a start frame `s` inside some episode `[ep_from, ep_to)`. Every frame can start a
   window (the index is built over all frames).
2. For each of the `W = 2` window slots `w`: take frame `s + w` (clamped to the episode's last
   frame if it would run past the end), and mark whether that slot is *real* (`valid = (s+w) <
   ep_to`) or padding. The image is resized to 256×256.
3. For each window slot, gather the next `H = 4` actions (`s+w` … `s+w+3`). Each present action
   is **normalized** (§6.4) and marked valid; any action past the episode end is filled with
   zeros and marked invalid (so the loss masks it out).

A batch of `B` examples is stacked. The wrist slot is filled with zeros (FR5 has one camera) and
masked. Language is emitted as raw **byte strings**, tokenized later (§6.5).

### 6.3 The exact batch — every key, shape, dtype

This is the table to keep next to the code. `B` = batch size, `W = window_size = 2`,
`H = action_horizon = 4`, `dim = action_dim = 7`.

| Key | Shape | Dtype | Meaning |
|---|---|---|---|
| `observation.image_primary` | (B, W, 256, 256, 3) | uint8 | FR5 wrist frame → Octo's primary camera slot |
| `observation.image_wrist` | (B, W, 128, 128, 3) | uint8 | **zeros** — FR5 has no 2nd camera; masked out |
| `observation.timestep_pad_mask` | (B, W) | bool | True where the window slot is a real frame (not padding) |
| `observation.pad_mask_dict.image_primary` | (B, W) | bool | primary present where timestep valid |
| `observation.pad_mask_dict.image_wrist` | (B, W) | bool | **all False** — wrist never present |
| `observation.pad_mask_dict.timestep` | (B, W) | bool | mirrors `timestep_pad_mask` |
| `task.language_instruction` | (B,) | bytes | raw UTF-8 instruction bytes (tokenized by `process_text`) |
| `action` | (B, W, H, 7) | float32 | **normalized** (mean/std) future-action chunks |
| `action_pad_mask` | (B, W, H, 7) | bool | True where the action is real (not past-episode padding) |

The **masks** are what let Octo train on short/edge windows safely: `timestep_pad_mask` tells the
transformer which history frames are real, `pad_mask_dict.image_wrist` (all False) tells it to
ignore the zeroed wrist camera entirely, and `action_pad_mask` zeroes the loss on any action that
fell past the end of an episode.

### 6.4 Normalization & the `action_stats` dict

Octo expects **normalized** actions during training and **unnormalizes** them at inference.
Octo's `NormalizationType.NORMAL` means **mean/std** (z-score) normalization:

```
normalized   = (action − mean) / std            # in fr5_octo_data._norm_action
unnormalized = normalized × std + mean          # in sample_actions at inference
```

The adapter computes the stats over **all** frames of the FR5 dataset (`octo_common.action_stats`):

```python
action_stats(actions) -> {
    "mean": actions.mean(0),            # per-dim mean (7,)
    "std":  actions.std(0).clip(1e-6),  # per-dim std (7,), clipped away from 0
    "min":  actions.min(0),             # per-dim min (7,)
    "max":  actions.max(0),             # per-dim max (7,)
    "mask": ones(7, dtype=bool),        # unnormalize EVERY dim (incl. the gripper)
}
```

- `mean`/`std` drive the z-score. `std` is clipped to `≥ 1e-6` so a constant dimension never
  divides by zero.
- `min`/`max` are stored for completeness (some Octo normalization types use bounds; NORMAL uses
  mean/std).
- `mask` is all-True: every dimension — including the gripper — is unnormalized at inference. (If
  a dimension's `mask` were False, `sample_actions` would pass it through unchanged.)

These stats are the **bridge back to FR5 units**. `finetune.py` saves them next to the checkpoint
as `fr5_action_stats.npz`; `inference_finetuned.py` loads them and hands them to `sample_actions`
as `unnormalization_statistics`, so the output lands in real FR5 joint angles + gripper (§7).
Contrast zero-shot, which borrowed a *source* dataset's stats (§4.2) — the stats are exactly what
make finetuned output FR5-correct and zero-shot output not.

### 6.5 Language tokenization

The adapter emits the instruction as **raw bytes** in `task.language_instruction`. The actual
tokenization happens through the model's own text processor:

```python
from octo.utils.train_utils import process_text
batch = process_text(raw_batch, model.text_processor)   # bytes → T5 token ids in batch["task"]
```

`process_text` runs Octo's `text_processor` (the **T5** tokenizer, §3.1 step 2): it splits each
instruction into sub-word pieces and writes integer token ids into the batch, which the model's
frozen T5 encoder then turns into language token vectors. Keeping tokenization inside the model's
own processor guarantees the FR5 path uses *exactly* the same vocabulary and conventions as
pretraining. Both `finetune.py` (every step) and `FR5OctoData.example_batch` call `process_text`.

---

## 7. Workflow 3 — inference with the finetuned model

**Goal:** load the finetuned checkpoint + the saved FR5 stats and produce actions **in FR5
space**. Code: [`inference_finetuned.py`](../policies/octo/inference_finetuned.py).

The inference call is *identical* to zero-shot (§4) with two changes: load the **finetuned**
checkpoint, and pass **FR5** stats:

```python
model = load_octo("policies/octo/checkpoints", step=5000)        # finetuned weights (orbax)
stats = load_stats("policies/octo/checkpoints/fr5_action_stats.npz")  # FR5 mean/std/min/max/mask
task  = model.create_tasks(texts=["pick up the block and place it in the bin"])
obs   = build_observation(rgb, window)                            # (1, 2, 256, 256, 3) + mask

actions = model.sample_actions(obs, task,
              unnormalization_statistics=stats, rng=key)          # → (1, 4, 7) in FR5 units
a0 = actions[0, 0]                                                # the next action to send
joints, gripper = a0[:6], a0[6]                                   # 6 FR5 joints (deg) + gripper
```

Because `stats` is now the FR5 dataset's own mean/std, `sample_actions` unnormalizes the head's
output back into **FR5 joint space**: `a0[:6]` are the six joint targets, `a0[6]` is the gripper
command. The script prints `joints=… gripper=…` per step. This is the **deployable** model — the
thing you wrap in a control loop (§8).

```
   finetuned head → normalized 7-vector × FR5_std + FR5_mean → [j1..j6 (deg), gripper]
                                                                └── real FR5 commands ──┘
```

If `fr5_action_stats.npz` is missing next to the checkpoint, the script stops with a clear error
(a checkpoint must have been written by `finetune.py`, which always saves the stats alongside the
weights).

---

## 8. Deployment on the real FR5

Octo is **not** in `common/deploy.py` (that loop is PyTorch). To run finetuned Octo on the real
arm you wrap `inference_finetuned.py`'s call the *same way* `common/deploy.py` wraps the torch
policies — a **30 Hz control loop** in the JAX venv.

### 8.1 The two clocks (same as every policy)

```
FR5 servo loop  — 1000 Hz — hardware. holds/interpolates to the last joint target every 1 ms.
policy loop     —   30 Hz — ours.     every ~33 ms: grab frame → sample_actions → send a target.
```

We run at **30 Hz** because the wrist camera produces 30 frames/s — no point predicting faster
than new images arrive. Between our 33 ms commands the FR5's own 1000 Hz servo loop keeps the arm
moving smoothly. The full two-clock model is in [`inference.md`](inference.md) §1 and
[`act.md`](act.md) §8.0.

### 8.2 The 30 Hz loop with Octo's 2-frame history window

Octo's `window_size = 2` means **you must feed it the current frame *and* the previous frame**.
Maintain a small rolling buffer of the last two wrist frames:

```
on start:                          history = deque(maxlen=2)   # the 2-frame window
each tick (every ~33 ms):
    rgb  = grab_wrist_frame()                    # (H,W,3) uint8 from the D405
    history.append(resize_primary(rgb))          # → 256×256
    frames = list(history)
    if len(frames) == 1:                         # first tick: tile the single frame
        frames = [frames[0], frames[0]]
    obs = {
        "image_primary":     np.stack(frames)[None],   # (1, 2, 256, 256, 3) uint8
        "timestep_pad_mask": np.ones((1, 2), bool),
    }
    chunk = np.asarray(model.sample_actions(obs, task,
                unnormalization_statistics=fr5_stats, rng=key))   # (1, 4, 7) FR5 space
    a0 = chunk[0, 0]
    robot.servo_j(a0[:6].tolist())               # 6 joint targets (deg)
    set_gripper(float(a0[6]))                     # gripper command
```

The first action of the 4-step chunk (`a0`) is the command for *this* tick. Octo predicts a chunk
(`action_horizon = 4`), so — exactly as with the other policies — you have a choice of how to
*consume* it:

### 8.3 How to consume the 4-step chunk

| Strategy | What you do | Trade-off |
|---|---|---|
| **Re-query every step** (send only `a0`) | call `sample_actions` each tick, send the first action, discard the rest | freshest observation every tick; pays the full forward pass at 30 Hz |
| **Open-loop chunk** | call once, execute all 4 actions blindly, then re-query | cheapest (one call per 4 ticks); small jerk at the chunk boundary |
| **Action queue / receding horizon** | execute the first `n_action_steps` of the chunk, then re-query | the diffusion/flow policies' default; balances cost vs reactivity |
| **Temporal ensembling (TE)** | re-query every tick, blend overlapping chunk predictions for "now" with exponentially decaying weights `w(age)=exp(−m·age)` | smoothest; removes the boundary jerk; pays the full pass every tick |

The chunk is only 4 steps (`4/30 ≈ 0.13 s`), so the open-loop window is short and re-querying
every tick is cheap relative to the long-horizon policies. The math of **temporal ensembling** is
explained once in [`act.md`](act.md) §8 and [`inference.md`](inference.md) §3 — it is action-space
agnostic and applies to Octo's chunk exactly as to ACT's. Note Octo's actions here are **absolute
joint angles**, so TE averages absolute configurations and you send the blended `q` straight to
`servo_j`; if you later switch Octo to a delta-EEF action space you would average deltas and add
to the current EEF first (see [`action_spaces.md`](action_spaces.md) §3).

### 8.4 The 30 Hz budget — can Octo keep up?

Each `sample_actions` call runs the 12-layer transformer **plus the diffusion head's 20-step
reverse loop** (`diffusion_steps = 20`). On a GPU this is comfortably under the ~33 ms budget; on
CPU it is far slower (fine for the offline `inference_*` scripts, not for real-time control). This
is the same lesson as the other denoising policies in [`inference.md`](inference.md) §5: anything
with a denoising loop wants a GPU to hold 30 Hz. Verify on the deploy machine before trusting the
loop.

---

## 9. The isolated environment & dependency pins

Octo runs in an **isolated Python 3.10 virtual environment**, `.venv-octo`, built by
[`setup_env.sh`](../policies/octo/setup_env.sh). This section explains *what* is pinned and
*why* — the pins are load-bearing.

### 9.1 Why isolation is necessary

The repo's main environment is a **PyTorch/CUDA** stack (lerobot, the five torch policies). Octo
is a **JAX/Flax** stack. Both want to own the same scarce native resources:

- **JAX vs PyTorch CUDA.** Each ships its own CUDA/cuDNN runtime and GPU memory allocator.
  Co-installing them in one env routinely produces version clashes and double-initialization of
  the GPU.
- **`numpy < 2` vs the rest.** TensorFlow 2.15 (a transitive Octo dependency, used by `dlimp`/the
  data tooling) and this JAX pinset require **NumPy 1.x** (`1.26.4`). NumPy 2 breaks them. The
  PyTorch side may prefer a newer NumPy — so the two cannot share one NumPy.
- **The `ml-dtypes` / `tensorstore` / `orbax` knot.** JAX, `tensorstore` (the checkpoint
  storage backend), and `orbax-checkpoint` all depend on tight, mutually-constrained versions of
  low-level libs like `ml-dtypes`. Mixing them with PyTorch's transitive deps tends to resolve to
  an incompatible set that *imports* but crashes at runtime.

Isolation makes these problems disappear: each subsystem gets the exact stack it was verified
against, and they never meet.

### 9.2 The pinned stack (one compatible set)

`setup_env.sh` installs JAX **first** (so the right `jaxlib` wins), then Octo + `dlimp` (its data
helper, from git), then the pinned compatible stack:

| Package | Pin | Role / why pinned |
|---|---|---|
| `jax` (+`jaxlib`) | **0.4.20** | the core array+autodiff engine; everything else is pinned to match it |
| `flax` | **0.7.5** | neural-net library (the Octo modules) — API-compatible with jax 0.4.20 |
| `optax` | **0.1.7** | optimizers — `create_optimizer`, the freeze partition, AdamW + cosine schedule |
| `distrax` | **0.1.5** | probability distributions used inside Octo |
| `chex` | **0.1.85** | jax testing/asserts utility, version-locked to the above |
| `orbax-checkpoint` | **0.4.0** | checkpoint save/load (the finetuned `checkpoints/<step>/`) |
| `tensorstore` | **0.1.45** | storage backend orbax uses; the `ml-dtypes` knot pins it here |
| `tensorflow` | **2.15.0** | pulled in by the data tooling; needs NumPy 1.x |
| `tensorflow-probability` | **0.23.0** | matched to TF 2.15 |
| `scipy` | **1.11.4** | matched to NumPy 1.26 |
| `numpy` | **1.26.4** (`<2`) | hard requirement of TF 2.15 and this JAX set |
| `transformers` | **4.34.1** | provides the frozen **T5** text encoder (`FlaxT5EncoderModel`) |
| `dlimp` (git) | — | Octo's data-loading helper |
| `einops`, `ml_collections` | — | tensor-reshape + config utilities Octo uses |

The FR5 adapter's lean extras (`pyarrow`, `opencv`, `pandas`, `pillow`) come from
`requirements-octo.txt` — just enough to read the LeRobot dataset without torch/lerobot.

### 9.3 Building it

```bash
# Linux + NVIDIA GPU (CUDA 12) — default, recommended for finetuning:
bash policies/octo/setup_env.sh
# CPU-only (any platform; fine for inference, slow for finetuning):
bash policies/octo/setup_env.sh cpu
source .venv-octo/bin/activate
```

The CUDA-12 build is the default; `cpu` is the fallback. On first use the Octo team weights
download automatically from HuggingFace (`octo-base-1.5` ~547 MB, or `octo-small-1.5`). The
script ends by verifying `from octo.model.octo_model import OctoModel` imports cleanly.

> **Do not upgrade these pins piecemeal.** `numpy<2`, `scipy 1.11`, `tensorflow 2.15`, and
> `jax 0.4.20` form **one** verified compatible set. Bumping any one in isolation is the most
> common way to break `.venv-octo`.

---

## 10. Practical guidance & troubleshooting

### 10.1 Small vs base — which checkpoint?

| Situation | Use | Why |
|---|---|---|
| Linux GPU box, real finetune/deploy | **`octo-base-1.5`** (93M) | the default; largest, best prior; fits a modest GPU |
| Laptop/CPU smoke test | **`octo-small-1.5`** (27M) | imports and runs CPU-feasibly; all 3 workflows were verified on it |
| Memory-constrained finetune | `octo-small-1.5` | a third of the params; far less GPU memory |

There is **no** `octo-large` — `octo-base-1.5` is the ceiling.

### 10.2 Which finetune mode? (the `head_only` default)

- **Default `head_only`** for the ~54-episode FR5 set: freeze the 800k-trajectory backbone, train
  only the head. Strongest prior preservation, least overfitting, trains in minutes.
- **Go `--mode full`** only once you have a **larger, more varied dataset** (e.g. fixed-home +
  workspace-grid coverage, per [`il_failure_modes.md`](il_failure_modes.md) §11). More data lets
  the backbone safely adapt to the FR5's camera/scene without overfitting.
- **`head_mlp_only`** is the most conservative middle ground — try it if even `head_only` shows
  overfitting. Full per-mode guidance: [`octo_finetuning.md`](octo_finetuning.md) §6.

### 10.3 Common pitfalls

| Symptom | Likely cause | Fix |
|---|---|---|
| `numpy`-related import crash in `.venv-octo` | a NumPy 2.x leaked in | pin back to `numpy==1.26.4`; never upgrade pins piecemeal (§9) |
| Frames load as black/zeros | missing JPGs *and* missing/short MP4 | check `frames/<image_key>/ep-*/` exist; verify `dataset.image_key` matches the dataset (`observation.images.wrist_cam`) |
| Finetune stops: "pretrained head (…) != config (…)" | you changed `action_dim`/`action_horizon` | keep them at `7`/`4` (FR5 matches Octo's head), or do head surgery via `from_config`+`merge_params` |
| Zero-shot output looks like noise / wrong scale | **expected** — it's pretrained-space (Δ-EEF, WidowX stats), not FR5 (§4) | finetune; zero-shot is a load/run baseline only |
| Finetuned arm reaches the same spot regardless of object | too few/biased episodes (start-pose leak) | fix the *data* first: fixed home + varied object (`il_failure_modes.md` §11) — vision must be the only way to lower the loss |
| Arm grasps mid-air / misses by a constant offset | object out of camera coverage, or insufficient finetune | check wrist-camera coverage of the object throughout the trajectory; collect more data; consider delta-EEF action space (`action_spaces.md`) |
| `inference_finetuned.py`: "missing fr5_action_stats.npz" | checkpoint not written by `finetune.py` | finetune writes the stats next to the weights; re-run finetune or point `--checkpoint` at the right dir |
| CPU inference too slow for 30 Hz | denoising loop (20 steps) on CPU | deploy on GPU (`setup_env.sh` CUDA build); CPU is fine only for the offline scripts (§8.4) |

### 10.4 Things to keep true (invariants)

- **`action_dim = 7`, `action_horizon = 4`, `window_size = 2`** — these match the FR5 and the
  pretrained head; changing them triggers head surgery.
- **Octo is vision + language only** — there is no proprioceptive state input. Don't try to add a
  state channel; the whole point is to stay state-free (§5.3).
- **Save & reload the FR5 stats with the checkpoint** — they are what put inference output back in
  FR5 units. Verified: save → reload → FR5-space `sample_actions` round-trips correctly.

---

## 11. Summary + commands cheat-sheet

**Summary.** Octo is a JAX/Flax generalist policy — a block-causal transformer + a DDPM diffusion
action head — pretrained on 800k Open-X-Embodiment trajectories across ~25 datasets and many
robots. That diversity gives it a transferable visual prior and (being vision+language only) no
joint-state shortcut. On the FR5 it runs three workflows: **zero-shot** (loads and runs, but
outputs Octo's pretrained-space Δ-EEF actions calibrated to a *source* robot, so it won't drive
the FR5 — a baseline that motivates finetuning); **finetune** (direct, since `action_dim=7`
already matches 6 joints + gripper — default `head_only` freezes the backbone and adapts only the
head, normalizing FR5 actions with their own mean/std saved as `fr5_action_stats.npz`); and
**finetuned inference** (the same `sample_actions` call, now unnormalizing back into real FR5
joint space — the deployable model). The FR5 data adapter reads the LeRobot v3 dataset directly
(pyarrow + opencv, no torch) into Octo's `(B, window, …)` batch with masked padding. Deployment is
a 30 Hz loop holding a 2-frame history window, sending the first action of each 4-step chunk
(open-loop or temporally ensembled). It all lives in an isolated `.venv-octo` whose pins
(`jax 0.4.20`, `tf 2.15`, `numpy<2`, …) form one verified compatible set.

**Deepest dives:** architecture internals → [`octo_model.md`](octo_model.md); finetuning →
[`octo_finetuning.md`](octo_finetuning.md). Related: [`action_spaces.md`](action_spaces.md),
[`il_failure_modes.md`](il_failure_modes.md), [`proprioception_modes.md`](proprioception_modes.md),
[`inference.md`](inference.md), [`act.md`](act.md), [`diffusion_policy.md`](diffusion_policy.md).

**Commands** (run inside `.venv-octo`; consistent with
[`policies/octo/README.md`](../policies/octo/README.md)):

```bash
# ── one-time setup ───────────────────────────────────────────────────────────
bash policies/octo/setup_env.sh            # Linux + CUDA 12 (default)
bash policies/octo/setup_env.sh cpu        # CPU-only (slow finetune)
source .venv-octo/bin/activate

# ── Workflow 1: zero-shot (loads + runs; NOT FR5-correct — a baseline) ───────
python policies/octo/inference_pretrained.py                       # random frame
python policies/octo/inference_pretrained.py --image frame.jpg     # a real wrist frame
python policies/octo/inference_pretrained.py --model hf://rail-berkeley/octo-small-1.5

# ── Workflow 2: finetune on the FR5 ─────────────────────────────────────────
python policies/octo/finetune.py --config policies/octo/config.yaml   # default: head_only
python policies/octo/finetune.py --mode full                          # train everything
python policies/octo/finetune.py --mode head_mlp_only                 # most conservative
python policies/octo/finetune.py --mode head_only --steps 50 \
    --batch-size 4 --save-dir /tmp/octo_fr5                           # quick smoke test

# ── Workflow 3: inference with the finetuned checkpoint (FR5 space) ──────────
python policies/octo/inference_finetuned.py \
    --checkpoint policies/octo/checkpoints --step 5000
```

> **Verified this session** (on `octo-small-1.5`, CPU, read-only — no weights loaded to *write*
> this doc): zero-shot output `(1,4,7)`; one finetune step (loss + gradient update through the
> bound module); the three modes' freeze behavior (`head_only`/`head_mlp_only` leave the
> transformer unchanged, `full` updates it); and a save → reload → FR5-space `sample_actions`
> round trip. The Linux GPU box only needs `setup_env.sh` (CUDA build) and the real dataset path.
