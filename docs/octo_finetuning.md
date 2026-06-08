# Finetuning Octo — full vs head-only vs LoRA (a complete, from-the-ground-up explainer)

This document answers one practical question completely: **when you finetune Octo on the FR5,
what actually gets trained — and is it LoRA?** Every technical term is defined the first time it
appears. Every equation lists its symbols and explains in plain English what it does and why.

The short answer up front, so nothing below surprises you:

> **Octo is NOT finetuned with LoRA.** Octo's official recipe *freezes parameters by name pattern*
> (`frozen_keys`) and trains the rest with full-rank gradient updates. Its "lightweight" option —
> `head_only` — means *freeze the whole transformer backbone, train the full action-head weights*.
> That is a different mechanism from LoRA (which adds a small low-rank update to frozen weights).
> Both are valid ways to adapt cheaply; Octo simply chose freezing.

This is the deep-dive companion to [`octo.md`](octo.md) (which covers the architecture and the
three FR5 workflows). Here we zoom in on **finetuning** specifically. The code being documented is
[`policies/octo/finetune.py`](../policies/octo/finetune.py), configured by
[`policies/octo/config.yaml`](../policies/octo/config.yaml).

---

## Table of contents

0. What "finetuning" means, from scratch
1. The finetuning spectrum — full / head-only / LoRA / other PEFT
2. LoRA explained properly (the `W + (α/r)·B·A` math)
3. What Octo actually does — the three `frozen_keys` modes
4. The optimizer recipe — cosine LR, warmup, AdamW, the weight-decay mask, grad clipping
5. Why Octo doesn't need LoRA (balanced) — and how you'd add it anyway
6. Practical guidance for the FR5
7. Commands recap

---

## 0. What "finetuning" means, from scratch

**Pretraining** is the expensive, one-time phase where a model learns general structure from a
huge dataset. Octo was *pretrained* by the Octo team (RAIL, Berkeley) on **800k trajectories**
from the Open-X-Embodiment dataset — a mix of many robots, cameras, and tasks. The result is a set
of learned numbers (**weights**, also called **parameters**) that already "know how to see"
manipulation scenes and produce reasonable robot actions.

**Finetuning** is the much cheaper second phase: you take those pretrained weights and continue
training them on *your* small dataset (here, ~54 FR5 teleop episodes) so the model adapts to your
robot, your camera, and your action space. You are not learning from zero — you are *nudging* an
already-competent model.

To talk about *what* gets nudged, split the model into two parts. Octo is a **transformer**
(a neural network built from attention layers — see [`act.md`](act.md) §2.3 if "attention" is new)
followed by a **diffusion action head**:

```
   image + language  ─►  ┌────────────────────────────┐ ─►  ┌──────────────────┐ ─► action chunk
                         │   Octo transformer          │     │  diffusion head  │    (horizon 4,
                         │   "the BACKBONE"            │     │  "the HEAD"      │     dim 7)
                         │   params: octo_transformer.*│     │  params: heads_*  │
                         └────────────────────────────┘     └──────────────────┘
                          ~90%+ of the parameters            small fraction of params
```

- **Backbone** = the big shared feature extractor. Its job is to turn raw inputs (images, language
  tokens) into rich internal representations (**embeddings** — fixed-size vectors that summarize the
  input). In Octo every backbone parameter's name starts with the prefix `octo_transformer.`.
- **Head** = the small task-specific output network bolted on top of the backbone. Octo's head is a
  **diffusion head**: it reads the backbone's "readout" embeddings and *denoises* a noise sample into
  a 7-D action chunk (same denoising idea as Diffusion Policy — see
  [`diffusion_policy.md`](diffusion_policy.md)). Its parameters' names start with `heads_`.

The whole finetuning question is: **of {backbone, head}, which weights do we let move?** That single
choice is the "finetuning spectrum."

---

## 1. The finetuning spectrum

There is a continuum from "change nothing" to "change everything." Four landmarks on it:

### 1.1 Full finetuning

Update **all** weights — backbone and head. Maximum capacity to adapt: the model can re-shape even
its lowest-level visual features to your data. Costs:

- **Most compute and memory.** Every parameter needs a gradient (its training signal) plus optimizer
  state (extra bookkeeping numbers per parameter — see §4 on AdamW).
- **Most overfitting risk on small data.** **Overfitting** = the model memorizes the few training
  examples instead of learning a general rule, so it does great on the training set and poorly on
  anything new. With only ~54 episodes, full finetuning can wash out the valuable 800k-trajectory
  prior.

### 1.2 Head-only finetuning (a.k.a. "linear probing")

**Freeze** the backbone — *frozen* means its weights are held fixed, receiving zero update — and
train **only the head**. The backbone becomes a fixed feature extractor; you only learn how to map
its (already good) features to *your* action space.

The classic version of this is called **linear probing**: freeze the backbone and train a single
linear (output) layer on top. The name comes from "probing" what information a frozen representation
already contains by seeing how well a trivial linear readout can use it. Octo's `head_only` is the
same idea but the trainable head is a full diffusion head, not just one linear layer — so it's
"head-only finetuning," a slightly richer cousin of pure linear probing.

Costs: cheap (few trainable params, trains in minutes), robust on small data (you can't damage the
pretrained features because they don't move). The bet you're making: **the frozen features are
already good enough** for your task. For Octo on the FR5 that bet is strong — the features came from
800k robot trajectories.

### 1.3 LoRA (Low-Rank Adaptation)

A **PEFT** method. *PEFT* = **Parameter-Efficient FineTuning**: a family of techniques that adapt a
big model by training only a *tiny* number of new parameters while keeping the original weights
frozen. LoRA freezes each pretrained weight matrix and learns a small **low-rank** "update" beside
it (full math in §2). Very few trainable parameters; the original weights are untouched and the
update can be folded back in at the end so inference has zero extra cost.

### 1.4 Other PEFT (one line each, for completeness)

- **Adapters** — insert small new bottleneck layers between the frozen layers; train only those.
- **Prefix / prompt tuning** — prepend a handful of trainable "virtual token" vectors to the input;
  the frozen model conditions on them. The model weights never change at all.

### 1.5 Comparison table

| Method | What trains | Trainable params | Memory | Overfit risk (small data) | Adds NEW params? | Mergeable into W? |
|---|---|---|---|---|---|---|
| **Full** | everything | 100% | highest | highest | no | n/a |
| **Head-only / linear probe** | head (or just final layer) | small (head size) | low | low | no | n/a (head stays separate) |
| **LoRA** | low-rank `A`,`B` beside frozen W | tiny (`r·(d+k)` per matrix) | low | low–medium | yes (then foldable) | **yes** — `W ← W + (α/r)BA` |
| **Adapters** | inserted bottleneck layers | small | low | low | yes | no (extra layers stay) |
| **Prefix/prompt** | a few virtual-token vectors | tiniest | lowest | low | yes (tokens) | no |

**Octo uses the first two columns of behavior (full or head-only).** It does *not* use LoRA,
adapters, or prefix tuning. The rest of this doc makes that precise.

---

## 2. LoRA explained properly (so you know exactly what Octo is NOT doing)

Many people assume "finetune a foundation model" means "LoRA," so it's worth understanding LoRA
fully — then the contrast with Octo is sharp.

### 2.1 The setup

Inside any transformer, most of the heavy lifting is done by **weight matrices**. A weight matrix
`W` of shape `d × k` maps a `k`-vector input to a `d`-vector output (`output = W · input`, the linear
layer from [`act.md`](act.md) §2.1). A big model has hundreds of these, some very large.

Ordinary finetuning replaces `W` with `W + ΔW`, where `ΔW` (the **update**, same shape `d × k`) is
everything finetuning changed. Note: `ΔW` has exactly as many numbers as `W` — `d·k` of them. For a
huge model that's billions of trainable numbers.

### 2.2 The key empirical observation

LoRA's founding insight (Hu et al., 2021): **the update `ΔW` learned during finetuning is
empirically low-rank.** *Rank* is the number of truly independent directions in a matrix — the
"intrinsic dimensionality" of the information it carries. A `d × k` matrix can have rank up to
`min(d,k)`, but a *low-rank* one (rank `r ≪ min(d,k)`) is highly redundant: it can be written as the
product of two skinny matrices. Intuitively, adapting a pretrained model to a new task doesn't
require re-learning everything — it only needs a few new "directions" of change. So `ΔW`, even though
it's shaped `d × k`, really only contains `r` directions' worth of information.

### 2.3 The decomposition

If `ΔW` is rank `r`, you can write it exactly as a product of two small matrices:

```
   ΔW   =   B · A
 (d×k)    (d×r)(r×k)
```

- `A` has shape `r × k` (a "down-projection": `k` numbers → `r` numbers),
- `B` has shape `d × r` (an "up-projection": `r` numbers → `d` numbers),
- `r` is the **rank** you choose — small, e.g. `r = 8`.

LoRA freezes `W` and trains only `A` and `B`. The layer computes:

```
   output  =  W·x  +  (α / r) · B · (A · x)
              └─┬─┘     └──────────┬────────┘
            frozen path        trainable low-rank path
```

written as a weight update:

```
   W_effective  =  W  +  (α / r) · B · A
```

- `α` (alpha) is a fixed **scaling constant** chosen by you; `α/r` keeps the size of the update
  stable when you change `r`. It is *not* trained.
- Only `A` and `B` carry gradients.

### 2.4 Why this saves so much

Trainable parameter count drops from `d·k` (full) to `r·(d + k)` (LoRA). Concretely, for a
`1024 × 1024` matrix with `r = 8`:

- full: `1024 · 1024 = 1,048,576` trainable numbers,
- LoRA: `8 · (1024 + 1024) = 16,384` trainable numbers — about **64× fewer**.

### 2.5 Two details that make it work in practice

1. **Initialize `B = 0`** (and `A` random). Then at step 0, `B·A = 0`, so `W_effective = W` exactly —
   training *starts from the pretrained model* and can only improve from there. No initial shock.
2. **Inference merge (mergeable).** When done, compute `W ← W + (α/r)·B·A` once and discard `A`,`B`.
   The model is now a normal model with no extra layers — **zero added inference latency**. This is
   why the table marks LoRA "mergeable."

### 2.6 ASCII picture of the low-rank path

```
        x  (k-dim input)
        │
   ┌────┴───────────────────────────────┐
   │                                     │
   ▼ frozen                              ▼ trainable
  W·x  (d-dim)                       A·x  (squeezes k → r, the "bottleneck")
   │                                     │
   │                                 B·(A·x)  (expands r → d)
   │                                     │
   │                              ×(α/r) scale
   └──────────────── + ──────────────────┘
                     │
                     ▼
              output (d-dim)
```

The whole point: information flows through a tiny `r`-wide **bottleneck** (the squeeze to `r`
dimensions), so only `r·(d+k)` numbers must be learned.

**Hold this picture.** In §3 you'll see Octo has *none* of it — no `A`, no `B`, no bottleneck, no
merge step.

---

## 3. What Octo actually does — freezing by name, not low-rank updates

Octo's official finetuning recipe lives in `octo-models/octo` at
`scripts/configs/finetune_config.py`. Our [`policies/octo/finetune.py`](../policies/octo/finetune.py)
mirrors it exactly. The mechanism is **freezing parameters by name pattern**, controlled by a
setting called `frozen_keys`, in three modes.

### 3.1 The three modes (the `FROZEN_KEYS` dict)

From `finetune.py`:

```python
FROZEN_KEYS = {
    "full": None,
    "head_only": ("octo_transformer.*",),
    "head_mlp_only": (
        "octo_transformer.*",
        "heads_*.map_head.probe",
        "heads_*.map_head.MultiHeadDotProductAttention_0.*",
    ),
}
```

| Mode | `frozen_keys` (frozen patterns) | What is FROZEN | What TRAINS |
|---|---|---|---|
| `full` | `None` | nothing | **all** weights — backbone + head |
| `head_only` *(FR5 default)* | `octo_transformer.*` | the entire transformer backbone | **the full head weights** |
| `head_mlp_only` | `octo_transformer.*` + the head's `map_head` probe & attention | backbone **and** the head's attention/probe | only the head's **MLP** (final feed-forward) |

A few definitions for that last row: the head's `map_head` is the small attention block that
*reads out* (maps) the backbone embeddings into the head; its `probe` is a learned query vector that
does the reading; the **MLP** (multi-layer perceptron) is the final stack of linear layers that
produces the output. `head_mlp_only` freezes everything except that final MLP — the most
conservative option, the fewest trainable params.

### 3.2 The freeze mechanism — gradient masking (optax partition)

How does naming a parameter actually stop it from training? **Gradient masking.** During training,
every parameter normally gets a **gradient** (the number telling the optimizer how to nudge it). To
freeze a parameter you simply **set its gradient to zero** before the optimizer step — zero gradient
means zero update, forever. The parameter stays *exactly* at its pretrained value.

Octo implements this with **optax** (the JAX optimizer library) using a **partition**: it splits all
parameters into two groups by matching their names against `frozen_keys` — matching names get a
"freeze" optimizer (always zero update), the rest get the real AdamW optimizer. That's it.

```
   parameter name                         matches frozen_keys?     optimizer applied
   ─────────────────────────────────      ────────────────────     ─────────────────
   octo_transformer.BlockTransformer...   yes (head_only)          freeze  → grad := 0
   heads_action.diffusion_model.Dense_0   no                       AdamW   → real update
```

### 3.3 What this is NOT

Compare to §2 point by point:

- **No low-rank decomposition.** When a head parameter trains, its *full* `d·k` weight matrix gets a
  real gradient — there is no rank-`r` bottleneck, no `A`/`B` factors.
- **No new parameters.** LoRA/adapters/prefix all *add* trainable tensors. Octo adds **zero** new
  parameters. It only chooses which *existing* ones move.
- **No merge step.** There's nothing to fold back in; the trained weights are already the model's
  real weights.

So the precise statement is:

> **`head_only` = "freeze the backbone, train the full-rank head."** It is the *head-only / linear-probe*
> column of the §1.5 table, **not** the LoRA column. They land in similar places on cost and
> overfitting risk, but the mechanism is completely different.

### 3.4 Verified empirically

Running all three modes on `octo-small` for a few steps:

- `full` — every transformer parameter changed (nonzero delta between before/after). ✔ backbone trains.
- `head_only` and `head_mlp_only` — **every** `octo_transformer.*` parameter was *exactly* unchanged
  (delta `0`), while the head parameters did move. ✔ the freeze works, and only the intended head
  params train.

This is the concrete confirmation that `frozen_keys` does what §3.2 claims.

---

## 4. The optimizer recipe

Freezing decides *which* params train. The **optimizer** decides *how* they're updated each step.
Octo uses its own official `create_optimizer` (called in `finetune.py`), and we keep it unchanged.
Here is every piece, defined.

### 4.1 AdamW

**AdamW** is the optimization algorithm. It's **Adam** (adaptive per-parameter step sizes, computed
from running averages of recent gradients) plus **decoupled weight decay** (the "W"). For this doc
the relevant facts are just: it's the standard, well-behaved choice for transformers, and it keeps a
little extra state per parameter (hence full finetuning's higher memory in §1.5).

### 4.2 Cosine learning-rate schedule with linear warmup

The **learning rate (LR)** is the step size — how far you move the weights along the gradient each
step. Octo does not hold it constant; it follows a **schedule**:

```
 LR
3e-4 ┤            ____
     │         __/    \__
     │       _/          \__
     │      /                \___
     │     /                     \____
 0.0 ┤____/                           \________
     └────┬──────────────────────────────────┬──►  step
        warmup                          cosine decay
     (linear 0 → 3e-4)              (3e-4 → 0 along a cosine)
```

- **Warmup** = a short opening phase where LR ramps **linearly** from `0.0` up to the peak. Starting
  at full LR on step 1 can violently disturb the delicate pretrained weights; warming up lets the
  optimizer's running averages stabilize first. (Official Octo: 2000 warmup steps. Our FR5 config
  uses fewer — `warmup_steps: 100` — because we run far fewer total steps.)
- **Peak LR = `3e-4`** (`0.0003`). This is Octo's value and ours (`config.yaml` `learning_rate`).
- **Cosine decay** = after the peak, LR follows the smooth downward half of a cosine curve back to
  `0.0` over the remaining steps. Cosine (vs a sudden drop) anneals gently, which tends to land in
  flatter, better-generalizing minima.

So the full shape is `init 0.0 → linear warmup → peak 3e-4 → cosine decay → 0.0`.

### 4.3 Weight decay — and the ViT/timm-style mask

**Weight decay** = a regularizer that, each step, shrinks every weight slightly toward zero (adds a
penalty proportional to the weight's size). It discourages large weights, which fights overfitting.
Octo uses **weight decay 0.01**.

But — crucially — Octo applies it with a **mask** (a per-parameter on/off switch), in the
**ViT/timm style** (Vision Transformer / the popular `timm` library convention):

> **Decay only `kernel` (weight-matrix) parameters. Do NOT decay biases or LayerNorm parameters.**

Definitions: a **`kernel`** is a weight matrix (the `W` of a linear layer). A **bias** is the additive
offset `b`. **LayerNorm** is a normalization layer with learned scale/shift parameters that
re-center and re-scale activations. **Why exclude biases and norms?** They are few in number and
serve a *calibration* role (setting offsets and scales), not a *capacity* role. Pulling them toward
zero doesn't reduce overfitting — it actively distorts the network's normalization and offsets,
hurting performance. So the standard practice (which Octo follows) is to decay only the big weight
matrices.

### 4.4 Gradient clipping by global norm = 1.0

A single bad batch can produce a huge gradient that takes a destructive step. **Gradient clipping by
global norm** prevents this. The **global norm** is the single overall magnitude of *all* gradients
stacked together (the square root of the sum of every gradient's square). If that global norm exceeds
a threshold (Octo: **1.0**), all gradients are scaled down proportionally so the norm equals exactly
`1.0` — preserving their *direction* but capping their *size*. This keeps finetuning stable,
especially important when nudging a precious pretrained model.

### 4.5 The pieces together (and our scale vs official)

| Knob | Official Octo | Our FR5 `config.yaml` | What it does |
|---|---|---|---|
| Optimizer | AdamW | AdamW | adaptive updates + decoupled weight decay |
| Peak LR | `3e-4` | `3e-4` | step size at the peak |
| LR schedule | warmup → cosine→0 | warmup → cosine→0 | gentle ramp up then anneal down |
| Warmup steps | 2000 | 100 | linear ramp `0 → peak` |
| Weight decay | 0.01 (kernels only) | 0.01 (kernels only) | shrink weight matrices; skip biases/norms |
| Grad clip (global norm) | 1.0 | 1.0 | cap update magnitude |
| Batch size | 256 | 16 | examples per step |
| Steps | ~50k | 5000 | total updates |

We run a smaller batch and fewer steps simply because the FR5 dataset and hardware are smaller — the
*recipe* (optimizer, schedule shape, decay mask, clipping) is identical to official Octo.

---

## 5. Why Octo doesn't need LoRA — and how you'd add it anyway

This is the honest, balanced version. LoRA is excellent technology; it's just aimed at a problem
Octo doesn't have.

**LoRA's big win is at huge scale.** PEFT methods exist because some models have *billions* of
parameters, and full finetuning then needs more GPU memory than most people have (gradients +
optimizer state for billions of weights). LoRA lets you adapt such a model by training a few million
low-rank numbers instead. That's a decisive advantage **when full finetuning is infeasible**.

**Octo is not at that scale.** `octo-base-1.5` is **93M** parameters — `octo-small-1.5` is 27M. Full
finetuning fits comfortably on a single modest GPU. And `head_only` already freezes **90%+** of the
parameters (the entire `octo_transformer.*` backbone) and trains in minutes. So:

- The "cheap and robust" niche LoRA usually fills is **already filled** by Octo's `head_only` /
  `head_mlp_only` freeze modes — at zero extra code and zero new parameters.
- Adding LoRA to Octo would mean writing **custom Flax adapter injection** (there is *no* official
  LoRA-for-Octo) for little practical gain at 93M params.

**But, to be fair and not dogmatic:**

- LoRA *can* be added as a custom extension — nothing about Octo forbids it; it's just not part of
  the official release.
- The broader literature notes full finetuning can outperform LoRA on long-horizon robustness when
  data and compute allow. So LoRA is not strictly "better than full" either.

The takeaway is **not** "LoRA is bad." It's: **at Octo's scale, the freeze modes are the natural
cheap/robust option, so LoRA is unnecessary — and it isn't part of official Octo.**

---

## 6. Practical guidance for the FR5

### 6.1 Default to `head_only`, and here's why

The FR5 dataset is **tiny — about 54 episodes**. That is squarely the regime where full finetuning
overfits and can erase the pretrained prior. So the default in `config.yaml` is:

```yaml
finetune:
  mode: head_only
```

Rationale:

- **Keep the 800k-trajectory visual prior intact.** Freezing `octo_transformer.*` means the
  backbone's hard-won "how to see manipulation" features can't be damaged by 54 episodes.
- **You only need to re-map the action head.** The FR5 action is 7-D (6 joint angles + 1 gripper),
  which *exactly matches* Octo's head (`action_dim=7`, `action_horizon=4`) — so no architectural
  "head surgery," just retraining the head to output FR5 actions instead of Octo's pretrained action
  space. (More on the action-space mismatch in [`action_spaces.md`](action_spaces.md) and
  [`octo.md`](octo.md) §3.)
- **Least overfitting, fastest training.** Few trainable params, robust on small data.

### 6.2 When to switch to `--mode full`

Move to full finetuning once you have collected a **larger, more diverse dataset** — e.g. a
fixed-home + workspace-grid sweep covering many object positions. With more data the backbone can be
safely adapted to the FR5's specific camera and scene without overfitting, unlocking the extra
capacity full finetuning provides. `head_mlp_only` sits in between as the most conservative option if
even `head_only` shows signs of overfitting.

### 6.3 The state-free connection (the proprioceptive shortcut)

Octo finetuning is **vision + language only — there is no proprioceptive (joint-state) input.**
*Proprioception* is the robot's sense of its own joint positions. This matters because of a classic
imitation-learning failure called the **proprioceptive shortcut**: when a policy is given the current
joint state *and* asked to predict the next joint command, the two are so tightly correlated that the
network learns to ignore the camera entirely and just echo the state — then fails the moment the
scene changes. It's documented in full in [`il_failure_modes.md`](il_failure_modes.md) §1.

Because Octo never receives joint state, **it cannot take that shortcut** — it is forced to use
vision. This is the exact "state-free" recipe that [`proprioception_modes.md`](proprioception_modes.md)
implements as an opt-in mode (`none`) for the PyTorch policies; Octo gets it **for free**, by design.
So finetuning Octo (`head_only`, vision-only) is structurally well-positioned to generalize
spatially — a frozen 800k-trajectory visual backbone *and* immunity to the proprioceptive shortcut.

---

## 7. Commands recap

Consistent with [`policies/octo/README.md`](../policies/octo/README.md) §3. Run inside `.venv-octo`
(ideally on the Linux GPU box):

```bash
# default — head_only (recommended for the tiny FR5 dataset)
python policies/octo/finetune.py --config policies/octo/config.yaml

# train everything (once you have a larger dataset)
python policies/octo/finetune.py --mode full

# most conservative — freeze backbone + head attention, train only the head MLP
python policies/octo/finetune.py --mode head_mlp_only

# quick smoke test (tiny, fast)
python policies/octo/finetune.py --mode head_only --steps 50 --batch-size 4 \
    --save-dir /tmp/octo_fr5
```

The `--mode` flag selects the `frozen_keys` entry from §3.1. There is no `--lora` flag, because
**Octo doesn't use LoRA** — `--mode` chooses *which existing weights are frozen*, and the optimizer
recipe from §4 trains the rest.

---

## 8. One-paragraph summary

Finetuning Octo means continuing to train its pretrained weights on your data. The choice is *which*
weights move: `full` (everything), `head_only` (freeze the `octo_transformer.*` backbone, train the
full-rank head — the FR5 default), or `head_mlp_only` (also freeze the head's attention, train only
its MLP). The freeze mechanism is **gradient masking via an optax partition** — frozen params get
zero gradient. This is emphatically **not LoRA**: there is no low-rank `B·A` update, no new
parameters, and no merge step. LoRA is a PEFT trick for billion-parameter models that can't afford
full finetuning; at Octo's 93M scale the freeze modes already provide the cheap, robust, overfit-safe
option, so LoRA is unnecessary (though addable as a custom extension). The optimizer is Octo's
official AdamW + cosine-with-warmup LR (peak `3e-4`) + weight decay `0.01` on kernels only + global-
norm gradient clipping at `1.0`. For the FR5, default to `head_only` to preserve the 800k-trajectory
visual prior on a ~54-episode dataset, run vision-only (which sidesteps the proprioceptive shortcut),
and graduate to `--mode full` once you have a larger dataset.
