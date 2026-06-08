# The Octo Model Architecture — how the model works inside, from the ground up

This document explains **the Octo *model itself*** — what it is structurally and how a forward pass
flows through it, from first principles. It assumes only that you know basic Python and a little
machine learning; every technical term is defined the first time it appears, with shapes and small
worked examples in the house style of [`act.md`](act.md).

It is the **internals** companion to the existing Octo docs:

- [`octo.md`](octo.md) — the high-level overview + the three FR5 workflows (zero-shot / finetune /
  finetuned inference). Read that first for *what Octo is for*.
- [`octo_finetuning.md`](octo_finetuning.md) — how Octo is *finetuned* (full vs head-only vs LoRA,
  `frozen_keys`, the optimizer recipe). This doc does **not** re-explain finetuning — it explains
  the architecture being finetuned.
- This doc fills the gap between them: **how does Octo actually compute an action, inside the box?**

The facts below are verified against the installed Octo source
(`.venv-octo/.../octo/model/`): `octo_module.py`, `components/tokenizers.py`,
`components/vit_encoders.py`, `components/block_transformer.py`, `components/action_heads.py`.

---

## Table of contents

1. The one-paragraph version of the Octo model
2. Prerequisites you need (token, embedding, attention) — quick, with cross-refs
3. Tokenization from scratch (images, language, readouts)
4. The token sequence + block-wise ("block-causal") attention — the heart of Octo
5. The transformer blocks (dims: small vs base)
6. The diffusion action head
7. End-to-end inference flow (the pipeline diagram)
8. How Octo differs from this repo's from-scratch policies
9. Glossary

---

## 1. The one-paragraph version of the Octo model

Octo turns **everything it sees into tokens** — a *token* is just a fixed-size vector. Each camera
image becomes a grid of image tokens (via a small convolutional "stem" then patchifying); the
language instruction becomes a sequence of language tokens (via a frozen T5 text encoder); and a
handful of learned **readout tokens** are added as empty "scratchpads." All these tokens are run
through a **transformer** — but not with one global attention mask. Instead tokens are organized
into named **groups** (a task *prefix*, and per-frame *observation* + *readout* groups) and attention
is declared by **rules between groups**: the task prefix is shared, each frame's observation attends
to the task and to observations at the current-or-earlier frame (**block-causal** over a 2-frame
history window), and the readout token attends to everything before it while **nothing attends back
to it** — so it passively *collects* information without disturbing the rest. The transformer's
output embedding at the readout token is fed to a **diffusion action head**, which starts from
Gaussian noise and iteratively **denoises** it — conditioned on that readout embedding — into a clean
chunk of 4 future actions of 7 numbers each (6 joints + gripper for the FR5). That modular,
group-and-rules design is *the whole point*: you can add or drop a camera, language, or a new head
without retraining from scratch — which is what makes one Octo a **generalist** across robots.

```
   image(s) + language ──► tokenizers ──► [task prefix | per-frame: obs + readout] tokens
                                                    │
                                       block-causal transformer (12 layers)
                                                    │  readout_action embedding
                                                    ▼
                              diffusion head: noise ──denoise──► action chunk (4 × 7)
```

---

## 2. Prerequisites (quick — full versions are cross-referenced)

A few ideas recur. If they're new, the linked sections define them in full; here are one-liners so
this doc stands on its own.

- **Token.** A token is a single fixed-size vector — a list of `D` numbers — that represents one
  "unit" of input (a patch of image, a word-piece of text, or a learned placeholder). In Octo
  `D = token_embedding_size` (**384** for octo-small, **768** for octo-base). Everything the model
  processes is a token; the model's job is to mix tokens together with attention and read an answer
  out of one of them.
- **Embedding.** A learned vector that represents some piece of data in a fixed-size numeric space.
  "Embedding X" = mapping X to a vector. (See [`act.md`](act.md) §2.2.)
- **Attention.** The mechanism that lets each token decide *which other tokens to read and how
  much*, via **Query/Key/Value** vectors and a `softmax`. Full derivation in [`act.md`](act.md)
  §2.3. You only need the intuition here: *attention mixes information between tokens, and a "mask"
  decides which pairs are allowed to mix.*
- **Transformer.** A neural net built mostly of attention layers stacked with small feed-forward
  ("MLP") layers. (See [`act.md`](act.md) §2.4.)
- **Multi-head attention.** Run attention `h` times in parallel with different learned projections,
  then concatenate — each "head" can focus on a different relationship.
- **DDPM / diffusion.** A way to *generate* data by learning to remove noise: corrupt a target with
  Gaussian noise, train a network to predict the noise, then at run time start from pure noise and
  denoise step by step. The full math (forward process, beta schedule, reverse sampling) is derived
  in [`diffusion_policy.md`](diffusion_policy.md); §6 here gives the intuition and links there.

---

## 3. Tokenization from scratch

**The unifying idea.** Octo never feeds raw pixels or raw text into the transformer. It first
converts each modality into a sequence of **tokens** (vectors of size `token_embedding_size`). There
are three token *sources*. Each is implemented as a small "tokenizer" module
(`octo/model/components/tokenizers.py`).

### 3.1 What "tokenize" means

To **tokenize** something is to chop it into discrete units and turn each unit into a vector. For
text, the units are sub-word pieces; for an image, the units are square **patches**. The output is
always the same: a sequence of vectors of length `token_embedding_size`, ready for attention.

### 3.2 Image tokenizer — conv stem + patchify

An image is a grid of pixels: `image_primary` is `256×256×3` (height × width × 3 color channels);
the optional `image_wrist` is `128×128×3`. We cannot feed a flat 196,608-number vector to a linear
layer (too many weights, all spatial structure lost — same reasoning as [`act.md`](act.md) §2.5).
Instead Octo's `ImageTokenizer` does two things: a small **conv stem**, then **patchify**.

**What a patch is.** A **patch** is a small square sub-region of the image (e.g. 16×16 pixels). To
**patchify** an image is to cut it into a non-overlapping grid of such patches and turn each patch
into one token vector. A 256×256 image cut into 16×16 patches gives a `16×16 = 256`-patch grid →
**256 image tokens**, each token summarizing one 16×16 pixel region.

**What a conv stem is.** A **convolution** slides a small learnable filter across the image,
computing a weighted sum of each local neighborhood — it detects local patterns (edges, textures)
cheaply and preserves spatial layout (see [`act.md`](act.md) §2.5). A **stem** is the first few
convolution layers a network runs *before* anything else — it "warms up" the raw pixels into
slightly more abstract features. Octo's stem is called **`SmallStem16`** (in `vit_encoders.py`),
from Xiao et al. 2021, *"Early Convolutions Help Transformers See Better"*: the empirical finding is
that putting a few conv layers in front of the patchify step **stabilizes ViT training** (a "ViT" =
Vision Transformer, a transformer that operates on image patches). Without the stem, patchify alone
(a single big strided conv) makes early ViT training brittle.

Concretely `SmallStem16` is: **3–4 small conv layers** (each `3×3`, stride 2, with GroupNorm +
ReLU — the source uses features `32 → 96 → 192 → 384`), which together downsample the image by
`2×2×2×2 = 16×`, followed by a final `1×1` "embedding" conv that emits the per-patch token vectors.
The net effect is exactly "downsample by 16, then one token per resulting cell" — i.e. patchify with
an effective patch size of 16, with a learned conv stem doing the work instead of a single dumb
strided conv.

**Worked size example.**

```
image_primary:  256 × 256 × 3
   SmallStem16 downsamples by 16×:   256 / 16 = 16
   → a 16 × 16 grid of patch features
   → flatten the grid → 16 × 16 = 256 image tokens, each a vector of size token_embedding_size

image_wrist:    128 × 128 × 3
   128 / 16 = 8
   → an 8 × 8 grid → 8 × 8 = 64 image tokens
```

| Camera | Input pixels | Stem downsample | Patch grid | # image tokens |
|---|---|---|---|---|
| `image_primary` | 256 × 256 | 16× | 16 × 16 | **256** |
| `image_wrist` | 128 × 128 | 16× | 8 × 8 | **64** |

Each token then gets a learned **positional embedding** added (so the transformer knows *which*
grid cell and *which* timestep a token came from — attention is otherwise order-blind; see
[`act.md`](act.md) §2.4). On the FR5 we have one camera, so its frame is mapped to `image_primary`
and `image_wrist` is zeroed + masked (see [`octo.md`](octo.md) §2).

> Note on the FR5 path: Octo's image tokenizer normalizes pixels internally (the stem maps `uint8`
> pixels to `[-1, 1]` or ImageNet-normalized floats — see `normalize_images` in `vit_encoders.py`),
> so you pass raw `uint8` images, not pre-normalized ones.

### 3.3 Language tokenizer — T5

The instruction string (e.g. `"pick up the block and place it in the bin"`) is turned into tokens by
`LanguageTokenizer`, which wraps a pretrained **T5** model.

- **Tokenizer (text sense).** A text tokenizer splits a string into sub-word units ("word-pieces")
  from a fixed vocabulary and maps each to an integer id. E.g. `"placing"` might split into
  `["plac", "ing"]`. This happens *outside* the model (the repo's data adapter calls the model's
  `text_processor`; see [`octo.md`](octo.md) §4).
- **T5.** *Text-To-Text Transfer Transformer* (Google) — a transformer pretrained on a huge text
  corpus. Octo uses only its **encoder** (`FlaxT5EncoderModel`): it reads the sequence of token ids
  and outputs one **token embedding** (a vector) per word-piece — its `last_hidden_state`. A "token
  embedding" is just the encoder's learned vector for that word-piece *in context*.
- **Frozen by default.** Unless `finetune_encoder=True`, the T5 outputs are wrapped in
  `jax.lax.stop_gradient` — the T5 weights don't train; Octo treats the language vectors as a
  fixed, high-quality representation of the instruction.

The result is a short sequence of language token vectors (one per word-piece), which become the
**task prefix** (§4).

### 3.4 Readout tokens — the CLS-like "scratchpad"

The third source isn't an input at all — it's a set of **learned special tokens** called
**readouts**. Octo's config declares `readouts = {"action": 1}`: **one** `readout_action` token per
timestep. In the source (`octo_module.py`) a readout token starts as a **zero vector plus a learned
positional embedding** — it carries no input information of its own.

The point of a readout token is to *collect* information so a head can read it out, **without
perturbing the other tokens**. This is exactly the role of a **CLS token** (the "classification"
token prepended to a sequence so its final embedding summarizes the whole input) or a **register
token** in modern transformers. In ACT, the CVAE encoder's `[CLS]` token plays the same aggregating
role — see [`act.md`](act.md) §5.1(a) ("what the CLS token is doing"): every other token attends
into it, and you throw the rest away and keep only its output.

The crucial asymmetry, enforced by the attention rules in §4:

```
  readout_action  ──attends to──►  task tokens + observation tokens (≤ its own timestep)
  task / observation tokens  ──do NOT attend to──►  readout_action
```

So the readout passively *reads* the scene into a single embedding that the diffusion head will use
as conditioning — and because nothing attends back to it, adding it (or adding a second readout,
e.g. a "value" head) **does not change** any other token's representation. Octo's own docstring puts
it precisely:

> "Readouts ... are designed to only *read* from the sequence before it, without the ability to
> influence (i.e. write) the computation for any of the non-readout tokens. By design, different
> readouts are completely independent of each other."

---

## 4. The token sequence + block-wise attention — the heart of Octo

This is the part that makes Octo *Octo*. A vanilla transformer applies **one** attention mask to one
flat sequence. Octo instead organizes tokens into named **groups** and declares attention with
**rules between groups**. This section unpacks that.

### 4.1 The two kinds of group

From `block_transformer.py`:

- A **`PrefixGroup`** sits at the front of the sequence and is **shared across all timesteps** —
  it is not tied to any single frame. The **task** tokens (the language instruction, and/or a goal
  image) are a `PrefixGroup`. Its tokens have shape `(batch, n_tokens, D)`.
- A **`TimestepGroup`** is **repeated once per timestep** in the history window. Each frame's
  **observation** (image) tokens are a `TimestepGroup`, and so is that frame's **readout** token.
  Its tokens have shape `(batch, horizon, n_tokens, D)` — note the extra `horizon` (= window) axis.

So with `window_size = 2`, the assembled sequence looks like (from `octo_module.py`'s docstring):

```
[ <task language tokens>                                            ← PrefixGroup (shared)
  <t=0 image_primary tokens> <t=0 readout_action token>            ← TimestepGroups at t=0
  <t=1 image_primary tokens> <t=1 readout_action token> ]          ← TimestepGroups at t=1
```

(If `image_wrist` is active, its tokens sit alongside `image_primary` in each timestep's observation
group; on the FR5 it's zeroed + masked.)

### 4.2 The `AttentionRule` vocabulary

Instead of a hand-built mask, each group says "here is how I attend to a group *named X*." The rule
options (the `AttentionRule` enum in `block_transformer.py`) — define each:

| `AttentionRule` | Meaning (from token at *self.timestep* toward a token at *other.timestep*) |
|---|---|
| `ALL` | attend to that group **always**, regardless of timestep. *(Breaks causal structure — used with care.)* |
| `CAUSAL` | attend iff `other.timestep ≤ self.timestep` (same frame or earlier) |
| `CURRENT` | attend iff `other.timestep == self.timestep` (same frame only) |
| `STRICT_PAST` | attend iff `other.timestep < self.timestep` (strictly earlier frames) |
| `NEVER` | never attend to that group |

A rule is matched by the **name** of the other group (via `fnmatch`, so patterns like `task_*` and
`obs_*` work). The default, if no pattern matches, is `NEVER`. Also: all tokens **within the same
group at the same timestep always attend to each other** (so the two image patches of one frame see
each other, and a frame's image tokens and that frame's wrist tokens see each other).

### 4.3 The actual rules Octo declares

Reading them straight out of `octo_module.py`:

```
task tokens          attention_rules = { "task_*": CAUSAL }
                        → task attends to other task tokens; (not to obs or readouts)

observation tokens   attention_rules = { "task_*": CAUSAL, "obs_*": CAUSAL }
                        → an obs token attends to the task prefix AND to obs tokens
                          at the same-or-earlier timestep  (block-causal)

readout_action       attention_rules = { "task_*": CAUSAL, "obs_*": CAUSAL,
                                          "readout_action": CAUSAL }
                        → a readout attends to task + all obs up to its timestep,
                          and only to its OWN readout group
```

The key fact you can now *derive*, not memorize: **no group lists `readout_*` in its rules** (except
the readout itself). Since the default is `NEVER`, **nothing attends to the readout tokens** — which
is exactly the CLS-like "read but don't write" property from §3.4. And because every rule is
`CAUSAL` over the timestep axis (the transformer is built with `enforce_causal=True`), a frame can
never attend to a *future* frame → **block-causal over the 2-frame window**.

### 4.4 ASCII attention diagram (who attends to whom, 2-frame window)

A ✓ means "the **row** token is allowed to attend to the **column** token." Read it row by row.

```
                          ATTENDS TO  ───────────────────────────────────────►
                          task     obs t0    readout t0   obs t1    readout t1
   ┌──────────────────┬────────┬──────────┬────────────┬─────────┬────────────┐
 A │ task             │   ✓    │    ·     │     ·      │    ·    │     ·      │
 T │ obs   t=0        │   ✓    │    ✓     │     ·      │    ·    │     ·      │
 T │ readout t=0      │   ✓    │    ✓     │     ✓      │    ·    │     ·      │
 E │ obs   t=1        │   ✓    │    ✓     │     ·      │    ✓    │     ·      │
 R │ readout t=1      │   ✓    │    ✓     │     ·      │    ✓    │     ✓      │
   └──────────────────┴────────┴──────────┴────────────┴─────────┴────────────┘
                                            ▲                       ▲
                              nothing attends to readouts (their columns are blank
                              except the readout's own row → "read but don't write")
```

Three things to notice:

1. **Block-causal over time.** `obs t=0` cannot see `obs t=1` (no ✓ in the t=1 columns) — the past
   can't peek at the future. `obs t=1` *can* see `obs t=0` (history). This is the 2-frame window.
2. **The task prefix is seen by everyone** (the whole `task` column is ✓ for every observation/
   readout row) — every frame is conditioned on the instruction.
3. **The readout columns are blank** except on the readout's own row → nothing writes into the
   readout; it only reads.

### 4.5 `repeat_task_tokens`

Octo sets **`repeat_task_tokens = True`**. In addition to the shared task `PrefixGroup`, the task
tokens are **copied into a `TimestepGroup` at every timestep** (the loop in `octo_module.py` that
tiles `task_tokens` across the window and re-files them under the name `obs_<task>`). Plainly: every
frame gets its *own local copy* of the instruction tokens so cross-modal attention between vision and
language happens *at each timestep*, not only through the distant prefix. The effect — "every frame
sees the instruction" — is the same; repeating it just makes that conditioning stronger and more
local.

### 4.6 WHY the modular block design matters (the generalist payoff)

This is the architectural thesis of Octo. Because attention is declared **per group, by rule**,
rather than as one frozen global mask:

- **You can add or remove input modalities** — a second camera, a goal image, the language
  instruction — by adding/removing a tokenizer and a group with its own attention rules. The rest of
  the model is untouched.
- **You can add new output heads** — e.g. a "value" readout next to the "action" readout — just by
  declaring another readout group. Because readouts don't write back, a new head **cannot disturb**
  the existing ones, and heads are independent (you can run one without the other).
- Each new group simply *declares how it attends*; no global mask surgery, no retraining from
  scratch.

That flexibility is **why one pretrained Octo can serve many robot embodiments** and be finetuned
cheaply onto a new one like the FR5 (swap the camera into `image_primary`, retrain just the head —
see [`octo_finetuning.md`](octo_finetuning.md)). The "backbone vs head" split there maps onto this
structure: the **backbone** is everything in §3–§5 (`octo_transformer.*`), the **head** is §6
(`heads_*`). The modular token/group/rule design is the concrete reason Octo is a **generalist**.

---

## 5. The transformer blocks

Once the groups and rules produce a token sequence + a block-causal mask, the rest is a **standard
transformer**: a stack of identical blocks, each = **multi-head self-attention** (under the block
mask) **+ a feed-forward MLP**, with residual connections and layer normalization around each
sub-layer (the standard recipe; see [`act.md`](act.md) §2.4 for "add & normalize"). Positional
information is supplied by the **learned** per-group positional embeddings added at tokenization
(§3.2/§3.4) — initialized `normal(stddev=0.02)` and sized to `max_horizon`, then truncated to the
actual window — encoding both *modality/group* and *timestep* (so the transformer is told "this is
the t=1 image patch #37," etc.).

There are **two** released sizes. Both have 12 layers; they differ in width:

| Spec | `octo-small-1.5` | `octo-base-1.5` | What it is |
|---|---|---|---|
| `token_embedding_size` (D) | **384** | **768** | width of every token vector |
| transformer layers | **12** | **12** | depth (number of attention+MLP blocks) |
| attention heads | **6** | **12** | parallel attention sub-computations per block |
| MLP dim | **1536** | **3072** | hidden width of each block's feed-forward MLP |
| ≈ ViT scale | ViT-Small | ViT-Base | the comparable image-transformer size |
| parameters | **≈ 27M** | **≈ 93M** | total weights |

`octo-base-1.5` (93M) is the **largest** official checkpoint — there is **no** "octo-large." Both
are far smaller than the π0-family VLMs. (Sizes and the no-large fact are confirmed in
[`octo.md`](octo.md) §7.)

The transformer's output is a dictionary of embeddings, one per token group — including the one we
care about: **`readout_action`**, shape `(batch, window, 1, D)` (one readout token per frame).

---

## 6. The diffusion action head

The backbone produced a `readout_action` embedding per timestep. The **`DiffusionActionHead`**
(`action_heads.py`) turns the **last frame's** readout embedding into an action chunk by a
**conditional diffusion** process.

### 6.1 Readout embedding → conditioning vector

The head takes the `readout_action` token group `(batch, window, n_tokens=1, D)` and pools it to one
vector per timestep: with `use_map=False` (Octo's default for this head) it **mean-pools** over the
token axis — trivially the single readout token itself — giving `(batch, window, D)`. That vector is
the **conditioning** `c`: "everything the transformer understood about the scene + instruction,
distilled into one vector." At inference only the **last** timestep's `c` is used (the most recent
frame).

### 6.2 Conditional denoising (DDPM-style)

The head is a small **score network** — an MLP with residual connections (3 blocks, hidden 256;
`create_diffusion_model` in `octo/model/components/diffusion.py`), **not** a U-Net (that's the
separate `UNetDDPMActionHead`; Octo's default is the MLP one). It implements the **same "corrupt the
future, learn to denoise it" idea as Diffusion Policy** — so rather than re-derive DDPM here, the
full forward process, beta schedule, and reverse sampling math are in
[`diffusion_policy.md`](diffusion_policy.md) §§5–9. The intuition, in five lines:

- **Train.** Take the ground-truth action chunk, flatten it to a vector of `action_horizon ×
  action_dim = 4 × 7 = 28` numbers, pick a random noise level, add Gaussian noise. Ask the score
  network — *conditioned on `c`* and given the noise level — to predict the noise that was added
  (`pred_eps`). Loss = MSE between predicted and true noise (`continuous_loss`, `loss_type="mse"`).
- **Infer.** Start from a pure-Gaussian-noise chunk and run the DDPM **reverse loop** (`scan_fn`):
  at each of `diffusion_steps` steps, the network predicts the noise, you subtract a scaled version
  of it (`current_x = (1/√αₜ)·(current_x − ((1−αₜ)/√(1−ᾱₜ))·eps_pred)`), add a touch of fresh noise
  for `t>0`, and repeat — gradually turning static into a clean action chunk. (`betas` come from a
  **cosine** schedule; defaults: `n_diffusion_samples=1`, so one sample is drawn.)

```
   inference (per query):

   c (readout embedding)
        │  conditions every step
        ▼
   noise (4×7=28 numbers)  ──step 1──► ··· ──step k──► clean action chunk (4 × 7)
   N(0, I)                  denoise          denoise
        └──────────────── diffusion_steps reverse iterations ────────────────┘
```

Why diffusion for actions? Because human demonstrations are **multimodal** (many valid ways to do a
task), and a stochastic denoiser naturally lands in *one* mode instead of averaging them into mush —
the same reason ACT needs its CVAE latent (see [`act.md`](act.md) §4 Idea 3, and
[`diffusion_policy.md`](diffusion_policy.md) §3). The difference is that Octo's conditioning vector
`c` comes from a **much larger pretrained transformer** than Diffusion Policy's from-scratch
encoder.

### 6.3 Action chunking and what reaches the robot

The head predicts a **chunk** of `action_horizon = 4` future actions, each `action_dim = 7`
(**action chunking** — predicting several future steps at once for smoother, less error-prone motion;
see [`act.md`](act.md) §4 Idea 1). Output shape per query: **`(4, 7)`**.

The raw 28 numbers live in **normalized** space. They are **un-normalized** with the dataset's
saved mean/std before use (see [`octo.md`](octo.md) §5 — at zero-shot these are a source dataset's
stats; after FR5 finetuning they're FR5 stats, so the numbers come back as real FR5 joint angles +
gripper). At deploy you send the **first** action of the chunk: `action[:6]` → the 6 joint targets
(`servo_j`), `action[6]` → the gripper. (You can also execute the chunk open-loop or with temporal
ensembling — see [`inference.md`](inference.md).)

---

## 7. End-to-end inference flow

Putting §§3–6 together — one forward pass, FR5 deployment:

```
 ┌─ INPUTS (a 2-frame history window) ──────────────────────────────────────────┐
 │  image_primary  (B, 2, 256, 256, 3) uint8     language: "pick up the block…"  │
 │  image_wrist    (B, 2, 128, 128, 3) [FR5: zeroed+masked]                       │
 │  timestep_pad_mask (B, 2)                                                      │
 └───────────────┬───────────────────────────────────────────┬───────────────────┘
                 │ image tokenizer                            │ language tokenizer
                 │  SmallStem16 conv stem → patchify          │  T5 encoder (frozen)
                 │  256→16×16=256 tok/frame                   │  word-pieces → token vecs
                 ▼                                            ▼
        per-frame OBSERVATION tokens                    TASK PREFIX tokens
                 │                                            │
                 │  + learned positional/group embeddings     │
                 └──────────────────┬─────────────────────────┘
                                    ▼
        assemble groups:  [ task prefix | t0: obs + readout | t1: obs + readout ]
                          (repeat_task_tokens=True → task copied into each frame)
                                    │
                          ┌─────────▼──────────────────────────────┐
                          │  BLOCK TRANSFORMER (12 layers)          │
                          │  block-causal over the 2-frame window   │
                          │  readouts read but are never read       │
                          └─────────┬──────────────────────────────┘
                                    │  readout_action embedding (last frame) = c
                                    ▼
                          ┌────────────────────────────────────────┐
                          │  DIFFUSION ACTION HEAD (MLP score net)  │
                          │  noise ──denoise (conditioned on c)──►  │
                          │  action chunk  (4, 7)                   │
                          └─────────┬──────────────────────────────┘
                                    │  un-normalize with dataset stats
                                    ▼
                       FR5: send first action → joints[:6] via servo_j, gripper[6]
```

---

## 8. How Octo differs from this repo's from-scratch policies

The other policies here ([`act.md`](act.md), [`diffusion_policy.md`](diffusion_policy.md),
[`pi0.md`](pi0.md)) are *trained from scratch* on ~54 FR5 episodes. Octo is *pretrained* and
finetuned. The internal architecture differs too. This table **complements** the framework/workflow
comparison in [`octo.md`](octo.md) §1 (which covers JAX-vs-PyTorch, pretraining source, and the FR5
workflows) by focusing on **architecture internals**:

| Internal aspect | **ACT** | **Diffusion Policy** | **π0** | **Octo** |
|---|---|---|---|---|
| Pretraining | none (scratch) | none (scratch) | VLM-pretrained | **800k Open-X trajectories** |
| Encoder core | CVAE + transformer enc/dec | ResNet + 1-D U-Net | 2B PaliGemma VLM | **tokenized block-causal transformer** |
| How inputs enter | image grid + state token + latent z | image features + FiLM cond. | image+text into VLM | **everything → tokens, group+rule attention** |
| Multimodality handled by | CVAE latent `z` | denoising (DDPM) | flow matching | **diffusion action head** |
| Action head | direct one-shot decode | DDPM U-Net | flow matching | **DDPM MLP score net** (shared family w/ Diffusion Policy) |
| Language | no | no | yes | **yes (frozen T5 tokens)** |
| Proprioceptive state in | yes (`state` token) | yes | yes | **no — vision+language only** |
| History window | 1 obs | 2 obs | — | **2 frames (block-causal)** |
| Action output | chunk (100×7) | chunk (16×7) | chunk | **chunk (4×7)** |
| Framework | PyTorch (lerobot) | PyTorch (lerobot) | PyTorch (lerobot) | **JAX / Flax** |

Two relationships worth holding onto:

- **Octo's head is the same diffusion family as Diffusion Policy** — both corrupt the action chunk
  with noise and learn to denoise (cross-ref [`diffusion_policy.md`](diffusion_policy.md)). The
  difference is *where the conditioning comes from*: a from-scratch ResNet+U-Net for Diffusion
  Policy vs Octo's 800k-pretrained block-causal transformer.
- **Octo is vision+language, state-free** — there is no joint-state input, so it structurally cannot
  take the *proprioceptive shortcut* that bites the state-based policies (see [`octo.md`](octo.md)
  §6 and [`octo_finetuning.md`](octo_finetuning.md) §6.3).

---

## 9. Glossary

- **Token** — a fixed-size vector (length `token_embedding_size` = 384 small / 768 base) representing
  one unit of input or a learned placeholder; everything Octo processes is a token.
- **token_embedding_size (D)** — the transformer's internal vector width.
- **Tokenizer** — a module that turns one modality into tokens (image / language / readout).
- **Patch** — a small square sub-region of an image (16×16 px here).
- **Patchify** — cut an image into a grid of non-overlapping patches, one token per patch.
- **Conv stem (`SmallStem16`)** — a few convolution layers run before patchify; stabilizes ViT
  training (Xiao et al. 2021). Downsamples 16×, so 256→16×16=256 tokens.
- **ViT** — Vision Transformer; a transformer operating on image patch tokens.
- **T5** — *Text-To-Text Transfer Transformer*; Octo uses its (frozen) encoder to turn the
  instruction into language token vectors.
- **Token embedding** — the encoder's learned vector for a token in context.
- **Readout token** — a learned, input-free "scratchpad" token that *reads* the sequence (CLS-like)
  so a head can use its output; nothing attends back to it.
- **CLS / register token** — a special token whose final embedding summarizes the sequence (cf.
  [`act.md`](act.md) §5.1(a)).
- **PrefixGroup** — a token group at the front, shared across timesteps (the task tokens).
- **TimestepGroup** — a token group repeated per timestep (a frame's observation, or readout).
- **Group** — a named set of tokens that declares how it attends to other groups.
- **Block-causal (block-wise) attention** — attention declared by rules between groups, causal over
  the timestep axis (a frame attends to current-or-earlier frames, never future ones).
- **AttentionRule** — `ALL` / `CAUSAL` / `CURRENT` / `STRICT_PAST` / `NEVER`: when one group may
  attend to another (matched by group name; default `NEVER`).
- **repeat_task_tokens** — copies the task tokens into each timestep so every frame sees the
  instruction locally.
- **Positional embedding** — a learned vector added per token encoding its group + timestep, so the
  order-blind transformer knows position.
- **Multi-head attention** — running attention in parallel with several learned projections (6 heads
  small / 12 base).
- **MLP / feed-forward** — the per-block fully-connected sub-layer (hidden 1536 small / 3072 base).
- **window_size** — the history length the model attends over (**2** frames; `max_horizon = 10`).
- **action_horizon** — how many future steps the head predicts at once (**4**).
- **action_dim** — numbers per action (**7** = 6 FR5 joints + gripper).
- **Diffusion / DDPM head** — generates the action chunk by iteratively denoising Gaussian noise,
  conditioned on the readout embedding (full math: [`diffusion_policy.md`](diffusion_policy.md)).
- **Conditioning (`c`)** — the readout embedding fed to the diffusion head as context.
- **Score network** — the network inside the diffusion head that predicts noise (here an MLP with
  residual blocks, not a U-Net).
- **Action chunking** — predicting several future actions per query for smoother, less error-prone
  motion (cf. [`act.md`](act.md) §4).
- **Backbone / head** — the shared transformer (`octo_transformer.*`) vs the action head (`heads_*`);
  the finetuning split (see [`octo_finetuning.md`](octo_finetuning.md)).
```
