# π0 (pi0) — Flow-Matching Vision-Language-Action model

---

## the one-line version

take the DiT + flow matching idea, but replace the small CLIP encoder with a **full 2-billion-parameter vision-language model** (PaliGemma — the same kind of model that can look at an image and answer questions about it) as the brain that understands the scene and the instruction. then bolt a small "action expert" onto it that produces robot actions via flow matching.

this is a **foundation model** for robots — pretrained on huge robot datasets, then fine-tuned on your task.

---

## what makes it a "VLA"

VLA = Vision-Language-Action. the lineage:

```
VLM (vision-language model)        →   sees image, reads text, outputs text
                                       e.g. "the block is on the left"

VLA (vision-language-ACTION model) →   sees image, reads text, outputs ROBOT ACTIONS
                                       e.g. joint trajectory to grab the block
```

pi0 takes a pretrained VLM (PaliGemma, which already understands images and language from internet-scale pretraining) and teaches it to output actions instead of words. the bet is that all that visual and language understanding transfers to robot control.

---

## architecture — two experts sharing attention

pi0 has a clever structure: two transformer "experts" that attend to each other.

```
        ┌─────────────────────────────────────────────────┐
        │  PaliGemma (2B) — the "vision-language expert"   │
        │                                                  │
        │  image  → SigLIP vision encoder → image tokens   │
        │  task   → Gemma tokenizer       → text tokens    │
        │                                                  │
        │  (these tokens form the "prefix")                │
        └────────────────────┬─────────────────────────────┘
                             │  shared attention
        ┌────────────────────┴─────────────────────────────┐
        │  Gemma 300M — the "action expert"                │
        │                                                  │
        │  state + noisy action chunk + timestep           │
        │  → action tokens (the "suffix")                  │
        │                                                  │
        │  outputs: velocity field for flow matching       │
        └──────────────────────────────────────────────────┘
```

the image+text tokens (prefix) and the state+action tokens (suffix) all go into one big attention computation. the action expert "reads" the visual-language understanding from PaliGemma through attention, then produces the flow-matching velocity for the action chunk.

the attention is **masked** so that image/language/state tokens don't attend to the action tokens (the actions are what you're trying to generate, they shouldn't leak back into the conditioning).

---

## the math — same flow matching as DiT

pi0 uses the exact same flow-matching objective as `dit_flow`:

```
x_t = t · a_clean + (1 − t) · ε              (straight-line path)
v   = a_clean − ε                            (velocity target)
loss = ‖ v_θ(x_t, t, c) − v ‖²               (regress velocity)
```

inference is the same Euler ODE integration (10 steps). the difference is entirely in **what computes `v_θ`** — here it's the 2B PaliGemma + 300M action expert instead of a small DiT.

one detail: pi0 uses a **beta-distribution timestep sampling** during training (not uniform) — it samples `t` more densely near the data end of the path, which the paper found helps.

---

## why state and actions get padded to 32

pi0 is designed to work across many robots with different numbers of joints. so its architecture has a fixed width: `max_state_dim = 32`, `max_action_dim = 32`. our FR5 has 6 state dims and 7 action dims — the wrapper pads these out to 32 with zeros, and the model ignores the padding. this is how one pretrained pi0 can be fine-tuned on a 6-DOF arm, a 7-DOF arm, or a bimanual setup without changing the architecture.

```
FR5 state:  [j1 j2 j3 j4 j5 j6]  →  pad  →  [j1 j2 j3 j4 j5 j6 0 0 ... 0]  (32 dims)
FR5 action: [j1..j6 gripper]     →  pad  →  [j1..j6 grip 0 0 ... 0]        (32 dims)
```

---

## how it differs from DiT + flow matching

| | DiT + Flow Matching | π0 |
|---|---|---|
| **vision/language brain** | CLIP ViT-B/16 (~150M) | PaliGemma 2B (full VLM) |
| **action head** | DiT transformer | Gemma 300M action expert |
| **pretraining** | CLIP weights only | pretrained on large robot datasets |
| **state/action width** | exact (6/7) | padded to 32 (multi-robot) |
| **objective** | flow matching | flow matching (identical) |
| **size** | ~150M | ~2.3B |
| **needs** | GPU | bigger GPU + gated PaliGemma download |

it's basically "DiT + flow matching, but the encoder is a giant pretrained brain instead of a small one."

---

## the practical reality

- **download**: PaliGemma weights are ~5 GB and gated (need a HuggingFace account + accept Google's license).
- **compute**: 2.3B params — needs a real GPU, won't run on a laptop.
- **payoff**: because it's pretrained on diverse robot data, it can generalize and follow language instructions far better than a model trained from scratch on your handful of demos. this is the "fine-tune a foundation model" approach vs "train from scratch."

---

## when to use it

- ✓ you want a pretrained foundation model that already understands manipulation
- ✓ language-conditioned, multi-task, want generalization to new objects/instructions
- ✓ you have a proper GPU and HuggingFace access
- ✗ single narrow task, limited compute, want something simple and self-contained
