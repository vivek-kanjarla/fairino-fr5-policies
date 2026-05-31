# DiT + Flow Matching

---

## the one-line version

same "denoise a corrupted action chunk" idea as diffusion policy, but with two upgrades: a **transformer** denoiser instead of a U-Net (the "DiT"), and **flow matching** instead of DDPM noise (a cleaner, straighter math that needs fewer inference steps).

---

## two ideas bolted together

### idea 1 — DiT (Diffusion Transformer)

diffusion policy uses a 1-D U-Net to denoise. DiT replaces it with a transformer. why? transformers scale better, handle long-range dependencies in the action sequence better, and the conditioning (observation, timestep) gets injected cleanly through a mechanism called **AdaLN** (Adaptive Layer Norm).

instead of FiLM layers in a U-Net, each transformer block's layer-norm gains and biases are *computed from the conditioning vector*. the conditioning literally reshapes how every layer normalizes its activations.

### idea 2 — flow matching (instead of DDPM diffusion)

this is the key conceptual upgrade. DDPM learns to reverse a noising process step by step over a curved path. flow matching learns a **straight-line path** from noise to data.

```
DDPM:           noise ～～～curvy path～～～→ data   (needs many steps)
Flow matching:  noise ───straight line───→ data   (needs few steps)
```

you define a straight interpolation between noise and the clean action, and train the network to predict the **velocity** (the direction and speed to move along that line).

---

## the math (flow matching)

**the path** — linear interpolation between noise and clean action:

```
x_t = t · a_clean + (1 − (1−σ)·t) · ε,     t ∈ [0, 1],  ε ~ N(0, I)
```

- at `t=0`: `x_0 = ε` (pure noise)
- at `t=1`: `x_1 = a_clean` (clean action)
- `σ` (sigma_min) is usually ~0, so it simplifies to `x_t = t·a + (1−t)·ε`

**the target** — the velocity field (the straight-line direction from noise to data):

```
v = a_clean − (1−σ)·ε     ≈   a_clean − ε
```

this is just "data minus noise" — the constant velocity that carries you in a straight line from `ε` to `a_clean`.

**the loss** — regress the predicted velocity onto the true velocity:

```
loss = ‖ v_θ(x_t, t, c) − v ‖²
```

**inference** — Euler integration of the ODE, t from 0 to 1:

```
start: x_0 = ε ~ N(0, I)
dt = 1 / num_steps        (num_steps = 10 in our config)
for t = 0, dt, 2·dt, ..., 1:
    x_{t+dt} = x_t + dt · v_θ(x_t, t, c)
end: x_1 = clean action chunk
```

because the path is straight, you can take big Euler steps. 10 steps (sometimes fewer) gets you a clean action vs DDPM's need for more.

---

## architecture

```
Observation                            Noisy action chunk x_t + timestep t
    │                                            │
    ├─ CLIP ViT-B/16 (vision)                    │
    │   ↓                                         │
    │  image embedding                            │
    │                                            │
    ├─ CLIP text encoder (language!)              │
    │   ↓                                         │
    │  task embedding ("pick up the block...")    │
    │                                            │
    ├─ linear (state)                             │
    │   ↓                                         │
    └──── conditioning vector c ──────────────────┤
                                                 │
                                    DiT (transformer blocks)
                                    each block: self-attention + MLP
                                    AdaLN injects c (+ timestep) by
                                    modulating layer-norm gains/shifts
                                                 │
                                    predicted velocity v_θ(x_t, t, c)
```

note the new ingredient vs diffusion policy: **language**. the CLIP text encoder turns the task string into an embedding that's part of the conditioning. so this model can in principle do different things based on the instruction.

---

## how it differs from diffusion policy

| | Diffusion Policy | DiT + Flow Matching |
|---|---|---|
| **denoiser** | 1-D U-Net | Transformer (DiT) |
| **conditioning injection** | FiLM | AdaLN |
| **corruption math** | DDPM (curved path) | flow matching (straight path) |
| **network predicts** | noise ε | velocity (a − ε) |
| **vision encoder** | ResNet18 | CLIP ViT-B/16 |
| **language** | none | CLIP text encoder ✓ |
| **inference steps** | ~10 DDIM | ~10 Euler (can go lower) |

---

## why flow matching is "better" (and the caveat)

**better**: straight-line paths mean the ODE is easier to integrate — fewer steps, more stable gradients, often smoother action chunks. it's the current trend in 2024-2025 (pi0 uses it too).

**caveat**: with only a handful of demonstrations and a big CLIP backbone, this model has a lot of capacity to overfit. it shines when you have language-conditioned multi-task data, which is exactly what the next models (pi0) push further.

---

## when to use it

- ✓ you want language conditioning (multi-task, instruction following)
- ✓ you want the modern flow-matching recipe with fast sampling
- ✓ you have a GPU (CLIP + transformer is heavier than ResNet + U-Net)
- ✗ single-task, tiny data, CPU-only deployment
