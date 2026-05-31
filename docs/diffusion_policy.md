# Diffusion Policy (DDPM / DDIM)

---

## the one-line version

instead of directly predicting "what action to take", the network learns to **denoise** a corrupted action chunk back to the clean one — the same math that makes image generators like DALL·E work.

---

## the core idea

you know how if you add noise to a photo gradually, you eventually get pure static? diffusion policy runs that in reverse. at training time you take your expert action chunk, add a controlled amount of gaussian noise to it, and train a network to predict how to undo that noise. at inference time you start from pure noise and run the denoising process to get a clean action sequence.

```
TRAINING
────────
clean actions  →  add noise (forward diffusion)  →  noisy actions
                                                           ↓
                                                    network tries to
                                                    predict the noise
                                                    (or clean actions)

INFERENCE
─────────
pure noise  →  denoise 10 steps  →  clean action chunk
```

---

## architecture

```
Observation                        Noisy action chunk
    │                                      │
    ├─ ResNet18                            │
    │   ↓                                  │
    │  image features (spatial softmax)    │
    │                                      │
    ├─ linear                              │
    │   ↓                                  │
    │  state features                      │
    │                                      │
    └──────────────────────────────────────┤
                                           │
                              global conditioning vector c
                                           │
                              1-D Conditional U-Net
                              (processes action chunk as a time signal)
                              (FiLM layers inject c at each resolution)
                                           │
                              predicted noise  ε_θ(a_t, t, c)
```

the U-Net treats the action chunk as a 1-D temporal signal (like audio). it has encoder and decoder stages at multiple resolutions, with skip connections. FiLM (Feature-wise Linear Modulation) layers inject the observation conditioning vector at each resolution — basically the observation tells the network "you're in this scene, denoise accordingly."

---

## the math

**forward diffusion** (training, adding noise):

```
a_t = √ᾱ_t · a_clean + √(1 - ᾱ_t) · ε,     ε ~ N(0, I)
```

- `a_clean` = the expert action chunk from the demonstration
- `t` = random timestep sampled from 0 to T (T=100 in our config)
- `ᾱ_t` = a schedule that controls how much noise — starts at 1 (no noise), ends near 0 (pure noise)
- the schedule is `squaredcos_cap_v2` — a cosine curve so noise is added slowly at first, faster in the middle, then slowly again

**training loss** (predict the noise, epsilon-prediction):

```
loss = ‖ ε_θ(a_t, t, c) − ε ‖²
```

just MSE between what the network predicted the noise to be and what it actually was.

**inference** (DDIM denoising, 10 steps):

```
start: a_T ~ N(0, I)
for t = T, T-1, ..., 0:
    predicted_noise = U-Net(a_t, t, c)
    a_{t-1} = DDIM_update(a_t, predicted_noise, t)
```

DDIM (Denoising Diffusion Implicit Models) is a faster version of DDPM — instead of 100 small steps, you can do 10 bigger deterministic steps and get the same quality. this is what makes diffusion policy fast enough to run at 30 Hz on a GPU.

---

## the action queue

diffusion policy predicts `horizon=16` actions but only executes `n_action_steps=8` of them. the rest are thrown away. this is intentional — predictions for the near future are more reliable than predictions far into the future. after 8 steps the model is re-queried.

```
predict chunk: [a0, a1, a2, a3, a4, a5, a6, a7,  a8, a9, a10, a11, a12, a13, a14, a15]
execute:        ↑   ↑   ↑   ↑   ↑   ↑   ↑   ↑   <- discard these ->
                  these 8 steps only
```

---

## how it differs from ACT

| | ACT | Diffusion Policy |
|---|---|---|
| **what the network predicts** | actions directly | noise to remove from corrupted actions |
| **training loss** | L1 + KL (CVAE) | MSE on noise |
| **chunk size** | 100 (3.3 s) | 16 (0.53 s) |
| **re-query every** | 100 steps | 8 steps |
| **inference** | single forward pass | 10 DDIM steps |
| **model size** | ~19M params | ~75M params |
| **strengths** | fast inference, simple | multimodal (handles multiple valid solutions) |

the biggest practical advantage of diffusion policy over ACT: it handles **multimodal behavior** naturally. if there are two equally valid ways to do something (pick up from left or right), ACT averages them and does neither well. diffusion policy can represent both modes because the denoising process can converge to different solutions from different noise samples.

---

## when to use it

- ✓ task has multiple valid ways to do it (different grasp poses, paths)
- ✓ you have enough GPU to afford 10 denoising steps per 8 actions
- ✗ you need pure CPU inference at 30 Hz (too slow without CUDA)
- ✗ very long horizon tasks (16-step chunk is short)
