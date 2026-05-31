# ACT — Action Chunking with Transformers

> you already know this one — included for completeness so the docs set is whole.

---

## the one-line version

a transformer that looks at the current observation and **directly predicts a chunk of future actions** in one shot, trained as a CVAE (conditional variational autoencoder) so it can capture the variability in human demonstrations.

---

## the core ideas

### action chunking
instead of predicting one action at a time (which compounds errors and produces jittery motion), ACT predicts a whole chunk of `chunk_size=100` future actions at once. you execute them open-loop, then re-predict.

### the CVAE
ACT is trained as a conditional VAE. during training a small encoder looks at the *actual* action sequence and compresses it into a latent `z` that captures "which style of demonstration was this." the decoder then reconstructs the actions from the observation + `z`. at test time you just set `z = 0` (the prior mean) and decode.

why: human demos are inconsistent — the same task done slightly differently each time. without the latent, the model averages all those styles into mush. the latent gives it a knob to represent the variation.

---

## architecture

```
TRAINING
────────
action sequence  →  CVAE encoder (transformer)  →  latent z (mean, var)
                                                        │
observation (images + joints)  →  ResNet18 + transformer encoder
                                                        │
                          transformer decoder ──────────┤
                                   │      conditioned on obs + z
                          predicted action chunk (100 steps)

loss = L1(predicted, actual)  +  β · KL(z ‖ N(0,I))

INFERENCE
─────────
z = 0  →  decoder(obs, z=0)  →  action chunk
```

---

## the math

```
L = L1_reconstruction  +  β · D_KL( q(z|actions) ‖ N(0, I) )
```

- L1 makes predicted actions match the demonstration
- KL keeps the latent close to a standard normal so `z=0` is valid at test time
- `β` (kl_weight=10 in our config) balances the two

---

## temporal ensembling (inference option)

instead of executing a chunk then re-predicting every 100 steps (which can jerk at boundaries), ACT can re-predict every single step and **blend overlapping predictions** with an exponential weight `exp(-coeff·age)`. smoother, but runs the model every step. see `CONTROL_FREQUENCIES.md`.

---

## how it differs from the diffusion/flow family

| | ACT | Diffusion / DiT / π0 |
|---|---|---|
| **how actions are produced** | direct prediction (one forward pass) | iterative denoising / ODE |
| **handles multimodality via** | CVAE latent z | the denoising process itself |
| **inference cost** | one forward pass (cheapest) | multiple steps |
| **language conditioning** | no | DiT/π0 yes |

ACT is the simplest and fastest of the bunch. the diffusion/flow models trade compute for better multimodal behavior and (for DiT/π0) language conditioning.

---

## when to use it

- ✓ you want the simplest, fastest-to-deploy policy
- ✓ single task, CPU inference acceptable
- ✓ a strong baseline before trying heavier models
- ✗ you need language conditioning or strong multimodal behavior
