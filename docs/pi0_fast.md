# π0-FAST — autoregressive VLA (no flow matching)

---

## the one-line version

same PaliGemma brain as π0, but instead of denoising actions with flow matching, it **turns actions into discrete tokens and generates them one at a time like text** — exactly how a language model writes a sentence word by word. this makes a single big difference: there's no ODE loop at inference, so it can be much faster.

read [`pi0.md`](pi0.md) for the shared PaliGemma backbone; this doc covers what's different.

---

## the key idea — actions as tokens

π0 and π0.5 treat actions as continuous numbers and denoise them. π0-FAST treats actions as a **language**.

```
flow matching (π0):     action = continuous vector, denoised over an ODE
π0-FAST:                action = sequence of discrete tokens, generated like text
```

so the question becomes: how do you turn a continuous action chunk (50 timesteps × 7 joints of real numbers) into discrete tokens you can generate autoregressively? that's what **FAST** does.

---

## FAST = Frequency-space Action Sequence Tokenization

naively, you could just round each action number into a bin (like 256 levels). but action chunks are smooth and highly correlated across time — naive binning wastes tokens encoding redundant information, and the token sequence gets very long.

FAST is cleverer. it:

1. takes the action chunk (a `[T × action_dim]` matrix of smooth trajectories)
2. applies a **Discrete Cosine Transform (DCT)** — the same transform JPEG uses to compress images. this converts the time-domain trajectory into frequency components. smooth trajectories have most of their energy in a few low-frequency components.
3. quantizes the frequency coefficients (most high-frequency ones are ~zero and get dropped)
4. compresses the result with byte-pair encoding (BPE) — the same compression tokenizers use for text

the result: a short sequence of discrete tokens that captures the action chunk efficiently. smooth = few tokens.

```
continuous action chunk  →  DCT  →  quantize  →  BPE  →  discrete tokens
   [50 × 7 floats]                                        [~handful of ints]
```

---

## how generation works

once actions are tokens, π0-FAST works exactly like a language model:

```
input:  [image tokens] [text/instruction tokens] [state]
                              ↓
              PaliGemma generates action tokens
              autoregressively (one token at a time,
              each conditioned on the previous ones)
                              ↓
              [action token 1, token 2, token 3, ...]
                              ↓
              de-tokenize (inverse BPE → inverse DCT)
                              ↓
              continuous action chunk
```

it uses a **KV cache** (the standard LLM trick) so generating each next token is cheap.

---

## the speed tradeoff

| | π0 (flow matching) | π0-FAST (autoregressive) |
|---|---|---|
| **inference** | 10 ODE steps, each a full forward pass | generate N tokens, KV-cached |
| **training** | regress velocity (parallel, fast) | next-token cross-entropy (also parallel) |
| **inference speed** | slower (multiple full passes) | faster per the FAST paper |
| **what's predicted** | continuous velocity field | discrete action tokens |
| **loss** | MSE on velocity | cross-entropy on tokens (like LLM) |

the FAST paper's claim is that autoregressive token generation, with the efficient DCT tokenization, ends up faster to train and competitive in quality with diffusion/flow approaches — while being able to reuse the entire LLM generation machinery (KV cache, sampling, etc.).

> note the name is a bit of a double meaning: "FAST" is the tokenization scheme (Frequency-space Action Sequence Tokenization), and it also happens to make inference fast.

---

## architecture

```
PaliGemma 2B (vision-language)
        ↓
image + text + state tokens  →  prefix
        ↓
autoregressive decoding (language-model head)
        ↓
action tokens  →  inverse FAST (BPE⁻¹ → DCT⁻¹)  →  continuous action chunk
```

no separate "action expert", no flow matching, no ODE. it's PaliGemma's own language-modeling head producing action tokens.

extra dependency: `lerobot/fast-action-tokenizer` (the trained FAST tokenizer, a small public download) on top of the gated PaliGemma weights.

---

## how the three π models relate

| | π0 | π0.5 | π0-FAST |
|---|---|---|---|
| backbone | PaliGemma 2B | PaliGemma 2B | PaliGemma 2B |
| action method | flow matching | flow matching | **autoregressive tokens (FAST)** |
| action expert | Gemma 300M | Gemma 300M | none (uses LM head) |
| inference | 10 ODE steps | 10 ODE steps | token generation (KV-cached) |
| tokenizer length | 48 | 200 | 200 |
| extra download | — | — | FAST action tokenizer |

think of them as: π0 = the original flow-matching VLA, π0.5 = π0 tuned for open-world generalization, π0-FAST = π0 but with the action head swapped from "denoise" to "generate tokens."

---

## when to use it

- ✓ inference speed matters and you want to avoid the ODE loop
- ✓ you like the "actions are just another language" framing (reuses LLM tooling)
- ✓ you have GPU + PaliGemma access
- ✗ tasks needing very high-precision continuous control where tokenization quantization could hurt
