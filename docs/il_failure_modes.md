# Imitation Learning — Known Failure Modes and How the FR5 Hit Them

A map of the structural ways behavioral cloning (BC) fails, grounded in the
literature, with a concrete column showing exactly where the FR5 pick-policy
experiments fell into each trap.

---

## 0. Why BC is fragile by design

Behavioral cloning trains a policy π(a|o) by supervised learning on
(observation, action) pairs from human demonstrations. It minimizes prediction
error on the *training distribution* — it has no mechanism to distinguish
between causes and correlations, no feedback loop, and no way to recover from
states it was never shown.

Every failure mode below is a consequence of one of those three gaps.

---

## 1. Causal Confusion / Proprioceptive Shortcut

### What it is

BC learns a *discriminative* model: "given this observation, what action was
demonstrated?" It does not learn a *causal* model: "which part of the
observation actually caused the demonstrated action?"

When a shortcut exists — an input that cheaply predicts the action without
being the true cause — the model takes it. This is the core result of:

> de Haan, Jayaraman, Levine — "Causal Confusion in Imitation Learning"
> NeurIPS 2019 · https://arxiv.org/abs/1905.11979

The proprioceptive form: the robot's joint state is numerically clean and
simple. Vision requires learning detection-like features from raw pixels.
If the joint state at episode-start correlates with the grasp location (because
the operator started near the object), the model learns a cheap
*proprioceptive autoregression* — "continue from where I am" — and never needs
to use the cameras.

### How it shows up at deployment

At training: varied start poses → state predicts grasp → low loss, vision
ignored.
At deploy: fixed home start → state is constant → same "average" reach every
time → grasps empty air regardless of object position.

The motion *looks* correct (clean reach-close-lift arc) but goes to the wrong
place.

### FR5 evidence — what we measured

We ran causal intervention probes on three trained models
(`experiments/shortcut_probe.py`):

| Intervention | What changes | What stays | Result |
|---|---|---|---|
| Image ablation | blank cameras | real state | action barely changes |
| State ablation | mean state | real cameras | action moves 3.6–5.2× more |
| Conflict swap | state from ep_i + image from ep_j | — | reach follows **state** (corr +0.85–0.88), not image (corr −0.14 to −0.29) |

The conflict swap is decisive: when the image and the proprioception point at
*different* object locations, the policy follows the state every single time.

| Model | State dim | Image-ablation Δ | State-ablation Δ | Ratio | Follows image | Follows state |
|---|---|---|---|---|---|---|
| ACT (paper-scale) | 6 | 5.0° | 26.4° | **5.2×** | −0.14 | **+0.85** |
| Diffusion v1 | 6 | 5.0° | 19.7° | **4.0×** | −0.29 | **+0.85** |
| Diffusion v2 (rich) | 13 | 5.2° | 18.8° | **3.6×** | −0.28 | **+0.88** |

**Notable:** Diffusion v2 had a *richer* state (13-D: joints + eef + gripper).
More state → stronger shortcut. A higher-capacity state representation gives
the model more to lean on and less reason to learn vision. Adding more
proprioception made it worse, not better.

### Root cause in the FR5 data

The operator tended to start the arm near the object. The probe measured this
directly:

```
corr(start-pose J1, grasp J1) = 0.45
J2 std across episodes = 17°,  J4 std = 21°   ← start poses were varied
```

So the initial joint configuration *half-predicted* the grasp location. With
only ~54 training episodes there was not enough pressure for the harder vision
pathway to compete with that cheap signal.

### Fix

1. Fixed home start + widely varied object position across demos (breaks the
   correlation at the source)
2. 100–150+ demos (raises the floor that state alone can reach, forcing vision
   to compete)
3. Proprioception dropout 20–40% during training (randomly zeros state so the
   loss cannot be driven by state alone)
4. State-free policy — drop proprioceptive input entirely; Zhao et al. 2025
   showed this improved height generalization 0% → 85%

---

## 2. Covariate Shift / Distributional Shift

### What it is

The policy is trained on states visited by the *expert*. Once deployed, small
prediction errors move the robot to slightly off-trajectory states — states the
expert never visited, so the policy has never been trained on them. From that
new state, the policy makes another error. Errors compound.

The formal bound (Ross & Bagnell 2010, DAgger paper): under BC, the expected
cost after T steps scales as **O(T²)** — quadratic in horizon. Under DAgger
(interactive correction), it's O(T). This is why long-horizon tasks fail
catastrophically even when short-horizon ones work.

### How it shows up at deployment

The robot starts performing the task correctly for the first few seconds, then
drifts — a joint goes slightly too far, a grasp misses by a few mm, and the
next prediction is made from a state that never appeared in training. The
policy has no recovery strategy; it either freezes or produces garbage.

### FR5 evidence

The "went absolute nuts" episodes: the model starts the reach plausibly but
then executes increasingly incoherent actions. This is the compounding-error
signature. The first 300ms (roughly one ACT chunk at 30 Hz) looks fine; after
that it breaks down.

The 3.3-second open-loop execution window in the VRworking version made this
especially severe — no re-planning, no state feedback, just 100 pre-computed
actions played back regardless of what actually happened.

### Fix

- Shorter re-planning interval (ACT temporal ensembling helps because it re-
  plans every tick rather than committing to the whole chunk)
- DAgger: collect new demos at states where the policy drifts (expensive, needs
  an interactive setup)
- Residual policies or hybrid controllers that fall back to a hard-coded
  heuristic when confidence is low

---

## 3. Mode Averaging / Multimodal Demonstration Collapse

### What it is

When demonstrations are multimodal — the expert sometimes reaches left,
sometimes right — a BC policy trained with L2 loss learns the *average* of
the two modes. The average is often a physically impossible or task-failing
trajectory (reaching to the midpoint between both options).

ACT specifically addresses this via the CVAE: the latent z encodes which mode
is intended. At inference z=0 is used, which is the prior — roughly the
"default" or "most common" mode, not the average of all modes.

### How it shows up at deployment

The robot reaches for an in-between point that corresponds to no real grasp
location, or wavers. It is especially visible when grasping from either side
of an object is demonstrated.

### FR5 evidence

Less severe here because pick-place demonstrations are mostly unimodal (one
object, one target). The "sometimes it works, sometimes absolute nuts" report
likely reflects covariate shift more than mode collapse. However, if demos were
recorded from multiple operator start positions covering different approach
angles, averaging is a contributing factor to the inconsistent success.

### Fix

- ACT's CVAE forces the policy to commit to one mode (z encodes intent)
- Diffusion policy handles multimodality naturally (the denoising process
  converges to one sample, not the mean)
- Ensure demonstrations are consistent: same operator, same approach angle,
  same grasp strategy

---

## 4. Data Insufficiency and Coverage Gaps

### What it is

A BC policy can only generalize across states it has seen during training —
or states that are close enough in the input space that the learned features
transfer. With too few demos, large regions of the workspace (object positions,
approach angles, lighting conditions) are uncovered.

The threshold observed in practice: below ~100 demonstrations, most policies
show unreliable behavior on held-out object positions. At 100–150 demos the
shortcut pressure drops enough for vision to start contributing.

### FR5 evidence

~54 training episodes. Object placed in a limited region of the workspace.
This directly enabled the proprioceptive shortcut (section 1) and amplified
distributional shift (section 2) — the model never saw enough visual diversity
to build a robust object detector in its ResNet features.

### Fix

- Collect more demos: target 100–150 as a minimum; 200+ if the workspace is
  large or the task has high variance
- Vary object position systematically across the full workspace (not just the
  center region)
- Vary lighting, object orientation, background if sim-to-real is in scope

---

## 5. Open-Loop Execution Blindness

### What it is

Once an action chunk is committed and sent to the robot, the policy receives
no feedback about whether the action succeeded. If a grasp missed by 2 mm, the
policy doesn't know — it continues executing the "lift" motion and picks up
nothing.

### How it shows up at deployment

Clean reach-close-lift motion with the gripper closing on empty air. This is
exactly what was observed on the FR5.

### FR5 evidence (VRworking version)

The VRworking repo had `state_dim=6` — no gripper state in the observation.
The policy executed 100 actions (3.3 seconds) open-loop with no gripper
position feedback. If the grasp failed, the lift proceeded anyway.

Compounded by the proprioceptive shortcut: the reach was already going to the
wrong location, so even with closed-loop gripper feedback the grasp would fail.

### Fix

- Add gripper state to observation (done — state_dim=7 now includes
  `gripper_norm`)
- Tactile / force feedback to detect grasp success before lifting
- A grasp-success check (gripper closed past threshold?) gating the lift phase
- Closed-loop replanning: if `gripper_norm < threshold` after "grasp" step,
  re-execute the grasp approach

---

## 6. Summary Table — FR5 Failure Mode Audit

| Failure mode | In literature since | Did FR5 hit it? | Severity | Primary fix |
|---|---|---|---|---|
| Causal confusion / proprioceptive shortcut | de Haan 2019 | **Yes — confirmed by probes** | High | Fixed home + varied object + dropout |
| Covariate shift / compounding errors | Ross 2010 (DAgger) | **Yes — "went absolute nuts"** | High | Shorter chunks, more demos, DAgger |
| Mode averaging | — | Partial | Low-Med | ACT CVAE / diffusion handles this |
| Data insufficiency | Community consensus | **Yes — 54 demos** | High | 100–150+ demos |
| Open-loop execution blindness | — | **Yes — 3.3s open-loop, no gripper fb** | High | gripper_norm in obs (done), tactile |

All five failure modes were active simultaneously on the FR5. The proprioceptive
shortcut is the most measurable and the most clearly confirmed — but it is
sitting on top of a data insufficiency problem, and both are compounded by
open-loop execution.

The fixes are ordered: **data first** (fixed home, varied object, 100+ demos),
then **architecture** (dropout, gripper in obs), then **inference** (shorter
chunks, closed-loop gating). Fixing only one layer while leaving the others
active is unlikely to produce reliable pick-place behavior.

---

## References

- de Haan, Jayaraman, Levine — "Causal Confusion in Imitation Learning" — NeurIPS 2019
  https://arxiv.org/abs/1905.11979
- Ross, Gordon, Bagnell — "A Reduction of Imitation Learning and Structured Prediction
  to No-Regret Online Learning" — AISTATS 2011 (DAgger)
- Zhao et al. — "Do You Need Proprioceptive States in Visuomotor Policies?" — 2025
  https://arxiv.org/abs/2509.18644
- Ma et al. — "Causal Diffusion Policy" (CDP) — PMLR 2025
  https://proceedings.mlr.press/v305/ma25c.html
- Zhao et al. — "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware"
  (ACT paper) — RSS 2023
