# Imitation Learning — Known Failure Modes and How the FR5 Hit Them

A comprehensive map of the structural ways behavioral cloning (BC) fails, grounded in the
literature, with a concrete column showing exactly where the FR5 pick-policy experiments
fell into each trap — and a prioritized remediation plan at the end.

---

## 0. Why BC is Fragile by Design

Behavioral cloning trains a policy π(a|o) by supervised learning on (observation, action)
pairs from human demonstrations. The training objective is:

```
L_BC = E_{(o,a) ~ D} [ ||π(o) - a||² ]
```

This formulation has three structural gaps that every failure mode below exploits:

| Gap | What it means | Which failure modes |
|---|---|---|
| No causal reasoning | Minimizes prediction error; can't distinguish cause from correlation | §1, §3 |
| No feedback loop | No mechanism to detect or recover from errors at deploy time | §2, §5 |
| No out-of-distribution handling | Undefined behavior on states not in D | §2, §4 |

BC is strictly worse than online RL at all three. It is used anyway because it is cheap,
safe, and fast to collect demonstrations for — but those advantages come with structural
fragility that must be explicitly designed around.

---

## 1. Causal Confusion / Proprioceptive Shortcut

### 1.1 What it is

BC learns a *discriminative* model: "given this observation, what action did the expert
take?" It does not learn a *causal* model: "which part of the observation *caused* the
expert to take that action?"

These two formulations are equivalent only when the training data has no spurious
correlations. In practice they almost always differ. This is the core result of:

> de Haan, Jayaraman, Levine — "Causal Confusion in Imitation Learning"
> NeurIPS 2019 · https://arxiv.org/abs/1905.11979

The paper formalizes it with Pearl's do-calculus: a BC policy learns P(a|o), but the
causal quantity is P(a|do(o)) — the distribution of actions if you *intervene* to set o,
rather than merely *observe* it. When a non-causal input (e.g. a dashboard light in the
driving demo) correlates with the action (e.g. braking), BC bakes in that correlation
and fails as soon as the correlation breaks at deploy time.

### 1.2 The proprioceptive shortcut specifically

The robot's joint state is numerically clean, low-dimensional (6–13 numbers), and
already perfectly aligned with the action space (the next joint angles to command).
Vision requires learning detection-like features from raw 224×224×3 pixels through a
full ResNet18 — far more capacity and gradient signal to train.

Gradient descent is lazy: if minimizing `||π(state) - a||²` gets the training loss low
enough, it never has to solve `||π(image) - a||²`. The vision pathway gets trained just
enough to not increase the loss, not enough to be reliable.

### 1.3 Why the shortcut forms: the math

In the FR5 setup, the operator started the arm *near* the object before recording. This
created a correlation:

```
corr(q_start, q_grasp) = 0.45     (q = joint angle vector)
```

From the policy's perspective, minimizing the loss requires predicting the grasp
trajectory. The cheapest predictor of q_grasp is q_start (corr=0.45). The next cheapest
is the vision pathway (corr≈0.67 with object position, but only recoverable when state
is neutralized). Since q_start is immediately available in the observation and has lower
prediction error, state wins.

At deploy time q_start is always the home pose — constant across episodes — so the
policy always predicts the same average trajectory.

### 1.4 FR5 evidence

Three trained models probed with causal interventions (`experiments/shortcut_probe.py`):

| Intervention | What changes | What stays | Result |
|---|---|---|---|
| Image ablation | blank cameras | real state | action barely changes |
| State ablation | mean state | real cameras | action moves 3.6–5.2× more |
| Conflict swap | state_i + image_j (different episode) | — | reach follows **state** (corr +0.85–0.88), not image (corr −0.14 to −0.29) |

| Model | State dim | Image-ablation Δ | State-ablation Δ | Ratio | Follows image | Follows state |
|---|---|---|---|---|---|---|
| ACT (paper-scale) | 6 | 5.0° | 26.4° | **5.2×** | −0.14 | **+0.85** |
| Diffusion v1 | 6 | 5.0° | 19.7° | **4.0×** | −0.29 | **+0.85** |
| Diffusion v2 (rich) | 13 | 5.2° | 18.8° | **3.6×** | −0.28 | **+0.88** |

**Counterintuitive result:** Diffusion v2 had a richer state (13-D: joints + eef +
gripper). More state → stronger shortcut (ratio 5.2× → 3.6× is better, but image
correlation −0.28 vs −0.14 is worse — meaning v2 relies on state even more absolutely
and the residual vision signal is even more suppressed). Adding proprioception helped
the shortcut, not the task.

This matches causal confusion theory exactly: a richer shortcut input means less gradient
pressure on the harder vision pathway.

### 1.5 What a healthy model looks like

For comparison, a policy that has learned to use vision should show:

| Metric | Shortcut model (FR5) | Healthy model |
|---|---|---|
| Image-ablation Δ | ~5° | large (policy can't act without cameras) |
| State-ablation Δ | ~26° | small (state is a minor refinement) |
| Ablation ratio | 5× | ≤1.5× |
| Conflict swap → image corr | −0.14 | +0.7 or higher |
| Conflict swap → state corr | +0.85 | ≤0.2 |

### 1.6 Fix

Priority order:

1. **Break the correlation in the data.** Fixed home start + widely varied object
   position. This is the only fix that attacks the root cause at the source.
2. **More demonstrations (100–150+).** Raises the difficulty floor for the state
   pathway, forcing vision to contribute.
3. **Proprioception dropout (20–40%).** During training only: randomly zero the
   entire state vector. The policy can't rely on state for zeroed samples, so the
   vision pathway is forced to carry the loss.

   ```python
   # training loop only — flag is False at eval/deploy
   if training and torch.rand(1).item() < proprio_dropout_rate:
       state = torch.zeros_like(state)
   ```

4. **State-free policy.** Drop proprioceptive input entirely. Zhao et al. 2025 showed
   this improved height generalization 0% → 85%, horizontal 6% → 64%. The vision-only
   policy has no shortcut available and is forced to solve the visual problem.

### 1.7 Modality laziness — why state wins the training race

The proprioceptive shortcut has a deeper mechanism than just "state correlates with
action." There is a specific dynamic in multimodal learning called **modality laziness**:
due to the greedy nature of deep models in joint optimization, the model prioritizes
modalities that converge faster (strong modalities), which suppresses learning of
slower modalities (weak modalities).

For the FR5 policy:

| Modality | Why it's "easy" to learn from | Why it wins gradient competition |
|---|---|---|
| 6 joint angles + gripper | Low-dimensional, noiseless, directly in the same units as the action | Loss drops fast, large gradients early in training |
| Wrist camera (D405) | High-dimensional pixels, changes with every arm movement | Loss drops slowly, gradients are weak |
| Scene camera (D435i) | Fixed viewpoint but lighting variation, background clutter | Weakest signal, most redundant with wrist |

What actually happens during training:

```
Epoch 1–5:   State branch converges quickly → loss drops sharply
Epoch 5–50:  State branch already doing well → image gradients are tiny
             (adding more from image barely changes the loss)
Epoch 50+:   State branch frozen in good solution → image encoder
             gets near-zero gradient → never learns to detect the object
```

The final policy is effectively: `action ≈ f(state) + ε · g(image_features)`
where ε is very small. When you zero out the images, `g(zeros)` maps to some
near-zero embedding (a consistent offset the policy has absorbed), and the policy
keeps working because the state branch carries almost all the weight.

This explains why the blank-image test works on the FR5: by the time training ended,
the image encoder had learned to represent nearly nothing useful. The training
objective was already satisfied without it.

**The fix for modality laziness** is to make the state branch unavailable for some
fraction of training (dropout), or to remove it entirely, forcing the optimizer to
route gradient through the image encoder.

### 1.8 The in-distribution vs OOD distinction — why original ACT "worked"

A critical nuance: ACT and similar policies are **not** pure trajectory replay. The
original ACT paper explicitly demonstrates disturbance recovery — physical perturbations
mid-task where the robot recovers using visual feedback. ACT does use vision
meaningfully within its training distribution.

The key phrase is *within its training distribution.* ACT was evaluated with object
positions randomized along a 15cm reference line — **the same 15cm range for both
training and testing**. That is in-distribution spatial variation. The robot was never
tested on table heights, object positions, or approach angles outside that range.

What happens outside training distribution? A 2025 benchmarking study specifically
tested ACT on spatial out-of-distribution settings:

| Task | ACT | π0 | Gap |
|---|---|---|---|
| Clean Dish (OOD) | 36% | 85% | −49% |
| Put Sponge in Pot (OOD) | 32% | 80% | −48% |
| Unzip Bag (OOD) | 3% | 56% | −53% |
| Folding Shorts (OOD) | 0% | 0–46% | — |

ACT degrades significantly OOD — it doesn't completely fail, but it clearly has much
weaker spatial generalization than a model trained on diverse data (π0).

**The fine-tuning regime problem.** π0's advantage comes from pre-training on 10,000+
hours of demonstration data across 68 tasks and 7 robot configurations. At that scale,
the same joint state maps to dozens of different correct actions depending on the task
and object position — the state shortcut can no longer memorize a single trajectory,
so the visual backbone is forced to develop. But — and this is what the Zhao et al.
2025 paper targets — when you fine-tune π0 on your narrow real-world data (a few
hundred episodes, same table height, same approximate object zone), the shortcut comes
back. Fine-tuning on low-diversity data overwrites the learned visual features with a
state-to-action mapping in the fine-tuning distribution.

**Your FR5 situation:** you are not running ACT or π0 with 10,000 hours of pre-training.
You are fine-tuning on ~54 pick-place demos, all at the same table height, in the same
approximate object region. This is the exact regime where state dominance is most severe.
The blank-image test confirming the shortcut is not a property of ACT or diffusion
policy as architectures — it is a property of your dataset.

---

## 2. Covariate Shift and Compounding Errors

### 2.1 What it is

BC trains on (o, a) pairs where the observation o comes from the *expert's* trajectory.
At deploy, the robot takes an action, lands in a new state, and generates a new
observation. If that new observation differs even slightly from what the expert would
have produced, the policy is now being queried on an out-of-distribution input.

The problem compounds: a small error at step t puts the robot in state s' ≠ s_expert.
The policy predicts an action from s', which may be wrong, landing in s'' further from
the expert trajectory. Each step the errors accumulate.

### 2.2 The formal bound

Ross & Bagnell 2011 (DAgger paper) prove:

```
E[cost_BC]    ≤  T · ε_BC + O(T²)        BC under covariate shift
E[cost_DAgger] ≤  T · ε_DAgger + O(T)    interactive imitation learning
```

Where ε is the per-step expected loss under the learner's own state distribution.
The O(T²) term for BC means: on a 10-second horizon (300 steps at 30 Hz), errors grow
with T² — catastrophically faster than linear. This is why long-horizon manipulation
tasks fail even when short demos look fine.

### 2.3 How it shows up

The robot starts plausibly — the first 200–300ms of ACT's chunk or the first few
diffusion steps look like the demo. Then a joint overshoots by 2°. The next observation
is slightly wrong. The policy was never trained on this exact configuration, so the next
action is also slightly wrong. By second 2, the arm is doing something that no human
ever demonstrated and the policy produces garbage.

Signature: **correct for 1 chunk, then rapidly degrades.** Often produces jerky or
oscillatory motion — the policy is jumping between nearby training trajectories that
share similar (but not identical) observations.

### 2.4 FR5 evidence

The "went absolute nuts" episodes. The model's first 300ms (one ACT chunk) looked
plausible, then the reach deviated, the gripper went to the wrong height, and the final
position was completely wrong.

The VRworking repo executed 100 actions (3.3 seconds) completely open-loop — no
replanning. This is the worst-case covariate shift scenario: 3.3 seconds of error
accumulation with T²=1089× penalty.

ACT's temporal ensembling partially mitigates this: instead of committing to a 100-step
chunk, each tick re-plans with the current state observation. The re-planning anchors the
trajectory back toward a reasonable expert state every 33ms, limiting how far the error
can compound before correction.

### 2.5 Interaction with proprioceptive shortcut

The two failure modes amplify each other:

```
Step 1: Proprioceptive shortcut → wrong reach direction
Step 2: Arm arrives at wrong location → never seen in training
Step 3: Covariate shift → policy produces incoherent recovery action
Step 4: Robot is now far from any training state → pure garbage
```

If you fix the shortcut (vision is now used) but don't fix the data coverage, covariate
shift is still active whenever the arm reaches a novel view angle or lighting condition.

### 2.6 Fix

| Fix | Effort | Effect |
|---|---|---|
| More demos covering the workspace | Medium | Reduces the size of the OOD region |
| Shorter action chunks + temporal ensembling | None (already in ACT) | Re-plans every 33ms, limits accumulation |
| DAgger | High | Directly collects demos at the states the policy actually reaches |
| Ensemble / uncertainty gating | Medium | Detects OOD states, triggers recovery behavior |

**DAgger in practice:** run the policy, record the states it reaches, ask the operator
to demonstrate from those states, add to the dataset, retrain. Repeat. Expensive but
the only method with a theoretical guarantee.

---

## 3. Mode Averaging / Multimodal Demonstration Collapse

### 3.1 What it is

When demonstrations are multimodal — the expert sometimes approaches from the left,
sometimes from the right — the dataset contains two action clusters for the same (or
similar) observation. BC with L2 loss minimizes:

```
E[ ||π(o) - a||² ]
```

When a has two modes a_L and a_R with equal probability, the loss-minimizing prediction
is:

```
π*(o) = E[a|o] = 0.5 * a_L + 0.5 * a_R
```

This is the average — a trajectory that goes *between* the two approaches, which is
physically wrong and fails the task. The model doesn't know it's supposed to pick one;
it just minimizes the squared error.

### 3.2 Why action chunking helps but doesn't fully solve it

ACT and diffusion policy predict an entire trajectory chunk, not a single action. For
chunked prediction, the multimodality is even more severe — two entire future trajectories
are averaged, not just two actions. The averaged chunk is a physically impossible
compromise path.

ACT's CVAE is the explicit fix: the latent z encodes which mode is intended. At inference,
z is sampled from the prior (or set to 0), which commits to one mode rather than averaging.
Diffusion handles it by converging to a single sample from the data distribution, not the
mean — the stochastic sampling process naturally lands in one mode or the other.

### 3.3 FR5 evidence

Less severe here because pick-place is mostly unimodal (single object, single target, one
reasonable approach direction). However:

- If different operators recorded demos with different preferred approach angles, those
  contribute to bimodality
- If the object was sometimes grasped from above and sometimes from the side, those are
  two modes
- The "sometimes works, sometimes absolute nuts" pattern is more consistent with
  covariate shift (§2) than mode collapse — but if the policy sometimes reaches left and
  sometimes right with no clear trigger, mode averaging is a contributing factor

### 3.4 Fix

- ACT (CVAE with z=0): already handles this in your setup
- Diffusion policy: handles it naturally through sampling
- For pure BC: Gaussian Mixture output heads, or energy-based models
- Data curation: ensure consistent demonstration style (same approach direction, same
  grasp strategy across all episodes)

---

## 4. Data Insufficiency and Coverage Gaps

### 4.1 What it is

BC can only generalize to states that are within the "support" of the training
distribution — i.e., states that are close enough in observation space that the
learned features transfer. With sparse demos, most of the workspace is uncovered.

The problem is especially acute for vision-based policies because the observation space
(224×224×3 = 150K dimensions) is astronomically large. The ResNet18 features must be
trained to be invariant to lighting, view angle, background clutter, and object pose
variation. This requires seeing enough *diverse* examples that the invariances are
learned — not just enough examples total.

### 4.2 Coverage in different axes

| Axis | FR5 status | Required coverage |
|---|---|---|
| Object X position | Limited region | Full workspace grid |
| Object Y position | Limited region | Full workspace grid |
| Object Z (height) | Fixed table | Vary ±5cm at minimum |
| Approach angle | Operator-dependent, variable | Fixed one approach OR all angles |
| Lighting | Lab lighting (consistent) | OK if deployment conditions match |
| Object orientation | Fixed | Vary if object can be placed in different orientations |
| Background clutter | None | Add some clutter to prevent background memorization |

### 4.3 The 100-demo threshold

Community consensus across the ACT, Diffusion Policy, and lerobot communities:

- **<50 demos:** vision pathway rarely wins against the proprioceptive shortcut
- **50–100 demos:** vision starts contributing but is unreliable; shortcut is still
  dominant for novel object positions
- **100–150 demos:** threshold where vision-primary behavior begins to emerge
- **150–300 demos:** reliable generalization across a defined workspace region
- **300+ demos:** needed for large workspaces or high object diversity

The FR5 had ~54 demos. This placed it firmly in the "shortcut dominant" regime.

### 4.4 Why quantity is not enough — diversity matters more

Consider two datasets:
- Dataset A: 200 demos, all with the object in the same 5cm region
- Dataset B: 80 demos, object uniformly spread across a 40×40cm workspace

Dataset B will produce a more vision-reliant policy despite having fewer total demos.
Coverage of the *state space* is what matters, not the total count.

Practical implication: before collecting 150 demos, define a grid of object positions and
ensure each grid cell has at least 3–5 demos. A structured sweep is better than a random
collection.

### 4.5 FR5 evidence

54 episodes, object placed in a limited region. The shortcut formed precisely because
there wasn't enough workspace diversity to make vision necessary. The bigger ACT model
(94M params) had *the same validation loss* as the small one — this is the classic
signature of data bottleneck rather than capacity bottleneck. More model capacity doesn't
help when the training data doesn't require it.

### 4.6 Fix

1. Define a workspace grid (e.g. 4×4 = 16 cells, 3cm per cell)
2. Collect at least 5 demos per grid cell = 80 demos minimum
3. Extend the grid further: 5×5 = 25 cells × 5 = 125 demos
4. After collecting, run the conflict-swap probe — if image correlation rises above +0.5,
   vision is starting to contribute

---

## 5. Open-Loop Execution Blindness

### 5.1 What it is

Action chunking policies commit to a sequence of actions without observing the result
of each one. If the environment deviates — the gripper slips, the object shifts, the
arm oscillates — the policy has no mechanism to detect this and continues executing the
pre-planned chunk regardless.

This is not a training failure — it is a fundamental design property of open-loop
execution. The policy doesn't know what it doesn't know.

### 5.2 The gripper feedback problem specifically

The most critical place to lack feedback is the grasp. The sequence is:

```
1. Reach (arm moves toward object)
2. Close gripper (gripper command sent)
3. Lift (arm moves upward)
```

If step 2 fails (gripper closed on empty air), step 3 executes anyway and lifts nothing.
The policy has no signal that the grasp failed. Without `gripper_norm` in the
observation, the policy at step 3 doesn't even know whether its own gripper is open or
closed.

### 5.3 FR5 evidence

The VRworking version had `state_dim=6` — no gripper state in the observation. The
policy executed 100 actions (3.3 seconds) open-loop. If the grasp failed, the lift
proceeded. The result: robot executes a perfect-looking grasp motion, lifts its arm,
and holds nothing.

This session's fix: `state_dim=7` now includes `gripper_norm` (current gripper opening,
0=closed, 1=open). The policy now observes its own gripper state and can condition on
it at every replanning step.

### 5.4 Residual problem after the fix

Even with `gripper_norm` in the observation:
- ACT's temporal ensembling re-plans every tick, so the policy *does* see the gripper
  state before each action chunk
- But if the shortcut is active (§1), it may not use the gripper observation correctly
- Force/tactile feedback would provide a more reliable grasp-success signal

### 5.5 Fix, ordered by cost

| Fix | Difficulty | What it gives |
|---|---|---|
| Add `gripper_norm` to obs | Done | Policy can observe own gripper state |
| Grasp-success gate | Low | After "close" command: if `gripper_norm > 0.5` after 200ms, abort and retry |
| Temporal ensembling (ACT) | Already on | Re-plans every 33ms, partially closes the loop |
| Force/torque sensing | Medium hardware | Detect contact, slip, load |
| Tactile sensor (finger pads) | High hardware | Rich contact signal |

---

## 6. Action Space Design Failures

### 6.1 Absolute vs relative actions

The action space design has a major impact on how well BC generalizes.

**Absolute joint angles** (what FR5 uses currently): the action a_t is the target joint
configuration in degrees. This is easy to implement and accurate to execute. But it has
a critical property: *the absolute value is tightly coupled to the workspace location*.
If the policy predicts `J1=45°` it means "reach to this specific position in the room,"
not "reach 10° to the right of where I am."

This coupling makes absolute actions a strong shortcut amplifier. The policy can memorize
"from home pose, the grasp is at J1=52°, J2=−30°, ..." — pure proprioceptive
autoregression with no vision required.

**Relative actions** (delta joint angles or delta EEF position): the action a_t is the
*change* from the current state. A_t = q_{t+1} - q_t. This breaks the coupling: the
same relative action "move forward 3cm" is valid from many starting positions, so the
policy must use vision to determine *which* relative action to take at each step.

Zhao et al. 2025 specifically attributes part of their generalization improvement to
relative EEF action space, combined with state-free observation.

### 6.2 FR5 evidence

The FR5 policies use absolute joint angle targets. Combined with the proprioceptive
shortcut, this creates a particularly tight shortcut loop:

```
obs: q_home (fixed) → predict: q_grasp (memorized) → action: go to q_grasp
```

No vision required at any step. Switching to delta actions would partially break this
because the policy can no longer directly output a memorized absolute target.

### 6.3 The action space generalization table

Zhao et al. 2025 Table III tested four action space choices, all with a state-free
policy (vision only), on pick-place tasks. The result is unambiguous:

| Action space | Height generalization | Horizontal generalization |
|---|---|---|
| **Relative EEF** (Δx, Δy, Δz) | **0.984** | **0.584** |
| Absolute EEF | 0 | 0 |
| Relative joint angle (Δq) | 0 | 0 |
| Absolute joint angle (q) | 0 | 0 |

Only relative end-effector actions generalize. The reason:

```
Absolute joint angles:   action = q_t+1                            depends on absolute pose
Relative joint angles:   action = Δq   = q_t+1 − q_t              depends on current joints via FK
Absolute EEF:            action = EEF_t+1                          still tied to absolute Cartesian
Relative EEF:            action = ΔEEF = EEF_t+1 − EEF_t          same visual scene → same action
```

Relative EEF is the only space where the same visual observation reliably produces the
same action regardless of where the robot's absolute pose is. A "move 3cm forward"
command means the same thing whether the table is 80cm high or 90cm high.

The FR5 currently uses **absolute joint angles** — the worst possible choice for
generalization. This directly amplifies the proprioceptive shortcut: the policy can
trivially memorize `home_pose → grasp_pose` as a fixed lookup table.

### 6.4 Fix

- Switch to relative EEF: `a_t = [Δx, Δy, Δz, Δrx, Δry, Δrz, Δgripper]` in Cartesian space
- This requires a forward kinematics layer to convert policy output to joint commands at
  execution time (most robot controllers support Cartesian EEF mode natively)
- Or as a cheaper intermediate: delta joint angles `a_t = q_{t+1} - q_t`, which at
  least removes the absolute memorization problem, though FK coupling remains

---

## 7. Observation Space Design Failures

### 7.1 Camera placement and occlusion

The wrist-mounted camera (D405) is critical for close-range grasping but has two
structural problems:

1. **Self-occlusion:** the gripper fingers block the view of the object at close range.
   At the exact moment the gripper closes, the camera can't see the object anymore.
   The policy must predict the grasp from a frame where the target is occluded.

2. **Viewpoint change during reach:** as the arm moves, the wrist camera's viewpoint
   changes continuously. The visual features at the start of the reach (t=0) are
   completely different from those at the grasp (t=T). The policy must track this
   viewpoint shift across the entire trajectory. With 54 demos, that's not enough
   visual diversity to learn a stable object representation across all approach angles.

The scene camera (D435i) provides a fixed viewpoint and is better for high-level
positioning, but lower resolution for fine manipulation.

### 7.2 Temporal context

ACT uses only the current observation — no history. But many manipulation situations
are ambiguous from a single frame:

- Is the gripper opening or closing? (can't tell from one frame)
- Is the arm still moving or stopped? (velocity is invisible in a single frame)
- Did the grasp succeed? (object may be occluded immediately after grasping)

Policies that receive a history of recent observations (a stack of the last k frames,
or a recurrent hidden state) can resolve these ambiguities. ACT's CVAE partially
compensates by encoding the entire demonstration history into z during training, but
z=0 at inference means this history is not available at deploy time.

### 7.3 Image resolution and crop

A 224×224 image covers the full scene. If the object is small (e.g. a 3cm cube at
50cm distance), it occupies roughly 20×20 pixels — about 0.8% of the total image area.
The ResNet18 features must localize this tiny region among 150K input pixels. With
limited demos, the network may never learn to reliably localize small objects because
the gradients from that 0.8% are swamped by the rest of the image.

Fix: crop the image to the region of interest (RoI) before feeding to ResNet, or use
a higher-resolution camera for fine manipulation.

### 7.4 The overhead camera problem — more cameras can hurt

Counterintuitive finding from Zhao et al. 2025 (Table VII): adding an overhead/scene
camera actually *hurt* spatial generalization in multiple conditions:

| Setup | In-domain (100cm) | Table raised (up 10cm) | Holder shifted (20cm) |
|---|---|---|---|
| **With overhead camera** | 0% | 46.7% | 0% |
| **Wrist camera only** | **100%** | **86.7%** | **80%** |

The overhead camera provides a global view of the workspace — which means when the
table height or object position changes, the overhead view changes too. The policy
trained with an overhead camera has learned to rely on specific visual features in
that fixed overhead view. Move the table 10cm and the overhead scene looks different,
creating distribution shift even in the visual branch.

The wrist camera, by contrast, naturally follows the end-effector. Its view of the
object stays roughly consistent regardless of table height (the camera moves with
the arm), making it more robust to spatial variations.

**For the FR5:** the D435i scene camera is a scene/overhead camera. Based on this
finding, removing it (or at least ablating it) and relying only on the wrist D405
may improve spatial generalization. Before adding more cameras, check whether each
one is actually helping or introducing a new source of distribution shift.

### 7.5 FR5 evidence

The scene camera + wrist camera combination is well-designed in principle. The issues
are:

- Wrist camera self-occlusion during grasp closing (unavoidable without redesign)
- 54 demos insufficient to learn stable visual features across approach angles
- No image crop or RoI — full 224×224 frames fed to ResNet
- Scene camera may be creating distribution shift as a second source of spatial context
  (remove it and test with wrist-only to validate)

---

## 8. Dataset Bias and Operator-Specific Behavior

### 8.1 Operator fingerprints

Each human operator has characteristic movement patterns: preferred approach angles,
velocity profiles, wrist rotation tendencies, grasp styles. If all demos are recorded
by one operator (or two operators with similar styles), the policy learns that operator's
*style*, not the *task*.

At deploy time, the policy will reproduce the operator's style regardless of whether it
is appropriate for the current object position. This is a subtle form of causal
confusion: the policy learned to imitate the person, not to solve the task.

### 8.2 Demonstration quality variance

Not all demonstrations are equally good. A tired or distracted operator may produce
demos with:

- Corrective sub-motions (hesitation, small back-and-forth)
- Non-smooth trajectories that look like noise to the policy
- Inconsistent grasp heights across the dataset

BC treats all demos equally. A dataset with 10 excellent demos and 40 inconsistent ones
may perform worse than a dataset with 40 consistently good demos.

### 8.3 FR5 evidence

The demos were recorded by a single operator (Vivek, from the VRworking context). The
operator's tendency to start near the object (creating the start-pose correlation)
is itself an operator-specific behavior. A different operator with a "home first, then
approach" style would not have created the same shortcut.

### 8.4 Fix

- **Multiple operators** recording demos, or at least one operator using a consistent
  and principled approach style
- **Demo quality filter:** throw out demos where the task failed or where the trajectory
  is clearly inconsistent (high jerk, grasp missed, etc.)
- **Standardize approach:** define a fixed approach direction and stick to it across all
  recordings

---

## 9. How All Failures Interact — the FR5 Compounding Chain

The failure modes don't operate in isolation. On the FR5, all five active modes
amplified each other:

```
DATA COLLECTION (54 demos, operator starts near object)
    │
    ├─► [§1] Proprioceptive shortcut forms
    │         (start-pose corr=0.45 → state predicts grasp → vision ignored)
    │
    ├─► [§4] Data insufficiency: workspace uncovered
    │         (vision has insufficient diversity to build reliable features)
    │
    │   At deployment:
    │
    ├─► [§1] Constant home pose → same average reach every time
    │
    ├─► [§5] Reach goes to wrong location → gripper closes on air
    │         (no gripper feedback in VRworking version)
    │
    └─► [§2] 3.3s open-loop execution → errors compound with T²
              robot has never been trained on the post-grasp-failure state
              → produces incoherent recovery actions → "absolute nuts"
```

The occasional success: when the object happened to be placed near the "average"
location that the shortcut predicts (~the same area the operator tended to use),
the policy worked. When the object was elsewhere, it failed completely.

This is why **fixing one mode while leaving others active rarely produces reliable
behavior.** The system needs all the active failure modes addressed simultaneously.

---

## 10. Diagnostic Guide — How to Tell Which Mode You're In

Before collecting more data or retraining, run these checks to identify which failure
modes are active.

### 10.1 Conflict-swap probe (causal confusion test)

```python
# Take state from episode i, image from episode j (different object position)
# Run policy, measure where the reach goes
# If reach follows state_i: shortcut active
# If reach follows image_j: vision is being used
action_swap = policy(state=ep_i.state, image=ep_j.image)
corr_with_state = pearsonr(action_swap, ep_i.expert_action)
corr_with_image = pearsonr(action_swap, ep_j.expert_action)
```

| Result | Diagnosis |
|---|---|
| `corr_state > 0.7` | Shortcut active (§1) |
| `corr_image > 0.7` | Vision is working |
| Both low | Mode averaging or high variance |

### 10.2 Ablation ratio (relative importance test)

```python
Δ_image = ||action(full_obs) - action(blanked_camera)||
Δ_state = ||action(full_obs) - action(mean_state)||
ratio = Δ_state / Δ_image
```

| Ratio | Diagnosis |
|---|---|
| >3× | State dominates → shortcut active (§1) |
| ~1× | Balanced use of both inputs |
| <0.5× | Vision dominates → healthy or shortcut-free |

### 10.3 Rollout timeline analysis

Record the exact step where the rollout first diverges from the expected trajectory:

| Divergence point | Diagnosis |
|---|---|
| From step 1 | Wrong reach direction → shortcut (§1) or mode collapse (§3) |
| After first chunk (~300ms) | Covariate shift kicking in (§2) |
| After grasp attempt | Open-loop blindness / grasp failure (§5) |
| Random, no pattern | Data coverage gaps (§4) or multimodal instability (§3) |

### 10.4 Training vs validation loss gap

| Gap | Diagnosis |
|---|---|
| Val loss ≈ Train loss, both high | Underfitting (not enough capacity or training) |
| Val loss >> Train loss | Overfitting to demo style (§8) |
| Val loss ≈ Train loss, both low, but deploy fails | Shortcut (§1) — metric doesn't capture it |
| Bigger model doesn't lower val loss | Data bottleneck (§4), not capacity |

---

## 11. Prioritized Remediation Plan for FR5

Ordered by impact-per-effort. Do these in sequence, not all at once — each fix has a
measurable test, so you can confirm it worked before adding the next.

### Phase 1: Fix the data (highest impact, must do first)

1. **Set a fixed home position for every recording.** The robot must start from exactly
   the same joint configuration before every demo. This breaks `corr(start, grasp)`.
2. **Define a 4×4 workspace grid** and place the object at each grid position across
   recordings. 4–5 demos per cell = 64–80 demos with guaranteed workspace coverage.
3. **Validate with the shortcut probe.** After retraining: if `corr_image > 0.5` in the
   conflict swap, the fix is working. If `corr_state` is still dominant, collect more demos.

Expected outcome: shortcut ratio drops from 5× toward 1–2×, image correlation rises.

### Phase 2: Architecture and training changes (add after Phase 1)

4. **Proprioception dropout 20–30%.** Add to the training loop.
5. **Verify `state_dim=7`** is active (gripper in observation). Already done in this
   session — confirm configs and model defaults.
6. **Consider relative action space.** Implement delta joint angles and retrain — this
   removes the absolute-position shortcut.

### Phase 3: Inference changes (add after Phase 2)

7. **Grasp-success gate.** After the "close gripper" step: check `gripper_norm`. If
   `gripper_norm > 0.5` after 200ms (gripper is open = missed), abort and retry the
   approach from 5cm above.
8. **Temporal ensembling** (ACT only — already implemented). Confirm `m=0.01` is being
   used so re-planning happens every tick.

### Phase 4: Structural improvements (if Phase 1–3 still insufficient)

9. **State-free policy.** Remove proprioceptive state input entirely and retrain.
   Only attempt this after the data coverage (Phase 1) is fixed — a vision-only policy
   on sparse data is worse than a shortcut model.
10. **Delta EEF action space.** Move from joint angles to delta Cartesian EEF pose.
    More generalizable and less coupled to absolute position.
11. **Image cropping / RoI.** Crop to the workspace region before passing to ResNet to
    reduce background memorization and improve object localization.

---

## 12. Summary Audit Table

| Failure mode | In literature since | FR5 hit it? | Severity | Status |
|---|---|---|---|---|
| Causal confusion / proprioceptive shortcut | de Haan 2019 | **Yes — probes confirm** | Critical | Partially fixed (state_dim=7, dropout in config); data fix needed |
| Modality laziness | CVPR 2024 | **Yes — state won gradient race** | Critical | Same fix as above |
| Covariate shift / compounding errors | Ross 2010 (DAgger) | **Yes — "went absolute nuts"** | High | Partially mitigated by TE; DAgger not done |
| Mode averaging / collapse | — | Partial | Low | Handled by ACT CVAE |
| Data insufficiency + coverage gaps | Community consensus | **Yes — 54 demos** | Critical | Not fixed — needs new data collection |
| Open-loop execution blindness | — | **Yes — 3.3s, no gripper fb** | High | gripper_norm added to obs; grasp gate not implemented |
| Absolute action space coupling | Zhao 2025 | Active | Medium | Not addressed; relative EEF recommended |
| Overhead camera distribution shift | Zhao 2025 | Possible | Medium | Not tested — ablate D435i and check |
| Operator-specific bias | — | Likely | Medium | Not addressed yet |

All critical failures (shortcut + modality laziness + data coverage) must be fixed
before the policy can generalize reliably. The other modes can be addressed incrementally.

---

## 13. World Models — Do They Have the Same Problem?

A natural question: do world model architectures (Dreamer, TD-MPC, Cosmos-Policy) suffer
from the proprioceptive shortcut and the other failure modes above? The answer depends
heavily on *which* type of world model.

### 13.1 Three types of world models in robotics

| Type | Examples | How it works |
|---|---|---|
| **RSSM-based** | DreamerV3, DayDreamer | Learn latent dynamics from visual + proprio; policy trained via RL in imagination |
| **Video-generation WAMs** | Cosmos-Policy, GE-Act, LingBot-VA | Built on video diffusion models pre-trained on internet video; actions decoded from video latents |
| **TD-MPC style** | TD-MPC2 | Task-oriented latent dynamics model; MPC planning with learned value function |

### 13.2 RSSM (Dreamer) — structurally more resistant, not immune

The RSSM training objective is fundamentally different from BC:

```
BC loss:    ||π(state, image) − action||²         ← state directly predicts action
RSSM loss:  ||future_image_predicted − future_image_actual||² + KL(posterior||prior)
```

To reconstruct a future image of "gripper approaching object from above," joint angles
alone are insufficient — you need to encode what the scene actually looks like. The
visual reconstruction objective **forces** the latent state to contain visual features.
The latent state compresses both visual and proprioceptive information, but since the
reconstruction target is visual, the latent cannot ignore the image pathway the way BC
can.

The policy that acts on this latent state gets a visually-feature-rich representation,
not raw joint angles. This is the structural reason RSSM models are less susceptible to
the state shortcut.

**But RSSM has its own failure mode:** distributional brittleness in imagination.

```
State shortcut risk: LOW
New failure mode: compounding prediction errors in imagined rollouts
                  → small errors in contact, friction, object properties accumulate
                  → imagined trajectory diverges from reality in OOD conditions
                  → planning with a wrong world model is worse than no planning
```

If the table moves 10cm, the imagined future frames look different from anything in
training. The world model's predictions diverge, the planned trajectory is wrong, and
the policy fails — not because it ignored vision, but because its visual dynamics model
was not trained on the new configuration.

### 13.3 Video WAMs (Cosmos-Policy) — most resistant

This is where the architecture is genuinely different. Video WAMs like Cosmos-Policy and
LingBot-VA are built on video diffusion backbones pre-trained on internet-scale video
**with no robot proprioception at all**. The visual dynamics representations were built
entirely from pixels across millions of videos.

When robot-specific fine-tuning is added:
- The visual backbone already has deep spatiotemporal understanding
- There is no history of the model relying on joint angles — the shortcut never formed
- Proprioception is added as a conditioning signal on top of a 2B-parameter visual model
  — it is very hard for a narrow fine-tuning dataset to override a backbone this strong

Empirical results from the 2025 WAM robustness benchmarking study:

| Architecture | Robustness under visual/language perturbations |
|---|---|
| LingBot-VA (WAM) | 74.2% on RoboTwin 2.0-Plus |
| Cosmos-Policy (WAM) | 82.2% on LIBERO-Plus |
| π0.5 (VLA) | Comparable on some tasks, weaker overall |
| ACT (BC) | 3–36% on OOD tasks |

The video WAM advantage comes from the pre-training making vision the dominant modality
*by construction*, not by trying to prevent state from dominating.

**The catch:** WAMs require massive compute (video diffusion backbone ≥2B params),
internet-scale pre-training infrastructure, and expensive fine-tuning. Not practical for
a lab-scale FR5 setup today.

### 13.4 TD-MPC — somewhere in between

TD-MPC learns task-oriented latent dynamics and plans via MPC with a learned value
function. It is less susceptible to BC-style state shortcuts because the dynamics model
must predict *transitions*, not just actions. But task-specific representations don't
transfer well, and the same distributional brittleness as RSSM applies.

### 13.5 The deeper principle

The proprioceptive shortcut is a special case of a general principle:

> **Whichever input modality provides the cheapest path to minimizing the training loss
> will dominate, regardless of which modality is causally relevant.**

| Architecture | Cheapest loss path | Dominant modality | Failure mode |
|---|---|---|---|
| BC (ACT, Diffusion) | State → action lookup | Proprioception | Trajectory memorization |
| RSSM (Dreamer) | State + visual → future visual | Both (forced by reconstruction) | OOD imagination errors |
| Video WAMs | Visual dynamics only (pre-training) | Vision | Fine-tuning distribution shift |

The state-free policy paper's fix — remove the shortcut-prone input entirely — is the
most surgical solution. World models achieve something similar but through a more
expensive route: making the training objective hard enough that state alone is
insufficient to satisfy it.

For the FR5 at current scale: RSSM world models would require online RL data collection
(expensive, dangerous on real hardware). Video WAMs are too compute-heavy. The practical
path is BC with state-free + relative EEF action space — the Zhao et al. 2025 approach,
which gives WAM-level spatial generalization at BC cost.

---

## References

- de Haan, Jayaraman, Levine — "Causal Confusion in Imitation Learning" — NeurIPS 2019
  https://arxiv.org/abs/1905.11979
- Ross, Gordon, Bagnell — "A Reduction of Imitation Learning and Structured Prediction
  to No-Regret Online Learning" — AISTATS 2011 (DAgger)
  https://arxiv.org/abs/1011.0686
- Zhao et al. — "Do You Need Proprioceptive States in Visuomotor Policies?" — 2025
  https://arxiv.org/abs/2509.18644
- Ma et al. — "Causal Diffusion Policy" (CDP) — PMLR 2025
  https://proceedings.mlr.press/v305/ma25c.html
- Chi et al. — "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"
  RSS 2023 · https://arxiv.org/abs/2303.04137
- Zhao et al. — "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware"
  (ACT paper) — RSS 2023 · https://arxiv.org/abs/2304.13705
- "Shortcut Learning in Generalist Robot Policies" — 2024
  (Open X-Embodiment shortcut analysis) · https://arxiv.org/abs/2508.06426
- Zhang et al. — "Multimodal Representation Learning by Alternating Unimodal Adaptation"
  (modality laziness) — CVPR 2024 · https://arxiv.org/abs/2311.10707
- "Experiences from Benchmarking Vision-Language-Action Models for Robotic Manipulation"
  2025 · https://arxiv.org/abs/2511.11298
- "Do World Action Models Generalize Better than VLAs? A Robustness Study" — 2025
  https://arxiv.org/abs/2603.22078
- Ha & Schmidhuber — "World Models" — NeurIPS 2018
- Hafner et al. — "Mastering diverse control tasks through world models" (DreamerV3)
  Nature 2025 · https://www.nature.com/articles/s41586-025-08744-2
- Hansen et al. — "TD-MPC2: Scalable, Robust World Models for Continuous Control" — 2024
