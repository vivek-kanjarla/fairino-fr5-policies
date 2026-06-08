# Action Space Design — What to Predict and Why It Matters

How the policy represents its output (absolute joint angles vs delta EEF vs absolute EEF)
is one of the most consequential design decisions in a manipulation pipeline. It directly
determines whether the policy can generalize to new object positions, whether temporal
ensembling works, and what the deploy-side robot control looks like.

---

## 0. The FR5 current situation

The FR5 policies currently predict **absolute joint angles**:

```
action = [q1_cmd, q2_cmd, q3_cmd, q4_cmd, q5_cmd, q6_cmd, gripper_norm]  # 7D
```

This is what gets stored in the dataset (from `fr5_cmd_j1..j6`), what the models are
trained on, and what `deploy.py` sends via `robot.servo_j(action[:6])`.

The EEF pose **is already recorded** in every episode CSV:

```python
EEF_COLS = ["fr5_eef_x_mm", "fr5_eef_y_mm", "fr5_eef_z_mm",
            "fr5_eef_rx_deg", "fr5_eef_ry_deg", "fr5_eef_rz_deg"]
```

It is extracted by `convert_episodes.py` and stored in the dataset as
`observation.eef_pose`, but it is not currently used as the action. **No new data
collection is needed to switch action spaces.**

---

## 1. What the field actually uses

ACT (Zhao et al. 2023) is an **outlier** in using absolute joint angles. It was designed
for ALOHA — a custom bimanual robot where direct joint control was natural. Most of the
broader manipulation learning field uses delta EEF:

| Paper / System | Action space | Notes |
|---|---|---|
| ACT (Zhao 2023) | Absolute joint angles | Designed for ALOHA bimanual; unusual choice |
| Diffusion Policy (Chi 2023) | Task-dependent; often absolute joint or EEF pos | Push-T uses 2D EEF pos |
| RT-1, RT-2 (Google 2022, 2023) | Delta EEF (Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper) | 7D standard |
| OpenVLA (Kim et al. 2024) | 7D delta EEF | De facto VLA standard |
| Octo (2023) | Delta EEF (DROID/Open X convention) | — |
| π0, π0.5 (Physical Intelligence) | Delta EEF in Cartesian space | — |
| DROID dataset (2024) | Delta EEF | Large-scale teleoperation dataset |
| Open X-Embodiment | Majority delta EEF; joint-based is minority | ACT-sourced episodes are joint |
| Zhao et al. 2025 (state-free) | Relative EEF — only space that generalizes | See §3 |

**The community standard is delta EEF.** When cross-robot pre-training datasets are
built (Open X, DROID, OXE), they use delta EEF as the common action representation
because it is robot-agnostic: a "move 3cm forward" delta means the same thing on
a Franka, a UR5, or an FR5.

---

## 2. Why action space determines generalization

Zhao et al. 2025 (Table III) tested all four common action spaces on the same task,
same policy, same cameras. Only relative EEF generalizes:

| Action space | Height generalization | Horizontal generalization |
|---|---|---|
| **Relative EEF** (Δx, Δy, Δz, Δrx, Δry, Δrz) | **0.984** | **0.584** |
| Absolute EEF | 0 | 0 |
| Relative joint angle (Δq) | 0 | 0 |
| Absolute joint angle (q) — **FR5 current** | 0 | 0 |

The reason is a single mathematical fact:

```
Absolute joint angles:   action = q_target
                         depends on where the arm is in the room

Relative joint angles:   action = Δq = q_target − q_current
                         still depends: Δp_EEF = J(q_current) · Δq
                         (Jacobian couples joint delta to Cartesian delta)

Absolute EEF:            action = [x, y, z, rx, ry, rz]_target
                         still absolute — height change breaks it

Relative EEF:            action = ΔEEF = EEF_target − EEF_current
                         same visual scene → same delta → same motion
                         table 10cm higher? same visual scene, same delta, works
```

Only relative EEF gives translation invariance: the same visual observation always
produces the same relative motion, regardless of the robot's absolute position in space.

---

## 3. Temporal ensembling with delta EEF — does it work?

Yes, completely. Temporal ensembling (TE) is action-space-agnostic. It averages
overlapping chunk predictions for the same timestep using exponential weighting:

```
w(age) = exp(−m × age)    (m=0.01 in ACT)
```

**With absolute joints (current):**
```
chunk t=0: [q₁,  q₂,  ..., q₁₀₀]   ← absolute arm configurations
chunk t=1: [q₁', q₂', ..., q₁₀₁]
At step t: smoothed_q = Σ w(age) · q_t_prediction  →  robot.servo_j(smoothed_q)
```

**With delta EEF (proposed):**
```
chunk t=0: [Δeef₁,  Δeef₂,  ..., Δeef₁₀₀]   ← relative movements
chunk t=1: [Δeef₁', Δeef₂', ..., Δeef₁₀₁]
At step t: smoothed_Δ = Σ w(age) · Δeef_t_prediction
           target_eef = current_eef + smoothed_Δ  →  robot.servo_cart(target_eef)
```

The math is identical. The only difference is the final "apply" step: absolute targets
go directly to `servo_j`, while deltas are added to the current EEF state first.

### 3.1 One practical issue: orientation wrapping

When averaging orientation deltas (Δrx, Δry, Δrz in degrees), you must normalize
each delta to `[−180°, +180°]` before averaging. Otherwise:

```
chunk A predicts: Δrz = +179°
chunk B predicts: Δrz = −181°    ← same physical rotation, different sign
naive average: (179 + (−181)) / 2 = −1°   ← wrong
correct (after wrapping both to [−180, +180]): average of +179° and +179° = +179°
```

For typical manipulation (small, smooth movements), deltas rarely exceed ±30° so
this almost never triggers in practice.

### 3.2 Does TE still make sense with delta chunks?

Yes, and the physical interpretation is actually cleaner:

| Chunk type | What a 100-step chunk means |
|---|---|
| Absolute joints | "Here are the next 100 absolute arm configurations" — committed to a specific path |
| Delta EEF | "Here are the next 100 incremental end-effector movements" — describes the *shape* of the motion |

The delta chunk is a relative trajectory. It describes how the end-effector should
move, not where it should be. TE averaging over multiple delta chunks smooths the
predicted motion shape. This is physically meaningful: old predictions describe a
slightly outdated motion plan, new predictions describe the updated plan — their
exponentially-weighted average transitions smoothly between them.

---

## 4. The three switchable options for FR5

### Option A — Delta joint angles (easiest, not enough)

Change `action = q_cmd` → `action = q_cmd − q_prev`.

- **Data:** one line in `convert_episodes.py` — diff the CMD columns
- **Deploy:** `robot.servo_j(current_joints + action[:6])`
- **Generalization:** still 0% per Zhao 2025 (FK coupling remains)
- **Worth doing?** Partially breaks absolute-position memorization as a quick test,
  but does not fix the fundamental generalization problem

### Option B — Delta EEF (correct fix)

Change `action = [Δx, Δy, Δz, Δrx, Δry, Δrz, gripper_norm]`.

- **Data:** `convert_episodes.py` — `diff(eef_cols)`, wrap to `[−180, +180]`
- **Deploy:** `target_eef = current_eef + action[:6]; robot.servo_cart(target_eef)`
  OR: `target_eef = current_eef + action[:6]` → IK → `robot.servo_j(ik_result)`
- **Generalization:** 0.984 height, 0.584 horizontal
- **Action dim:** stays 7 (6 EEF deltas + 1 gripper)
- **Units:** mm for position, degrees for orientation — normalise during training

### Option C — Absolute EEF (not worth it)

Per Zhao 2025: 0% generalization, same as absolute joints. Skip.

---

## 5. How it's implemented (config-driven, backward-compatible)

This is **built and wired end-to-end**, not just a proposal. The action space is a
single switch that flows through the whole pipeline; the model architecture, training
loop, and temporal ensembling are **completely unchanged** (action_dim stays 7).

The switch lives at **conversion time** and is then carried automatically through the
dataset → checkpoint → deploy, so a checkpoint always knows how to execute itself.

```
convert_episodes.py --action-space {joint|delta_eef}
        │  writes the chosen action representation into the dataset
        │  AND stamps  meta/info.json["action_space"]
        ▼
train.py  reads dataset info, copies action_space into the checkpoint
        ▼
deploy.py  reads checkpoint["action_space"] and dispatches execution
```

### 5.1 `common/convert_episodes.py` — `--action-space`

```bash
# default — absolute joint angles (no IK at deploy)
python common/convert_episodes.py --episodes episodes --out <ds> --extract-frames

# delta EEF — Cartesian TCP pose deltas
python common/convert_episodes.py --episodes episodes --out <ds> --extract-frames \
    --action-space delta_eef
```

For `delta_eef` it builds `action[t] = eef[t+1] − eef[t]` (per episode; the last frame's
delta is 0), wraps orientation deltas to `[−180, 180]`, appends `gripper_norm`, and stamps
`info.json["action_space"] = "delta_eef"` with the matching feature names
(`delta_eef_x_mm … delta_eef_rz_deg, gripper_norm`). `joint` (default) is the old behavior.

### 5.2 `common/train.py` — carries the tag into the checkpoint

`train.py` reads `train_ds.info["action_space"]` and saves it in the checkpoint
(`ckpt["action_space"]`). Nothing else in training changes — the model trains on whatever
7-D action the dataset contains. **Single source of truth: the dataset's `info.json`.**

### 5.3 `common/deploy.py` — dispatches execution

```python
model, cfg, action_space = load_policy(ckpt)          # action_space from the checkpoint
...
if action_space == "delta_eef":
    current_eef = np.asarray(robot.get_eef_pose())     # [x,y,z (mm), rx,ry,rz (deg)]
    target_eef  = (current_eef + action[:6]).tolist()  # apply the predicted delta
    try:
        joints = robot.inverse_kin(target_eef)         # IK -> joints
        robot.servo_j(joints)                          # proven joint-servo execution
    except IOError as e:                               # unreachable/singular target
        print(f"[IK] {e} — holding pose")             # hold instead of crashing the loop
else:  # "joint" (default)
    robot.servo_j(action[:6].tolist())
```

The **observation side is unchanged**: the policy still receives the 7-D state
(joint angles + gripper). Only the *output* representation and how it's executed change.

Backward compatible: a checkpoint with no `action_space` tag (anything trained before
this change) defaults to `"joint"`, so existing models deploy exactly as before.

---

## 6. Deploy-side: IK vs ServoCart (now in the robot wrapper)

The `FR5Controller` (`so101-fr5-teleop/fr5.py`) gained two methods for the delta-EEF path:

| Method | Wraps (Fairino SDK) | Use |
|---|---|---|
| `inverse_kin(desc_pos)` | `GetInverseKin(0, pose, -1)` → joints | IK then `servo_j` — the **default** delta-EEF path. `config=-1` seeds from the current joints so successive solves stay continuous (no elbow flips). Raises on unreachable/singular targets so deploy can hold pose. |
| `servo_cart(desc_pos, mode)` | `ServoCart(mode, pose, …)` | Direct Cartesian servo. `mode=0` absolute, `mode=1` incremental (feed the delta straight in). Lower latency (no IK round-trip), but can't pre-detect unreachable targets. |

`deploy.py` uses **`inverse_kin` + `servo_j`** by default: it reuses the already-proven
joint-servo execution path and lets you catch unreachable targets before commanding motion.
`servo_cart` is available if you prefer Cartesian servo (especially `mode=1` for deltas).

> **Validate on the robot.** These two wrappers were written against the Fairino SDK API but
> could not be exercised here (the `fairino` package only runs on the robot machine). Confirm
> `GetInverseKin` / `ServoCart` behave/return as expected on your firmware before a full run —
> the IK path's `try/except` already degrades safely (holds the last pose) if a call fails.

---

## 7. Normalization

Delta EEF values have very different scales:

| Dimension | Typical range per step (30Hz) | Unit |
|---|---|---|
| Δx, Δy, Δz | ±0.5 to ±5 mm | mm |
| Δrx, Δry, Δrz | ±0.1 to ±5 degrees | degrees |
| gripper_norm | 0 to 1 | normalized |

Before training, normalize each action dimension to zero mean and unit std (compute
from the training dataset). This prevents the larger-magnitude position deltas from
dominating the loss over the smaller-magnitude orientation deltas.

The existing `dataset.py` already computes `action_mean` and `action_std` — the
normalization code is already there and will just pick up the new action values.

---

## 8. Recommended migration path

```
Step 1  Collect new data with fixed home + workspace grid (Phase 1 from il_failure_modes.md)
        ↓  no action space change yet — validate data fix first
Step 2  Confirm conflict-swap probe improves (image corr > 0.5) with same absolute joints
        ↓  validates that the data is the bottleneck, not the action space
Step 3  Switch convert_episodes.py to delta EEF (Option B above)
        Retrain ACT and diffusion on same new dataset
        ↓
Step 4  Implement ServoCart or IK path in deploy.py
        ↓
Step 5  Re-run conflict-swap probe + spatial generalization test
        Target: height generalization > 80%, image corr > 0.7
```

Do not switch action spaces and fix the data at the same time. You want to know which
change caused the improvement.

---

## References

- Zhao et al. — "Do You Need Proprioceptive States in Visuomotor Policies?" — 2025
  https://arxiv.org/abs/2509.18644  (Table III: action space ablation)
- Chi et al. — "Diffusion Policy" — RSS 2023  https://arxiv.org/abs/2303.04137
- Zhao et al. — "ACT: Learning Fine-Grained Bimanual Manipulation" — RSS 2023
  https://arxiv.org/abs/2304.13705
- Kim et al. — "OpenVLA: An Open-Source Vision-Language-Action Model" — 2024
  https://arxiv.org/abs/2406.09246
- Khazatsky et al. — "DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset" — 2024
  https://arxiv.org/abs/2403.12945
