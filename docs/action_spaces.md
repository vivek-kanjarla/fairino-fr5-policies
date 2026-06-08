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

## 5. What changes in the codebase

The model architecture, training loop, and TE inference are **completely unchanged**.
Action dim stays 7. Chunk size stays 100. The only two files that touch action
representation:

### 5.1 `common/convert_episodes.py`

```python
# Current — absolute joint commands
action = np.concatenate(
    [rows[CMD_COLS].to_numpy(np.float32),
     rows[[GRIPPER_COL]].to_numpy(np.float32)],
    axis=1,
)

# New — delta EEF (Option B)
eef = rows[EEF_COLS].to_numpy(np.float32)
delta_eef = np.diff(eef, axis=0, prepend=eef[:1])  # Δ[t] = eef[t] - eef[t-1]; Δ[0]=0

# wrap orientation columns (cols 3,4,5) to [-180, 180]
delta_eef[:, 3:6] = (delta_eef[:, 3:6] + 180) % 360 - 180

gripper = rows[[GRIPPER_COL]].to_numpy(np.float32)
action = np.concatenate([delta_eef, gripper], axis=1)   # (N, 7)
```

Also update `info.json` feature names:
```python
"action": {"dtype": "float32", "shape": [7],
           "names": ["delta_eef_x_mm", "delta_eef_y_mm", "delta_eef_z_mm",
                     "delta_eef_rx_deg", "delta_eef_ry_deg", "delta_eef_rz_deg",
                     "gripper_norm"]}
```

### 5.2 `common/deploy.py`

```python
# Current
joints_cmd  = action[:6].tolist()
gripper_cmd = float(action[6])
robot.servo_j(joints_cmd)

# New — delta EEF with ServoCart
current_eef = robot.get_eef_pose()           # [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg]
delta        = action[:6]
target_eef   = current_eef + delta
robot.servo_cart(target_eef)                 # if FR5 SDK supports ServoCart

# Alternative — delta EEF with online IK
target_eef   = current_eef + delta
joints_cmd   = robot.ik(target_eef)          # Fairino SDK has IK methods
robot.servo_j(joints_cmd)
```

The observation side is unchanged: `robot.get_joint_positions()` still feeds the 7D
state (joint angles + gripper_norm). The policy only changes its *output* representation.

---

## 6. Deploy-side: ServoCart vs IK

The Fairino FR5 SDK provides both options:

| Method | API | Notes |
|---|---|---|
| `ServoCart` | `robot.ServoCart(desc_pos, pos_gain, att_gain)` | Cartesian servo — direct EEF velocity/position servo |
| IK → `ServoJ` | `robot.GetInverseKin(desc_pos)` → `robot.ServoJ(joint_pos)` | Two-step: IK then joint servo |

`ServoCart` is cleaner (no IK round-trip, lower latency). `GetInverseKin → ServoJ` is
safer if the EEF target is near a singularity (IK can warn you).

For a simple pick-place task well away from singularities, `ServoCart` is the right
choice. Check that the FR5 firmware version supports it — it was added in SDK v3.x.

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
