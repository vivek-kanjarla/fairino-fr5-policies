"""
Workflow 2 — finetune Octo on the FR5 LeRobot dataset.

Loads the Octo team's pretrained weights and adapts them to the FR5 by finetuning
on the recorded teleop episodes. FR5's action is 7-D (6 joints + gripper), which
matches Octo's default action head (action_dim=7, action_horizon=4) — so NO head
surgery is needed: we finetune the pretrained model directly.

Finetuning modes — Octo does NOT use LoRA. The official recipe freezes parameters
by key pattern (octo-models/octo scripts/configs/finetune_config.py):

  full           train everything            (frozen_keys = None)
  head_only      freeze the transformer,     (frozen_keys = "octo_transformer.*")
                 train only the heads          ← best for a tiny dataset like the FR5
  head_mlp_only  freeze transformer + head    (+ head map_head probe/attention)
                 attention, train head MLP only

We use Octo's official `create_optimizer` (cosine LR, grad-clip 1.0, AdamW wd=0.01,
ViT-style no-wd on biases/norms) + `frozen_keys` — exactly the pretrained recipe.

Key choices:
  * FR5 wrist camera -> Octo's image_primary (256x256); the unused image_wrist
    slot is fed zeros and masked out.
  * Actions are normalized with the dataset's own mean/std (saved alongside the
    checkpoint so inference can unnormalize back to FR5 joint commands).
  * Vision + language only, NO proprioceptive state input — which, by design,
    sidesteps the proprioceptive shortcut documented in docs/il_failure_modes.md.

Run (inside .venv-octo, ideally on the Linux GPU box):
    python policies/octo/finetune.py --config policies/octo/config.yaml
    python policies/octo/finetune.py --mode full                       # train everything
    python policies/octo/finetune.py --mode head_only --steps 50 --batch-size 4   # smoke
"""

import argparse
from pathlib import Path

import jax
import yaml

from octo_common import load_octo, save_stats, DEFAULT_MODEL
from fr5_octo_data import FR5OctoData

REPO_ROOT = Path(__file__).resolve().parents[2]

# Octo's official frozen-key patterns per finetuning mode
# (octo-models/octo scripts/configs/finetune_config.py). No LoRA — freeze by key.
FROZEN_KEYS = {
    "full": None,
    "head_only": ("octo_transformer.*",),
    "head_mlp_only": (
        "octo_transformer.*",
        "heads_*.map_head.probe",
        "heads_*.map_head.MultiHeadDotProductAttention_0.*",
    ),
}


def make_train_step(model):
    """Build the jitted train step. Loss = Octo action head loss on normalized actions."""

    def loss_fn(params, batch, dropout_rng):
        bound = model.module.bind({"params": params}, rngs={"dropout": dropout_rng})
        embeddings = bound.octo_transformer(
            batch["observation"],
            batch["task"],
            batch["observation"]["timestep_pad_mask"],
            train=True,
        )
        loss, metrics = bound.heads["action"].loss(
            embeddings,
            batch["action"],
            batch["observation"]["timestep_pad_mask"],
            batch["action_pad_mask"],
            train=True,
        )
        return loss, metrics

    @jax.jit
    def train_step(state, batch):
        rng, dropout_rng = jax.random.split(state.rng)
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.model.params, batch, dropout_rng)
        new_state = state.apply_gradients(grads=grads, rng=rng)
        return new_state, loss, metrics

    return train_step


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    ap.add_argument("--model", default=None, help="override pretrained weights")
    ap.add_argument("--root", default=None, help="override dataset root")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--save-dir", default=None, help="override checkpoint dir")
    ap.add_argument("--mode", choices=list(FROZEN_KEYS), default=None,
                    help="full | head_only | head_mlp_only (Octo official; no LoRA)")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    m, d, ft = cfg["model"], cfg["dataset"], cfg["finetune"]

    def resolve(p):
        return p if Path(p).is_absolute() else str((REPO_ROOT / p).resolve())

    model_name = args.model or m.get("pretrained", DEFAULT_MODEL)
    root = resolve(args.root or d["root"])
    steps = args.steps or ft["steps"]
    batch_size = args.batch_size or ft["batch_size"]
    mode = args.mode or ft.get("mode", "head_only")
    if mode not in FROZEN_KEYS:
        raise SystemExit(f"unknown mode {mode!r}; use one of {list(FROZEN_KEYS)}")
    save_dir = resolve(args.save_dir or ft["save_dir"])

    # ── data ────────────────────────────────────────────────────────────────--
    data = FR5OctoData(
        root,
        window_size=m["window_size"],
        action_horizon=m["action_horizon"],
        action_dim=m["action_dim"],
        image_key=d.get("image_key", "observation.images.wrist_cam"),
        instruction=d.get("instruction"),
        seed=ft["seed"],
    )
    print(f"[finetune] dataset={root}  samples={len(data.index)}")

    # ── model ───────────────────────────────────────────────────────────────--
    model = load_octo(model_name)

    # sanity: action dims must match the pretrained head (else head surgery needed)
    head_dim = model.config["model"]["heads"]["action"]["kwargs"]["action_dim"]
    head_h = model.config["model"]["heads"]["action"]["kwargs"]["action_horizon"]
    if (head_dim, head_h) != (m["action_dim"], m["action_horizon"]):
        raise SystemExit(
            f"pretrained head ({head_dim}d,{head_h}h) != config "
            f"({m['action_dim']}d,{m['action_horizon']}h). Reinitialize the head "
            "(from_config + merge_params) — see README §head-surgery.")

    # ── optimizer (Octo's official create_optimizer: cosine LR + grad-clip +
    #    AdamW no-wd-on-bias/norm + frozen_keys for the chosen mode) ──────────--
    from octo.utils.train_utils import TrainState, process_text, create_optimizer
    frozen_keys = FROZEN_KEYS[mode]
    tx, lr_callable, _ = create_optimizer(
        model.params,
        learning_rate=dict(
            name="cosine",
            init_value=0.0,
            peak_value=ft["learning_rate"],
            warmup_steps=ft["warmup_steps"],
            decay_steps=steps,
            end_value=0.0,
        ),
        weight_decay=ft.get("weight_decay", 0.01),
        clip_gradient=ft.get("grad_clip", 1.0),
        frozen_keys=frozen_keys,
    )
    state = TrainState.create(rng=jax.random.PRNGKey(ft["seed"]), model=model, tx=tx)
    train_step = make_train_step(model)

    # ── loop ────────────────────────────────────────────────────────────────--
    print(f"[finetune] mode={mode}  frozen_keys={frozen_keys}")
    print(f"[finetune] {steps} steps  batch={batch_size}  peak_lr={ft['learning_rate']}\n")
    log_every, save_every = ft.get("log_every", 100), ft.get("save_every", 1000)
    for step in range(1, steps + 1):
        batch = process_text(data.sample_batch(batch_size), model.text_processor)
        state, loss, metrics = train_step(state, batch)
        if step % log_every == 0 or step == 1:
            mae = float(metrics.get("mae", metrics.get("loss", loss)))
            print(f"  step {step}/{steps}  loss={float(loss):.4f}  mae={mae:.4f}")
        if step % save_every == 0 or step == steps:
            state.model.save_pretrained(step=step, checkpoint_path=save_dir)
            save_stats(Path(save_dir) / "fr5_action_stats.npz", data.stats)
            print(f"  ↳ saved checkpoint + action stats @ step {step} -> {save_dir}")

    print(f"\n[finetune] done. checkpoint at {save_dir} (load with --model {save_dir} --step {steps})")


if __name__ == "__main__":
    main()
