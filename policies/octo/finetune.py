"""
Workflow 2 — finetune Octo on the FR5 LeRobot dataset.

Loads the Octo team's pretrained weights and adapts them to the FR5 by finetuning
on the recorded teleop episodes. FR5's action is 7-D (6 joints + gripper), which
matches Octo's default action head (action_dim=7, action_horizon=4) — so NO head
surgery is needed: we finetune the pretrained model directly.

Key choices:
  * FR5 wrist camera -> Octo's image_primary (256x256); the unused image_wrist
    slot is fed zeros and masked out.
  * Actions are normalized with the dataset's own mean/std (saved alongside the
    checkpoint so inference can unnormalize back to FR5 joint commands).
  * Vision + language only, NO proprioceptive state input — which, by design,
    sidesteps the proprioceptive shortcut documented in docs/il_failure_modes.md.
  * Optional --freeze-transformer trains only the action head (faster, less prone
    to overfitting on small datasets).

Run (inside .venv-octo, ideally on the Linux GPU box):
    python policies/octo/finetune.py --config policies/octo/config.yaml
    python policies/octo/finetune.py --steps 50 --batch-size 4   # quick smoke
"""

import argparse
from pathlib import Path

import jax
import numpy as np
import optax
import yaml

from octo_common import load_octo, save_stats, DEFAULT_MODEL
from fr5_octo_data import FR5OctoData

REPO_ROOT = Path(__file__).resolve().parents[2]


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


def freeze_transformer_mask(params):
    """optax.masked mask: True = train (action head), False = freeze (transformer)."""
    import flax.traverse_util as ftu
    flat = ftu.flatten_dict(params)
    mask = {k: ("octo_transformer" not in "/".join(k)) for k in flat}
    return ftu.unflatten_dict(mask)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    ap.add_argument("--model", default=None, help="override pretrained weights")
    ap.add_argument("--root", default=None, help="override dataset root")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--save-dir", default=None, help="override checkpoint dir")
    ap.add_argument("--freeze-transformer", action="store_true")
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
    freeze = args.freeze_transformer or ft.get("freeze_transformer", False)
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

    # ── optimizer ───────────────────────────────────────────────────────────--
    lr = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=ft["learning_rate"],
        warmup_steps=ft["warmup_steps"], decay_steps=steps,
        end_value=ft["learning_rate"] * 0.1,
    )
    tx = optax.adamw(lr, weight_decay=ft.get("weight_decay", 0.01))
    if freeze:
        tx = optax.masked(tx, freeze_transformer_mask(model.params))
        print("[finetune] freezing transformer — training action head only")

    from octo.utils.train_utils import TrainState, process_text
    state = TrainState.create(rng=jax.random.PRNGKey(ft["seed"]), model=model, tx=tx)
    train_step = make_train_step(model)

    # ── loop ────────────────────────────────────────────────────────────────--
    print(f"[finetune] {steps} steps  batch={batch_size}  lr={ft['learning_rate']}  freeze={freeze}\n")
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
