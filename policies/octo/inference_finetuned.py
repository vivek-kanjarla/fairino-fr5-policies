"""
Workflow 3 — inference with the FINETUNED Octo checkpoint.

Loads the checkpoint written by finetune.py plus the FR5 action stats saved next
to it, and produces actions in FR5 space (6 joint deltas/targets + gripper),
unnormalized with the dataset's own mean/std. This is the model you would deploy.

Run (inside .venv-octo):
    python policies/octo/inference_finetuned.py --checkpoint policies/octo/checkpoints --step 5000
    python policies/octo/inference_finetuned.py --checkpoint policies/octo/checkpoints --step 5000 --image frame.jpg

For real robot deployment, wrap this the same way common/deploy.py does: at 30 Hz,
read the wrist frame, keep a 2-frame history window, call sample_actions, and send
the first action of the returned chunk to the FR5 (action[:6] joints, action[6]
gripper). Octo predicts `action_horizon` steps; you can also execute the chunk
open-loop or with temporal ensembling (see docs/inference.md).
"""

import argparse
from pathlib import Path

import jax
import numpy as np

from octo_common import (
    load_octo, model_window_size, resize_primary, read_image_rgb,
    load_stats, FR5_INSTRUCTION, PRIMARY_IMAGE_SIZE,
)


def build_observation(rgb_primary, window):
    img = resize_primary(rgb_primary)
    return {
        "image_primary": np.tile(img[None, None], (1, window, 1, 1, 1)),
        "timestep_pad_mask": np.ones((1, window), bool),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, help="finetune.py save_dir")
    ap.add_argument("--step", type=int, required=True, help="checkpoint step to load")
    ap.add_argument("--image", default=None)
    ap.add_argument("--instruction", default=FR5_INSTRUCTION)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model = load_octo(args.checkpoint, step=args.step)
    window = model_window_size(model)

    stats_path = Path(args.checkpoint) / "fr5_action_stats.npz"
    if not stats_path.exists():
        raise SystemExit(f"missing {stats_path} — was this checkpoint written by finetune.py?")
    stats = load_stats(stats_path)

    if args.image:
        rgb = read_image_rgb(args.image)
    else:
        rgb = np.random.randint(0, 255, (PRIMARY_IMAGE_SIZE, PRIMARY_IMAGE_SIZE, 3), np.uint8)
        print("[octo] no --image given; using a random frame")

    task = model.create_tasks(texts=[args.instruction])
    obs = build_observation(rgb, window)

    print(f"\nfinetuned rollout ({args.steps} steps), instruction={args.instruction!r}\n")
    rng = jax.random.PRNGKey(args.seed)
    for step in range(args.steps):
        rng, key = jax.random.split(rng)
        actions = np.asarray(model.sample_actions(
            obs, task, unnormalization_statistics=stats, rng=key))    # (1, horizon, 7) FR5 space
        a0 = actions[0, 0]
        print(f"  step {step}: chunk={actions.shape}  joints={np.round(a0[:6], 3)}  gripper={a0[6]:.3f}")


if __name__ == "__main__":
    main()
