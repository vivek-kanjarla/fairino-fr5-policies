"""
Workflow 1 — Octo ZERO-SHOT inference (NO finetuning).

Loads the Octo team's pretrained weights and runs the model on an FR5 wrist-cam
image + language instruction.

IMPORTANT: zero-shot actions come out in Octo's *pretrained* action space
(normalized delta end-effector poses learned from its 800k-trajectory Open-X
training mix), NOT FR5 joint commands. They will NOT correctly drive the FR5
until you finetune (see finetune.py). This script proves the pretrained model
loads and runs end-to-end on FR5-shaped inputs, and is the baseline for the
finetuned comparison.

Run (inside .venv-octo):
    python policies/octo/inference_pretrained.py                      # random image
    python policies/octo/inference_pretrained.py --image frame.jpg    # a real wrist frame
    python policies/octo/inference_pretrained.py --model hf://rail-berkeley/octo-small-1.5
"""

import argparse

import jax
import numpy as np

from octo_common import (
    load_octo, model_window_size, resize_primary, read_image_rgb,
    pick_pretrained_action_stats, DEFAULT_MODEL, FR5_INSTRUCTION, PRIMARY_IMAGE_SIZE,
)


def build_observation(rgb_primary: np.ndarray, window: int) -> dict:
    """Tile a single 256x256 frame across the history window (fresh-start padding)."""
    img = resize_primary(rgb_primary)                       # (256,256,3)
    images = np.tile(img[None, None], (1, window, 1, 1, 1))  # (1, W, 256,256,3)
    return {
        "image_primary": images,
        "timestep_pad_mask": np.ones((1, window), bool),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--image", default=None, help="wrist-cam image path; random if omitted")
    ap.add_argument("--instruction", default=FR5_INSTRUCTION)
    ap.add_argument("--steps", type=int, default=5, help="simulated rollout steps")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model = load_octo(args.model)
    window = model_window_size(model)

    if args.image:
        rgb = read_image_rgb(args.image)
    else:
        rgb = np.random.randint(0, 255, (PRIMARY_IMAGE_SIZE, PRIMARY_IMAGE_SIZE, 3), np.uint8)
        print("[octo] no --image given; using a random frame")

    task = model.create_tasks(texts=[args.instruction])
    stats = pick_pretrained_action_stats(model)
    obs = build_observation(rgb, window)

    print(f"\nzero-shot rollout ({args.steps} steps), instruction={args.instruction!r}\n")
    rng = jax.random.PRNGKey(args.seed)
    for step in range(args.steps):
        rng, key = jax.random.split(rng)
        actions = np.asarray(model.sample_actions(
            obs, task, unnormalization_statistics=stats, rng=key))   # (1, horizon, dim)
        a0 = actions[0, 0]
        print(f"  step {step}: action_chunk={actions.shape}  a0={np.round(a0, 4)}")

    print("\nNOTE: these are Octo-pretrained-space actions, not FR5 joint commands. "
          "Finetune (finetune.py) to map them onto the FR5.")


if __name__ == "__main__":
    main()
