"""
FR5 LeRobot dataset  ->  Octo numpy batches.

Reads the LeRobot v3.0 dataset (parquet + extracted jpg frames) directly with
pyarrow + opencv — NO torch / lerobot dependency, so the JAX venv stays lean.
Produces batches in exactly the structure Octo's transformer + action head read
(verified against octo-{small,base}-1.5):

    batch = {
      "observation": {
        "image_primary":     (B, W, 256, 256, 3) uint8,   # FR5 wrist cam -> primary slot
        "image_wrist":       (B, W, 128, 128, 3) uint8,   # zeros (FR5 has no 2nd cam) -> masked out
        "timestep_pad_mask": (B, W) bool,
        "pad_mask_dict": {"image_primary": (B,W) bool, "image_wrist": (B,W) bool, "timestep": (B,W) bool},
      },
      "task": {"language_instruction": (B,) bytes},        # process_text() tokenizes this
      "action":          (B, W, action_horizon, 7) float32,  # normalized (mean/std)
      "action_pad_mask": (B, W, action_horizon, 7) bool,
    }

W = window_size (2), action_horizon = 4 by default.  Actions are normalized with
dataset mean/std; the same stats are saved by finetune.py and reused to
unnormalize at inference.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from octo_common import resize_primary, read_image_rgb, action_stats, ACTION_DIM

WRIST_KEY = "observation.images.wrist_cam"
WRIST_SLOT_SIZE = 128   # Octo image_wrist H=W (we feed zeros here — FR5 has one camera)


class FR5OctoData:
    def __init__(self, root, window_size=2, action_horizon=4, action_dim=ACTION_DIM,
                 image_key=WRIST_KEY, instruction=None, seed=0):
        self.root = Path(root)
        self.W = window_size
        self.H = action_horizon
        self.action_dim = action_dim
        self.image_key = image_key
        self.instruction = instruction
        self.rng = np.random.default_rng(seed)

        with open(self.root / "meta/info.json") as f:
            self.info = json.load(f)
        self.df = pq.read_table(self.root / "data/chunk-000/file-000.parquet").to_pandas()
        self.eps = pq.read_table(
            self.root / "meta/episodes/chunk-000/file-000.parquet").to_pandas()
        tasks_path = self.root / "meta/tasks.parquet"
        self.task_map = {}
        if tasks_path.exists():
            t = pq.read_table(tasks_path).to_pandas()
            self.task_map = dict(zip(t["task_index"].tolist(), t["task"].tolist()))

        # action stats over ALL frames (for normalization + inference unnormalization)
        acts = np.array(self.df["action"].tolist(), np.float32)[:, :action_dim]
        self.stats = action_stats(acts)

        # sample index: (ep_idx, start_abs, ep_from, ep_to) with room for >=1 action
        self.index = []
        for _, ep in self.eps.iterrows():
            f, t = int(ep["dataset_from_index"]), int(ep["dataset_to_index"])
            ep_idx = int(ep["episode_index"])
            for s in range(f, t):                 # every frame can start a window
                self.index.append((ep_idx, s, f, t))
        if not self.index:
            raise SystemExit(f"no usable frames under {self.root}")

    # ── helpers ────────────────────────────────────────────────────────────--

    def _task_bytes(self, task_idx: int) -> bytes:
        s = self.instruction or self.task_map.get(task_idx, "")
        return s.encode("utf-8")

    def _frame_primary(self, ep_idx: int, frame_idx: int) -> np.ndarray:
        jpg = self.root / "frames" / self.image_key / f"ep-{ep_idx:03d}" / f"{frame_idx:06d}.jpg"
        if jpg.exists():
            return resize_primary(read_image_rgb(jpg))
        rgb = self._video_frame(ep_idx, frame_idx)
        return resize_primary(rgb)

    def _video_frame(self, ep_idx: int, frame_idx: int) -> np.ndarray:
        try:
            import cv2
        except ImportError:
            return np.zeros((256, 256, 3), np.uint8)
        vid = self.root / "videos" / self.image_key / "chunk-000" / f"file-{ep_idx:03d}.mp4"
        if not vid.exists():
            return np.zeros((256, 256, 3), np.uint8)
        cap = cv2.VideoCapture(str(vid))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, bgr = cap.read()
        cap.release()
        if not ok:
            return np.zeros((256, 256, 3), np.uint8)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _norm_action(self, a: np.ndarray) -> np.ndarray:
        return (a[:self.action_dim] - self.stats["mean"]) / self.stats["std"]

    # ── one example (a window of W frames, each with H future actions) ───────--

    def _example(self, ep_idx, s, ep_from, ep_to):
        imgs, ts_valid, chunks, cmasks = [], [], [], []
        for w in range(self.W):
            f_abs = min(s + w, ep_to - 1)
            valid = (s + w) < ep_to
            frame_idx = int(self.df.iloc[f_abs]["frame_index"])
            imgs.append(self._frame_primary(ep_idx, frame_idx))
            ts_valid.append(valid)

            chunk, cmask = [], []
            for h in range(self.H):
                a_abs = f_abs + h
                if a_abs < ep_to:
                    a = np.array(self.df.iloc[a_abs]["action"], np.float32)
                    chunk.append(self._norm_action(a))
                    cmask.append(True)
                else:
                    chunk.append(np.zeros(self.action_dim, np.float32))
                    cmask.append(False)
            chunks.append(np.stack(chunk))                       # (H, dim)
            cmasks.append(np.array(cmask))                       # (H,)

        task_idx = int(self.df.iloc[s].get("task_index", 0))
        return {
            "image_primary": np.stack(imgs),                     # (W,256,256,3)
            "timestep_valid": np.array(ts_valid, bool),          # (W,)
            "action": np.stack(chunks).astype(np.float32),       # (W,H,dim)
            "action_cmask": np.stack(cmasks),                    # (W,H)
            "task_bytes": self._task_bytes(task_idx),
        }

    def sample_batch(self, batch_size: int) -> dict:
        replace = len(self.index) < batch_size
        picks = self.rng.choice(len(self.index), size=batch_size, replace=replace)
        ex = [self._example(*self.index[i]) for i in picks]

        img_primary = np.stack([e["image_primary"] for e in ex])           # (B,W,256,256,3)
        ts_valid    = np.stack([e["timestep_valid"] for e in ex])          # (B,W)
        action      = np.stack([e["action"] for e in ex])                  # (B,W,H,dim)
        cmask       = np.stack([e["action_cmask"] for e in ex])            # (B,W,H)
        B, W = ts_valid.shape

        # FR5 has one camera -> primary; the wrist slot is zeros and masked out.
        img_wrist = np.zeros((B, W, WRIST_SLOT_SIZE, WRIST_SLOT_SIZE, 3), np.uint8)
        action_pad_mask = np.broadcast_to(cmask[..., None],
                                          action.shape).astype(bool)        # (B,W,H,dim)

        return {
            "observation": {
                "image_primary":     img_primary,
                "image_wrist":       img_wrist,
                "timestep_pad_mask": ts_valid,
                "pad_mask_dict": {
                    "image_primary": ts_valid,                              # present where timestep valid
                    "image_wrist":   np.zeros((B, W), bool),               # never present
                    "timestep":      ts_valid,
                },
            },
            "task": {
                "language_instruction": np.array([e["task_bytes"] for e in ex]),  # (B,) bytes
            },
            "action":          action,
            "action_pad_mask": action_pad_mask,
        }

    def example_batch(self, text_processor) -> dict:
        """A single processed batch (B=1) used to init shapes / finetune from_config."""
        from octo.utils.train_utils import process_text
        b = self.sample_batch(1)
        return process_text(b, text_processor)
