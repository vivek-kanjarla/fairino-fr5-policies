import json
import random
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset
from torchvision import transforms


VIDEO_KEY = "observation.images.wrist_cam"

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


# ── Tier 2 augmentation: background texture replacement ───────────────────────

class _BackgroundAugment:
    """Replace low-texture (table/background) regions with random synthetic textures.

    Works on raw [0, 1] float tensors (C, H, W) BEFORE ImageNet normalisation.

    How it works:
      1. Estimate foreground via Laplacian edge magnitude — objects have high
         local edge density; flat table surface has low edge density.
      2. Dilate + soften the mask so object boundaries are well-covered.
      3. Blend the background region with a random procedural texture.

    Why this matters: confirmed research shows tabletop texture change alone
    drops policy success from 0.58 → 0.04. This augment attacks that failure
    mode without needing depth data or a segmentation model.
    """

    def __init__(self, p: float = 0.8, blend: float = 0.85):
        self.p     = p       # probability of applying per sample
        self.blend = blend   # how strongly to replace background (1.0 = full replace)

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return img

        _, H, W = img.shape

        # -- foreground mask via Laplacian edge density --
        gray = img.mean(0).numpy()                             # (H, W)
        lap  = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        edge = np.abs(lap)
        edge = cv2.GaussianBlur(edge, (15, 15), 0)
        if edge.max() > 1e-6:
            edge /= edge.max()

        fg_mask = (edge > 0.08).astype(np.uint8)
        fg_mask = cv2.dilate(fg_mask, np.ones((13, 13), np.uint8), iterations=3)
        fg_soft = cv2.GaussianBlur(fg_mask.astype(np.float32), (21, 21), 0)
        fg      = torch.from_numpy(fg_soft).unsqueeze(0)      # (1, H, W), in [0, 1]

        tex    = self._random_texture(H, W)                   # (3, H, W), in [0, 1]
        bg_out = (self.blend * tex + (1 - self.blend) * img).clamp(0, 1)
        return (fg * img + (1 - fg) * bg_out).clamp(0, 1)

    @staticmethod
    def _random_texture(H: int, W: int) -> torch.Tensor:
        choice = random.randint(0, 4)

        if choice == 0:
            # solid colour with slight noise
            base = torch.rand(3, 1, 1).expand(3, H, W).clone()
            return (base + torch.randn(3, H, W) * 0.03).clamp(0, 1)

        elif choice == 1:
            # linear gradient
            lo, hi = sorted([random.random(), random.random()])
            g = torch.linspace(lo, hi, W).unsqueeze(0).expand(H, W)
            c = torch.rand(3, 1, 1)
            return (g.unsqueeze(0) * c).clamp(0, 1)

        elif choice == 2:
            # colour noise (simulates fabric/carpet texture)
            base = torch.rand(3, 1, 1).expand(3, H, W).clone()
            return (base + torch.randn(3, H, W) * 0.18).clamp(0, 1)

        elif choice == 3:
            # checker pattern (hard tabletop visual shift)
            sz   = random.randint(10, 40)
            xs   = (torch.arange(W) // sz) % 2
            ys   = (torch.arange(H) // sz) % 2
            grid = (xs.unsqueeze(0) ^ ys.unsqueeze(1)).float()   # (H, W)
            c1, c2 = torch.rand(3, 1, 1), torch.rand(3, 1, 1)
            return (grid.unsqueeze(0) * c1 + (1 - grid.unsqueeze(0)) * c2).clamp(0, 1)

        else:
            # low-frequency sinusoidal (wood-grain / cloth-like)
            fx = random.uniform(1, 5)
            fy = random.uniform(1, 5)
            x  = torch.linspace(0, fx * 2 * 3.14159, W)
            y  = torch.linspace(0, fy * 2 * 3.14159, H)
            gx, gy = torch.meshgrid(y, x, indexing='ij')
            n  = (torch.sin(gx + random.random() * 6) *
                  torch.cos(gy + random.random() * 6) * 0.5 + 0.5)
            c  = torch.rand(3, 1, 1)
            return (n.unsqueeze(0) * c).clamp(0, 1)


# ── transform factory ─────────────────────────────────────────────────────────

def _build_transform(image_size: tuple, aug_level: str) -> transforms.Compose:
    """
    aug_level:
      "none"   — deterministic resize + ImageNet norm (val / no-aug baseline)
      "crops"  — random resized crop + ImageNet norm  (Tier 1, confirmed benefit)
      "full"   — random crop + background replacement + mild colour jitter + norm
                 (Tier 1 + Tier 2; colour jitter kept very mild since photometric
                 distortion alone was refuted — it only supplements the spatial augs)
    """
    h, w = image_size

    if aug_level == "none":
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])

    if aug_level == "crops":
        return transforms.Compose([
            transforms.Resize((int(h * 1.12), int(w * 1.12))),
            transforms.RandomCrop(image_size),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])

    if aug_level == "full":
        # background replacement runs on raw [0,1] tensor before norm;
        # colour jitter applied first (on tensor), then background aug, then norm
        return transforms.Compose([
            transforms.Resize((int(h * 1.12), int(w * 1.12))),
            transforms.RandomCrop(image_size),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.1, hue=0.03),
            _BackgroundAugment(p=0.8, blend=0.85),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])

    raise ValueError(f"unknown aug_level {aug_level!r}; use 'none', 'crops', or 'full'")


# ── dataset ───────────────────────────────────────────────────────────────────

class FR5Dataset(Dataset):
    def __init__(self, root, chunk_size=100, use_image=True,
                 image_size=(224, 224), episode_indices=None,
                 aug_level="none"):
        self.root      = Path(root)
        self.chunk_size = chunk_size
        self.use_image  = use_image
        self.image_size = image_size
        self.aug_level  = aug_level

        with open(self.root / "meta/info.json") as f:
            self.info = json.load(f)

        self.df = pq.read_table(
            self.root / "data/chunk-000/file-000.parquet"
        ).to_pandas()

        self.episodes = pq.read_table(
            self.root / "meta/episodes/chunk-000/file-000.parquet"
        ).to_pandas()

        if episode_indices is not None:
            self.episodes = self.episodes[
                self.episodes["episode_index"].isin(episode_indices)
            ].reset_index(drop=True)

        self._samples = self._build_index()

        # task index → language string (used by language-conditioned policies)
        tasks_path = self.root / "meta" / "tasks.parquet"
        if tasks_path.exists():
            tasks_df   = pq.read_table(tasks_path).to_pandas()
            self._task_map = dict(zip(tasks_df["task_index"].tolist(),
                                      tasks_df["task"].tolist()))
        else:
            self._task_map = {}

        self._img_transform = _build_transform(image_size, aug_level)

    def _build_index(self):
        samples = []
        for _, ep in self.episodes.iterrows():
            ep_idx = int(ep["episode_index"])
            from_i = int(ep["dataset_from_index"])
            to_i   = int(ep["dataset_to_index"])
            for t in range(from_i, to_i):
                samples.append((ep_idx, t, from_i, to_i))
        return samples

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        ep_idx, frame_abs, ep_from, ep_to = self._samples[idx]

        row   = self.df.iloc[frame_abs]
        state = torch.tensor(row["observation.state"], dtype=torch.float32)

        chunk_end  = min(frame_abs + self.chunk_size, ep_to)
        chunk_rows = self.df.iloc[frame_abs:chunk_end]
        actions    = torch.tensor(
            np.array(chunk_rows["action"].tolist()), dtype=torch.float32
        )

        pad_len = self.chunk_size - len(actions)
        is_pad  = torch.zeros(self.chunk_size, dtype=torch.bool)
        if pad_len > 0:
            padding = actions[-1:].expand(pad_len, -1)
            actions = torch.cat([actions, padding], dim=0)
            is_pad[self.chunk_size - pad_len:] = True

        task_idx = int(row["task_index"]) if "task_index" in row.index else 0
        sample = {
            "observation.state": state,
            "action":            actions,
            "action_is_pad":     is_pad,
            "task":              self._task_map.get(task_idx, ""),
        }

        if self.use_image:
            frame_in_ep = int(row["frame_index"])
            img = self._load_frame(ep_idx, frame_in_ep)
            if img is not None:
                sample["observation.image"] = img

        return sample

    def _load_frame(self, ep_idx, frame_idx):
        # fast path: pre-extracted JPEG
        jpg = (self.root / "frames" / VIDEO_KEY /
               f"ep-{ep_idx:03d}" / f"{frame_idx:06d}.jpg")
        frame = cv2.imread(str(jpg)) if jpg.exists() else self._read_video_frame(ep_idx, frame_idx)

        if frame is None:
            return None

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img   = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        return self._img_transform(img)

    def _read_video_frame(self, ep_idx, frame_idx):
        vid = self.root / "videos" / VIDEO_KEY / "chunk-000" / f"file-{ep_idx:03d}.mp4"
        if not vid.exists():
            return None
        cap = cv2.VideoCapture(str(vid))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        return frame if ret else None

    def get_stats(self):
        ep_mask = self.df["episode_index"].isin(
            self.episodes["episode_index"].tolist()
        )
        sub    = self.df[ep_mask]
        state  = np.array(sub["observation.state"].tolist())
        action = np.array(sub["action"].tolist())
        return {
            "state_mean":  state.mean(0).astype(np.float32),
            "state_std":   state.std(0).clip(1e-6).astype(np.float32),
            "state_min":   state.min(0).astype(np.float32),
            "state_max":   state.max(0).astype(np.float32),
            "action_mean": action.mean(0).astype(np.float32),
            "action_std":  action.std(0).clip(1e-6).astype(np.float32),
            "action_min":  action.min(0).astype(np.float32),
            "action_max":  action.max(0).astype(np.float32),
        }

    @staticmethod
    def episode_split(n_episodes, val_frac=0.1, seed=42):
        indices = list(range(n_episodes))
        random.seed(seed)
        random.shuffle(indices)
        n_val = max(1, int(val_frac * n_episodes))
        return indices[n_val:], indices[:n_val]
