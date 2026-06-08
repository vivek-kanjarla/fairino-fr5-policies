"""
common/proprio.py — proprioception handling modes, shared across all policies.

This is a single, policy-agnostic utility for benchmarking *how much a policy
leans on proprioceptive state vs vision* — the proprioceptive shortcut described
in docs/il_failure_modes.md. The same three modes plug into ACT, Diffusion, and
any future wrapper without per-policy surgery.

Three modes
───────────
  full     State passed through unchanged. Current/default behaviour.

  dropout  State is randomly zeroed PER-SAMPLE during TRAINING with probability
           `dropout_rate`; full state is used at eval/deploy. For the zeroed
           samples the loss can no longer be driven by state, so the vision
           pathway is forced to contribute. Typical rate 0.2–0.4.

  none     State is ALWAYS zeroed — training, eval and deploy. Functionally
           state-free / vision-only. The state encoder still exists but receives
           a constant, so it carries zero per-sample information (the closest we
           get to Zhao et al. 2025's state-free policy without rebuilding each
           lerobot backbone's input_features).

Design
──────
All three reduce to one operation — mask the (already-normalised) state vector
just before it enters the underlying lerobot policy, in BOTH forward() (train)
and predict() (deploy). Because the *model wrapper* owns the masking:
  • train.py and deploy.py stay completely policy-agnostic — they always pass
    the real state and the model decides what to do with it.
  • The mode is saved inside the checkpoint config, so deploy reproduces the
    exact training-time proprio handling automatically.

See docs/proprioception_modes.md for the benchmark recipe.
"""

from dataclasses import dataclass

import torch


VALID_MODES = ("full", "dropout", "none")


@dataclass
class ProprioConfig:
    mode: str = "full"
    dropout_rate: float = 0.3

    def __post_init__(self):
        if self.mode not in VALID_MODES:
            raise ValueError(
                f"proprio_mode {self.mode!r} invalid; use one of {VALID_MODES}")
        if not 0.0 <= self.dropout_rate <= 1.0:
            raise ValueError(
                f"proprio_dropout_rate must be in [0, 1], got {self.dropout_rate}")

    @classmethod
    def from_model_cfg(cls, model_cfg) -> "ProprioConfig":
        """Build from a policy's model-config dataclass or a plain dict."""
        get = (model_cfg.get if isinstance(model_cfg, dict)
               else lambda k, d: getattr(model_cfg, k, d))
        return cls(
            mode=get("proprio_mode", "full"),
            dropout_rate=float(get("proprio_dropout_rate", 0.3)),
        )

    @property
    def active(self) -> bool:
        """True if this mode ever alters the state (i.e. not plain 'full')."""
        return self.mode != "full"


def mask_state(state: torch.Tensor, cfg: ProprioConfig, training: bool) -> torch.Tensor:
    """Apply proprio masking to a normalised state tensor.

    state    : (B, state_dim) or (B, T, state_dim) — masking broadcasts over all
               non-batch dims so the whole state vector of a sample is zeroed or
               kept together.
    training : the wrapper's nn.Module.training flag (True during the train pass,
               False during validation/eval/deploy).

    Returns a tensor of the same shape.
    """
    if cfg.mode == "none":
        return torch.zeros_like(state)

    if cfg.mode == "dropout" and training:
        # per-sample Bernoulli keep mask over the batch dim, broadcast over the rest
        keep_shape = [state.shape[0]] + [1] * (state.dim() - 1)
        keep = (torch.rand(keep_shape, device=state.device) >= cfg.dropout_rate)
        return state * keep.to(state.dtype)

    # full, or dropout during eval/deploy → unchanged
    return state


def describe(cfg: ProprioConfig) -> str:
    """One-line human summary for training/deploy logs."""
    if cfg.mode == "dropout":
        return f"proprio=dropout (rate={cfg.dropout_rate:.2f}, train-only)"
    if cfg.mode == "none":
        return "proprio=none (state-free: state zeroed everywhere)"
    return "proprio=full (state used as-is)"
