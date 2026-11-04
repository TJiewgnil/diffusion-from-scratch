"""Forward diffusion process and schedules.

Used by both Part 1 (1D) and Part 2 / Part 3 (2D, NN).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


def make_schedule(kind: str, T: int, beta_start: float = 1e-4, beta_end: float = 0.02,
                  beta_const: float = 0.25) -> np.ndarray:
    """Return betas as np.ndarray of shape (T,)."""
    if kind == "constant":
        return np.full(T, beta_const, dtype=np.float64)
    if kind == "linear":
        return np.linspace(beta_start, beta_end, T, dtype=np.float64)
    if kind == "quadratic":
        return np.linspace(beta_start ** 0.5, beta_end ** 0.5, T, dtype=np.float64) ** 2
    raise ValueError(f"unknown schedule: {kind}")


@dataclass
class DiffusionSchedule:
    """Precomputed schedule tensors. Works on any backend (np or torch) via .torch()."""
    betas: np.ndarray             # (T,)
    alphas: np.ndarray            # 1 - beta
    alpha_bars: np.ndarray        # cumulative product
    sqrt_alpha_bars: np.ndarray
    sqrt_one_minus_alpha_bars: np.ndarray

    @classmethod
    def from_betas(cls, betas: np.ndarray) -> "DiffusionSchedule":
        alphas = 1.0 - betas
        alpha_bars = np.cumprod(alphas)
        return cls(
            betas=betas,
            alphas=alphas,
            alpha_bars=alpha_bars,
            sqrt_alpha_bars=np.sqrt(alpha_bars),
            sqrt_one_minus_alpha_bars=np.sqrt(1.0 - alpha_bars),
        )

    @property
    def T(self) -> int:
        return len(self.betas)

    def torch(self, device: torch.device | str = "cpu", dtype=torch.float32):
        """Return a dict of torch tensors on given device."""
        return {
            k: torch.as_tensor(getattr(self, k), dtype=dtype, device=device)
            for k in ("betas", "alphas", "alpha_bars",
                     "sqrt_alpha_bars", "sqrt_one_minus_alpha_bars")
        }


def q_sample(x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor,
             sched_t: dict) -> torch.Tensor:
    """Closed-form forward sample x_t = sqrt(abar_t) x_0 + sqrt(1-abar_t) eps.

    x0, noise: (B, D)   t: (B,) long   sched_t: dict from DiffusionSchedule.torch()
    """
    sa = sched_t["sqrt_alpha_bars"][t].unsqueeze(-1)
    so = sched_t["sqrt_one_minus_alpha_bars"][t].unsqueeze(-1)
    return sa * x0 + so * noise
