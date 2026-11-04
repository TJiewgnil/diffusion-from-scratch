"""Training and DDPM ancestral sampling for Part 2 / Part 3.

The training loop is the standard DDPM noise-prediction objective:
    L = || eps - eps_theta(x_t, t) ||^2,  x_t = sqrt(abar_t) x_0 + sqrt(1-abar_t) eps.

The sampler is DDPM ancestral sampling (Ho 2020, Algorithm 2):
    x_{t-1} = (1/sqrt(alpha_t)) (x_t - (beta_t/sqrt(1-abar_t)) eps_theta(x_t, t)) + sigma_t z
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .forward_process import DiffusionSchedule, q_sample


# ---------------------------------------------------------------------------
# Part 2: per-point training (each x_0 is a single 2D point)
# ---------------------------------------------------------------------------

def train_pointwise(model: nn.Module, data: np.ndarray, sched: DiffusionSchedule,
                    device: torch.device, *, steps: int = 20000,
                    batch_size: int = 256, lr: float = 1e-3,
                    log_every: int = 1000, snapshot_steps: tuple[int, ...] = (),
                    sample_n: int = 1000) -> dict:
    """Train a per-point noise-prediction MLP. Returns history + snapshots.

    snapshot_steps: at each listed step (after the update), sample `sample_n` points
    and store them under `history['snapshots'][step]`.
    """
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched_t = sched.torch(device=device)

    data_t = torch.as_tensor(data, dtype=torch.float32, device=device)
    N = data_t.shape[0]
    T = sched.T

    losses: list[float] = []
    snapshots: dict[int, np.ndarray] = {}
    snap_set = set(snapshot_steps)

    for step in range(1, steps + 1):
        idx = torch.randint(0, N, (batch_size,), device=device)
        x0 = data_t[idx]
        t = torch.randint(0, T, (batch_size,), device=device)
        noise = torch.randn_like(x0)
        xt = q_sample(x0, t, noise, sched_t)
        pred = model(xt, t)
        loss = ((pred - noise) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

        if step % log_every == 0:
            recent = float(np.mean(losses[-log_every:]))
            print(f"  step {step:6d}  loss(last {log_every})={recent:.4f}")

        if step in snap_set:
            samples = sample_pointwise(model, sched, device, n=sample_n)
            snapshots[step] = samples
            model.train()

    return {"losses": losses, "snapshots": snapshots}


@torch.no_grad()
def sample_pointwise(model: nn.Module, sched: DiffusionSchedule,
                     device: torch.device, *, n: int = 1000) -> np.ndarray:
    """DDPM ancestral sampling for the per-point model. Returns (n, 2) np array."""
    model.eval()
    sched_t = sched.torch(device=device)
    x = torch.randn(n, 2, device=device)
    T = sched.T
    for t in range(T - 1, -1, -1):
        t_b = torch.full((n,), t, device=device, dtype=torch.long)
        eps = model(x, t_b)
        beta_t = sched_t["betas"][t]
        alpha_t = sched_t["alphas"][t]
        sob = sched_t["sqrt_one_minus_alpha_bars"][t]
        mean = (x - (beta_t / sob) * eps) / torch.sqrt(alpha_t)
        if t > 0:
            abar_tm1 = sched_t["alpha_bars"][t - 1]
            abar_t = sched_t["alpha_bars"][t]
            sigma = torch.sqrt((1 - abar_tm1) / (1 - abar_t) * beta_t)
            x = mean + sigma * torch.randn_like(x)
        else:
            x = mean
    return x.cpu().numpy()


# ---------------------------------------------------------------------------
# Part 3: set-level training (each x_0 is a (N_pts, 2) point cloud)
# ---------------------------------------------------------------------------

def train_setwise(model: nn.Module, data: np.ndarray, sched: DiffusionSchedule,
                  device: torch.device, *, steps: int = 20000,
                  batch_size: int = 16, lr: float = 1e-3,
                  log_every: int = 500, snapshot_steps: tuple[int, ...] = (),
                  sample_clouds: int = 4) -> dict:
    """Train a permutation-equivariant set model. data: (S, N, 2)."""
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched_t = sched.torch(device=device)

    data_t = torch.as_tensor(data, dtype=torch.float32, device=device)
    S, Npts, _ = data_t.shape
    T = sched.T

    losses: list[float] = []
    snapshots: dict[int, np.ndarray] = {}
    snap_set = set(snapshot_steps)

    for step in range(1, steps + 1):
        idx = torch.randint(0, S, (batch_size,), device=device)
        x0 = data_t[idx]                                # (B, N, 2)
        t = torch.randint(0, T, (batch_size,), device=device)
        noise = torch.randn_like(x0)
        sa = sched_t["sqrt_alpha_bars"][t].view(-1, 1, 1)
        so = sched_t["sqrt_one_minus_alpha_bars"][t].view(-1, 1, 1)
        xt = sa * x0 + so * noise
        pred = model(xt, t)
        loss = ((pred - noise) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

        if step % log_every == 0:
            recent = float(np.mean(losses[-log_every:]))
            print(f"  step {step:6d}  loss(last {log_every})={recent:.4f}")

        if step in snap_set:
            snapshots[step] = sample_setwise(model, sched, device,
                                             n_clouds=sample_clouds, n_pts=Npts)
            model.train()

    return {"losses": losses, "snapshots": snapshots}


@torch.no_grad()
def sample_setwise(model: nn.Module, sched: DiffusionSchedule,
                   device: torch.device, *, n_clouds: int = 4,
                   n_pts: int = 1024) -> np.ndarray:
    """DDPM ancestral sampling for the set model. Returns (n_clouds, n_pts, 2)."""
    model.eval()
    sched_t = sched.torch(device=device)
    x = torch.randn(n_clouds, n_pts, 2, device=device)
    T = sched.T
    for t in range(T - 1, -1, -1):
        t_b = torch.full((n_clouds,), t, device=device, dtype=torch.long)
        eps = model(x, t_b)
        beta_t = sched_t["betas"][t]
        alpha_t = sched_t["alphas"][t]
        sob = sched_t["sqrt_one_minus_alpha_bars"][t]
        mean = (x - (beta_t / sob) * eps) / torch.sqrt(alpha_t)
        if t > 0:
            abar_tm1 = sched_t["alpha_bars"][t - 1]
            abar_t = sched_t["alpha_bars"][t]
            sigma = torch.sqrt((1 - abar_tm1) / (1 - abar_t) * beta_t)
            x = mean + sigma * torch.randn_like(x)
        else:
            x = mean
    return x.cpu().numpy()
