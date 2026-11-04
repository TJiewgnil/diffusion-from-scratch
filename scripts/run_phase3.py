"""Phase 3 (Part 3): permutation-equivariant diffusion on stars.

Trains a PointNet-style model that takes a full point cloud (1024 pts) and
predicts per-point noise. Saves checkpoint and figures.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

import numpy as np
import matplotlib.pyplot as plt
import torch

from src.utils import set_seed, get_device, load_stars, FIGURES_DIR, CKPT_DIR
from src.forward_process import DiffusionSchedule, make_schedule
from src.model_advanced import PermInvNoiseModel
from src.train import train_setwise, sample_setwise

torch.set_num_threads(8)


T = 500              # use fewer steps than Part 2 — saves time and quality is fine
BATCH = 8            # # of clouds per batch (memory & speed)
LR = 1e-3
TOTAL_STEPS = 8000
SNAPSHOTS = (1000, 3000, 6000, 8000)


def main():
    set_seed(0)
    device = get_device()
    data = load_stars()                       # (S, N, 2)
    print(f"data: {data.shape}, range [{data.min():.3f}, {data.max():.3f}]")
    sched = DiffusionSchedule.from_betas(make_schedule("linear", T))

    model = PermInvNoiseModel(hidden=128)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params}")

    t0 = time.time()
    hist = train_setwise(model, data, sched, device,
                         steps=TOTAL_STEPS, batch_size=BATCH, lr=LR,
                         log_every=1000, snapshot_steps=SNAPSHOTS,
                         sample_clouds=4)
    dt = time.time() - t0
    print(f"trained in {dt:.1f}s ({dt/TOTAL_STEPS*1000:.1f} ms/step)")
    torch.save(model.state_dict(), CKPT_DIR / "stars.pt")

    # Loss curve
    losses = hist["losses"]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    win = 100
    sm = np.convolve(losses, np.ones(win) / win, mode="valid")
    ax.plot(np.arange(len(sm)) + win, sm)
    ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_ylabel("MSE loss (smoothed)")
    ax.set_title("Part 3 — training loss (stars)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "phase3_loss.png", dpi=120); plt.close(fig)

    # Progress snapshots (1 cloud per snapshot, side-by-side)
    fig, axes = plt.subplots(1, len(SNAPSHOTS), figsize=(3.6 * len(SNAPSHOTS), 3.6))
    for ax, step in zip(axes, SNAPSHOTS):
        pts = hist["snapshots"][step][0]      # first of the 4 samples
        ax.scatter(pts[:, 0], pts[:, 1], s=2, c="C3")
        ax.set_title(f"step {step}")
        ax.set_aspect("equal"); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "phase3_progress.png", dpi=120); plt.close(fig)

    # Final generation: 6 fresh samples + 3 training examples
    final = sample_setwise(model, sched, device, n_clouds=6, n_pts=1024)
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    for k, ax in enumerate(axes[0]):
        ax.plot(data[k, :, 0], data[k, :, 1], '-', lw=0.5)
        ax.scatter(data[k, :, 0], data[k, :, 1], s=2)
        ax.set_title(f"data sample #{k}")
        ax.set_aspect("equal"); ax.grid(alpha=0.3)
    for k, ax in enumerate(axes[1]):
        ax.scatter(final[k, :, 0], final[k, :, 1], s=2, c="C3")
        ax.set_title(f"generated #{k}")
        ax.set_aspect("equal"); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "phase3_final.png", dpi=120); plt.close(fig)
    print("Saved phase3_*.png")

    np.savez(CKPT_DIR / "phase3_artifacts.npz",
             final=final,
             **{f"snap_{k}": hist["snapshots"][k] for k in SNAPSHOTS})
    print("Saved phase3_artifacts.npz")


if __name__ == "__main__":
    main()
