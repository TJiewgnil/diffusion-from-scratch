"""Phase 2 — Part 2: train a noise-prediction MLP per shape.

Trains one model for Circle, Square, Dinohead and saves:
  - checkpoints/<shape>.pt
  - figures/phase2_<shape>_progress.png  (snapshots at 1k/5k/10k/15k)
  - figures/phase2_loss.png              (training losses, all 3 shapes)
  - figures/phase2_final.png             (final samples vs data, all 3)
  - figures/phase2_ablation_T.png        (ablation on T, on Circle)
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

from src.utils import set_seed, get_device, load_shape, FIGURES_DIR, CKPT_DIR
from src.forward_process import DiffusionSchedule, make_schedule
from src.model import NoiseMLP
from src.train import train_pointwise, sample_pointwise

torch.set_num_threads(8)


SHAPES = ("circle", "square", "dinohead")
SNAPSHOTS = (1000, 5000, 10000, 15000)
TOTAL_STEPS = 15000
T = 1000
BATCH = 256
LR = 1e-3
SAMPLE_N = 1500


def train_one(shape: str, sched, device, seed: int = 0) -> dict:
    print(f"\n=== Training {shape} ===")
    set_seed(seed)
    data = load_shape(shape)
    model = NoiseMLP(hidden=128, n_layers=3)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  data: {data.shape}   params: {n_params}")
    t0 = time.time()
    hist = train_pointwise(model, data, sched, device,
                           steps=TOTAL_STEPS, batch_size=BATCH, lr=LR,
                           log_every=2500,
                           snapshot_steps=SNAPSHOTS, sample_n=SAMPLE_N)
    dt = time.time() - t0
    print(f"  trained in {dt:.1f}s ({dt/TOTAL_STEPS*1000:.2f} ms/step)")
    torch.save(model.state_dict(), CKPT_DIR / f"{shape}.pt")
    final = sample_pointwise(model, sched, device, n=SAMPLE_N)
    hist["final"] = final
    hist["data"] = data
    hist["params"] = n_params
    return hist


def progress_plot(shape: str, hist: dict, outfile: Path) -> None:
    fig, axes = plt.subplots(1, len(SNAPSHOTS) + 1, figsize=(3 * (len(SNAPSHOTS) + 1), 3.2))
    axes[0].scatter(hist["data"][:, 0], hist["data"][:, 1], s=2, c="C0")
    axes[0].set_title(f"{shape}: data")
    for ax in axes:
        ax.set_aspect("equal"); ax.grid(alpha=0.3)
    for ax, step in zip(axes[1:], SNAPSHOTS):
        pts = hist["snapshots"][step]
        ax.scatter(pts[:, 0], pts[:, 1], s=2, c="C3")
        ax.set_title(f"step {step}")
    plt.tight_layout()
    plt.savefig(outfile, dpi=120); plt.close(fig)
    print(f"  saved {outfile.name}")


def main():
    device = get_device()
    sched = DiffusionSchedule.from_betas(make_schedule("linear", T))

    histories: dict[str, dict] = {}
    for shape in SHAPES:
        histories[shape] = train_one(shape, sched, device)
        progress_plot(shape, histories[shape],
                      FIGURES_DIR / f"phase2_{shape}_progress.png")

    # Combined loss plot
    fig, ax = plt.subplots(figsize=(8, 4))
    for shape in SHAPES:
        losses = histories[shape]["losses"]
        # smoothed
        win = 200
        sm = np.convolve(losses, np.ones(win) / win, mode="valid")
        ax.plot(np.arange(len(sm)) + win, sm, label=shape)
    ax.set_xlabel("step"); ax.set_ylabel("MSE loss (smoothed)")
    ax.set_yscale("log")
    ax.set_title("Phase 2 training losses (window=200)")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "phase2_loss.png", dpi=120); plt.close(fig)

    # Final comparison
    fig, axes = plt.subplots(2, len(SHAPES), figsize=(4 * len(SHAPES), 7))
    for j, shape in enumerate(SHAPES):
        d = histories[shape]["data"]
        s = histories[shape]["final"]
        axes[0, j].scatter(d[:, 0], d[:, 1], s=2, c="C0"); axes[0, j].set_title(f"{shape}: data")
        axes[1, j].scatter(s[:, 0], s[:, 1], s=2, c="C3"); axes[1, j].set_title(f"{shape}: samples")
        for r in range(2):
            axes[r, j].set_aspect("equal"); axes[r, j].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "phase2_final.png", dpi=120); plt.close(fig)
    print("Saved phase2_final.png")

    # Ablation on T (Circle only, short training each)
    print("\n=== Ablation: T on Circle ===")
    abl_T = (100, 250, 1000)
    abl_results: dict[int, np.ndarray] = {}
    abl_steps = 6000
    for Tv in abl_T:
        set_seed(0)
        sch = DiffusionSchedule.from_betas(make_schedule("linear", Tv))
        m = NoiseMLP(hidden=128, n_layers=3)
        train_pointwise(m, load_shape("circle"), sch, device,
                        steps=abl_steps, batch_size=BATCH, lr=LR,
                        log_every=abl_steps)
        abl_results[Tv] = sample_pointwise(m, sch, device, n=SAMPLE_N)
        print(f"  T={Tv} done")
    fig, axes = plt.subplots(1, len(abl_T), figsize=(4 * len(abl_T), 3.6))
    for ax, Tv in zip(axes, abl_T):
        ax.scatter(abl_results[Tv][:, 0], abl_results[Tv][:, 1], s=2, c="C3")
        ax.set_title(f"T={Tv}")
        ax.set_aspect("equal"); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "phase2_ablation_T.png", dpi=120); plt.close(fig)
    print("Saved phase2_ablation_T.png")

    # Save numerical summaries
    np.savez(CKPT_DIR / "phase2_artifacts.npz",
             **{f"{s}_final": histories[s]["final"] for s in SHAPES},
             **{f"{s}_data": histories[s]["data"] for s in SHAPES},
             **{f"{s}_snap_{k}": histories[s]["snapshots"][k]
                for s in SHAPES for k in SNAPSHOTS})
    print("Saved phase2_artifacts.npz")


if __name__ == "__main__":
    main()
