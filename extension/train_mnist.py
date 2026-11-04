"""Train DDPM on MNIST. Supports unconditional and class-conditional + CFG.

Examples
--------
    # unconditional, T4 GPU
    python extension/train_mnist.py --steps 50000

    # class-conditional with classifier-free guidance
    python extension/train_mnist.py --steps 50000 --conditional

    # quick local CPU dry-run (no AMP, tiny batch)
    python extension/train_mnist.py --steps 50 --batch-size 8 --no-amp \
        --sample-every 20 --ckpt-every 50 --num-workers 0
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from unet import UNet, count_parameters  # noqa: E402
from ddpm import DDPM  # noqa: E402


# --------------------------- utilities ---------------------------


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EMA:
    """Exponential moving average of model parameters."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {
            n: p.detach().clone()
            for n, p in model.named_parameters()
            if p.requires_grad
        }
        self.backup: dict[str, torch.Tensor] | None = None

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    def copy_to(self, model: torch.nn.Module):
        for n, p in model.named_parameters():
            if n in self.shadow:
                p.data.copy_(self.shadow[n])

    def store(self, model: torch.nn.Module):
        self.backup = {
            n: p.detach().clone()
            for n, p in model.named_parameters()
            if p.requires_grad
        }

    def restore(self, model: torch.nn.Module):
        assert self.backup is not None
        for n, p in model.named_parameters():
            if n in self.backup:
                p.data.copy_(self.backup[n])
        self.backup = None


def get_mnist_loader(
    data_dir: str, batch_size: int, num_workers: int, pin_memory: bool
) -> DataLoader:
    transform = transforms.Compose([
        transforms.ToTensor(),                       # [0, 1]
        transforms.Lambda(lambda x: x * 2.0 - 1.0),  # [-1, 1]
    ])
    train_set = datasets.MNIST(
        data_dir, train=True, download=True, transform=transform
    )
    return DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        drop_last=True,
    )


def save_sample_grid(
    samples: torch.Tensor,
    path: Path,
    nrow: int = 8,
    title: str | None = None,
):
    """samples: (B, 1, 28, 28) in [-1, 1]."""
    samples = (samples.clamp(-1, 1) + 1) / 2
    B = samples.shape[0]
    ncol = math.ceil(B / nrow)
    fig, axes = plt.subplots(
        ncol, nrow, figsize=(nrow * 1.1, ncol * 1.1), squeeze=False
    )
    for i, ax in enumerate(axes.flat):
        if i < B:
            ax.imshow(
                samples[i, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1
            )
        ax.axis("off")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_loss(loss_log: list[tuple[int, float]], path: Path):
    if not loss_log:
        return
    steps, losses = zip(*loss_log)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(steps, losses)
    ax.set_xlabel("step")
    ax.set_ylabel("loss (avg per 100 steps)")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


# --------------------------- main ---------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data", type=str)
    parser.add_argument(
        "--out-dir", default="./extension/runs/mnist_uncond", type=str
    )
    parser.add_argument("--steps", default=50000, type=int)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--lr", default=2e-4, type=float)
    parser.add_argument("--num-timesteps", default=1000, type=int)
    parser.add_argument("--ema-decay", default=0.9999, type=float)
    parser.add_argument(
        "--conditional", action="store_true",
        help="train class-conditional model with CFG",
    )
    parser.add_argument(
        "--cfg-dropout", default=0.1, type=float,
        help="probability of replacing label with null token during training",
    )
    parser.add_argument(
        "--cfg-scale", default=3.0, type=float,
        help="guidance scale used in periodic visualization sampling",
    )
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True,
        help="mixed-precision training (CUDA only)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        type=str,
    )
    parser.add_argument("--sample-every", default=2000, type=int)
    parser.add_argument("--ckpt-every", default=5000, type=int)
    parser.add_argument("--num-workers", default=2, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default=None, type=str)
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    use_amp = args.amp and device.type == "cuda"
    pin_memory = device.type == "cuda"

    # ------- data
    loader = get_mnist_loader(
        args.data_dir, args.batch_size, args.num_workers, pin_memory
    )
    iter_loader = iter(loader)

    # ------- model
    num_classes = 10 if args.conditional else None
    model = UNet(num_classes=num_classes).to(device)
    n_params = count_parameters(model)
    print(
        f"UNet({'cond' if args.conditional else 'uncond'}): "
        f"{n_params:,} params  on {device}  amp={use_amp}"
    )

    diffusion = DDPM(num_timesteps=args.num_timesteps).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ema = EMA(model, decay=args.ema_decay)

    # fixed noise / labels for periodic sampling visualization
    if args.conditional:
        vis_labels = torch.arange(10, device=device).repeat_interleave(8)  # 80
        vis_shape = (80, 1, 28, 28)
    else:
        vis_labels = None
        vis_shape = (64, 1, 28, 28)

    # ------- resume
    start_step = 0
    loss_log: list[tuple[int, float]] = []
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        for k, v in ckpt["ema_shadow"].items():
            ema.shadow[k] = v.to(device)
        start_step = ckpt["step"] + 1
        loss_log = ckpt.get("loss_log", [])
        print(f"resumed from {args.resume} at step {start_step}")

    # ------- training loop
    pbar = tqdm(
        range(start_step, args.steps),
        initial=start_step,
        total=args.steps,
        dynamic_ncols=True,
    )
    running_loss = 0.0
    running_count = 0
    log_every = 100
    t_train_start = time.time()

    for step in pbar:
        try:
            x, y = next(iter_loader)
        except StopIteration:
            iter_loader = iter(loader)
            x, y = next(iter_loader)
        x = x.to(device, non_blocking=True)

        if args.conditional:
            y = y.to(device, non_blocking=True)
            if args.cfg_dropout > 0:
                drop = torch.rand(y.shape[0], device=device) < args.cfg_dropout
                y = torch.where(drop, torch.full_like(y, num_classes), y)
        else:
            y = None

        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            loss = diffusion.loss(model, x, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        ema.update(model)

        running_loss += loss.item()
        running_count += 1
        if (step + 1) % log_every == 0:
            avg = running_loss / running_count
            running_loss = 0.0
            running_count = 0
            loss_log.append((step + 1, avg))
            pbar.set_postfix(loss=f"{avg:.4f}")

        if (step + 1) % args.sample_every == 0:
            ema.store(model)
            ema.copy_to(model)
            model.eval()
            samples = diffusion.p_sample_loop(
                model,
                shape=vis_shape,
                device=device,
                y=vis_labels,
                cfg_scale=args.cfg_scale if args.conditional else 1.0,
                num_classes=num_classes,
            )
            tag = (
                f"step {step+1}  EMA"
                + (f"  cfg={args.cfg_scale}" if args.conditional else "")
            )
            save_sample_grid(
                samples,
                out_dir / "samples" / f"step_{step+1:06d}.png",
                nrow=10 if args.conditional else 8,
                title=tag,
            )
            ema.restore(model)

        if (step + 1) % args.ckpt_every == 0:
            torch.save({
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "ema_shadow": {k: v.cpu() for k, v in ema.shadow.items()},
                "args": vars(args),
                "loss_log": loss_log,
            }, out_dir / "last.pt")
            plot_loss(loss_log, out_dir / "loss.png")

    # ------- final EMA-only checkpoint (small, for distribution)
    ema.store(model)
    ema.copy_to(model)
    torch.save({
        "step": args.steps - 1,
        "model": model.state_dict(),       # already EMA-weighted
        "ema_shadow": {k: v.cpu() for k, v in ema.shadow.items()},
        "args": vars(args),
    }, out_dir / "final_ema.pt")
    ema.restore(model)

    plot_loss(loss_log, out_dir / "loss.png")
    elapsed = time.time() - t_train_start
    print(f"done in {elapsed/60:.1f} min  ->  {out_dir}")


if __name__ == "__main__":
    main()
