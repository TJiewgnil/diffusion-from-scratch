"""DDPM forward process (q_sample), training loss, and ancestral sampling.

Schedule defaults: linear β from 1e-4 to 0.02, T = 1000 (Ho et al. 2020).
Sampling variance: fixed_large = β_t (DDPM paper default).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_beta_schedule(
    schedule: str = "linear",
    num_timesteps: int = 1000,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
) -> torch.Tensor:
    if schedule == "linear":
        return torch.linspace(
            beta_start, beta_end, num_timesteps, dtype=torch.float64
        )
    if schedule == "cosine":
        # Nichol & Dhariwal 2021
        s = 0.008
        t = torch.arange(num_timesteps + 1, dtype=torch.float64) / num_timesteps
        f = torch.cos((t + s) / (1 + s) * torch.pi / 2) ** 2
        alpha_bar = f / f[0]
        betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
        return betas.clamp(max=0.999)
    raise ValueError(f"unknown schedule: {schedule}")


class DDPM(nn.Module):
    """Gaussian DDPM with ε-prediction objective."""

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_schedule: str = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        super().__init__()
        self.num_timesteps = num_timesteps

        betas = make_beta_schedule(
            beta_schedule, num_timesteps, beta_start, beta_end
        )
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], dtype=torch.float64), alphas_cumprod[:-1]]
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        # store as float32 buffers
        self.register_buffer("betas", betas.float())
        self.register_buffer("alphas", alphas.float())
        self.register_buffer("alphas_cumprod", alphas_cumprod.float())
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev.float())
        self.register_buffer(
            "sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod).float()
        )
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alphas_cumprod).float(),
        )
        self.register_buffer(
            "sqrt_recip_alphas", torch.sqrt(1.0 / alphas).float()
        )
        self.register_buffer("posterior_variance", posterior_variance.float())

    @staticmethod
    def _extract(values: torch.Tensor, t: torch.Tensor, shape) -> torch.Tensor:
        out = values.gather(0, t)
        return out.reshape(t.shape[0], *([1] * (len(shape) - 1)))

    # ---------------- forward / training ----------------

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Single-step forward to arbitrary t in closed form."""
        if noise is None:
            noise = torch.randn_like(x0)
        a = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        b = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return a * x0 + b * noise

    def loss(
        self,
        model: nn.Module,
        x0: torch.Tensor,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B = x0.shape[0]
        t = torch.randint(0, self.num_timesteps, (B,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        eps_pred = model(x_t, t, y) if y is not None else model(x_t, t)
        return F.mse_loss(eps_pred, noise)

    # ---------------- ancestral sampling ----------------

    @torch.no_grad()
    def p_sample_step(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: int,
        y: torch.Tensor | None = None,
        cfg_scale: float = 1.0,
        num_classes: int | None = None,
    ) -> torch.Tensor:
        B = x_t.shape[0]
        t_batch = torch.full((B,), t, device=x_t.device, dtype=torch.long)

        if cfg_scale != 1.0 and y is not None and num_classes is not None:
            # one batched forward for [uncond, cond]
            null_y = torch.full_like(y, num_classes)
            x_in = torch.cat([x_t, x_t], dim=0)
            t_in = torch.cat([t_batch, t_batch], dim=0)
            y_in = torch.cat([null_y, y], dim=0)
            eps_all = model(x_in, t_in, y_in)
            eps_uncond, eps_cond = eps_all.chunk(2, dim=0)
            eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
        else:
            eps = model(x_t, t_batch, y) if y is not None else model(x_t, t_batch)

        beta_t = self.betas[t]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t]
        sqrt_recip_alpha = self.sqrt_recip_alphas[t]
        mean = sqrt_recip_alpha * (x_t - beta_t / sqrt_one_minus * eps)

        if t == 0:
            return mean
        noise = torch.randn_like(x_t)
        return mean + beta_t.sqrt() * noise

    @torch.no_grad()
    def p_sample_loop(
        self,
        model: nn.Module,
        shape: tuple[int, ...],
        device: torch.device | str,
        y: torch.Tensor | None = None,
        cfg_scale: float = 1.0,
        num_classes: int | None = None,
        return_trajectory: bool = False,
        progress: bool = False,
    ):
        model.eval()
        x = torch.randn(shape, device=device)
        traj = [x.cpu()] if return_trajectory else None
        snap_every = max(1, self.num_timesteps // 20)

        iterator = reversed(range(self.num_timesteps))
        if progress:
            from tqdm import tqdm
            iterator = tqdm(list(iterator), desc="sampling", leave=False)

        for t in iterator:
            x = self.p_sample_step(
                model, x, t, y=y, cfg_scale=cfg_scale, num_classes=num_classes
            )
            if return_trajectory and (t % snap_every == 0 or t == 0):
                traj.append(x.cpu())

        return (x, traj) if return_trajectory else x


if __name__ == "__main__":
    torch.manual_seed(0)
    diff = DDPM(num_timesteps=1000)
    # check schedule: ᾱ_T close to 0
    abar_T = diff.alphas_cumprod[-1].item()
    print(f"alpha_bar_T = {abar_T:.2e}  (should be ~4e-5)")

    # q_sample roundtrip shape
    x0 = torch.randn(4, 1, 28, 28)
    t = torch.tensor([0, 250, 500, 999])
    x_t = diff.q_sample(x0, t)
    print(f"q_sample({tuple(x0.shape)}, t={t.tolist()}) -> {tuple(x_t.shape)}")

    # cumulative variance at t=999 ≈ 1
    big_x0 = torch.randn(10000, 1)
    t_T = torch.full((10000,), 999, dtype=torch.long)
    x_T = diff.q_sample(big_x0, t_T)
    print(f"x_T mean={x_T.mean().item():+.4f}  var={x_T.var().item():.4f}  (want 0, 1)")

    # mini sampling sanity (small T to keep fast)
    from unet import UNet
    model = UNet()
    small_diff = DDPM(num_timesteps=10)
    out = small_diff.p_sample_loop(model, shape=(2, 1, 28, 28), device="cpu")
    print(f"sample loop OK: out={tuple(out.shape)}  range=[{out.min():.2f}, {out.max():.2f}]")
