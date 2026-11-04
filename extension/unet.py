"""U-Net for MNIST DDPM (28x28 grayscale) with optional class conditioning.

Architecture (with default ch_mults=(1, 2, 4) and num_res_blocks=2):
    conv_in (1 -> 64) @ 28
    Stage 0: 2 DownBlocks @ 28 (64 ch)            -> Downsample -> 14
    Stage 1: 2 DownBlocks @ 14 (128 ch)           -> Downsample -> 7
    Stage 2: 2 DownBlocks @ 7  (256 ch, w/ attn)
    Mid:     ResBlock + Attn + ResBlock @ 7 (256 ch)
    Stage 2': 3 UpBlocks @ 7  (256 ch, w/ attn)   -> Upsample -> 14
    Stage 1': 3 UpBlocks @ 14 (128 ch)            -> Upsample -> 28
    Stage 0': 3 UpBlocks @ 28 (64 ch)
    conv_out (64 -> 1) @ 28

Class conditioning (set num_classes=10 to enable):
    embedding index `num_classes` is the "null" token used by
    Classifier-Free Guidance — drop labels with this token during
    training, and use it as the unconditional branch at inference.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"dim must be even, got {dim}")
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / half
        )
        args = t.float()[:, None] * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)


def _gn(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, dropout: float = 0.1):
        super().__init__()
        self.in_layers = nn.Sequential(
            _gn(in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_ch),
        )
        self.out_layers = nn.Sequential(
            _gn(out_ch),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )
        self.skip = (
            nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )
        # Zero-init last conv → identity at init for stable diffusion training.
        nn.init.zeros_(self.out_layers[-1].weight)
        nn.init.zeros_(self.out_layers[-1].bias)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.in_layers(x)
        h = h + self.emb_layers(emb)[:, :, None, None]
        h = self.out_layers(h)
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels {channels} not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.norm = _gn(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, C // self.num_heads, H * W)
        q, k, v = qkv.unbind(dim=1)
        scale = 1.0 / math.sqrt(C // self.num_heads)
        attn = torch.einsum("bhcn,bhcm->bhnm", q * scale, k).softmax(dim=-1)
        out = torch.einsum("bhnm,bhcm->bhcn", attn, v).reshape(B, C, H, W)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        target_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        if target_size is None:
            x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        else:
            x = F.interpolate(x, size=target_size, mode="nearest")
        return self.conv(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim, with_attn: bool, dropout: float):
        super().__init__()
        self.res = ResBlock(in_ch, out_ch, emb_dim, dropout)
        self.attn = AttentionBlock(out_ch) if with_attn else None

    def forward(self, x, emb):
        x = self.res(x, emb)
        if self.attn is not None:
            x = self.attn(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, emb_dim, with_attn: bool, dropout: float):
        super().__init__()
        self.res = ResBlock(in_ch + skip_ch, out_ch, emb_dim, dropout)
        self.attn = AttentionBlock(out_ch) if with_attn else None

    def forward(self, x, skip, emb):
        x = torch.cat([x, skip], dim=1)
        x = self.res(x, emb)
        if self.attn is not None:
            x = self.attn(x)
        return x


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        ch_mults: tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        attn_resolutions: tuple[int, ...] = (7,),
        time_emb_dim: int = 256,
        num_classes: int | None = None,
        dropout: float = 0.1,
        image_size: int = 28,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.null_class = num_classes  # last embedding index is the CFG null token
        self.image_size = image_size

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        if num_classes is not None:
            self.class_embed = nn.Embedding(num_classes + 1, time_emb_dim)
            with torch.no_grad():
                self.class_embed.weight[num_classes].zero_()
        else:
            self.class_embed = None

        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.down_path = nn.ModuleList()
        skip_chs: list[int] = [base_channels]
        cur_ch = base_channels
        cur_res = image_size
        for i, mult in enumerate(ch_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.down_path.append(
                    DownBlock(
                        cur_ch, out_ch, time_emb_dim,
                        with_attn=(cur_res in attn_resolutions),
                        dropout=dropout,
                    )
                )
                cur_ch = out_ch
                skip_chs.append(cur_ch)
            if i != len(ch_mults) - 1:
                self.down_path.append(Downsample(cur_ch))
                skip_chs.append(cur_ch)
                cur_res = cur_res // 2

        self.mid1 = ResBlock(cur_ch, cur_ch, time_emb_dim, dropout)
        self.mid_attn = AttentionBlock(cur_ch)
        self.mid2 = ResBlock(cur_ch, cur_ch, time_emb_dim, dropout)

        self.up_path = nn.ModuleList()
        for i, mult in enumerate(reversed(ch_mults)):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks + 1):
                skip_ch = skip_chs.pop()
                self.up_path.append(
                    UpBlock(
                        cur_ch, skip_ch, out_ch, time_emb_dim,
                        with_attn=(cur_res in attn_resolutions),
                        dropout=dropout,
                    )
                )
                cur_ch = out_ch
            if i != len(ch_mults) - 1:
                self.up_path.append(Upsample(cur_ch))
                cur_res = cur_res * 2

        assert not skip_chs, f"unused skip channels: {skip_chs}"

        self.norm_out = _gn(cur_ch)
        self.conv_out = nn.Conv2d(cur_ch, in_channels, 3, padding=1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        emb = self.time_embed(t)
        if self.class_embed is not None:
            if y is None:
                y = torch.full(
                    (x.shape[0],), self.null_class, dtype=torch.long, device=x.device
                )
            emb = emb + self.class_embed(y)

        h = self.conv_in(x)
        skips: list[torch.Tensor] = [h]
        sizes: list[tuple[int, int]] = []

        for layer in self.down_path:
            if isinstance(layer, DownBlock):
                h = layer(h, emb)
                skips.append(h)
            else:  # Downsample
                sizes.append(tuple(h.shape[-2:]))
                h = layer(h)
                skips.append(h)

        h = self.mid1(h, emb)
        h = self.mid_attn(h)
        h = self.mid2(h, emb)

        for layer in self.up_path:
            if isinstance(layer, UpBlock):
                h = layer(h, skips.pop(), emb)
            else:  # Upsample
                h = layer(h, target_size=sizes.pop())

        h = F.silu(self.norm_out(h))
        return self.conv_out(h)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    torch.manual_seed(0)

    for num_classes in (None, 10):
        model = UNet(num_classes=num_classes)
        n = count_parameters(model)
        print(f"UNet(num_classes={num_classes}): {n:,} parameters")

        x = torch.randn(4, 1, 28, 28)
        t = torch.randint(0, 1000, (4,))
        if num_classes is None:
            out = model(x, t)
        else:
            y = torch.randint(0, num_classes, (4,))
            out = model(x, t, y)
            uncond = model(x, t)  # null token path
            assert uncond.shape == out.shape
        assert out.shape == x.shape, f"shape mismatch: {out.shape} vs {x.shape}"
        print(f"  forward OK: in={tuple(x.shape)}  out={tuple(out.shape)}")
