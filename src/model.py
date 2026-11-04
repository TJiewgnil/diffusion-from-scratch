"""Noise-predicting MLP for Part 2 (per-point model).

Inputs:  x_t in R^2, t in {0..T-1}
Output:  predicted noise eps_theta(x_t, t) in R^2
"""
from __future__ import annotations

import torch
from torch import nn

from .sinusoidal_embedding import SinusoidalEmbedding


class NoiseMLP(nn.Module):
    def __init__(self, pos_emb_size: int = 64, t_emb_size: int = 64,
                 hidden: int = 128, n_layers: int = 3,
                 pos_scale: float = 25.0):
        super().__init__()
        self.pos_emb_x = SinusoidalEmbedding(pos_emb_size, scale=pos_scale)
        self.pos_emb_y = SinusoidalEmbedding(pos_emb_size, scale=pos_scale)
        self.t_emb = SinusoidalEmbedding(t_emb_size, scale=1.0)

        in_dim = 2 * pos_emb_size + t_emb_size
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.GELU()]
            d = hidden
        layers += [nn.Linear(hidden, 2)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: (B, 2), t: (B,) long or float
        ex = self.pos_emb_x(x[:, 0])
        ey = self.pos_emb_y(x[:, 1])
        et = self.t_emb(t.float())
        h = torch.cat([ex, ey, et], dim=-1)
        return self.net(h)
