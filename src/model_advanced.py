"""Permutation-equivariant noise-predicting model for Part 3.

A PointNet-style network: per-point MLP -> global mean pool -> per-point head.
Input:  x in R^{B, N, 2}, t in {0..T-1} per sample (B,)
Output: eps_theta of same shape (B, N, 2)
"""
from __future__ import annotations

import torch
from torch import nn

from .sinusoidal_embedding import SinusoidalEmbedding


class PermInvNoiseModel(nn.Module):
    def __init__(self, pos_emb_size: int = 64, t_emb_size: int = 64,
                 hidden: int = 128, pos_scale: float = 25.0):
        super().__init__()
        self.pos_emb_x = SinusoidalEmbedding(pos_emb_size, scale=pos_scale)
        self.pos_emb_y = SinusoidalEmbedding(pos_emb_size, scale=pos_scale)
        self.t_emb = SinusoidalEmbedding(t_emb_size, scale=1.0)

        in_dim = 2 * pos_emb_size + t_emb_size
        # Per-point local encoder
        self.local = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        # Output head: concat(local_feat, global_feat, t_emb)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden + t_emb_size, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: (B, N, 2), t: (B,)
        B, N, _ = x.shape
        ex = self.pos_emb_x(x[..., 0])              # (B, N, P)
        ey = self.pos_emb_y(x[..., 1])              # (B, N, P)
        et = self.t_emb(t.float())                  # (B, T)
        et_n = et.unsqueeze(1).expand(B, N, et.size(-1))
        feat_in = torch.cat([ex, ey, et_n], dim=-1)  # (B, N, in_dim)
        local = self.local(feat_in)                  # (B, N, H)
        glob = local.mean(dim=1, keepdim=True)       # (B, 1, H)  permutation-invariant
        glob_n = glob.expand(B, N, local.size(-1))
        cat = torch.cat([local, glob_n, et_n], dim=-1)
        return self.head(cat)
