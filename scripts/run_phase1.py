"""Phase 1 — Part 1: 1D analytical diffusion.

Forward q(x_t | x_{t-1}) with constant beta=0.25, T=50, x_0 ~ N(m,p).
Verify x_T ~ N(0,1).
Closed-form reverse using known x_0 prior, sample from N(0,1) and recover N(m,p).
Saves figures to project/figures/.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from src.utils import set_seed, FIGURES_DIR
from src.forward_process import DiffusionSchedule, make_schedule
from src.reverse_closed import reverse_step_marginal


def main():
    set_seed(0)
    rng = np.random.default_rng(0)

    # --- config ---
    m_prior, p_prior = 4.0, 0.5
    T = 50
    sched = DiffusionSchedule.from_betas(make_schedule("constant", T, beta_const=0.25))

    # --- forward trajectories ---
    N = 1000
    x0 = m_prior + np.sqrt(p_prior) * rng.standard_normal(N)
    traj = np.empty((T + 1, N))
    traj[0] = x0
    for t in range(1, T + 1):
        noise = rng.standard_normal(N)
        traj[t] = np.sqrt(1 - sched.betas[t - 1]) * traj[t - 1] + np.sqrt(sched.betas[t - 1]) * noise
    xT = traj[T]
    print(f"Forward: x_T mean={xT.mean():.3f}, var={xT.var():.3f}  "
          f"(target ~0, ~1).  abar_T={sched.alpha_bars[-1]:.4e}")

    # KS test vs N(0,1)
    ks_fwd = stats.kstest(xT, 'norm')
    print(f"  KS vs N(0,1): stat={ks_fwd.statistic:.4f}, p={ks_fwd.pvalue:.3f}")

    # --- reverse trajectories ---
    N_rev = 1000
    xt = rng.standard_normal(N_rev)
    rev = np.empty((T + 1, N_rev))
    rev[T] = xt
    for t in range(T, 0, -1):
        xt = reverse_step_marginal(xt, t, sched, m_prior, p_prior, rng)
        rev[t - 1] = xt
    x0_recovered = rev[0]
    print(f"Reverse: x_0 mean={x0_recovered.mean():.3f}, var={x0_recovered.var():.3f}  "
          f"(target {m_prior}, {p_prior})")
    ks_rev = stats.kstest((x0_recovered - m_prior) / np.sqrt(p_prior), 'norm')
    print(f"  KS vs N(m,p): stat={ks_rev.statistic:.4f}, p={ks_rev.pvalue:.3f}")

    # --- plots ---
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # forward trajectories
    ax = axes[0, 0]
    for i in range(15):
        ax.plot(np.arange(T + 1), traj[:, i], alpha=0.5, lw=1)
    ax.set_xlabel('t'); ax.set_ylabel('x_t')
    ax.set_title(f'Forward trajectories  (m={m_prior}, p={p_prior}, beta=0.25, T={T})')
    ax.grid(alpha=0.3)

    # x_T histogram vs N(0,1)
    ax = axes[0, 1]
    ax.hist(xT, bins=40, density=True, alpha=0.6, color='C0', label=r'$x_T$ samples')
    xs = np.linspace(-4, 4, 200)
    ax.plot(xs, stats.norm.pdf(xs), 'r--', lw=2, label='N(0,1)')
    ax.set_xlabel('x_T'); ax.set_ylabel('density')
    ax.set_title(rf'Forward: $x_T$ distribution  (KS p={ks_fwd.pvalue:.3f})')
    ax.legend(); ax.grid(alpha=0.3)

    # reverse trajectories
    ax = axes[1, 0]
    for i in range(15):
        ax.plot(np.arange(T + 1), rev[:, i], alpha=0.5, lw=1)
    ax.set_xlabel('t'); ax.set_ylabel('x_t')
    ax.set_title('Reverse trajectories (closed form, from N(0,1))')
    ax.grid(alpha=0.3)

    # x_0 histogram vs N(m,p)
    ax = axes[1, 1]
    ax.hist(x0_recovered, bins=40, density=True, alpha=0.6, color='C2',
            label=r'reverse $x_0$')
    ax.hist(x0, bins=40, density=True, alpha=0.3, color='C1', label=r'original $x_0$')
    xs = np.linspace(m_prior - 3 * np.sqrt(p_prior), m_prior + 3 * np.sqrt(p_prior), 200)
    ax.plot(xs, stats.norm.pdf(xs, loc=m_prior, scale=np.sqrt(p_prior)),
            'r--', lw=2, label=f'N({m_prior},{p_prior})')
    ax.set_xlabel('x_0'); ax.set_ylabel('density')
    ax.set_title(rf'Reverse: recovered $x_0$  (KS p={ks_rev.pvalue:.3f})')
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    out = FIGURES_DIR / 'phase1_task1.png'
    plt.savefig(out, dpi=120)
    print(f"Saved {out}")

    # --- ablation: alternative schedules ---
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, kind in zip(axes, ('constant', 'linear', 'quadratic')):
        s = DiffusionSchedule.from_betas(make_schedule(kind, T, beta_const=0.25))
        x = m_prior + np.sqrt(p_prior) * rng.standard_normal(N)
        for t in range(1, T + 1):
            x = np.sqrt(1 - s.betas[t - 1]) * x + np.sqrt(s.betas[t - 1]) * rng.standard_normal(N)
        ax.hist(x, bins=40, density=True, alpha=0.6)
        xs = np.linspace(-4, 4, 200)
        ax.plot(xs, stats.norm.pdf(xs), 'r--', lw=2, label='N(0,1)')
        ax.set_title(f'{kind} schedule\nmean={x.mean():.2f}, var={x.var():.2f}')
        ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    out = FIGURES_DIR / 'phase1_schedules.png'
    plt.savefig(out, dpi=120)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
