"""Closed-form reverse process for Part 1 (1D, x0 prior known).

Math (Ho 2020 eq. 6-7), with arrays for the batch dim:

    q(x_{t-1} | x_t, x_0) = N(mu_tilde, beta_tilde)
        mu_tilde = c0(t) * x_0  +  ct(t) * x_t
        c0(t)   = sqrt(abar_{t-1}) * beta_t / (1 - abar_t)
        ct(t)   = sqrt(alpha_t) * (1 - abar_{t-1}) / (1 - abar_t)
        beta_tilde = (1 - abar_{t-1}) / (1 - abar_t) * beta_t

For Part 1 the x_0 prior is N(m_prior, p_prior); combined with x_t = sqrt(abar_t) x_0 +
sqrt(1-abar_t) eps we obtain a Gaussian posterior

    q(x_0 | x_t) = N(mu0, sigma0^2)
        sigma0^2 = p * (1 - abar_t) / (1 - abar_t + p * abar_t)
        mu0      = ((1 - abar_t) * m + sqrt(abar_t) * p * x_t) / (1 - abar_t + p * abar_t)

Marginalising x_0 yields a closed-form q(x_{t-1} | x_t):
        mean = c0 * mu0 + ct * x_t
        var  = c0^2 * sigma0^2 + beta_tilde
"""
from __future__ import annotations

import numpy as np

from .forward_process import DiffusionSchedule


def _coeffs(t: int, sched: DiffusionSchedule) -> tuple[float, float, float]:
    """Return (c0, ct, beta_tilde) used for q(x_{t-1} | x_t, x_0). t in [1..T]."""
    assert t >= 1, "reverse step only defined for t >= 1"
    beta_t = sched.betas[t - 1]
    alpha_t = sched.alphas[t - 1]
    abar_t = sched.alpha_bars[t - 1]
    abar_tm1 = sched.alpha_bars[t - 2] if t >= 2 else 1.0

    c0 = (np.sqrt(abar_tm1) * beta_t) / (1.0 - abar_t)
    ct = (np.sqrt(alpha_t) * (1.0 - abar_tm1)) / (1.0 - abar_t)
    beta_tilde = (1.0 - abar_tm1) / (1.0 - abar_t) * beta_t
    return float(c0), float(ct), float(beta_tilde)


def posterior_mean_var(x_t: np.ndarray, x_0: np.ndarray, t: int,
                       sched: DiffusionSchedule) -> tuple[np.ndarray, float]:
    """q(x_{t-1} | x_t, x_0)."""
    c0, ct, beta_tilde = _coeffs(t, sched)
    return c0 * x_0 + ct * x_t, beta_tilde


def posterior_x0_given_xt(x_t: np.ndarray, t: int, sched: DiffusionSchedule,
                          m_prior: float, p_prior: float) -> tuple[np.ndarray, float]:
    """q(x_0 | x_t) under prior x_0 ~ N(m_prior, p_prior)."""
    abar_t = sched.alpha_bars[t - 1]
    denom = (1.0 - abar_t) + p_prior * abar_t
    sigma0_sq = p_prior * (1.0 - abar_t) / denom
    mu0 = ((1.0 - abar_t) * m_prior + np.sqrt(abar_t) * p_prior * x_t) / denom
    return mu0, float(sigma0_sq)


def reverse_step_marginal(x_t: np.ndarray, t: int, sched: DiffusionSchedule,
                          m_prior: float, p_prior: float,
                          rng: np.random.Generator) -> np.ndarray:
    """Sample x_{t-1} from q(x_{t-1} | x_t) under the known x_0 prior."""
    c0, ct, beta_tilde = _coeffs(t, sched)
    mu0, sigma0_sq = posterior_x0_given_xt(x_t, t, sched, m_prior, p_prior)
    mean = c0 * mu0 + ct * x_t
    var = c0 ** 2 * sigma0_sq + beta_tilde
    # NB: this is the marginal q(x_{t-1} | x_t). Even at t=1, q(x_0 | x_1) is a
    # Gaussian with variance sigma0^2 (the posterior over x_0 given x_1), so we
    # add noise at every step. (Unlike DDPM noise-prediction sampling which
    # drops noise at t=1 because it returns the predicted x_0 mean.)
    return mean + np.sqrt(var) * rng.standard_normal(size=x_t.shape)


def reverse_step_conditional(x_t: np.ndarray, x_0: np.ndarray, t: int,
                             sched: DiffusionSchedule,
                             rng: np.random.Generator) -> np.ndarray:
    """Sample x_{t-1} ~ q(x_{t-1} | x_t, x_0) for a known x_0 (used when the
    original x_0 is available, e.g. when checking the forward trajectory)."""
    mean, var = posterior_mean_var(x_t, x_0, t, sched)
    if t == 1:
        return mean
    return mean + np.sqrt(var) * rng.standard_normal(size=x_t.shape)
