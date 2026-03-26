"""Portfolio optimization (dependency-free, long-only via simplex sampling)."""
from __future__ import annotations

import numpy as np

from .stats import sharpe_ratio


def equal_weight(n: int) -> np.ndarray:
    return np.ones(n) / n


def portfolio_return(weights, mean_returns) -> float:
    return float(np.asarray(weights) @ np.asarray(mean_returns))


def portfolio_volatility(weights, cov) -> float:
    w = np.asarray(weights, dtype=float)
    c = np.asarray(cov, dtype=float)
    return float(np.sqrt(w @ c @ w))


def min_variance_weights(cov) -> np.ndarray:
    """Analytical global minimum-variance weights (may include shorts)."""
    c = np.asarray(cov, dtype=float)
    inv = np.linalg.inv(c)
    ones = np.ones(c.shape[0])
    w = inv @ ones
    return w / w.sum()


def max_sharpe_weights(mean_returns, cov, risk_free: float = 0.0, n_samples: int = 5000, seed: int | None = None) -> np.ndarray:
    """Long-only max-Sharpe portfolio via Dirichlet simplex sampling."""
    mu = np.asarray(mean_returns, dtype=float)
    c = np.asarray(cov, dtype=float)
    n = mu.size
    rng = np.random.default_rng(seed)
    best_w = equal_weight(n)
    best_s = -np.inf
    rf_daily = risk_free / 252.0
    for _ in range(n_samples):
        w = rng.dirichlet(np.ones(n))
        ret = float(w @ mu)
        vol = float(np.sqrt(w @ c @ w))
        if vol == 0:
            continue
        s = (ret - rf_daily) / vol
        if s > best_s:
            best_s, best_w = s, w
    return best_w


def frontier_samples(mean_returns, cov, n_samples: int = 2000, seed: int | None = None):
    """Random (vol, ret) cloud on the long-only simplex; useful for plotting."""
    mu = np.asarray(mean_returns, dtype=float)
    c = np.asarray(cov, dtype=float)
    n = mu.size
    rng = np.random.default_rng(seed)
    pts = np.empty((n_samples, 2))
    for i in range(n_samples):
        w = rng.dirichlet(np.ones(n))
        pts[i, 0] = float(np.sqrt(w @ c @ w))
        pts[i, 1] = float(w @ mu)
    return pts
