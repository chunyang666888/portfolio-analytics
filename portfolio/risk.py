"""Risk analytics: correlation, beta, historical VaR / CVaR."""
from __future__ import annotations

import numpy as np


def correlation_matrix(returns_matrix):
    """Pearson correlation of asset returns. ``returns_matrix`` is (T, N)."""
    m = np.asarray(returns_matrix, dtype=float)
    return np.corrcoef(m, rowvar=False)


def beta(asset_returns, market_returns) -> float:
    """Slope of asset returns on market returns (CAPM beta)."""
    a = np.asarray(asset_returns, dtype=float)
    mk = np.asarray(market_returns, dtype=float)
    cov = np.cov(a, mk)
    var_mkt = cov[1, 1]
    return float(cov[0, 1] / var_mkt) if var_mkt != 0 else 0.0


def value_at_risk(returns, alpha: float = 0.05) -> float:
    """Historical VaR at confidence ``1-alpha`` (returned as a positive loss)."""
    r = np.asarray(returns, dtype=float)
    q = float(np.percentile(r, alpha * 100))
    return -q


def conditional_var(returns, alpha: float = 0.05) -> float:
    """Expected shortfall: mean loss in the worst ``alpha`` tail."""
    r = np.asarray(returns, dtype=float)
    q = float(np.percentile(r, alpha * 100))
    tail = r[r <= q]
    return float(-tail.mean()) if tail.size else 0.0
