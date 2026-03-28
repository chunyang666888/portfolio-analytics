"""Portfolio return / risk summary statistics."""
from __future__ import annotations

import numpy as np


def returns_from_prices(prices) -> np.ndarray:
    """Simple period returns from a price series."""
    arr = np.asarray(prices, dtype=float)
    return arr[1:] / arr[:-1] - 1.0


def annualized_return(r, periods_per_year: int = 252) -> float:
    r = np.asarray(r, dtype=float)
    if r.size == 0:
        return 0.0
    return float(np.prod(1.0 + r) ** (periods_per_year / r.size) - 1.0)


def annualized_volatility(r, periods_per_year: int = 252) -> float:
    r = np.asarray(r, dtype=float)
    if r.size < 2:
        return 0.0
    return float(np.std(r, ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(r, risk_free: float = 0.0, periods_per_year: int = 252) -> float:
    r = np.asarray(r, dtype=float)
    excess = r - risk_free / periods_per_year
    if excess.std(ddof=1) == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * excess.mean() / excess.std(ddof=1))


def sortino_ratio(r, risk_free: float = 0.0, periods_per_year: int = 252) -> float:
    r = np.asarray(r, dtype=float)
    excess = r - risk_free / periods_per_year
    downside = excess[excess < 0]
    if downside.size == 0 or downside.std(ddof=1) == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * excess.mean() / downside.std(ddof=1))


def max_drawdown(equity_curve) -> float:
    eq = np.asarray(equity_curve, dtype=float)
    running_max = np.maximum.accumulate(eq)
    return float((eq / running_max - 1.0).min())
