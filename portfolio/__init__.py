"""portfolio-analytics — portfolio statistics, risk metrics and optimization.

Return/risk stats, correlation, beta, historical VaR/CVaR, and a dependency-free
mean-variance optimizer (min-variance + max-Sharpe via simplex sampling)."""
from __future__ import annotations

from .stats import (
    returns_from_prices,
    annualized_return,
    annualized_volatility,
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
)
from .risk import correlation_matrix, beta, value_at_risk, conditional_var
from .optimizer import (
    equal_weight,
    portfolio_return,
    portfolio_volatility,
    min_variance_weights,
    max_sharpe_weights,
    frontier_samples,
)

__all__ = [
    "returns_from_prices",
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "correlation_matrix",
    "beta",
    "value_at_risk",
    "conditional_var",
    "equal_weight",
    "portfolio_return",
    "portfolio_volatility",
    "min_variance_weights",
    "max_sharpe_weights",
    "frontier_samples",
]
