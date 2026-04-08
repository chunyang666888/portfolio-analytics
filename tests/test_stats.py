import numpy as np

from portfolio.stats import (
    returns_from_prices,
    annualized_return,
    sharpe_ratio,
    max_drawdown,
)


def test_returns_from_prices():
    assert np.allclose(returns_from_prices([100, 110, 121]), [0.1, 0.1])


def test_annualized_return_flat():
    # two 1% periods, 1 period/year => CAGR ~1%
    assert abs(annualized_return([0.01, 0.01], periods_per_year=1) - 0.01) < 1e-9


def test_sharpe_zero_when_no_vol():
    assert sharpe_ratio([0.01, 0.01, 0.01]) == 0.0


def test_max_drawdown():
    eq = [100, 120, 60, 90]
    assert abs(max_drawdown(eq) - (-0.5)) < 1e-9
