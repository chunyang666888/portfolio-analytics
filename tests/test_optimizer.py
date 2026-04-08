import numpy as np

from portfolio.optimizer import (
    equal_weight,
    portfolio_volatility,
    min_variance_weights,
    max_sharpe_weights,
    frontier_samples,
)
from portfolio.stats import sharpe_ratio

MU = np.array([0.0010, 0.0015, 0.0008])
COV = np.array([
    [0.0004, 0.0001, 0.00005],
    [0.0001, 0.0009, 0.00008],
    [0.00005, 0.00008, 0.0003],
])


def test_equal_weight_sums_to_one():
    w = equal_weight(3)
    assert np.isclose(w.sum(), 1.0)
    assert (w > 0).all()


def test_portfolio_volatility_positive():
    assert portfolio_volatility(equal_weight(3), COV) > 0


def test_min_variance_sums_to_one():
    w = min_variance_weights(COV)
    assert np.isclose(w.sum(), 1.0)


def _port_sharpe(weights):
    # Daily Sharpe (rf=0) for a portfolio with mean vector MU and cov COV.
    ret = float(np.asarray(weights) @ MU)
    vol = float(np.sqrt(np.asarray(weights) @ COV @ np.asarray(weights)))
    return ret / vol


def test_max_sharpe_no_worse_than_equal_weight():
    w_ms = max_sharpe_weights(MU, COV, seed=42)
    assert np.isclose(w_ms.sum(), 1.0, atol=1e-9)
    s_equal = _port_sharpe(equal_weight(3))
    s_ms = _port_sharpe(w_ms)
    assert s_ms >= s_equal - 1e-6


def test_frontier_samples_shape():
    pts = frontier_samples(MU, COV, n_samples=50, seed=1)
    assert pts.shape == (50, 2)
