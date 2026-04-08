import numpy as np

from portfolio.risk import correlation_matrix, beta, value_at_risk, conditional_var


def test_correlation_identical_series_is_one():
    m = np.column_stack([np.arange(10, dtype=float)] * 2)
    corr = correlation_matrix(m)
    assert np.allclose(np.diag(corr), 1.0)
    assert abs(corr[0, 1] - 1.0) < 1e-9


def test_beta_scaled_series():
    market = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
    asset = 2.0 * market
    assert abs(beta(asset, market) - 2.0) < 1e-9


def test_var_and_cvar_positive_and_ordered():
    r = np.array([-0.1, -0.05, 0.0, 0.05, 0.1])
    var = value_at_risk(r, alpha=0.05)
    cvar = conditional_var(r, alpha=0.05)
    assert var > 0
    assert cvar >= var - 1e-9
