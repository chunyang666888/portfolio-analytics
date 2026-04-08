"""Portfolio analytics on synthetic multi-asset returns (offline)."""
import numpy as np

from portfolio import (
    annualized_return, annualized_volatility, sharpe_ratio, max_drawdown,
    correlation_matrix, beta, value_at_risk, conditional_var,
    equal_weight, min_variance_weights, max_sharpe_weights, portfolio_volatility,
)

NAMES = ["股票A", "股票B", "股票C", "股票D"]


def main():
    rng = np.random.default_rng(11)
    n, t = 4, 250
    mu = np.array([0.0008, 0.0011, 0.0006, 0.0013])
    # random positive-definite-ish covariance
    base = rng.normal(0, 0.01, (t, n))
    returns = mu * np.ones((t, n)) + base

    ann_ret = [annualized_return(returns[:, i]) for i in range(n)]
    ann_vol = [annualized_volatility(returns[:, i]) for i in range(n)]
    print("=== Per-asset ===")
    for i, name in enumerate(NAMES):
        print(f"{name}: 年化收益 {ann_ret[i]:.1%}  年化波动 {ann_vol[i]:.1%}  "
              f"Sharpe {sharpe_ratio(returns[:, i]):.2f}")

    print("\n=== Risk ===")
    print("相关性矩阵:\n", np.round(correlation_matrix(returns), 2))
    mkt = returns.mean(axis=1)
    for i, name in enumerate(NAMES):
        print(f"{name} beta: {beta(returns[:, i], mkt):.2f}")
    print(f"组合 VaR(95%): {value_at_risk(returns.mean(axis=1)):.2%}")
    print(f"组合 CVaR(95%): {conditional_var(returns.mean(axis=1)):.2%}")

    print("\n=== Optimization ===")
    cov = np.cov(returns, rowvar=False)
    mean = returns.mean(axis=0)
    w_eq = equal_weight(n)
    w_mv = min_variance_weights(cov)
    w_ms = max_sharpe_weights(mean, cov, seed=7)

    for label, w in [("等权", w_eq), ("最小方差", w_mv), ("最大夏普", w_ms)]:
        ret = float(w @ mean) * 252
        vol = portfolio_volatility(w, cov) * np.sqrt(252)
        shp = ret / vol if vol else 0
        print(f"{label}: 权重 {np.round(w, 3)}  年化收益 {ret:.1%}  波动 {vol:.1%}  Sharpe {shp:.2f}")


if __name__ == "__main__":
    main()
