# portfolio-analytics
![tests](https://github.com/chunyang666888/portfolio-analytics/actions/workflows/ci.yml/badge.svg)


> **Portfolio statistics, risk analytics and a mean-variance optimizer** — all in typed numpy with zero solver dependencies. Compute return/risk stats, correlation & beta, historical VaR/CVaR, and build minimum-variance or max-Sharpe portfolios (plus an efficient-frontier point cloud for plotting).

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](#running-tests)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](#license)

## Why this repo exists

Modern portfolio theory is table stakes for quant roles, but most candidates only *cite* it. This repo *implements* it — correlation/beta/VaR, plus an optimizer that needs no `scipy` (closed-form min-variance + Dirichlet simplex sampling for max-Sharpe). That signals you understand the math, not just the buzzwords.

## Features

- **Stats** — annualized return/volatility, Sharpe, Sortino, Max Drawdown.
- **Risk** — correlation matrix, CAPM beta, historical VaR & CVaR (expected shortfall).
- **Optimizer** — `equal_weight`, analytical `min_variance_weights`, long-only `max_sharpe_weights`, and `frontier_samples` for visualization.
- Dependency-light: `numpy` only.

## Installation

```bash
pip install -r requirements.txt
# or
pip install -e .
```

## Quick start

```python
import numpy as np
from portfolio import min_variance_weights, max_sharpe_weights, correlation_matrix

returns = np.random.default_rng(0).normal(0, 0.01, (250, 4))
cov = np.cov(returns, rowvar=False)
mean = returns.mean(axis=0)

w_min = min_variance_weights(cov)
w_sharpe = max_sharpe_weights(mean, cov, seed=7)
print(correlation_matrix(returns))
```

Run the bundled demo:

```bash
python examples/portfolio_demo.py
```

## Architecture

| Module | Responsibility |
|--------|----------------|
| `stats.py`     | Return/risk summary statistics |
| `risk.py`      | Correlation, beta, VaR, CVaR |
| `optimizer.py` | Weights & efficient frontier |

## Notes

- `min_variance_weights` is the unconstrained analytical solution and may include short positions.
- `max_sharpe_weights` / `frontier_samples` enforce long-only weights via Dirichlet sampling — increase `n_samples` for smoother frontiers.
- All risk numbers are **point estimates**; pair with the backtester for robustness checks.

## Running tests

```bash
pytest -q
```

## Project structure

```
portfolio-analytics/
├── portfolio/
│   ├── __init__.py
│   ├── stats.py
│   ├── risk.py
│   └── optimizer.py
├── examples/
│   └── portfolio_demo.py
├── tests/
│   ├── test_stats.py
│   ├── test_risk.py
│   └── test_optimizer.py
├── requirements.txt
├── pyproject.toml
└── README.md
```

## License

MIT — free for personal and commercial use.
