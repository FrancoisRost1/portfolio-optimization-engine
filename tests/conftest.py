"""
Shared test fixtures — synthetic data, config, and pre-computed objects
used across all test modules.

Uses deterministic random seed for reproducibility.
"""

import numpy as np
import pandas as pd
import pytest

from utils.config_loader import load_config, reset_cache


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    """Load the real config.yaml."""
    reset_cache()
    return load_config()


@pytest.fixture
def minimal_config():
    """Minimal config dict for unit tests that don't need all fields."""
    return {
        "universe": {
            "tickers": ["A", "B", "C", "D", "E"],
            "asset_classes": {
                "A": "equities",
                "B": "equities",
                "C": "bonds",
                "D": "commodities",
                "E": "fx",
            },
        },
        "constraints": {
            "long_only": True,
            "per_asset": {"floor": 0.0, "cap": 0.40},
            "per_class": {
                "equities": 0.60,
                "bonds": 0.40,
                "commodities": 0.30,
                "fx": 0.20,
            },
            "max_turnover": 1.0,
        },
        "covariance": {
            "default_method": "ledoit_wolf",
            "rmt": {
                "noise_estimate": "median",
                "preserve_trace": True,
                "eigenvalue_floor": 1e-10,
            },
        },
        "expected_returns": {
            "default_method": "shrinkage",
            "shrinkage": {"intensity": 0.5},
        },
        "optimization": {
            "risk_aversion": 2.5,
            "mean_variance": {"solver": "SLSQP", "frontier_points": 10},
            "risk_parity": {"solver": "SLSQP", "max_iter": 1000, "tolerance": 1e-10},
            "hrp": {"linkage_method": "single", "distance_metric": "correlation"},
        },
        "black_litterman": {
            "tau": 0.05,
            "risk_aversion": 2.5,
            "default_views": {"A": 0.08, "C": -0.02},
            "view_confidence": {"A": 0.70, "C": 0.50},
            "default_confidence": 0.50,
            "manual_market_weights": {"A": 0.30, "B": 0.25, "C": 0.20, "D": 0.15, "E": 0.10},
        },
        "vol_targeting": {
            "enabled": False,
            "target_annual_vol": 0.10,
            "max_scale": 1.0,
        },
        "backtest": {
            "rebalance_frequency": "monthly",
            "window_type": "rolling",
            "window_days": 126,
            "min_history": 126,
            "transaction_cost_bps": 10,
        },
        "metrics": {
            "risk_free_rate": 0.04,
            "cvar_percentile": 0.05,
            "rolling_sharpe_window": 126,
        },
        "benchmarks": {
            "spy_buy_hold": {"ticker": "SPY"},
            "sixty_forty": {
                "equity_ticker": "SPY",
                "bond_ticker": "AGG",
                "equity_weight": 0.60,
                "bond_weight": 0.40,
                "rebalance": "monthly",
            },
            "equal_weight": {"rebalance": "monthly"},
            "static_risk_parity": {"rebalance": "none"},
        },
    }


# ---------------------------------------------------------------------------
# Synthetic price / return data
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_prices():
    """500-day synthetic prices for 5 assets with realistic structure."""
    np.random.seed(42)
    T, N = 500, 5
    tickers = ["A", "B", "C", "D", "E"]
    dates = pd.bdate_range("2020-01-01", periods=T)

    # Correlated returns via Cholesky
    corr = np.array([
        [1.0, 0.6, 0.2, 0.1, 0.0],
        [0.6, 1.0, 0.3, 0.1, 0.0],
        [0.2, 0.3, 1.0, 0.1, 0.0],
        [0.1, 0.1, 0.1, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
    ])
    L = np.linalg.cholesky(corr)
    daily_rets = (L @ np.random.randn(N, T)).T * 0.01 + 0.0003

    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(daily_rets, axis=0)),
        columns=tickers,
        index=dates,
    )
    return prices


@pytest.fixture
def synthetic_returns(synthetic_prices):
    """Daily arithmetic returns from synthetic prices."""
    return synthetic_prices.pct_change().dropna()


@pytest.fixture
def synthetic_log_returns(synthetic_prices):
    """Daily log returns from synthetic prices."""
    return np.log(synthetic_prices / synthetic_prices.shift(1)).dropna()


@pytest.fixture
def synthetic_cov(synthetic_log_returns):
    """Ledoit-Wolf covariance from synthetic returns."""
    from src.covariance import ledoit_wolf_covariance
    cov, _ = ledoit_wolf_covariance(synthetic_log_returns)
    return cov


@pytest.fixture
def synthetic_market_weights():
    """Synthetic market-cap weights for 5 assets."""
    w = pd.Series({"A": 0.30, "B": 0.25, "C": 0.20, "D": 0.15, "E": 0.10})
    return w / w.sum()


@pytest.fixture
def large_synthetic_prices():
    """800-day synthetic prices for 14 real tickers (for backtest / integration)."""
    np.random.seed(123)
    T = 800
    tickers = ["SPY", "EFA", "EEM", "IWM", "TLT", "IEF", "LQD",
               "HYG", "EMB", "GLD", "SLV", "DBC", "VNQ", "UUP"]
    N = len(tickers)
    dates = pd.bdate_range("2018-01-01", periods=T)
    daily_rets = np.random.randn(T, N) * 0.01 + 0.0002
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(daily_rets, axis=0)),
        columns=tickers,
        index=dates,
    )
    return prices
