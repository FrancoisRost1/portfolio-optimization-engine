"""
Tests for src/backtester.py — rolling backtest engine.

Uses synthetic data with shortened windows for fast execution.
"""

import numpy as np
import pandas as pd
import pytest

from src.backtester import run_backtest, run_all_backtests


@pytest.fixture
def bt_config(minimal_config):
    """Config tuned for fast backtest tests."""
    cfg = minimal_config.copy()
    cfg["backtest"] = {
        "rebalance_frequency": "monthly",
        "window_type": "rolling",
        "window_days": 126,
        "min_history": 126,
        "transaction_cost_bps": 10,
    }
    cfg["covariance"] = {"default_method": "ledoit_wolf", "rmt": {}}
    return cfg


@pytest.fixture
def bt_prices():
    """Longer synthetic prices for backtest tests (600 days, 5 assets)."""
    np.random.seed(55)
    T, N = 600, 5
    tickers = ["A", "B", "C", "D", "E"]
    dates = pd.bdate_range("2019-01-01", periods=T)
    rets = np.random.randn(T, N) * 0.01 + 0.0003
    return pd.DataFrame(
        100 * np.exp(np.cumsum(rets, axis=0)),
        columns=tickers,
        index=dates,
    )


@pytest.fixture
def bt_market_weights():
    w = pd.Series({"A": 0.30, "B": 0.25, "C": 0.20, "D": 0.15, "E": 0.10})
    return w / w.sum()


class TestRunBacktest:
    def test_returns_dict_keys(self, bt_prices, bt_config):
        result = run_backtest(bt_prices, "rp", bt_config)
        assert set(result.keys()) == {"returns", "weights_history", "turnover", "rebalance_dates"}

    def test_returns_not_empty(self, bt_prices, bt_config):
        result = run_backtest(bt_prices, "rp", bt_config)
        assert len(result["returns"]) > 0

    def test_returns_series_type(self, bt_prices, bt_config):
        result = run_backtest(bt_prices, "rp", bt_config)
        assert isinstance(result["returns"], pd.Series)

    def test_weights_history_populated(self, bt_prices, bt_config):
        result = run_backtest(bt_prices, "rp", bt_config)
        assert len(result["weights_history"]) > 0
        # Each entry is (date, Series)
        date, w = result["weights_history"][0]
        assert isinstance(w, pd.Series)
        assert abs(w.sum() - 1.0) < 1e-4

    def test_turnover_recorded(self, bt_prices, bt_config):
        result = run_backtest(bt_prices, "rp", bt_config)
        assert len(result["turnover"]) > 0
        # First rebalance from zero → turnover ≈ 2.0
        _, first_turnover = result["turnover"][0]
        assert first_turnover > 0

    def test_mv_method(self, bt_prices, bt_config):
        result = run_backtest(bt_prices, "mv", bt_config)
        assert len(result["returns"]) > 0

    def test_hrp_method(self, bt_prices, bt_config):
        result = run_backtest(bt_prices, "hrp", bt_config)
        assert len(result["returns"]) > 0

    def test_bl_method(self, bt_prices, bt_config, bt_market_weights):
        result = run_backtest(bt_prices, "bl", bt_config, market_weights=bt_market_weights)
        assert len(result["returns"]) > 0

    def test_invalid_method_returns_empty(self, bt_prices, bt_config):
        """Invalid method is caught internally — returns empty result."""
        result = run_backtest(bt_prices, "invalid", bt_config)
        assert len(result["returns"]) == 0

    def test_transaction_costs_reduce_returns(self, bt_prices, bt_config):
        """Backtest with higher TC should produce lower cumulative return."""
        bt_config["backtest"]["transaction_cost_bps"] = 0
        result_no_tc = run_backtest(bt_prices, "rp", bt_config)

        bt_config["backtest"]["transaction_cost_bps"] = 100
        result_high_tc = run_backtest(bt_prices, "rp", bt_config)

        cum_no_tc = (1 + result_no_tc["returns"]).prod()
        cum_high_tc = (1 + result_high_tc["returns"]).prod()
        assert cum_high_tc <= cum_no_tc + 1e-6

    def test_insufficient_history_empty(self, bt_config):
        """Too few days → empty returns."""
        np.random.seed(1)
        short_prices = pd.DataFrame(
            100 + np.random.randn(50, 5) * 0.1,
            columns=["A", "B", "C", "D", "E"],
            index=pd.bdate_range("2020-01-01", periods=50),
        )
        bt_config["backtest"]["min_history"] = 252
        result = run_backtest(short_prices, "rp", bt_config)
        assert len(result["returns"]) == 0


class TestRunAllBacktests:
    def test_all_methods_present(self, bt_prices, bt_config, bt_market_weights):
        results = run_all_backtests(bt_prices, bt_config, bt_market_weights)
        assert set(results.keys()) == {"mv", "rp", "hrp", "bl"}

    def test_all_produce_returns(self, bt_prices, bt_config, bt_market_weights):
        results = run_all_backtests(bt_prices, bt_config, bt_market_weights)
        for method, result in results.items():
            assert len(result["returns"]) > 0, f"{method} produced empty returns"
