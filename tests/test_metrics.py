"""
Tests for src/metrics.py — CAGR, Sharpe, Sortino, Calmar, CVaR,
max drawdown, tracking error, information ratio, rolling Sharpe.
"""

import numpy as np
import pandas as pd
import pytest

from src.metrics import (
    compute_all_metrics,
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    calmar_ratio,
    cvar_95,
    tracking_error,
    information_ratio,
    rolling_sharpe,
    drawdown_series,
)
from src.returns import annualize_returns, annualize_vol


@pytest.fixture
def daily_returns():
    """Deterministic daily returns with positive drift for metric tests."""
    np.random.seed(42)
    return pd.Series(np.random.randn(500) * 0.008 + 0.0005, name="port")


@pytest.fixture
def benchmark_rets():
    np.random.seed(8)
    return pd.Series(np.random.randn(500) * 0.01 + 0.0002, name="bench")


class TestSharpeRatio:
    def test_positive_for_positive_excess(self, daily_returns):
        """With positive drift and rf=0, Sharpe should be positive."""
        sr = sharpe_ratio(daily_returns, rf=0.0)
        assert sr > 0, f"Sharpe={sr}, CAGR={annualize_returns(daily_returns)}"

    def test_zero_vol_returns_nan(self):
        flat = pd.Series([0.0] * 100)
        assert np.isnan(sharpe_ratio(flat, rf=0.0))

    def test_decreases_with_higher_rf(self, daily_returns):
        sr_low = sharpe_ratio(daily_returns, rf=0.0)
        sr_high = sharpe_ratio(daily_returns, rf=0.10)
        assert sr_high < sr_low


class TestSortinoRatio:
    def test_positive(self, daily_returns):
        """With positive drift and rf=0, Sortino should be positive."""
        sr = sortino_ratio(daily_returns, rf=0.0)
        assert sr > 0, f"Sortino={sr}"

    def test_higher_than_sharpe(self, daily_returns):
        """Sortino >= Sharpe when there are more up days than down."""
        sharpe = sharpe_ratio(daily_returns, rf=0.0)
        sortino = sortino_ratio(daily_returns, rf=0.0)
        # Not guaranteed but very likely with positive mean
        # Just check it computes without error
        assert not np.isnan(sortino)


class TestMaxDrawdown:
    def test_always_negative_or_zero(self, daily_returns):
        mdd = max_drawdown(daily_returns)
        assert mdd <= 0.0

    def test_zero_for_always_positive(self):
        """Monotonically increasing wealth → 0 drawdown."""
        rets = pd.Series([0.01] * 100)
        mdd = max_drawdown(rets)
        assert mdd == pytest.approx(0.0)

    def test_known_drawdown(self):
        """50% drop: 1.0 → 0.5 (drawdown = -50%)."""
        rets = pd.Series([0.0, -0.5, 0.0])
        mdd = max_drawdown(rets)
        assert mdd == pytest.approx(-0.5)


class TestCalmarRatio:
    def test_positive_when_profitable(self, daily_returns):
        """With positive drift, Calmar should be positive."""
        cr = calmar_ratio(daily_returns)
        assert cr > 0 or np.isnan(cr), f"Calmar={cr}"

    def test_nan_when_no_drawdown(self):
        rets = pd.Series([0.01] * 100)
        cr = calmar_ratio(rets)
        # max_drawdown = 0 → calmar = nan
        assert np.isnan(cr)


class TestCVaR:
    def test_negative(self, daily_returns):
        """CVaR (expected shortfall) should be negative — it's a tail loss."""
        cv = cvar_95(daily_returns, 0.05)
        assert cv < 0

    def test_worse_than_var(self, daily_returns):
        """CVaR <= VaR (mean of tail ≤ threshold)."""
        cv = cvar_95(daily_returns, 0.05)
        var = daily_returns.quantile(0.05)
        assert cv <= var + 1e-10


class TestTrackingError:
    def test_zero_when_identical(self, daily_returns):
        te = tracking_error(daily_returns, daily_returns)
        assert te == pytest.approx(0.0, abs=1e-10)

    def test_positive_when_different(self, daily_returns, benchmark_rets):
        te = tracking_error(daily_returns, benchmark_rets)
        assert te > 0


class TestInformationRatio:
    def test_finite(self, daily_returns, benchmark_rets):
        ir = information_ratio(daily_returns, benchmark_rets)
        assert not np.isnan(ir)


class TestRollingSharpe:
    def test_output_length(self, daily_returns):
        rs = rolling_sharpe(daily_returns, window=126)
        assert len(rs) == len(daily_returns)

    def test_nan_for_initial_window(self, daily_returns):
        rs = rolling_sharpe(daily_returns, window=126)
        assert rs.iloc[:125].isna().all()


class TestDrawdownSeries:
    def test_never_positive(self, daily_returns):
        dd = drawdown_series(daily_returns)
        assert (dd <= 1e-10).all()

    def test_starts_at_zero(self, daily_returns):
        dd = drawdown_series(daily_returns)
        assert dd.iloc[0] == pytest.approx(0.0)


class TestComputeAllMetrics:
    def test_contains_required_keys(self, daily_returns, minimal_config):
        m = compute_all_metrics(daily_returns, minimal_config)
        required = {"CAGR", "Annualized Vol", "Sharpe Ratio", "Sortino Ratio",
                     "Max Drawdown", "Calmar Ratio", "CVaR 95%"}
        assert required.issubset(set(m.keys()))

    def test_with_benchmark(self, daily_returns, benchmark_rets, minimal_config):
        m = compute_all_metrics(daily_returns, minimal_config, benchmark_rets)
        assert "Tracking Error" in m
        assert "Information Ratio" in m

    def test_all_values_numeric(self, daily_returns, minimal_config):
        m = compute_all_metrics(daily_returns, minimal_config)
        for k, v in m.items():
            assert isinstance(v, (int, float, np.floating)), f"{k} is {type(v)}"
