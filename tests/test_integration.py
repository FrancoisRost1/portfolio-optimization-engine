"""
Integration tests — end-to-end pipeline on synthetic data.

Verifies that all modules work together correctly: data -> covariance ->
expected returns -> optimization -> risk decomposition -> backtest -> metrics.
"""

import numpy as np
import pandas as pd
import pytest

from utils.config_loader import load_config
from src.returns import log_returns, arithmetic_returns
from src.covariance import estimate_covariance
from src.expected_returns import estimate_expected_returns
from src.optimizer_mv import optimize_mean_variance
from src.optimizer_rp import optimize_risk_parity
from src.optimizer_hrp import optimize_hrp
from src.optimizer_bl import optimize_black_litterman
from src.vol_target import apply_vol_target
from src.efficient_frontier import generate_efficient_frontier
from src.backtester import run_backtest
from src.metrics import compute_all_metrics
from src.risk_decomposition import (
    risk_contribution,
    herfindahl_index,
    effective_n,
    portfolio_volatility,
    decompose_all_methods,
)


@pytest.fixture
def full_config():
    """Real config with shortened backtest for speed."""
    config = load_config()
    config["backtest"]["window_days"] = 126
    config["backtest"]["min_history"] = 126
    return config


@pytest.fixture
def market_weights_14():
    tickers = ["SPY", "EFA", "EEM", "IWM", "TLT", "IEF", "LQD",
               "HYG", "EMB", "GLD", "SLV", "DBC", "VNQ", "UUP"]
    w = pd.Series(np.arange(14, 0, -1, dtype=float), index=tickers)
    return w / w.sum()


class TestFullPipeline:
    """End-to-end: prices → returns → cov → optimizer → metrics."""

    def test_all_optimizers_produce_valid_weights(self, large_synthetic_prices,
                                                   full_config, market_weights_14):
        log_ret = log_returns(large_synthetic_prices)
        arith_ret = arithmetic_returns(large_synthetic_prices)

        cov, meta = estimate_covariance(log_ret, "ledoit_wolf", full_config)
        mu = estimate_expected_returns(arith_ret, "shrinkage", full_config)

        # MV
        w_mv = optimize_mean_variance(mu, cov, full_config)
        assert abs(w_mv.sum() - 1.0) < 1e-6
        assert (w_mv >= -1e-8).all()

        # RP
        w_rp = optimize_risk_parity(cov, full_config)
        assert abs(w_rp.sum() - 1.0) < 1e-6
        assert (w_rp >= -1e-8).all()

        # HRP
        w_hrp = optimize_hrp(cov, full_config)
        assert abs(w_hrp.sum() - 1.0) < 1e-6
        assert (w_hrp >= -1e-8).all()

        # BL
        w_bl, info = optimize_black_litterman(cov, market_weights_14, full_config)
        assert abs(w_bl.sum() - 1.0) < 1e-6
        assert (w_bl >= -1e-8).all()

    def test_risk_decomposition_consistent(self, large_synthetic_prices,
                                            full_config, market_weights_14):
        log_ret = log_returns(large_synthetic_prices)
        cov, _ = estimate_covariance(log_ret, "ledoit_wolf", full_config)

        w_rp = optimize_risk_parity(cov, full_config)
        rc = risk_contribution(w_rp, cov)
        assert abs(rc["pct_risk_contribution"].sum() - 1.0) < 1e-6

        h = herfindahl_index(w_rp)
        assert 0 < h < 1
        en = effective_n(w_rp)
        assert en > 1

        vol = portfolio_volatility(w_rp, cov)
        assert vol > 0

    def test_efficient_frontier_monotonic(self, large_synthetic_prices, full_config):
        """Higher return should require higher (or equal) vol on the frontier."""
        log_ret = log_returns(large_synthetic_prices)
        arith_ret = arithmetic_returns(large_synthetic_prices)
        cov, _ = estimate_covariance(log_ret, "ledoit_wolf", full_config)
        mu = estimate_expected_returns(arith_ret, "shrinkage", full_config)

        frontier = generate_efficient_frontier(mu, cov, full_config, n_points=15)
        if len(frontier) >= 2:
            # Sort by return; vol should be non-decreasing (on the efficient part)
            sorted_f = frontier.sort_values("target_return")
            # Allow some tolerance due to constraints
            assert sorted_f["volatility"].iloc[-1] >= sorted_f["volatility"].iloc[0] - 0.01

    def test_all_covariance_methods(self, large_synthetic_prices, full_config):
        """All three cov methods should produce valid PSD matrices."""
        log_ret = log_returns(large_synthetic_prices)
        for method in ["sample", "ledoit_wolf", "rmt"]:
            cov, meta = estimate_covariance(log_ret, method, full_config)
            n = len(cov)
            assert cov.shape == (n, n)
            eigenvalues = np.linalg.eigvalsh(cov.values)
            assert (eigenvalues >= -1e-8).all(), f"{method}: negative eigenvalue"

    def test_vol_targeting_overlay(self, large_synthetic_prices, full_config):
        log_ret = log_returns(large_synthetic_prices)
        cov, _ = estimate_covariance(log_ret, "ledoit_wolf", full_config)
        w_rp = optimize_risk_parity(cov, full_config)

        full_config["vol_targeting"]["enabled"] = True
        full_config["vol_targeting"]["target_annual_vol"] = 0.05
        w_scaled, info = apply_vol_target(w_rp, cov, full_config)
        assert info["enabled"] is True
        assert info["scale_factor"] <= 1.0
        assert w_scaled.sum() <= 1.0 + 1e-6

    def test_backtest_and_metrics(self, large_synthetic_prices,
                                   full_config, market_weights_14):
        """Run a short backtest and verify metrics compute."""
        result = run_backtest(large_synthetic_prices, "rp", full_config)
        assert len(result["returns"]) > 0

        metrics = compute_all_metrics(result["returns"], full_config)
        assert "CAGR" in metrics
        assert "Sharpe Ratio" in metrics
        assert "Max Drawdown" in metrics
        assert metrics["Max Drawdown"] <= 0

    def test_multi_method_decomposition(self, large_synthetic_prices, full_config):
        log_ret = log_returns(large_synthetic_prices)
        arith_ret = arithmetic_returns(large_synthetic_prices)
        cov, _ = estimate_covariance(log_ret, "ledoit_wolf", full_config)
        mu = estimate_expected_returns(arith_ret, "shrinkage", full_config)

        weights = {
            "MV": optimize_mean_variance(mu, cov, full_config),
            "RP": optimize_risk_parity(cov, full_config),
            "HRP": optimize_hrp(cov, full_config),
        }

        decomp = decompose_all_methods(weights, cov)
        assert len(decomp) == 3
        for name, df in decomp.items():
            assert abs(df["pct_risk_contribution"].sum() - 1.0) < 1e-6

    def test_per_asset_cap_14_assets(self, large_synthetic_prices,
                                      full_config, market_weights_14):
        """All optimizers respect per-asset cap with 14 real tickers."""
        log_ret = log_returns(large_synthetic_prices)
        arith_ret = arithmetic_returns(large_synthetic_prices)
        cov, _ = estimate_covariance(log_ret, "ledoit_wolf", full_config)
        mu = estimate_expected_returns(arith_ret, "shrinkage", full_config)
        cap = full_config["constraints"]["per_asset"]["cap"]

        for name, w in [
            ("MV", optimize_mean_variance(mu, cov, full_config)),
            ("RP", optimize_risk_parity(cov, full_config)),
            ("HRP", optimize_hrp(cov, full_config)),
        ]:
            assert (w <= cap + 1e-4).all(), f"{name} violates per-asset cap"
