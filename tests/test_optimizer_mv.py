"""
Tests for src/optimizer_mv.py — mean-variance optimization and
efficient frontier point solver.
"""

import numpy as np
import pandas as pd
import pytest

from src.optimizer_mv import optimize_mean_variance, optimize_for_target_return
from src.expected_returns import shrinkage_returns


class TestMeanVarianceOptimizer:
    def test_weights_sum_to_one(self, synthetic_returns, synthetic_cov, minimal_config):
        mu = shrinkage_returns(synthetic_returns)
        w = optimize_mean_variance(mu, synthetic_cov, minimal_config)
        assert abs(w.sum() - 1.0) < 1e-6

    def test_long_only(self, synthetic_returns, synthetic_cov, minimal_config):
        mu = shrinkage_returns(synthetic_returns)
        w = optimize_mean_variance(mu, synthetic_cov, minimal_config)
        assert (w >= -1e-8).all()

    def test_per_asset_cap_respected(self, synthetic_returns, synthetic_cov, minimal_config):
        mu = shrinkage_returns(synthetic_returns)
        w = optimize_mean_variance(mu, synthetic_cov, minimal_config)
        cap = minimal_config["constraints"]["per_asset"]["cap"]
        assert (w <= cap + 1e-6).all()

    def test_output_is_series(self, synthetic_returns, synthetic_cov, minimal_config):
        mu = shrinkage_returns(synthetic_returns)
        w = optimize_mean_variance(mu, synthetic_cov, minimal_config)
        assert isinstance(w, pd.Series)
        assert list(w.index) == list(synthetic_cov.index)

    def test_higher_risk_aversion_lowers_risk(self, synthetic_returns, synthetic_cov, minimal_config):
        """Higher γ should produce lower portfolio variance."""
        mu = shrinkage_returns(synthetic_returns)
        sigma = synthetic_cov.values * 252

        minimal_config["optimization"]["risk_aversion"] = 1.0
        w_low = optimize_mean_variance(mu, synthetic_cov, minimal_config)
        var_low = w_low.values @ sigma @ w_low.values

        minimal_config["optimization"]["risk_aversion"] = 10.0
        w_high = optimize_mean_variance(mu, synthetic_cov, minimal_config)
        var_high = w_high.values @ sigma @ w_high.values

        assert var_high <= var_low + 1e-6

    def test_with_prev_weights(self, synthetic_returns, synthetic_cov, minimal_config):
        """Should work with previous weights provided."""
        mu = shrinkage_returns(synthetic_returns)
        prev = np.ones(len(synthetic_cov.index)) / len(synthetic_cov.index)
        w = optimize_mean_variance(mu, synthetic_cov, minimal_config, prev_weights=prev)
        assert abs(w.sum() - 1.0) < 1e-6


class TestOptimizeForTargetReturn:
    def test_achieves_target_return(self, synthetic_returns, synthetic_cov, minimal_config):
        mu = shrinkage_returns(synthetic_returns)
        target = mu.median()
        w, vol = optimize_for_target_return(target, mu, synthetic_cov, minimal_config)
        achieved = float(w.values @ mu.reindex(w.index).values)
        assert achieved >= target - 1e-4

    def test_weights_valid(self, synthetic_returns, synthetic_cov, minimal_config):
        mu = shrinkage_returns(synthetic_returns)
        target = mu.median()
        w, vol = optimize_for_target_return(target, mu, synthetic_cov, minimal_config)
        assert abs(w.sum() - 1.0) < 1e-6
        assert (w >= -1e-8).all()

    def test_returns_volatility(self, synthetic_returns, synthetic_cov, minimal_config):
        mu = shrinkage_returns(synthetic_returns)
        target = mu.median()
        _, vol = optimize_for_target_return(target, mu, synthetic_cov, minimal_config)
        assert vol > 0
