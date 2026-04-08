"""
Tests for src/expected_returns.py — shrinkage returns, BL implied returns,
and the dispatch function.
"""

import numpy as np
import pandas as pd
import pytest

from src.expected_returns import (
    shrinkage_returns,
    bl_implied_returns,
    estimate_expected_returns,
)


class TestShrinkageReturns:
    def test_output_length(self, synthetic_returns):
        mu = shrinkage_returns(synthetic_returns, intensity=0.5)
        assert len(mu) == len(synthetic_returns.columns)

    def test_intensity_zero_equals_historical(self, synthetic_returns):
        """δ=0 → pure historical arithmetic annualised means (daily_mean × 252)."""
        mu = shrinkage_returns(synthetic_returns, intensity=0.0)
        daily = synthetic_returns.mean()
        expected = daily * 252  # Arithmetic annualisation, not CAGR
        pd.testing.assert_series_equal(mu, expected, check_names=False)

    def test_intensity_one_equals_grand_mean(self, synthetic_returns):
        """δ=1 → all assets get the cross-sectional grand mean."""
        mu = shrinkage_returns(synthetic_returns, intensity=1.0)
        assert mu.std() < 1e-12  # all values identical

    def test_shrinkage_reduces_spread(self, synthetic_returns):
        """Shrinkage should reduce the spread between asset return estimates."""
        mu_raw = shrinkage_returns(synthetic_returns, intensity=0.0)
        mu_shrunk = shrinkage_returns(synthetic_returns, intensity=0.5)
        assert mu_shrunk.std() < mu_raw.std()

    def test_midpoint_shrinkage(self, synthetic_returns):
        """δ=0.5 should be average of historical and grand mean."""
        mu_0 = shrinkage_returns(synthetic_returns, intensity=0.0)
        mu_1 = shrinkage_returns(synthetic_returns, intensity=1.0)
        mu_half = shrinkage_returns(synthetic_returns, intensity=0.5)
        expected = 0.5 * mu_1 + 0.5 * mu_0
        pd.testing.assert_series_equal(mu_half, expected, check_names=False)


class TestBLImpliedReturns:
    def test_output_length(self, synthetic_cov, synthetic_market_weights):
        pi = bl_implied_returns(synthetic_cov, synthetic_market_weights)
        assert len(pi) == len(synthetic_cov.index)

    def test_higher_weight_higher_return(self, synthetic_cov, synthetic_market_weights):
        """Assets with higher market-cap weight should generally have higher
        implied returns (all else equal in a diagonal-dominant cov)."""
        pi = bl_implied_returns(synthetic_cov, synthetic_market_weights, risk_aversion=2.5)
        # At minimum, the highest-weight asset shouldn't have the lowest return
        max_weight_ticker = synthetic_market_weights.idxmax()
        assert pi[max_weight_ticker] > pi.min()

    def test_risk_aversion_scales_returns(self, synthetic_cov, synthetic_market_weights):
        """Doubling risk aversion should double implied returns (π = λΣw)."""
        pi_low = bl_implied_returns(synthetic_cov, synthetic_market_weights, risk_aversion=1.0)
        pi_high = bl_implied_returns(synthetic_cov, synthetic_market_weights, risk_aversion=2.0)
        np.testing.assert_array_almost_equal(pi_high.values, 2 * pi_low.values)

    def test_zero_risk_aversion_returns_zero(self, synthetic_cov, synthetic_market_weights):
        pi = bl_implied_returns(synthetic_cov, synthetic_market_weights, risk_aversion=0.0)
        np.testing.assert_array_almost_equal(pi.values, 0.0)


class TestEstimateExpectedReturns:
    def test_shrinkage_dispatch(self, synthetic_returns, minimal_config):
        mu = estimate_expected_returns(synthetic_returns, "shrinkage", minimal_config)
        assert len(mu) == len(synthetic_returns.columns)

    def test_bl_dispatch(self, synthetic_returns, synthetic_cov,
                         synthetic_market_weights, minimal_config):
        mu = estimate_expected_returns(
            synthetic_returns, "bl_implied", minimal_config,
            cov=synthetic_cov, market_weights=synthetic_market_weights,
        )
        assert len(mu) == len(synthetic_returns.columns)

    def test_bl_without_cov_raises(self, synthetic_returns, minimal_config):
        with pytest.raises(ValueError, match="requires cov"):
            estimate_expected_returns(synthetic_returns, "bl_implied", minimal_config)

    def test_invalid_method_raises(self, synthetic_returns, minimal_config):
        with pytest.raises(ValueError, match="Unknown method"):
            estimate_expected_returns(synthetic_returns, "garbage", minimal_config)
