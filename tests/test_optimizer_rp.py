"""
Tests for src/optimizer_rp.py — risk parity (equal risk contribution).
"""

import numpy as np
import pandas as pd
import pytest

from src.optimizer_rp import optimize_risk_parity, risk_contributions


class TestRiskContributions:
    def test_sum_to_one(self, synthetic_cov):
        n = len(synthetic_cov.index)
        w = np.ones(n) / n
        rc = risk_contributions(w, synthetic_cov.values * 252)
        assert abs(rc.sum() - 1.0) < 1e-6

    def test_nonnegative(self, synthetic_cov):
        n = len(synthetic_cov.index)
        w = np.ones(n) / n
        rc = risk_contributions(w, synthetic_cov.values * 252)
        assert (rc >= -1e-8).all()


class TestRiskParityOptimizer:
    def test_weights_sum_to_one(self, synthetic_cov, minimal_config):
        w = optimize_risk_parity(synthetic_cov, minimal_config)
        assert abs(w.sum() - 1.0) < 1e-6

    def test_long_only(self, synthetic_cov, minimal_config):
        w = optimize_risk_parity(synthetic_cov, minimal_config)
        assert (w >= -1e-8).all()

    def test_equal_risk_contributions(self, synthetic_cov, minimal_config):
        """Risk contributions should be approximately equal (1/N each)."""
        # Use relaxed config to avoid class caps interfering
        relaxed = minimal_config.copy()
        relaxed["constraints"] = {
            "long_only": True,
            "per_asset": {"floor": 0.0, "cap": 1.0},
            "per_class": {},
            "max_turnover": 1.0,
        }
        w = optimize_risk_parity(synthetic_cov, relaxed)
        rc = risk_contributions(w.values, synthetic_cov.values * 252)
        n = len(w)
        target = 1.0 / n
        # Allow 5% tolerance — RP is approximate with constraints
        assert np.abs(rc - target).max() < 0.05

    def test_output_type(self, synthetic_cov, minimal_config):
        w = optimize_risk_parity(synthetic_cov, minimal_config)
        assert isinstance(w, pd.Series)
        assert list(w.index) == list(synthetic_cov.index)

    def test_per_asset_cap(self, synthetic_cov, minimal_config):
        w = optimize_risk_parity(synthetic_cov, minimal_config)
        cap = minimal_config["constraints"]["per_asset"]["cap"]
        assert (w <= cap + 1e-6).all()

    def test_with_prev_weights(self, synthetic_cov, minimal_config):
        n = len(synthetic_cov.index)
        prev = np.ones(n) / n
        w = optimize_risk_parity(synthetic_cov, minimal_config, prev_weights=prev)
        assert abs(w.sum() - 1.0) < 1e-6

    def test_different_from_equal_weight(self, synthetic_cov, minimal_config):
        """RP should differ from naive equal weight unless cov is identity."""
        w = optimize_risk_parity(synthetic_cov, minimal_config)
        ew = pd.Series(1.0 / len(w), index=w.index)
        assert not np.allclose(w.values, ew.values, atol=0.01)
