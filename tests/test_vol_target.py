"""
Tests for src/vol_target.py — post-optimization volatility targeting overlay.
"""

import numpy as np
import pandas as pd
import pytest

from src.vol_target import apply_vol_target


class TestApplyVolTarget:
    def test_disabled_returns_unchanged(self, synthetic_cov, minimal_config):
        tickers = synthetic_cov.index
        w = pd.Series(np.ones(len(tickers)) / len(tickers), index=tickers)
        minimal_config["vol_targeting"]["enabled"] = False
        w_out, info = apply_vol_target(w, synthetic_cov, minimal_config)
        pd.testing.assert_series_equal(w_out, w)
        assert info["enabled"] is False

    def test_enabled_returns_scaled_weights(self, synthetic_cov, minimal_config):
        tickers = synthetic_cov.index
        w = pd.Series(np.ones(len(tickers)) / len(tickers), index=tickers)
        minimal_config["vol_targeting"]["enabled"] = True
        minimal_config["vol_targeting"]["target_annual_vol"] = 0.05
        w_out, info = apply_vol_target(w, synthetic_cov, minimal_config)
        assert info["enabled"] is True
        assert info["scale_factor"] <= 1.0

    def test_scale_capped_at_max(self, synthetic_cov, minimal_config):
        """Scale factor should never exceed max_scale (no leverage)."""
        tickers = synthetic_cov.index
        w = pd.Series(np.ones(len(tickers)) / len(tickers), index=tickers)
        minimal_config["vol_targeting"]["enabled"] = True
        minimal_config["vol_targeting"]["target_annual_vol"] = 1.0  # Very high target
        minimal_config["vol_targeting"]["max_scale"] = 1.0
        _, info = apply_vol_target(w, synthetic_cov, minimal_config)
        assert info["scale_factor"] <= 1.0

    def test_lower_target_reduces_weights(self, synthetic_cov, minimal_config):
        tickers = synthetic_cov.index
        w = pd.Series(np.ones(len(tickers)) / len(tickers), index=tickers)
        minimal_config["vol_targeting"]["enabled"] = True
        minimal_config["vol_targeting"]["target_annual_vol"] = 0.01  # Very low
        w_out, info = apply_vol_target(w, synthetic_cov, minimal_config)
        assert w_out.sum() < w.sum()

    def test_info_dict_keys(self, synthetic_cov, minimal_config):
        tickers = synthetic_cov.index
        w = pd.Series(np.ones(len(tickers)) / len(tickers), index=tickers)
        minimal_config["vol_targeting"]["enabled"] = True
        _, info = apply_vol_target(w, synthetic_cov, minimal_config)
        assert "target_vol" in info
        assert "portfolio_vol_before" in info
        assert "portfolio_vol_after" in info
        assert "scale_factor" in info
        assert "cash_weight" in info
