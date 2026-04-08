"""
Tests for src/covariance.py — sample, Ledoit-Wolf, RMT estimation,
dispatch, and correlation conversion.
"""

import numpy as np
import pandas as pd
import pytest

from src.covariance import (
    sample_covariance,
    ledoit_wolf_covariance,
    rmt_covariance,
    estimate_covariance,
    covariance_to_correlation,
)


# ── Sample covariance ──────────────────────────────────────────────

class TestSampleCovariance:
    def test_shape_matches_assets(self, synthetic_log_returns):
        cov = sample_covariance(synthetic_log_returns)
        n = len(synthetic_log_returns.columns)
        assert cov.shape == (n, n)

    def test_symmetric(self, synthetic_log_returns):
        cov = sample_covariance(synthetic_log_returns)
        np.testing.assert_array_almost_equal(cov.values, cov.values.T)

    def test_positive_diagonal(self, synthetic_log_returns):
        cov = sample_covariance(synthetic_log_returns)
        assert (np.diag(cov.values) > 0).all()

    def test_preserves_ticker_labels(self, synthetic_log_returns):
        cov = sample_covariance(synthetic_log_returns)
        assert list(cov.index) == list(synthetic_log_returns.columns)
        assert list(cov.columns) == list(synthetic_log_returns.columns)

    def test_matches_pandas_cov(self, synthetic_log_returns):
        cov = sample_covariance(synthetic_log_returns)
        expected = synthetic_log_returns.cov()
        pd.testing.assert_frame_equal(cov, expected)


# ── Ledoit-Wolf shrinkage ──────────────────────────────────────────

class TestLedoitWolf:
    def test_returns_tuple(self, synthetic_log_returns):
        result = ledoit_wolf_covariance(synthetic_log_returns)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_shape(self, synthetic_log_returns):
        cov, _ = ledoit_wolf_covariance(synthetic_log_returns)
        n = len(synthetic_log_returns.columns)
        assert cov.shape == (n, n)

    def test_shrinkage_between_0_and_1(self, synthetic_log_returns):
        _, alpha = ledoit_wolf_covariance(synthetic_log_returns)
        assert 0.0 <= alpha <= 1.0

    def test_symmetric(self, synthetic_log_returns):
        cov, _ = ledoit_wolf_covariance(synthetic_log_returns)
        np.testing.assert_array_almost_equal(cov.values, cov.values.T)

    def test_positive_semidefinite(self, synthetic_log_returns):
        """LW covariance must have non-negative eigenvalues."""
        cov, _ = ledoit_wolf_covariance(synthetic_log_returns)
        eigenvalues = np.linalg.eigvalsh(cov.values)
        assert (eigenvalues >= -1e-10).all()

    def test_differs_from_sample(self, synthetic_log_returns):
        """Shrinkage should modify the sample covariance."""
        sample = sample_covariance(synthetic_log_returns)
        lw, _ = ledoit_wolf_covariance(synthetic_log_returns)
        assert not np.allclose(sample.values, lw.values)


# ── RMT / Marchenko-Pastur ────────────────────────────────────────

class TestRMTCovariance:
    def test_shape(self, synthetic_log_returns, minimal_config):
        rmt_cfg = minimal_config["covariance"]["rmt"]
        cov, info = rmt_covariance(synthetic_log_returns, rmt_cfg)
        n = len(synthetic_log_returns.columns)
        assert cov.shape == (n, n)

    def test_symmetric(self, synthetic_log_returns, minimal_config):
        rmt_cfg = minimal_config["covariance"]["rmt"]
        cov, _ = rmt_covariance(synthetic_log_returns, rmt_cfg)
        np.testing.assert_array_almost_equal(cov.values, cov.values.T)

    def test_preserves_trace(self, synthetic_log_returns, minimal_config):
        """With preserve_trace=True, total variance should be conserved."""
        rmt_cfg = minimal_config["covariance"]["rmt"]
        cov_rmt, _ = rmt_covariance(synthetic_log_returns, rmt_cfg)
        sample = sample_covariance(synthetic_log_returns)
        np.testing.assert_almost_equal(
            np.trace(cov_rmt.values), np.trace(sample.values), decimal=6
        )

    def test_no_negative_eigenvalues(self, synthetic_log_returns, minimal_config):
        rmt_cfg = minimal_config["covariance"]["rmt"]
        cov, _ = rmt_covariance(synthetic_log_returns, rmt_cfg)
        eigenvalues = np.linalg.eigvalsh(cov.values)
        assert (eigenvalues >= 0).all()

    def test_info_dict_keys(self, synthetic_log_returns, minimal_config):
        rmt_cfg = minimal_config["covariance"]["rmt"]
        _, info = rmt_covariance(synthetic_log_returns, rmt_cfg)
        expected_keys = {"lambda_max", "lambda_min", "sigma2", "n_cleaned",
                         "n_total", "q_ratio", "eigenvalues_raw", "eigenvalues_clean"}
        assert expected_keys == set(info.keys())

    def test_mp_bounds_ordering(self, synthetic_log_returns, minimal_config):
        rmt_cfg = minimal_config["covariance"]["rmt"]
        _, info = rmt_covariance(synthetic_log_returns, rmt_cfg)
        assert info["lambda_min"] < info["lambda_max"]

    def test_n_cleaned_nonnegative(self, synthetic_log_returns, minimal_config):
        rmt_cfg = minimal_config["covariance"]["rmt"]
        _, info = rmt_covariance(synthetic_log_returns, rmt_cfg)
        assert info["n_cleaned"] >= 0
        assert info["n_cleaned"] <= info["n_total"]


# ── Dispatch ───────────────────────────────────────────────────────

class TestEstimateCovariance:
    def test_sample_dispatch(self, synthetic_log_returns, minimal_config):
        cov, meta = estimate_covariance(synthetic_log_returns, "sample", minimal_config)
        assert cov.shape[0] == cov.shape[1]
        assert meta == {}

    def test_lw_dispatch(self, synthetic_log_returns, minimal_config):
        cov, meta = estimate_covariance(synthetic_log_returns, "ledoit_wolf", minimal_config)
        assert "shrinkage_intensity" in meta

    def test_rmt_dispatch(self, synthetic_log_returns, minimal_config):
        cov, meta = estimate_covariance(synthetic_log_returns, "rmt", minimal_config)
        assert "lambda_max" in meta

    def test_invalid_method_raises(self, synthetic_log_returns, minimal_config):
        with pytest.raises(ValueError, match="Unknown covariance method"):
            estimate_covariance(synthetic_log_returns, "invalid", minimal_config)


# ── Correlation conversion ─────────────────────────────────────────

class TestCovarianceToCorrelation:
    def test_diagonal_is_one(self, synthetic_log_returns):
        cov = sample_covariance(synthetic_log_returns)
        corr = covariance_to_correlation(cov)
        np.testing.assert_array_almost_equal(np.diag(corr.values), 1.0)

    def test_values_in_range(self, synthetic_log_returns):
        cov = sample_covariance(synthetic_log_returns)
        corr = covariance_to_correlation(cov)
        assert (corr.values >= -1.0 - 1e-10).all()
        assert (corr.values <= 1.0 + 1e-10).all()

    def test_symmetric(self, synthetic_log_returns):
        cov = sample_covariance(synthetic_log_returns)
        corr = covariance_to_correlation(cov)
        np.testing.assert_array_almost_equal(corr.values, corr.values.T)
