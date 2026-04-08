"""
Tests for src/optimizer_bl.py — Black-Litterman pipeline:
views, uncertainty, posterior, and full optimization.
"""

import numpy as np
import pandas as pd
import pytest

from src.optimizer_bl import (
    build_views,
    compute_view_uncertainty,
    bl_posterior,
    optimize_black_litterman,
)
from src.expected_returns import bl_implied_returns


class TestBuildViews:
    def test_correct_dimensions(self):
        tickers = ["A", "B", "C"]
        views = {"A": 0.08, "C": -0.02}
        conf = {"A": 0.7, "C": 0.5}
        P, Q, c = build_views(tickers, views, conf)
        assert P.shape == (2, 3)
        assert Q.shape == (2,)
        assert c.shape == (2,)

    def test_pick_matrix_entries(self):
        tickers = ["A", "B", "C"]
        views = {"B": 0.10}
        P, Q, _ = build_views(tickers, views, {})
        assert P[0, 1] == 1.0  # B is at index 1
        assert P[0, 0] == 0.0
        assert P[0, 2] == 0.0
        assert Q[0] == 0.10

    def test_empty_views(self):
        P, Q, c = build_views(["A", "B"], {}, {})
        assert P.shape[0] == 0
        assert Q.shape[0] == 0

    def test_unknown_ticker_filtered(self):
        tickers = ["A", "B"]
        views = {"A": 0.05, "Z": 0.10}  # Z not in universe
        P, Q, c = build_views(tickers, views, {})
        assert P.shape[0] == 1  # Only A valid

    def test_default_confidence(self):
        tickers = ["A", "B"]
        views = {"A": 0.08}
        P, Q, c = build_views(tickers, views, {}, default_confidence=0.60)
        assert c[0] == 0.60


class TestComputeViewUncertainty:
    def test_diagonal_matrix(self):
        P = np.array([[1, 0, 0], [0, 0, 1]])
        cov = np.eye(3) * 0.01
        conf = np.array([0.5, 0.5])
        omega = compute_view_uncertainty(P, cov, tau=0.05, confidences=conf)
        assert omega.shape == (2, 2)
        # Off-diagonal should be zero (diagonal matrix)
        assert omega[0, 1] == 0.0
        assert omega[1, 0] == 0.0

    def test_higher_confidence_lower_uncertainty(self):
        P = np.array([[1, 0]])
        cov = np.eye(2) * 0.01
        omega_low = compute_view_uncertainty(P, cov, 0.05, np.array([0.3]))
        omega_high = compute_view_uncertainty(P, cov, 0.05, np.array([0.9]))
        assert omega_low[0, 0] > omega_high[0, 0]


class TestBLPosterior:
    def test_no_views_returns_prior(self, synthetic_cov, synthetic_market_weights):
        cov_annual = synthetic_cov * 252
        prior = bl_implied_returns(synthetic_cov, synthetic_market_weights)
        P = np.empty((0, len(prior)))
        Q = np.empty(0)
        omega = np.empty((0, 0))
        post_mu, post_cov = bl_posterior(prior, cov_annual, P, Q, omega, tau=0.05)
        pd.testing.assert_series_equal(post_mu, prior)

    def test_posterior_shifts_toward_views(self, synthetic_cov, synthetic_market_weights):
        """A bullish view on asset A should increase A's posterior return."""
        tickers = synthetic_cov.index.tolist()
        cov_annual = synthetic_cov * 252
        prior = bl_implied_returns(synthetic_cov, synthetic_market_weights)

        P = np.zeros((1, len(tickers)))
        P[0, 0] = 1.0
        Q = np.array([prior["A"] + 0.10])  # Very bullish
        conf = np.array([0.9])
        omega = compute_view_uncertainty(P, cov_annual.values, 0.05, conf)

        post_mu, _ = bl_posterior(prior, cov_annual, P, Q, omega, tau=0.05)
        assert post_mu["A"] > prior["A"]

    def test_posterior_cov_shape(self, synthetic_cov, synthetic_market_weights):
        cov_annual = synthetic_cov * 252
        prior = bl_implied_returns(synthetic_cov, synthetic_market_weights)
        P = np.empty((0, len(prior)))
        Q = np.empty(0)
        omega = np.empty((0, 0))
        _, post_cov = bl_posterior(prior, cov_annual, P, Q, omega, tau=0.05)
        assert post_cov.shape == cov_annual.shape


class TestOptimizeBlackLitterman:
    def test_weights_sum_to_one(self, synthetic_cov, synthetic_market_weights, minimal_config):
        w, info = optimize_black_litterman(synthetic_cov, synthetic_market_weights, minimal_config)
        assert abs(w.sum() - 1.0) < 1e-6

    def test_long_only(self, synthetic_cov, synthetic_market_weights, minimal_config):
        w, _ = optimize_black_litterman(synthetic_cov, synthetic_market_weights, minimal_config)
        assert (w >= -1e-8).all()

    def test_info_contains_prior_and_posterior(self, synthetic_cov,
                                               synthetic_market_weights, minimal_config):
        _, info = optimize_black_litterman(synthetic_cov, synthetic_market_weights, minimal_config)
        assert "prior_returns" in info
        assert "posterior_returns" in info
        assert "views" in info

    def test_no_views_posterior_equals_prior(self, synthetic_cov,
                                             synthetic_market_weights, minimal_config):
        """With empty views, posterior should equal prior."""
        _, info = optimize_black_litterman(
            synthetic_cov, synthetic_market_weights, minimal_config,
            views={}, view_confidence={},
        )
        pd.testing.assert_series_equal(
            info["posterior_returns"], info["prior_returns"],
            check_names=False,
        )

    def test_custom_views_override_config(self, synthetic_cov,
                                          synthetic_market_weights, minimal_config):
        """Passing explicit views should override config defaults."""
        custom_views = {"B": 0.15}
        w, info = optimize_black_litterman(
            synthetic_cov, synthetic_market_weights, minimal_config,
            views=custom_views,
        )
        assert info["views"] == custom_views
        assert abs(w.sum() - 1.0) < 1e-6
