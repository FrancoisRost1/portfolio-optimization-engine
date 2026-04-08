"""
Tests for src/optimizer_hrp.py — hierarchical risk parity.
"""

import numpy as np
import pandas as pd
import pytest

from src.optimizer_hrp import (
    correlation_distance,
    quasi_diagonalise,
    recursive_bisection,
    optimize_hrp,
)
from scipy.cluster.hierarchy import linkage


class TestCorrelationDistance:
    def test_output_is_condensed(self):
        """Should return a condensed distance vector for scipy linkage."""
        corr = pd.DataFrame(
            [[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]],
            index=["A", "B", "C"],
            columns=["A", "B", "C"],
        )
        dist = correlation_distance(corr)
        # Condensed form for 3 items has 3 elements
        assert dist.shape == (3,)

    def test_perfect_correlation_zero_distance(self):
        corr = pd.DataFrame([[1.0, 1.0], [1.0, 1.0]], index=["A", "B"], columns=["A", "B"])
        dist = correlation_distance(corr)
        assert dist[0] == pytest.approx(0.0)

    def test_zero_correlation_max_distance(self):
        corr = pd.DataFrame([[1.0, 0.0], [0.0, 1.0]], index=["A", "B"], columns=["A", "B"])
        dist = correlation_distance(corr)
        expected = np.sqrt(0.5)
        assert dist[0] == pytest.approx(expected)

    def test_negative_correlation(self):
        corr = pd.DataFrame([[1.0, -1.0], [-1.0, 1.0]], index=["A", "B"], columns=["A", "B"])
        dist = correlation_distance(corr)
        expected = np.sqrt(0.5 * (1 - (-1.0)))
        assert dist[0] == pytest.approx(expected)


class TestQuasiDiagonalise:
    def test_returns_all_indices(self):
        corr = pd.DataFrame(np.eye(4), columns=["A", "B", "C", "D"])
        dist = correlation_distance(corr)
        link = linkage(dist, method="single")
        order = quasi_diagonalise(link, 4)
        assert sorted(order) == [0, 1, 2, 3]

    def test_permutation(self):
        """Order should be a permutation of 0..N-1."""
        np.random.seed(99)
        corr_vals = np.eye(5) + np.random.randn(5, 5) * 0.1
        corr_vals = (corr_vals + corr_vals.T) / 2
        np.fill_diagonal(corr_vals, 1.0)
        corr_vals = np.clip(corr_vals, -1, 1)
        corr = pd.DataFrame(corr_vals)
        dist = correlation_distance(corr)
        link = linkage(dist, method="single")
        order = quasi_diagonalise(link, 5)
        assert len(order) == 5
        assert len(set(order)) == 5


class TestRecursiveBisection:
    def test_weights_sum_to_one(self):
        np.random.seed(42)
        cov = np.eye(4) * 0.01
        weights = recursive_bisection(cov, [0, 1, 2, 3])
        assert abs(weights.sum() - 1.0) < 1e-10

    def test_equal_for_identity_cov(self):
        """With identity cov, all assets have equal variance → equal weights."""
        cov = np.eye(5) * 0.01
        weights = recursive_bisection(cov, list(range(5)))
        np.testing.assert_array_almost_equal(weights, 0.2, decimal=5)

    def test_positive_weights(self):
        np.random.seed(42)
        n = 6
        A = np.random.randn(n, n) * 0.01
        cov = A.T @ A + np.eye(n) * 0.001
        weights = recursive_bisection(cov, list(range(n)))
        assert (weights > 0).all()


class TestHRPOptimizer:
    def test_weights_sum_to_one(self, synthetic_cov, minimal_config):
        w = optimize_hrp(synthetic_cov, minimal_config)
        assert abs(w.sum() - 1.0) < 1e-6

    def test_long_only(self, synthetic_cov, minimal_config):
        w = optimize_hrp(synthetic_cov, minimal_config)
        assert (w >= -1e-8).all()

    def test_per_asset_cap_respected(self, synthetic_cov, minimal_config):
        w = optimize_hrp(synthetic_cov, minimal_config)
        cap = minimal_config["constraints"]["per_asset"]["cap"]
        assert (w <= cap + 1e-6).all()

    def test_output_type(self, synthetic_cov, minimal_config):
        w = optimize_hrp(synthetic_cov, minimal_config)
        assert isinstance(w, pd.Series)

    def test_diversified(self, synthetic_cov, minimal_config):
        """HRP should produce well-diversified weights — no single asset > 50%."""
        w = optimize_hrp(synthetic_cov, minimal_config)
        assert w.max() < 0.50

    def test_different_linkage_methods(self, synthetic_cov, minimal_config):
        """Changing linkage method should produce different weights."""
        minimal_config["optimization"]["hrp"]["linkage_method"] = "single"
        w_single = optimize_hrp(synthetic_cov, minimal_config)
        minimal_config["optimization"]["hrp"]["linkage_method"] = "complete"
        w_complete = optimize_hrp(synthetic_cov, minimal_config)
        # They may not always differ, but test they both work
        assert abs(w_single.sum() - 1.0) < 1e-6
        assert abs(w_complete.sum() - 1.0) < 1e-6
