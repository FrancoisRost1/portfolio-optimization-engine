"""
Tests for src/risk_decomposition.py — risk contributions, Herfindahl,
effective N, portfolio volatility, and multi-method decomposition.
"""

import numpy as np
import pandas as pd
import pytest

from src.risk_decomposition import (
    risk_contribution,
    herfindahl_index,
    effective_n,
    portfolio_volatility,
    decompose_all_methods,
)


class TestRiskContribution:
    def test_pct_rc_sums_to_one(self, synthetic_cov):
        tickers = synthetic_cov.index
        n = len(tickers)
        w = pd.Series(np.ones(n) / n, index=tickers)
        rc = risk_contribution(w, synthetic_cov)
        assert abs(rc["pct_risk_contribution"].sum() - 1.0) < 1e-6

    def test_output_columns(self, synthetic_cov):
        tickers = synthetic_cov.index
        w = pd.Series(np.ones(len(tickers)) / len(tickers), index=tickers)
        rc = risk_contribution(w, synthetic_cov)
        expected_cols = {"weight", "marginal_risk", "risk_contribution", "pct_risk_contribution"}
        assert set(rc.columns) == expected_cols

    def test_zero_weight_zero_contribution(self, synthetic_cov):
        tickers = synthetic_cov.index.tolist()
        w = pd.Series([0.5, 0.5, 0.0, 0.0, 0.0], index=tickers)
        rc = risk_contribution(w, synthetic_cov)
        # Zero-weight assets should have zero absolute risk contribution
        assert rc.loc[tickers[2], "risk_contribution"] == pytest.approx(0.0, abs=1e-10)

    def test_nonnegative_for_long_only(self, synthetic_cov):
        tickers = synthetic_cov.index
        n = len(tickers)
        w = pd.Series(np.ones(n) / n, index=tickers)
        rc = risk_contribution(w, synthetic_cov)
        assert (rc["pct_risk_contribution"] >= -1e-8).all()


class TestHerfindahlIndex:
    def test_equal_weight(self):
        w = pd.Series([0.2] * 5)
        h = herfindahl_index(w)
        assert h == pytest.approx(0.2, abs=1e-10)

    def test_concentrated(self):
        w = pd.Series([1.0, 0.0, 0.0])
        h = herfindahl_index(w)
        assert h == pytest.approx(1.0)

    def test_higher_concentration_higher_index(self):
        w_diverse = pd.Series([0.25, 0.25, 0.25, 0.25])
        w_concentrated = pd.Series([0.70, 0.10, 0.10, 0.10])
        assert herfindahl_index(w_concentrated) > herfindahl_index(w_diverse)


class TestEffectiveN:
    def test_equal_weight(self):
        w = pd.Series([0.2] * 5)
        assert effective_n(w) == pytest.approx(5.0, abs=1e-10)

    def test_concentrated(self):
        w = pd.Series([1.0, 0.0, 0.0])
        assert effective_n(w) == pytest.approx(1.0)


class TestPortfolioVolatility:
    def test_positive(self, synthetic_cov):
        tickers = synthetic_cov.index
        w = pd.Series(np.ones(len(tickers)) / len(tickers), index=tickers)
        vol = portfolio_volatility(w, synthetic_cov)
        assert vol > 0

    def test_single_asset(self, synthetic_cov):
        """Vol of single asset = its own vol × √252."""
        tickers = synthetic_cov.index.tolist()
        w = pd.Series([1.0] + [0.0] * (len(tickers) - 1), index=tickers)
        vol = portfolio_volatility(w, synthetic_cov)
        expected = np.sqrt(synthetic_cov.values[0, 0] * 252)
        assert vol == pytest.approx(expected, rel=1e-4)


class TestDecomposeAllMethods:
    def test_returns_dict(self, synthetic_cov):
        tickers = synthetic_cov.index
        n = len(tickers)
        weights = {
            "A": pd.Series(np.ones(n) / n, index=tickers),
            "B": pd.Series(np.ones(n) / n, index=tickers),
        }
        result = decompose_all_methods(weights, synthetic_cov)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"A", "B"}
        for df in result.values():
            assert isinstance(df, pd.DataFrame)
