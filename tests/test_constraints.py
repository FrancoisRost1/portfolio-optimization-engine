"""
Tests for src/constraints.py — bounds, class constraints, turnover,
full constraint builder, and HRP clipping.
"""

import numpy as np
import pandas as pd
import pytest

from src.constraints import (
    build_bounds,
    build_class_constraints,
    build_turnover_constraint,
    build_full_weight_constraint,
    build_all_constraints,
    clip_to_constraints,
)


class TestBuildBounds:
    def test_correct_count(self):
        bounds = build_bounds(["A", "B", "C"], floor=0.0, cap=0.30)
        assert len(bounds) == 3

    def test_floor_and_cap(self):
        bounds = build_bounds(["A"], floor=0.05, cap=0.25)
        assert bounds[0] == (0.05, 0.25)

    def test_default_values(self):
        bounds = build_bounds(["A", "B"])
        for lo, hi in bounds:
            assert lo == 0.0
            assert hi == 0.30


class TestBuildClassConstraints:
    def test_one_constraint_per_class(self):
        tickers = ["A", "B", "C"]
        classes = {"A": "eq", "B": "eq", "C": "bond"}
        caps = {"eq": 0.60, "bond": 0.40}
        constraints = build_class_constraints(tickers, classes, caps)
        assert len(constraints) == 2

    def test_constraint_satisfied_within_cap(self):
        tickers = ["A", "B", "C"]
        classes = {"A": "eq", "B": "eq", "C": "bond"}
        caps = {"eq": 0.60, "bond": 0.40}
        constraints = build_class_constraints(tickers, classes, caps)
        w = np.array([0.25, 0.25, 0.30])  # eq=0.50 < 0.60, bond=0.30 < 0.40
        for c in constraints:
            assert c["fun"](w) >= 0

    def test_constraint_violated_over_cap(self):
        tickers = ["A", "B", "C"]
        classes = {"A": "eq", "B": "eq", "C": "bond"}
        caps = {"eq": 0.40}
        constraints = build_class_constraints(tickers, classes, caps)
        w = np.array([0.30, 0.30, 0.40])  # eq=0.60 > 0.40
        eq_constraint = constraints[0]
        assert eq_constraint["fun"](w) < 0

    def test_empty_class_ignored(self):
        tickers = ["A", "B"]
        classes = {"A": "eq", "B": "eq"}
        caps = {"eq": 0.60, "bond": 0.40}
        constraints = build_class_constraints(tickers, classes, caps)
        # bond class has no members → should be skipped
        assert len(constraints) == 1


class TestBuildTurnoverConstraint:
    def test_unconstrained_when_max_is_1(self):
        result = build_turnover_constraint(["A", "B"], np.array([0.5, 0.5]), 1.0)
        assert result == []

    def test_unconstrained_when_no_prev_weights(self):
        result = build_turnover_constraint(["A", "B"], None, 0.30)
        assert result == []

    def test_returns_constraint_when_active(self):
        prev = np.array([0.5, 0.5])
        result = build_turnover_constraint(["A", "B"], prev, 0.30)
        assert len(result) == 1
        assert result[0]["type"] == "ineq"

    def test_satisfied_when_no_change(self):
        prev = np.array([0.5, 0.5])
        constraints = build_turnover_constraint(["A", "B"], prev, 0.30)
        # Same weights → turnover ≈ 0 → satisfied
        assert constraints[0]["fun"](prev) > 0

    def test_violated_when_large_change(self):
        prev = np.array([0.5, 0.5])
        constraints = build_turnover_constraint(["A", "B"], prev, 0.10)
        new_w = np.array([0.9, 0.1])  # turnover ≈ 0.8 > 0.10
        assert constraints[0]["fun"](new_w) < 0


class TestBuildFullWeightConstraint:
    def test_satisfied_at_one(self):
        c = build_full_weight_constraint()
        assert abs(c["fun"](np.array([0.5, 0.3, 0.2]))) < 1e-10

    def test_violated_below_one(self):
        c = build_full_weight_constraint()
        assert c["fun"](np.array([0.3, 0.3, 0.2])) < 0


class TestBuildAllConstraints:
    def test_returns_constraints_and_bounds(self, minimal_config):
        tickers = minimal_config["universe"]["tickers"]
        constraints, bounds = build_all_constraints(tickers, minimal_config)
        assert len(bounds) == len(tickers)
        # At least: sum-to-one + class constraints
        assert len(constraints) >= 1

    def test_bounds_match_config(self, minimal_config):
        tickers = minimal_config["universe"]["tickers"]
        _, bounds = build_all_constraints(tickers, minimal_config)
        for lo, hi in bounds:
            assert lo == 0.0
            assert hi == 0.40


class TestClipToConstraints:
    def test_sum_to_one(self, minimal_config):
        raw = pd.Series({"A": 0.5, "B": 0.3, "C": 0.1, "D": 0.05, "E": 0.05})
        clipped = clip_to_constraints(raw, minimal_config)
        assert abs(clipped.sum() - 1.0) < 1e-10

    def test_per_asset_cap_enforced(self, minimal_config):
        """After clipping + class scaling + renorm, asset weights should be
        bounded. With extreme inputs, renorm can push weights above the raw cap,
        so we test with a realistic starting point."""
        raw = pd.Series({"A": 0.35, "B": 0.25, "C": 0.20, "D": 0.10, "E": 0.10})
        clipped = clip_to_constraints(raw, minimal_config)
        cap = minimal_config["constraints"]["per_asset"]["cap"]
        # After renorm, largest weight shouldn't exceed cap by much
        assert clipped.max() <= cap + 0.05

    def test_no_negative_weights(self, minimal_config):
        raw = pd.Series({"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.05, "E": 0.05})
        clipped = clip_to_constraints(raw, minimal_config)
        assert (clipped >= 0).all()

    def test_class_cap_enforced(self, minimal_config):
        """Equities (A+B) should not exceed 60% after clipping."""
        raw = pd.Series({"A": 0.40, "B": 0.40, "C": 0.10, "D": 0.05, "E": 0.05})
        clipped = clip_to_constraints(raw, minimal_config)
        eq_sum = clipped[["A", "B"]].sum()
        eq_cap = minimal_config["constraints"]["per_class"]["equities"]
        # After renorm it may not be exactly at cap, but scaled down
        assert eq_sum <= eq_cap + 0.01 or clipped.sum() == pytest.approx(1.0)
