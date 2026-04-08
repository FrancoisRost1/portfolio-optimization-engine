from __future__ import annotations

"""
Constraint builder — constructs scipy-compatible constraints and bounds
for portfolio optimization.

Handles per-asset bounds, per-class caps, and optional turnover limits.
All thresholds come from config.yaml.
"""

import numpy as np
import pandas as pd
from scipy.optimize import LinearConstraint


def build_bounds(
    tickers: list[str],
    floor: float = 0.0,
    cap: float = 0.30,
) -> list[tuple[float, float]]:
    """Build per-asset (lower, upper) bounds for scipy.optimize.

    Parameters
    ----------
    tickers : list[str]
        Asset tickers (determines number of variables).
    floor : float
        Minimum weight per asset (default 0, long-only).
    cap : float
        Maximum weight per asset (default 0.30).

    Returns
    -------
    list[tuple[float, float]]
        One (lower, upper) tuple per asset.
    """
    return [(floor, cap) for _ in tickers]


def build_class_constraints(
    tickers: list[str],
    asset_classes: dict[str, str],
    class_caps: dict[str, float],
) -> list[dict]:
    """Build per-asset-class inequality constraints.

    Each constraint enforces: sum of weights in class k <= cap_k.

    Parameters
    ----------
    tickers : list[str]
        Ordered list of asset tickers.
    asset_classes : dict[str, str]
        Mapping of ticker -> asset class name.
    class_caps : dict[str, float]
        Mapping of class name -> maximum aggregate weight.

    Returns
    -------
    list[dict]
        Scipy-compatible inequality constraint dicts.
        Each uses convention: constraint(w) >= 0.
    """
    constraints = []
    for cls, cap in class_caps.items():
        # Indices of tickers belonging to this class
        mask = np.array([1.0 if asset_classes.get(t) == cls else 0.0 for t in tickers])
        if mask.sum() == 0:
            continue
        # cap - sum(w_i for i in class) >= 0
        constraints.append({
            "type": "ineq",
            "fun": lambda w, m=mask, c=cap: c - np.dot(m, w),
        })
    return constraints


def build_turnover_constraint(
    tickers: list[str],
    prev_weights: np.ndarray | None,
    max_turnover: float,
) -> list[dict]:
    """Build a turnover constraint: sum(|w_new - w_old|) <= max_turnover.

    Turnover is linearised as two sets of auxiliary variables for scipy.
    Since SLSQP doesn't handle absolute values natively, we approximate
    by penalising the L1 distance in the objective instead when needed.

    For simplicity, we skip this constraint when max_turnover >= 1.0
    (effectively unconstrained) or when prev_weights is None (first period).

    Parameters
    ----------
    tickers : list[str]
        Asset tickers.
    prev_weights : np.ndarray or None
        Previous period weights. None on first rebalance.
    max_turnover : float
        Maximum allowed turnover (1.0 = unconstrained).

    Returns
    -------
    list[dict]
        Empty list if unconstrained; otherwise a single inequality constraint.
    """
    if max_turnover >= 1.0 or prev_weights is None:
        return []

    # Approximate: sum(|w - w_old|) <= max_turnover
    # This is a non-smooth constraint; we use a differentiable approximation
    # sqrt((w_i - w_old_i)^2 + eps) ≈ |w_i - w_old_i|
    eps = 1e-8

    def turnover_ineq(w, w_old=prev_weights, mt=max_turnover):
        diff = w - w_old
        approx_abs = np.sqrt(diff**2 + eps)
        return mt - approx_abs.sum()

    return [{"type": "ineq", "fun": turnover_ineq}]


def build_full_weight_constraint() -> dict:
    """Equality constraint: weights must sum to 1.

    Returns
    -------
    dict
        Scipy-compatible equality constraint.
    """
    return {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}


def build_all_constraints(
    tickers: list[str],
    config: dict,
    prev_weights: np.ndarray | None = None,
) -> tuple[list[dict], list[tuple[float, float]]]:
    """Build the complete set of constraints and bounds from config.

    Parameters
    ----------
    tickers : list[str]
        Asset tickers.
    config : dict
        Full config dict.
    prev_weights : np.ndarray, optional
        Previous weights for turnover constraint.

    Returns
    -------
    tuple[list[dict], list[tuple[float, float]]]
        (list of constraint dicts, list of per-asset bounds).
    """
    c = config.get("constraints", {})

    # Bounds
    floor = c.get("per_asset", {}).get("floor", 0.0)
    cap = c.get("per_asset", {}).get("cap", 0.30)
    bounds = build_bounds(tickers, floor=floor, cap=cap)

    # Constraints list
    constraints = [build_full_weight_constraint()]

    # Class caps
    asset_classes = config.get("universe", {}).get("asset_classes", {})
    class_caps = c.get("per_class", {})
    if asset_classes and class_caps:
        constraints.extend(build_class_constraints(tickers, asset_classes, class_caps))

    # Turnover
    max_turnover = c.get("max_turnover", 1.0)
    constraints.extend(build_turnover_constraint(tickers, prev_weights, max_turnover))

    return constraints, bounds


def clip_to_constraints(
    weights: pd.Series, config: dict
) -> pd.Series:
    """Post-hoc clipping for non-optimizer methods (HRP).

    Clips per-asset weights to [floor, cap], then scales down asset classes
    that exceed their class cap, then renormalises to sum to 1.

    Parameters
    ----------
    weights : pd.Series
        Raw weights from HRP (index = tickers).
    config : dict
        Full config dict.

    Returns
    -------
    pd.Series
        Clipped and renormalised weights.
    """
    c = config.get("constraints", {})
    floor = c.get("per_asset", {}).get("floor", 0.0)
    cap = c.get("per_asset", {}).get("cap", 0.30)
    asset_classes = config.get("universe", {}).get("asset_classes", {})
    class_caps = c.get("per_class", {})

    w = weights.copy()

    # Iterate clip → scale → renorm until stable (max 20 iterations)
    # Single-pass clip+renorm can re-violate caps; iteration converges.
    for _ in range(20):
        w_prev = w.copy()

        # Per-asset clipping
        w = w.clip(lower=floor, upper=cap)

        # Per-class cap enforcement
        for cls, cls_cap in class_caps.items():
            members = [t for t in w.index if asset_classes.get(t) == cls]
            if not members:
                continue
            cls_sum = w[members].sum()
            if cls_sum > cls_cap and cls_sum > 0:
                w[members] *= cls_cap / cls_sum

        # Renormalise to sum to 1
        total = w.sum()
        if total > 0:
            w = w / total

        # Check convergence
        if (w - w_prev).abs().max() < 1e-10:
            break

    return w
