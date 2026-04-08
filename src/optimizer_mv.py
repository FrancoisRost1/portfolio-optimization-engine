from __future__ import annotations

"""
Mean-Variance (Markowitz) optimizer.

Maximises: w'μ - (γ/2) × w'Σw
subject to full-investment, long-only, per-asset and per-class constraints.

CRITICAL: Never uses raw historical mean returns. Expected returns must come
from shrinkage_returns() or bl_implied_returns().
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.constraints import build_all_constraints


def optimize_mean_variance(
    expected_returns: pd.Series,
    cov: pd.DataFrame,
    config: dict,
    prev_weights: np.ndarray | None = None,
) -> pd.Series:
    """Run mean-variance optimization with constraints.

    Objective: max w'μ - (γ/2) × w'Σw  (equivalent to min negative)

    Parameters
    ----------
    expected_returns : pd.Series
        Annualized expected returns per asset.
    cov : pd.DataFrame
        N x N covariance matrix (daily frequency — annualised internally).
    config : dict
        Full config dict.
    prev_weights : np.ndarray, optional
        Previous weights for turnover constraint.

    Returns
    -------
    pd.Series
        Optimal weights (index = tickers).
    """
    tickers = cov.index.tolist()
    n = len(tickers)
    mu = expected_returns.reindex(tickers).values
    sigma = cov.values * 252  # Annualise daily covariance

    gamma = config.get("optimization", {}).get("risk_aversion", 2.5)
    constraints, bounds = build_all_constraints(tickers, config, prev_weights)

    # Objective: minimise -(w'μ) + (γ/2) × w'Σw
    def objective(w):
        ret = w @ mu
        risk = w @ sigma @ w
        return -ret + (gamma / 2) * risk

    def gradient(w):
        return -mu + gamma * sigma @ w

    # Initial guess: equal weight (feasible for most constraint sets)
    w0 = np.ones(n) / n

    result = minimize(
        objective,
        w0,
        method="SLSQP",
        jac=gradient,
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    if not result.success:
        import warnings
        warnings.warn(f"MV optimizer did not converge: {result.message}")

    weights = pd.Series(result.x, index=tickers, name="mv_weights")
    # Clean up tiny negative values from numerical noise
    weights = weights.clip(lower=0.0)
    total = weights.sum()
    if total > 0:
        weights = weights / total
    else:
        # Fallback to equal weight if optimizer produced all zeros
        weights = pd.Series(1.0 / n, index=tickers, name="mv_weights")

    return weights


def optimize_for_target_return(
    target_return: float,
    expected_returns: pd.Series,
    cov: pd.DataFrame,
    config: dict,
) -> tuple[pd.Series, float]:
    """Find the minimum-variance portfolio achieving a target return.

    Used to trace the efficient frontier. Minimises w'Σw subject to
    w'μ >= target_return and all standard constraints.

    Parameters
    ----------
    target_return : float
        Target annualized return.
    expected_returns : pd.Series
        Annualized expected returns per asset.
    cov : pd.DataFrame
        N x N covariance matrix (daily — annualised internally).
    config : dict
        Full config dict.

    Returns
    -------
    tuple[pd.Series, float]
        (optimal weights, achieved portfolio volatility).
    """
    tickers = cov.index.tolist()
    n = len(tickers)
    mu = expected_returns.reindex(tickers).values
    sigma = cov.values * 252

    constraints, bounds = build_all_constraints(tickers, config)

    # Add return target constraint
    constraints.append({
        "type": "ineq",
        "fun": lambda w: w @ mu - target_return,
    })

    def objective(w):
        return w @ sigma @ w

    def gradient(w):
        return 2 * sigma @ w

    w0 = np.ones(n) / n

    result = minimize(
        objective,
        w0,
        method="SLSQP",
        jac=gradient,
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    if not result.success:
        raise ValueError(f"Frontier solver did not converge: {result.message}")

    weights = pd.Series(result.x, index=tickers)
    weights = weights.clip(lower=0.0)
    weights = weights / weights.sum()

    port_vol = np.sqrt(weights.values @ sigma @ weights.values)

    return weights, port_vol
