from __future__ import annotations

"""
Risk Parity (Equal Risk Contribution) optimizer.

Each asset contributes equally to total portfolio variance:
  RC_i = w_i × (Σw)_i / (w'Σw) = 1/N for all i.

Minimises: Σ_i Σ_j (RC_i - RC_j)² subject to full-investment and long-only.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.constraints import build_all_constraints


def risk_contributions(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Compute each asset's risk contribution to total portfolio variance.

    RC_i = w_i × (Σw)_i
    Percentage RC = RC_i / portfolio_variance

    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights (N,).
    cov : np.ndarray
        N x N covariance matrix.

    Returns
    -------
    np.ndarray
        Percentage risk contributions (sum to 1).
    """
    port_var = weights @ cov @ weights
    marginal = cov @ weights
    rc = weights * marginal

    if port_var > 0:
        return rc / port_var
    return rc


def optimize_risk_parity(
    cov: pd.DataFrame,
    config: dict,
    prev_weights: np.ndarray | None = None,
) -> pd.Series:
    """Run risk parity optimization.

    Objective: minimise Σ_i Σ_j (w_i × (Σw)_i - w_j × (Σw)_j)²
    subject to Σ w_i = 1, w_i >= 0, plus per-class caps.

    Parameters
    ----------
    cov : pd.DataFrame
        N x N covariance matrix (daily frequency, annualised internally).
    config : dict
        Full config dict.
    prev_weights : np.ndarray, optional
        Previous weights for turnover constraint.

    Returns
    -------
    pd.Series
        Risk parity weights (index = tickers).
    """
    tickers = cov.index.tolist()
    n = len(tickers)
    sigma = cov.values * 252  # Annualise

    rp_config = config.get("optimization", {}).get("risk_parity", {})
    max_iter = rp_config.get("max_iter", 1000)
    tol = rp_config.get("tolerance", 1e-10)

    constraints, bounds = build_all_constraints(tickers, config, prev_weights)

    def objective(w):
        """Sum of squared differences between all pairs of risk contributions."""
        marginal = sigma @ w
        rc = w * marginal  # Absolute risk contributions
        # All pairs (RC_i - RC_j)^2, equivalent to N * var(RC)
        target_rc = (w @ sigma @ w) / n
        return np.sum((rc - target_rc) ** 2)

    def gradient(w):
        """Analytical gradient of the ERC objective."""
        Sw = sigma @ w
        port_var = w @ Sw
        target_rc = port_var / n
        rc = w * Sw
        diff = rc - target_rc

        # d(rc_i)/d(w_j) = delta_ij * (Σw)_i + w_i * Σ_ij
        # d(target)/d(w_j) = (1/n) * 2 * (Σw)_j
        grad = np.zeros(n)
        for i in range(n):
            # Derivative of rc_i w.r.t. w
            drc_i = np.zeros(n)
            drc_i[i] += Sw[i]
            drc_i += w[i] * sigma[i, :]

            # Derivative of target w.r.t. w
            dtarget = 2 * Sw / n

            grad += 2 * diff[i] * (drc_i - dtarget)

        return grad

    # Initial guess: inverse-volatility weights (good starting point for RP)
    vols = np.sqrt(np.diag(sigma))
    vols = np.where(vols == 0, 1e-10, vols)
    w0 = (1 / vols) / (1 / vols).sum()

    result = minimize(
        objective,
        w0,
        method="SLSQP",
        jac=gradient,
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": max_iter, "ftol": tol},
    )

    if not result.success:
        import warnings
        warnings.warn(f"RP optimizer did not converge: {result.message}")

    weights = pd.Series(result.x, index=tickers, name="rp_weights")
    weights = weights.clip(lower=0.0)
    total = weights.sum()
    if total > 0:
        weights = weights / total
    else:
        weights = pd.Series(1.0 / n, index=tickers, name="rp_weights")

    return weights
