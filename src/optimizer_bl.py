from __future__ import annotations

"""
Black-Litterman optimizer.

Four steps:
  1. Implied equilibrium returns: π = λ × Σ × w_mkt
  2. Incorporate investor views: Q = P × μ + ε, ε ~ N(0, Ω)
  3. Posterior returns: μ_BL = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ × [(τΣ)⁻¹π + P'Ω⁻¹Q]
  4. Feed μ_BL into mean-variance optimizer (same constraints)
"""

import numpy as np
import pandas as pd

from src.expected_returns import bl_implied_returns
from src.optimizer_mv import optimize_mean_variance


def build_views(
    tickers: list[str],
    views: dict[str, float],
    view_confidence: dict[str, float],
    default_confidence: float = 0.50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the pick matrix P, view vector Q, and confidence vector.

    Supports absolute views only (e.g., "SPY will return 8%").
    Each view maps one ticker to one expected return.

    Parameters
    ----------
    tickers : list[str]
        Full asset universe.
    views : dict[str, float]
        Ticker -> expected annual return (absolute view).
    view_confidence : dict[str, float]
        Ticker -> confidence level (0 to 1).
    default_confidence : float
        Fallback confidence for views without explicit entry.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        P (K x N pick matrix), Q (K x 1 view vector),
        confidences (K x 1 confidence levels).
    """
    if not views:
        return np.empty((0, len(tickers))), np.empty(0), np.empty(0)

    K = len(views)
    N = len(tickers)
    P = np.zeros((K, N))
    Q = np.zeros(K)
    confidences = np.zeros(K)

    for i, (ticker, expected_ret) in enumerate(views.items()):
        if ticker in tickers:
            j = tickers.index(ticker)
            P[i, j] = 1.0
            Q[i] = expected_ret
            confidences[i] = view_confidence.get(ticker, default_confidence)

    # Filter out rows where the ticker wasn't in the universe
    valid = P.sum(axis=1) > 0
    return P[valid], Q[valid], confidences[valid]


def compute_view_uncertainty(
    P: np.ndarray,
    cov: np.ndarray,
    tau: float,
    confidences: np.ndarray,
) -> np.ndarray:
    """Compute the view uncertainty matrix Ω.

    Default: Ω = diag(τ × P × Σ × P') scaled by inverse confidence.
    Higher confidence -> lower uncertainty (tighter views).

    Parameters
    ----------
    P : np.ndarray
        K x N pick matrix.
    cov : np.ndarray
        N x N covariance matrix (annualised).
    tau : float
        Scaling factor for prior uncertainty.
    confidences : np.ndarray
        K x 1 confidence levels (0 to 1).

    Returns
    -------
    np.ndarray
        K x K diagonal uncertainty matrix.
    """
    base_omega = np.diag(np.diag(tau * P @ cov @ P.T))

    # Scale uncertainty inversely with confidence:
    # confidence=1 -> scale=1 (tight), confidence~0 -> scale very high (loose)
    # Use 1/confidence as multiplier, clipped to avoid division by zero
    confidence_scale = np.where(confidences > 0.01, 1.0 / confidences, 100.0)

    return base_omega * np.diag(confidence_scale)


def bl_posterior(
    prior_returns: pd.Series,
    cov: pd.DataFrame,
    P: np.ndarray,
    Q: np.ndarray,
    omega: np.ndarray,
    tau: float,
) -> tuple[pd.Series, pd.DataFrame]:
    """Compute Black-Litterman posterior returns and covariance.

    μ_BL = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ × [(τΣ)⁻¹π + P'Ω⁻¹Q]
    Σ_BL = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ + Σ

    If no views are provided (P is empty), posterior = prior.

    Parameters
    ----------
    prior_returns : pd.Series
        Implied equilibrium returns π.
    cov : pd.DataFrame
        N x N covariance matrix (annualised).
    P : np.ndarray
        K x N pick matrix.
    Q : np.ndarray
        K x 1 view vector.
    omega : np.ndarray
        K x K view uncertainty matrix.
    tau : float
        Scaling factor.

    Returns
    -------
    tuple[pd.Series, pd.DataFrame]
        (posterior expected returns, posterior covariance).
    """
    tickers = cov.index
    sigma = cov.values
    pi = prior_returns.reindex(tickers).values

    # No views case: posterior = prior
    if P.shape[0] == 0:
        return prior_returns.copy(), cov.copy()

    tau_sigma = tau * sigma

    # Use solve() instead of inv() for numerical stability with near-singular matrices
    # τΣ⁻¹ is computed implicitly via solve(τΣ, ·)
    tau_sigma_inv_pi = np.linalg.solve(tau_sigma, pi)
    tau_sigma_inv_full = np.linalg.solve(tau_sigma, np.eye(sigma.shape[0]))

    omega_inv_Q = np.linalg.solve(omega, Q)
    omega_inv_P = np.linalg.solve(omega, P)

    # Posterior precision: (τΣ)⁻¹ + P'Ω⁻¹P
    posterior_precision = tau_sigma_inv_full + P.T @ omega_inv_P
    # Posterior covariance adjustment via solve (avoids explicit inv)
    posterior_cov_adj = np.linalg.solve(posterior_precision, np.eye(sigma.shape[0]))

    # Posterior mean: M⁻¹ × [(τΣ)⁻¹π + P'Ω⁻¹Q]
    mu_bl = posterior_cov_adj @ (tau_sigma_inv_pi + P.T @ omega_inv_Q)

    # Posterior covariance (full)
    sigma_bl = posterior_cov_adj + sigma

    mu_bl_series = pd.Series(mu_bl, index=tickers, name="bl_posterior_returns")
    sigma_bl_df = pd.DataFrame(sigma_bl, index=tickers, columns=tickers)

    return mu_bl_series, sigma_bl_df


def optimize_black_litterman(
    cov: pd.DataFrame,
    market_weights: pd.Series,
    config: dict,
    views: dict[str, float] | None = None,
    view_confidence: dict[str, float] | None = None,
    prev_weights: np.ndarray | None = None,
) -> tuple[pd.Series, dict]:
    """Run the full Black-Litterman pipeline: implied returns -> views -> posterior -> MV.

    Parameters
    ----------
    cov : pd.DataFrame
        N x N covariance matrix (daily frequency, annualised internally).
    market_weights : pd.Series
        Market-cap weights for equilibrium returns.
    config : dict
        Full config dict.
    views : dict, optional
        Ticker -> expected return. If None, uses config defaults.
    view_confidence : dict, optional
        Ticker -> confidence. If None, uses config defaults.
    prev_weights : np.ndarray, optional
        Previous weights for turnover constraint.

    Returns
    -------
    tuple[pd.Series, dict]
        (optimal weights, info dict with prior/posterior returns).
    """
    tickers = cov.index.tolist()
    bl_config = config.get("black_litterman", {})
    tau = bl_config.get("tau", 0.05)
    risk_aversion = bl_config.get("risk_aversion", 2.5)

    # Step 1: Implied equilibrium returns
    prior = bl_implied_returns(cov, market_weights, risk_aversion=risk_aversion)

    # Step 2: Build views
    if views is None:
        views = bl_config.get("default_views", {})
    if view_confidence is None:
        view_confidence = bl_config.get("view_confidence", {})
    default_conf = bl_config.get("default_confidence", 0.50)

    P, Q, confidences = build_views(tickers, views, view_confidence, default_conf)

    # Step 3: View uncertainty and posterior
    sigma_annual = cov.values * 252
    omega = compute_view_uncertainty(P, sigma_annual, tau, confidences)

    cov_annual = pd.DataFrame(sigma_annual, index=tickers, columns=tickers)
    posterior_returns, posterior_cov = bl_posterior(
        prior, cov_annual, P, Q, omega, tau
    )

    # Step 4: Feed posterior into MV optimizer
    # Use posterior covariance (already annualised), so pass daily=False by
    # dividing back by 252 since MV internally annualises
    posterior_cov_daily = posterior_cov / 252
    weights = optimize_mean_variance(
        posterior_returns, posterior_cov_daily, config, prev_weights
    )
    weights.name = "bl_weights"

    info = {
        "prior_returns": prior,
        "posterior_returns": posterior_returns,
        "posterior_cov": posterior_cov,
        "views": views,
        "view_confidence": view_confidence,
        "tau": tau,
    }

    return weights, info
