from __future__ import annotations

"""
Expected return estimation — two methods:
  1. Shrinkage returns (default for MV): pull historical means toward the grand mean
  2. Black-Litterman implied returns: equilibrium returns from market-cap weights

CRITICAL: Raw historical mean returns are NEVER used as expected returns.
They are too noisy and lead to extreme, unstable allocations.
"""

import numpy as np
import pandas as pd


def shrinkage_returns(
    returns: pd.DataFrame, intensity: float = 0.5
) -> pd.Series:
    """Compute shrinkage expected returns.

    μ_shrink = δ × μ_grand + (1 - δ) × μ_historical
    where μ_grand = cross-sectional mean of all asset mean returns.

    This pulls extreme return estimates toward the cross-sectional mean,
    producing more stable inputs for mean-variance optimization.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily arithmetic returns, columns = tickers.
    intensity : float
        Shrinkage intensity δ (0 = pure historical, 1 = pure grand mean).
        Default 0.5 from config.

    Returns
    -------
    pd.Series
        Annualized expected returns per asset.
    """
    # Annualize daily arithmetic means: μ_annual = daily_mean × 252
    # This is the correct input for mean-variance optimization (arithmetic expectation),
    # not the CAGR-style (1+μ)^252 - 1 which introduces compounding bias.
    daily_means = returns.mean()
    annual_means = daily_means * 252

    grand_mean = annual_means.mean()
    shrunk = intensity * grand_mean + (1 - intensity) * annual_means

    return shrunk


def bl_implied_returns(
    cov: pd.DataFrame,
    market_weights: pd.Series,
    risk_aversion: float = 2.5,
) -> pd.Series:
    """Compute Black-Litterman implied equilibrium returns.

    π = λ × Σ × w_mkt

    These are the returns that, given the covariance matrix and market-cap
    weights, make the market portfolio the mean-variance optimal portfolio.

    Parameters
    ----------
    cov : pd.DataFrame
        N x N covariance matrix (annualized or daily — must be consistent).
    market_weights : pd.Series
        Market-cap weights, must sum to ~1.0 and align with cov index.
    risk_aversion : float
        Risk aversion coefficient λ (default 2.5).

    Returns
    -------
    pd.Series
        Implied equilibrium expected returns (same frequency as cov).
    """
    tickers = cov.index
    w = market_weights.reindex(tickers).values
    sigma = cov.values

    # π = λ × Σ × w_mkt (annualize by multiplying daily cov by 252)
    pi = risk_aversion * (sigma * 252) @ w

    return pd.Series(pi, index=tickers, name="implied_returns")


def estimate_expected_returns(
    returns: pd.DataFrame,
    method: str,
    config: dict,
    cov: pd.DataFrame | None = None,
    market_weights: pd.Series | None = None,
) -> pd.Series:
    """Dispatch to the requested expected return estimation method.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily arithmetic returns.
    method : str
        'shrinkage' or 'bl_implied'.
    config : dict
        Full config dict.
    cov : pd.DataFrame, optional
        Covariance matrix (required for 'bl_implied').
    market_weights : pd.Series, optional
        Market-cap weights (required for 'bl_implied').

    Returns
    -------
    pd.Series
        Annualized expected returns per asset.
    """
    if method == "shrinkage":
        intensity = config.get("expected_returns", {}).get("shrinkage", {}).get("intensity", 0.5)
        return shrinkage_returns(returns, intensity=intensity)

    elif method == "bl_implied":
        if cov is None or market_weights is None:
            raise ValueError("bl_implied requires cov and market_weights")
        risk_aversion = config.get("black_litterman", {}).get("risk_aversion", 2.5)
        return bl_implied_returns(cov, market_weights, risk_aversion=risk_aversion)

    else:
        raise ValueError(f"Unknown method '{method}'. Use: shrinkage, bl_implied")
