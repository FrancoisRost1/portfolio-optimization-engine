"""
Risk decomposition — risk contribution, marginal risk, Herfindahl index.

Decomposes total portfolio variance into per-asset contributions, enabling
comparison of diversification quality across optimization methods.
"""

import numpy as np
import pandas as pd


def risk_contribution(
    weights: pd.Series, cov: pd.DataFrame
) -> pd.DataFrame:
    """Compute absolute and percentage risk contributions per asset.

    RC_i = w_i × (Σw)_i
    Pct RC_i = RC_i / portfolio_variance

    Parameters
    ----------
    weights : pd.Series
        Portfolio weights (index = tickers).
    cov : pd.DataFrame
        N x N DAILY covariance matrix (annualised internally by × 252).

    Returns
    -------
    pd.DataFrame
        Columns: weight, marginal_risk, risk_contribution, pct_risk_contribution.
    """
    tickers = weights.index
    w = weights.reindex(tickers).values
    sigma = cov.reindex(index=tickers, columns=tickers).values * 252

    port_var = w @ sigma @ w
    port_vol = np.sqrt(port_var) if port_var > 0 else 0.0

    # Marginal risk contribution: (Σw) / σ_portfolio
    marginal = sigma @ w
    if port_vol > 0:
        marginal_rc = marginal / port_vol
    else:
        marginal_rc = np.zeros(len(tickers))

    # Absolute risk contribution
    abs_rc = w * marginal

    # Percentage risk contribution
    if port_var > 0:
        pct_rc = abs_rc / port_var
    else:
        pct_rc = np.zeros(len(tickers))

    return pd.DataFrame(
        {
            "weight": w,
            "marginal_risk": marginal_rc,
            "risk_contribution": abs_rc,
            "pct_risk_contribution": pct_rc,
        },
        index=tickers,
    )


def herfindahl_index(weights: pd.Series) -> float:
    """Herfindahl index of portfolio concentration: Σ w_i².

    Lower = more diversified. Minimum = 1/N (equal weight).
    Maximum = 1.0 (single asset).

    Parameters
    ----------
    weights : pd.Series
        Portfolio weights.

    Returns
    -------
    float
        Herfindahl index.
    """
    return float((weights ** 2).sum())


def effective_n(weights: pd.Series) -> float:
    """Effective number of assets: 1 / Herfindahl.

    Higher = more diversified.

    Parameters
    ----------
    weights : pd.Series
        Portfolio weights.

    Returns
    -------
    float
        Effective number of assets.
    """
    h = herfindahl_index(weights)
    if h == 0:
        return np.nan
    return 1.0 / h


def portfolio_volatility(weights: pd.Series, cov: pd.DataFrame) -> float:
    """Compute annualised portfolio volatility.

    σ_p = √(w'Σw × 252)

    Parameters
    ----------
    weights : pd.Series
        Portfolio weights.
    cov : pd.DataFrame
        N x N daily covariance matrix.

    Returns
    -------
    float
        Annualised portfolio volatility.
    """
    tickers = weights.index
    w = weights.reindex(tickers).values
    sigma = cov.reindex(index=tickers, columns=tickers).values
    port_var = w @ sigma @ w * 252
    return float(np.sqrt(max(port_var, 0)))


def decompose_all_methods(
    all_weights: dict[str, pd.Series], cov: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """Run risk decomposition for multiple methods at once.

    Parameters
    ----------
    all_weights : dict[str, pd.Series]
        Method name -> weights Series.
    cov : pd.DataFrame
        N x N daily covariance matrix.

    Returns
    -------
    dict[str, pd.DataFrame]
        Method name -> risk decomposition DataFrame.
    """
    return {name: risk_contribution(w, cov) for name, w in all_weights.items()}
