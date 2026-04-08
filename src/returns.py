"""
Return computation — log returns for covariance estimation,
arithmetic returns for performance measurement.

Consistent with the signal-timing rule: returns measured from t to t+1.
"""

import numpy as np
import pandas as pd


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns from adjusted close prices.

    Log returns are additive over time and better suited for covariance
    estimation (closer to normally distributed for short horizons).

    Zero or negative prices are replaced with NaN before computing
    to avoid inf/nan contamination in the estimation pipeline.

    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted close prices, columns = tickers, index = dates.

    Returns
    -------
    pd.DataFrame
        Daily log returns (first row is NaN, dropped).
    """
    clean = prices.where(prices > 0)  # Replace zero/negative with NaN
    ret = np.log(clean / clean.shift(1))
    return ret.dropna(how="all")


def arithmetic_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily simple (arithmetic) returns from adjusted close prices.

    Arithmetic returns are used for portfolio performance measurement
    because portfolio return = weighted sum of arithmetic returns.

    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted close prices, columns = tickers, index = dates.

    Returns
    -------
    pd.DataFrame
        Daily arithmetic returns (first row is NaN, dropped).
    """
    ret = prices.pct_change()
    return ret.dropna(how="all")


def annualize_returns(daily_returns: pd.Series, trading_days: int = 252) -> float:
    """Annualize a series of daily returns via compounding (CAGR).

    Parameters
    ----------
    daily_returns : pd.Series
        Daily arithmetic returns.
    trading_days : int
        Trading days per year.

    Returns
    -------
    float
        Annualized compound return.
    """
    total = (1 + daily_returns).prod()
    n_days = len(daily_returns)
    if n_days == 0:
        return np.nan
    return total ** (trading_days / n_days) - 1


def annualize_vol(daily_returns: pd.Series, trading_days: int = 252) -> float:
    """Annualize daily return volatility.

    Parameters
    ----------
    daily_returns : pd.Series
        Daily arithmetic returns.
    trading_days : int
        Trading days per year.

    Returns
    -------
    float
        Annualized standard deviation.
    """
    return daily_returns.std() * np.sqrt(trading_days)
