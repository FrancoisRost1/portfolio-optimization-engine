"""
Shared utilities — date handling, formatting, and common helpers.
"""

import pandas as pd
import numpy as np


def get_month_end_dates(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return the last trading day of each month present in *dates*.

    Parameters
    ----------
    dates : pd.DatetimeIndex
        Full set of trading dates (e.g. from a price DataFrame index).

    Returns
    -------
    pd.DatetimeIndex
        One date per calendar month — the last date in *dates* for that month.
    """
    s = pd.Series(dates, index=dates)
    return pd.DatetimeIndex(s.groupby(s.dt.to_period("M")).last().values)


def get_quarter_end_dates(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return the last trading day of each quarter present in *dates*."""
    s = pd.Series(dates, index=dates)
    return pd.DatetimeIndex(s.groupby(s.dt.to_period("Q")).last().values)


def get_week_end_dates(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return the last trading day of each week present in *dates*."""
    s = pd.Series(dates, index=dates)
    return pd.DatetimeIndex(s.groupby(s.dt.to_period("W")).last().values)


def get_rebalance_dates(
    dates: pd.DatetimeIndex, frequency: str
) -> pd.DatetimeIndex:
    """Return rebalance dates based on the specified frequency.

    Parameters
    ----------
    dates : pd.DatetimeIndex
        All available trading dates.
    frequency : str
        One of 'monthly', 'quarterly', 'weekly'.

    Returns
    -------
    pd.DatetimeIndex
    """
    dispatch = {
        "monthly": get_month_end_dates,
        "quarterly": get_quarter_end_dates,
        "weekly": get_week_end_dates,
    }
    if frequency not in dispatch:
        raise ValueError(f"Unknown frequency '{frequency}'. Use: {list(dispatch)}")
    return dispatch[frequency](dates)


def safe_div(numerator: float, denominator: float) -> float:
    """Division that returns np.nan instead of raising on zero denominator."""
    if denominator == 0 or np.isnan(denominator):
        return np.nan
    return numerator / denominator


def format_pct(value: float, decimals: int = 1) -> str:
    """Format a decimal as a percentage string, e.g. 0.1234 -> '12.3%'."""
    if np.isnan(value):
        return "N/A"
    return f"{value * 100:.{decimals}f}%"


def format_ratio(value: float, decimals: int = 2) -> str:
    """Format a ratio, e.g. 1.234 -> '1.23x'."""
    if np.isnan(value):
        return "N/A"
    return f"{value:.{decimals}f}x"
