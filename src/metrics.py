from __future__ import annotations

"""
Performance and risk metrics, computed on daily returns, annualized where applicable.

All metrics follow standard institutional finance conventions:
  - Sharpe/Sortino use configurable risk-free rate
  - CVaR uses left-tail of daily returns
  - Drawdown computed on cumulative wealth path
"""

import numpy as np
import pandas as pd

from src.returns import annualize_returns, annualize_vol


def compute_all_metrics(
    daily_returns: pd.Series, config: dict, benchmark_returns: pd.Series | None = None
) -> dict:
    """Compute the full suite of portfolio performance metrics.

    Parameters
    ----------
    daily_returns : pd.Series
        Daily arithmetic returns of the portfolio.
    config : dict
        Full config dict (for risk-free rate, CVaR percentile, etc.).
    benchmark_returns : pd.Series, optional
        Daily returns of the benchmark (for tracking error / IR).

    Returns
    -------
    dict
        Metric name -> value.
    """
    m = config.get("metrics", {})
    rf = m.get("risk_free_rate", 0.04)
    cvar_pct = m.get("cvar_percentile", 0.05)

    cagr = annualize_returns(daily_returns)
    vol = annualize_vol(daily_returns)
    sharpe = sharpe_ratio(daily_returns, rf)
    sortino = sortino_ratio(daily_returns, rf)
    mdd = max_drawdown(daily_returns)
    calmar = calmar_ratio(daily_returns)
    cvar = cvar_95(daily_returns, cvar_pct)

    result = {
        "CAGR": cagr,
        "Annualized Vol": vol,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Max Drawdown": mdd,
        "Calmar Ratio": calmar,
        "CVaR 95%": cvar,
    }

    if benchmark_returns is not None:
        te = tracking_error(daily_returns, benchmark_returns)
        ir = information_ratio(daily_returns, benchmark_returns)
        result["Tracking Error"] = te
        result["Information Ratio"] = ir

    return result


def sharpe_ratio(
    daily_returns: pd.Series, rf: float = 0.04, trading_days: int = 252
) -> float:
    """Annualized Sharpe ratio: (mean_excess_return × 252) / (vol × √252).

    Uses arithmetic mean excess return, not CAGR, for consistency with
    standard Sharpe ratio definition (mean / std of excess returns).
    """
    daily_rf = (1 + rf) ** (1 / trading_days) - 1
    excess = daily_returns - daily_rf
    mean_excess = excess.mean() * trading_days
    vol = annualize_vol(daily_returns, trading_days)
    if vol == 0 or np.isnan(vol):
        return np.nan
    return mean_excess / vol


def sortino_ratio(
    daily_returns: pd.Series, rf: float = 0.04, trading_days: int = 252
) -> float:
    """Annualized Sortino ratio: mean_excess_return / downside_vol.

    Downside deviation is computed on returns below the daily risk-free rate
    (MAR = daily rf), not just negative returns.
    """
    daily_rf = (1 + rf) ** (1 / trading_days) - 1
    excess = daily_returns - daily_rf
    mean_excess = excess.mean() * trading_days
    # Downside deviation: std of returns below MAR (daily rf)
    downside = excess[excess < 0]
    if len(downside) == 0:
        return np.nan
    downside_vol = downside.std() * np.sqrt(trading_days)
    if downside_vol == 0 or np.isnan(downside_vol):
        return np.nan
    return mean_excess / downside_vol


def max_drawdown(daily_returns: pd.Series) -> float:
    """Maximum peak-to-trough decline on cumulative wealth path.

    Returns a negative number (e.g., -0.25 = -25% drawdown).
    """
    wealth = (1 + daily_returns).cumprod()
    peak = wealth.cummax()
    dd = (wealth - peak) / peak
    return float(dd.min())


def calmar_ratio(daily_returns: pd.Series) -> float:
    """CAGR / |Max Drawdown|."""
    cagr = annualize_returns(daily_returns)
    mdd = max_drawdown(daily_returns)
    if mdd == 0 or np.isnan(mdd):
        return np.nan
    return cagr / abs(mdd)


def cvar_95(daily_returns: pd.Series, percentile: float = 0.05) -> float:
    """Conditional Value at Risk: mean of returns below the given percentile.

    Returns a negative number representing expected loss in the tail.
    """
    cutoff = daily_returns.quantile(percentile)
    tail = daily_returns[daily_returns <= cutoff]
    if len(tail) == 0:
        return np.nan
    return float(tail.mean())


def tracking_error(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    trading_days: int = 252,
) -> float:
    """Annualized tracking error: std of excess returns × √252."""
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    if len(aligned) == 0:
        return np.nan
    excess = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    return float(excess.std() * np.sqrt(trading_days))


def information_ratio(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """(CAGR_port - CAGR_bench) / Tracking Error."""
    cagr_p = annualize_returns(portfolio_returns)
    cagr_b = annualize_returns(benchmark_returns)
    te = tracking_error(portfolio_returns, benchmark_returns)
    if te == 0 or np.isnan(te):
        return np.nan
    return (cagr_p - cagr_b) / te


def rolling_sharpe(
    daily_returns: pd.Series,
    window: int = 252,
    rf: float = 0.04,
) -> pd.Series:
    """Rolling annualized Sharpe ratio.

    Parameters
    ----------
    daily_returns : pd.Series
        Daily arithmetic returns.
    window : int
        Rolling window in trading days.
    rf : float
        Annualized risk-free rate.

    Returns
    -------
    pd.Series
        Rolling Sharpe with same index as input (NaN for first window-1 obs).
    """
    daily_rf = (1 + rf) ** (1 / 252) - 1
    excess = daily_returns - daily_rf
    roll_mean = excess.rolling(window).mean() * 252
    roll_std = daily_returns.rolling(window).std() * np.sqrt(252)
    return roll_mean / roll_std


def drawdown_series(daily_returns: pd.Series) -> pd.Series:
    """Compute the drawdown time series (all values <= 0)."""
    wealth = (1 + daily_returns).cumprod()
    peak = wealth.cummax()
    return (wealth - peak) / peak


def vol_normalize(
    daily_returns: pd.Series,
    target_vol: float = 0.10,
    lookback: int = 63,
) -> pd.Series:
    """Scale a return series to a target annualised volatility.

    Uses a rolling vol estimate to scale returns day by day, producing
    a return stream that targets *target_vol* annualised. This enables
    fair comparison of strategies with different risk levels.

    Parameters
    ----------
    daily_returns : pd.Series
        Raw daily arithmetic returns.
    target_vol : float
        Target annualised volatility (default 10%).
    lookback : int
        Rolling window for vol estimate in trading days (default 63 ≈ 3 months).

    Returns
    -------
    pd.Series
        Vol-normalised daily returns.
    """
    rolling_vol = daily_returns.rolling(lookback).std() * np.sqrt(252)
    # Use expanding vol for the first `lookback` days
    expanding_vol = daily_returns.expanding(min_periods=10).std() * np.sqrt(252)
    vol_est = rolling_vol.fillna(expanding_vol)
    # Avoid division by zero / extreme leverage
    vol_est = vol_est.clip(lower=0.01)
    scale = target_vol / vol_est
    # Cap leverage at 3x to avoid blowups
    scale = scale.clip(upper=3.0)
    return daily_returns * scale


def compute_gross_net_stats(
    daily_returns: pd.Series,
    turnover_history: list,
    tc_bps: int,
    config: dict,
) -> dict:
    """Compute gross vs net performance statistics.

    Parameters
    ----------
    daily_returns : pd.Series
        Net daily returns (already include TC).
    turnover_history : list
        List of (date, turnover) tuples from backtest.
    tc_bps : int
        Transaction cost in basis points.
    config : dict
        Full config dict.

    Returns
    -------
    dict
        Keys: avg_annual_turnover, total_tc_drag, gross_sharpe, net_sharpe.
    """
    if not turnover_history or len(daily_returns) == 0:
        return {
            "avg_annual_turnover": np.nan,
            "total_tc_drag": np.nan,
            "gross_sharpe": np.nan,
            "net_sharpe": np.nan,
        }

    turnovers = [t for _, t in turnover_history]
    n_years = len(daily_returns) / 252

    # Average annual turnover
    total_turnover = sum(turnovers)
    avg_annual = total_turnover / max(n_years, 0.01)

    # Total TC drag
    total_tc = total_turnover * tc_bps / 10000

    # Gross returns: add back the TC that was deducted
    rf = config.get("metrics", {}).get("risk_free_rate", 0.04)
    net_sharpe = sharpe_ratio(daily_returns, rf=rf)

    # Reconstruct gross returns by adding TC back on rebalance days
    gross = daily_returns.copy()
    for date, to in turnover_history:
        if date in gross.index:
            gross.loc[date] += to * tc_bps / 10000
    gross_sr = sharpe_ratio(gross, rf=rf)

    return {
        "avg_annual_turnover": avg_annual,
        "total_tc_drag": total_tc,
        "gross_sharpe": gross_sr,
        "net_sharpe": net_sharpe,
    }
