"""
Benchmark portfolio construction, SPY buy-and-hold, 60/40, equal-weight,
and static risk parity.

Benchmarks are computed on the same date range as the backtest for fair
comparison. AGG is used in the 60/40 benchmark only, not in the optimisation
universe.
"""

import numpy as np
import pandas as pd

from src.returns import arithmetic_returns
from src.optimizer_rp import optimize_risk_parity
from utils.helpers import get_rebalance_dates


def spy_buy_hold(prices: pd.DataFrame) -> pd.Series:
    """SPY buy-and-hold benchmark: 100% SPY, no rebalance.

    Parameters
    ----------
    prices : pd.DataFrame
        Full price DataFrame (must include 'SPY' column).

    Returns
    -------
    pd.Series
        Daily arithmetic returns for the SPY benchmark.
    """
    return prices["SPY"].pct_change().dropna()


def sixty_forty(
    prices: pd.DataFrame, config: dict
) -> pd.Series:
    """60/40 benchmark: 60% SPY + 40% AGG, monthly rebalance.

    AGG is fetched separately since it is not in the optimisation universe.

    Parameters
    ----------
    prices : pd.DataFrame
        Full price DataFrame (must include 'SPY').
    config : dict
        Full config dict.

    Returns
    -------
    pd.Series
        Daily arithmetic returns for the 60/40 benchmark.
    """
    import yfinance as yf
    from datetime import timedelta

    bench_config = config.get("benchmarks", {}).get("sixty_forty", {})
    eq_w = bench_config.get("equity_weight", 0.60)
    bd_w = bench_config.get("bond_weight", 0.40)
    eq_ticker = bench_config.get("equity_ticker", "SPY")
    bd_ticker = bench_config.get("bond_ticker", "AGG")
    rebal_freq = bench_config.get("rebalance", "monthly")

    # Fetch AGG if not in prices
    if bd_ticker not in prices.columns:
        start = prices.index.min() - timedelta(days=5)
        end = prices.index.max() + timedelta(days=1)
        agg = yf.download(bd_ticker, start=start, end=end, auto_adjust=False, progress=False)
        if isinstance(agg.columns, pd.MultiIndex):
            agg_prices = agg["Adj Close"]
            if isinstance(agg_prices, pd.DataFrame):
                agg_prices = agg_prices.iloc[:, 0]
        else:
            agg_prices = agg["Adj Close"]
        combined = pd.DataFrame({
            eq_ticker: prices[eq_ticker],
            bd_ticker: agg_prices,
        }).dropna()
    else:
        combined = prices[[eq_ticker, bd_ticker]].dropna()

    rets = combined.pct_change().dropna()
    rebal_dates = get_rebalance_dates(rets.index, rebal_freq)

    # Simulate with monthly rebalance
    portfolio_returns = []
    weights = np.array([eq_w, bd_w])

    for date in rets.index:
        daily_ret = rets.loc[date].values
        port_ret = weights @ daily_ret
        portfolio_returns.append(port_ret)

        # Drift weights
        weights = weights * (1 + daily_ret)
        weights = weights / weights.sum()

        # Rebalance
        if date in rebal_dates:
            weights = np.array([eq_w, bd_w])

    return pd.Series(portfolio_returns, index=rets.index, name="60/40")


def equal_weight(
    prices: pd.DataFrame, config: dict
) -> pd.Series:
    """Equal-weight (1/N) benchmark over the optimisation universe.

    Parameters
    ----------
    prices : pd.DataFrame
        Full price DataFrame with universe tickers.
    config : dict
        Full config dict.

    Returns
    -------
    pd.Series
        Daily arithmetic returns for the equal-weight benchmark.
    """
    rebal_freq = config.get("benchmarks", {}).get("equal_weight", {}).get("rebalance", "monthly")
    rets = prices.pct_change().dropna()
    n = len(rets.columns)
    rebal_dates = get_rebalance_dates(rets.index, rebal_freq)

    weights = np.ones(n) / n
    portfolio_returns = []

    for date in rets.index:
        daily_ret = rets.loc[date].values
        port_ret = weights @ daily_ret
        portfolio_returns.append(port_ret)

        weights = weights * (1 + daily_ret)
        weights = weights / weights.sum()

        if date in rebal_dates:
            weights = np.ones(n) / n

    return pd.Series(portfolio_returns, index=rets.index, name="Equal Weight")


def static_risk_parity(
    prices: pd.DataFrame, cov: pd.DataFrame, config: dict
) -> pd.Series:
    """Static risk parity: compute RP weights once on first-year data, hold static.

    Uses only the first 252 trading days to estimate covariance, avoiding
    full-sample lookahead contamination. This makes the benchmark
    point-in-time feasible.

    Parameters
    ----------
    prices : pd.DataFrame
        Full price DataFrame.
    cov : pd.DataFrame
        Full-sample covariance matrix (unused, kept for API compat).
    config : dict
        Full config dict.

    Returns
    -------
    pd.Series
        Daily arithmetic returns for the static RP benchmark.
    """
    from src.returns import log_returns as _lr
    from src.covariance import ledoit_wolf_covariance
    # Use first 252 days only to avoid lookahead
    log_ret = _lr(prices)
    first_year = log_ret.iloc[:min(252, len(log_ret))]
    init_cov, _ = ledoit_wolf_covariance(first_year)
    rp_weights = optimize_risk_parity(init_cov, config)
    rets = prices.pct_change().dropna()

    # Simple weighted returns, no rebalance (weights drift)
    weights = rp_weights.reindex(rets.columns).values

    portfolio_returns = []
    for date in rets.index:
        daily_ret = rets.loc[date].values
        port_ret = weights @ daily_ret
        portfolio_returns.append(port_ret)

        weights = weights * (1 + daily_ret)
        total = weights.sum()
        if total > 0:
            weights = weights / total

    return pd.Series(portfolio_returns, index=rets.index, name="Static RP")


def compute_all_benchmarks(
    prices: pd.DataFrame, cov: pd.DataFrame, config: dict
) -> dict[str, pd.Series]:
    """Compute all benchmark return series.

    Parameters
    ----------
    prices : pd.DataFrame
        Full price DataFrame.
    cov : pd.DataFrame
        Full-sample covariance matrix.
    config : dict
        Full config dict.

    Returns
    -------
    dict[str, pd.Series]
        Benchmark name -> daily return series.
    """
    benchmarks = {}
    benchmarks["SPY B&H"] = spy_buy_hold(prices)
    benchmarks["60/40"] = sixty_forty(prices, config)
    benchmarks["Equal Weight"] = equal_weight(prices, config)
    benchmarks["Static RP"] = static_risk_parity(prices, cov, config)
    return benchmarks
