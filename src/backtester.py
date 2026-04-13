from __future__ import annotations

"""
Rolling backtest engine, walks forward through time, re-estimating
covariance and re-optimizing weights at each rebalance date.

Signal timing (CRITICAL):
  At close(t): observe prices, estimate cov, run optimizer -> target weights
  At close(t+1): execute trades, measure returns from t+1 forward
  No lookahead. No same-day trading. Consistent with Projects 5, 6, 7.
"""

import numpy as np
import pandas as pd

from src.returns import log_returns, arithmetic_returns
from src.covariance import estimate_covariance
from src.expected_returns import estimate_expected_returns
from src.optimizer_mv import optimize_mean_variance
from src.optimizer_rp import optimize_risk_parity
from src.optimizer_hrp import optimize_hrp
from src.optimizer_bl import optimize_black_litterman
from src.vol_target import apply_vol_target
from utils.helpers import get_rebalance_dates


def run_backtest(
    prices: pd.DataFrame,
    method: str,
    config: dict,
    market_weights: pd.Series | None = None,
) -> dict:
    """Run a rolling backtest for a given optimization method.

    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted close prices, columns = tickers.
    method : str
        One of 'mv', 'rp', 'hrp', 'bl'.
    config : dict
        Full config dict.
    market_weights : pd.Series, optional
        Market-cap weights (required for 'bl' method).

    Returns
    -------
    dict with keys:
        'returns': pd.Series of daily portfolio returns
        'weights_history': list of (date, pd.Series) tuples
        'turnover': list of (date, float) tuples
        'rebalance_dates': list of dates
    """
    bt = config.get("backtest", {})
    rebal_freq = bt.get("rebalance_frequency", "monthly")
    window_type = bt.get("window_type", "rolling")
    window_days = bt.get("window_days", 252)
    min_history = bt.get("min_history", 252)
    tc_bps = bt.get("transaction_cost_bps", 10)
    cov_method = config.get("covariance", {}).get("default_method", "ledoit_wolf")

    # Compute returns
    log_ret = log_returns(prices)
    arith_ret = arithmetic_returns(prices)

    # Identify rebalance dates (need at least min_history days before first)
    all_dates = arith_ret.index
    rebal_candidates = get_rebalance_dates(all_dates, rebal_freq)
    if min_history >= len(all_dates):
        return {
            "returns": pd.Series(dtype=float),
            "weights_history": [],
            "turnover": [],
            "rebalance_dates": [],
        }
    rebal_dates = rebal_candidates[rebal_candidates >= all_dates[min_history]]

    tickers = prices.columns.tolist()
    n = len(tickers)

    # State
    current_weights = np.zeros(n)  # Start with no position
    prev_opt_weights = None
    portfolio_returns = []
    weights_history = []
    turnover_history = []

    rebal_set = set(rebal_dates)
    started = False
    # Pending weights: computed on rebalance day t, applied on day t+1
    # This enforces strict no-lookahead: signal at close(t), trade at close(t+1)
    pending_weights = None
    pending_turnover = None

    for i, date in enumerate(all_dates):
        # ── Step 1: Apply pending weights from PREVIOUS rebalance day ──
        # This ensures weights computed at close(t) take effect at close(t+1)
        if pending_weights is not None:
            turnover = float(np.abs(pending_weights - current_weights).sum())
            turnover_history.append((date, turnover))
            current_weights = pending_weights.copy()
            pending_weights = None
            started = True

        # ── Step 2: Compute today's return with current weights ──
        if started:
            daily_ret = arith_ret.loc[date].values
            port_ret = float(current_weights @ daily_ret)

            # Deduct transaction costs on execution day (day after rebalance signal)
            if turnover_history and turnover_history[-1][0] == date:
                tc = turnover_history[-1][1] * tc_bps / 10000
                port_ret -= tc

            portfolio_returns.append((date, port_ret))

            # Drift weights to reflect market moves (cash earns 0, simplifying assumption)
            # If vol targeting is active, weights sum to < 1 (implicit cash).
            # We preserve that structure by NOT renormalising to 1.
            risky_sum_before = current_weights.sum()
            drifted = current_weights * (1 + daily_ret)
            risky_sum_after = drifted.sum()
            if risky_sum_before > 0 and risky_sum_after > 0:
                # Scale so the risky portion maintains its share vs cash
                current_weights = drifted * (risky_sum_before / risky_sum_after)
            else:
                current_weights = drifted

        # ── Step 3: On rebalance days, compute new weights (signal at close t) ──
        # These will be applied TOMORROW (t+1), not today
        if date in rebal_set:
            date_loc = all_dates.get_loc(date)
            if window_type == "rolling":
                start_loc = max(0, date_loc - window_days)
            else:  # expanding
                start_loc = 0
            est_returns = log_ret.iloc[start_loc:date_loc]

            if len(est_returns) < min_history:
                continue

            # Estimate covariance
            cov, _ = estimate_covariance(est_returns, cov_method, config)

            # Run optimizer
            try:
                new_weights = _run_optimizer(
                    method, est_returns, cov, config,
                    market_weights, prev_opt_weights,
                )
            except Exception:
                continue

            # Apply vol targeting overlay
            new_weights, vt_info = apply_vol_target(new_weights, cov, config)

            # Stage for next-day execution (t+1)
            prev_opt_weights = new_weights.values.copy()
            pending_weights = new_weights.values.copy()
            weights_history.append((date, new_weights.copy()))

            if not started:
                # First rebalance: force start on the next trading day
                pass

    # Build return series
    if not portfolio_returns:
        return {
            "returns": pd.Series(dtype=float),
            "weights_history": [],
            "turnover": [],
            "rebalance_dates": [],
        }

    dates, rets = zip(*portfolio_returns)
    return_series = pd.Series(rets, index=pd.DatetimeIndex(dates), name=method)

    return {
        "returns": return_series,
        "weights_history": weights_history,
        "turnover": turnover_history,
        "rebalance_dates": list(rebal_dates),
    }


def run_all_backtests(
    prices: pd.DataFrame,
    config: dict,
    market_weights: pd.Series | None = None,
) -> dict[str, dict]:
    """Run backtests for all four optimization methods.

    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted close prices.
    config : dict
        Full config dict.
    market_weights : pd.Series, optional
        Market-cap weights for Black-Litterman.

    Returns
    -------
    dict[str, dict]
        Method name -> backtest result dict.
    """
    methods = ["mv", "rp", "hrp", "bl"]
    results = {}
    for method in methods:
        results[method] = run_backtest(prices, method, config, market_weights)
    return results


def _run_optimizer(
    method: str,
    returns: pd.DataFrame,
    cov: pd.DataFrame,
    config: dict,
    market_weights: pd.Series | None,
    prev_weights: np.ndarray | None,
) -> pd.Series:
    """Dispatch to the appropriate optimizer.

    Parameters
    ----------
    method : str
        'mv', 'rp', 'hrp', or 'bl'.
    returns : pd.DataFrame
        Estimation window log returns.
    cov : pd.DataFrame
        Estimated covariance matrix.
    config : dict
        Full config dict.
    market_weights : pd.Series or None
        Market-cap weights for BL.
    prev_weights : np.ndarray or None
        Previous period weights.

    Returns
    -------
    pd.Series
        Optimised weights.
    """
    if method == "mv":
        arith = np.exp(returns) - 1  # Convert log returns to arithmetic
        mu_method = config.get("expected_returns", {}).get("default_method", "shrinkage")
        mu = estimate_expected_returns(arith, mu_method, config, cov, market_weights)
        return optimize_mean_variance(mu, cov, config, prev_weights)

    elif method == "rp":
        return optimize_risk_parity(cov, config, prev_weights)

    elif method == "hrp":
        return optimize_hrp(cov, config)

    elif method == "bl":
        if market_weights is None:
            raise ValueError("Black-Litterman requires market_weights")
        weights, _ = optimize_black_litterman(
            cov, market_weights, config, prev_weights=prev_weights
        )
        return weights

    else:
        raise ValueError(f"Unknown method '{method}'. Use: mv, rp, hrp, bl")
