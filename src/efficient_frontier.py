from __future__ import annotations

"""
Efficient frontier generation, sweeps target returns to trace the
risk-return frontier for mean-variance portfolios.
"""

import numpy as np
import pandas as pd

from src.optimizer_mv import optimize_for_target_return


def generate_efficient_frontier(
    expected_returns: pd.Series,
    cov: pd.DataFrame,
    config: dict,
    n_points: int | None = None,
) -> pd.DataFrame:
    """Generate the efficient frontier by sweeping target returns.

    For each target return between the minimum and maximum achievable,
    finds the minimum-variance portfolio and records its volatility.

    Parameters
    ----------
    expected_returns : pd.Series
        Annualized expected returns per asset.
    cov : pd.DataFrame
        N x N covariance matrix (daily frequency).
    config : dict
        Full config dict.
    n_points : int, optional
        Number of points on the frontier. Defaults to config value (50).

    Returns
    -------
    pd.DataFrame
        Columns: target_return, volatility, weights (as dict).
    """
    if n_points is None:
        n_points = config.get("optimization", {}).get("mean_variance", {}).get("frontier_points", 50)

    # Per-asset cap constrains the max achievable return
    mu = expected_returns.values
    cap = config.get("constraints", {}).get("per_asset", {}).get("cap", 0.30)

    # Feasible return range: equal-weight return to max single-asset return
    # (constrained by per-asset cap, so max = weighted combination)
    min_return = mu.min()
    max_return = mu.max()

    # Add small buffer to avoid infeasible endpoints
    targets = np.linspace(min_return * 0.95, max_return * 1.0, n_points)

    results = []
    for target in targets:
        try:
            weights, vol = optimize_for_target_return(
                target, expected_returns, cov, config
            )
            achieved_return = float(weights.values @ expected_returns.reindex(weights.index).values)
            results.append({
                "target_return": achieved_return,
                "volatility": vol,
                "weights": weights.to_dict(),
            })
        except Exception:
            continue

    if not results:
        return pd.DataFrame(columns=["target_return", "volatility", "weights"])

    frontier = pd.DataFrame(results)

    # Sort by volatility; infeasible/non-convergent targets already filtered above
    frontier = frontier.sort_values("volatility").drop_duplicates(
        subset=["volatility"], keep="last"
    ).reset_index(drop=True)

    return frontier
