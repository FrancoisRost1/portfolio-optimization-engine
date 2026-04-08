"""
Volatility targeting — optional post-optimization overlay.

Scales optimised weights so that the portfolio's annualised volatility
matches a target level. Capped at scale_factor <= 1.0 (no leverage
in a long-only context).
"""

import numpy as np
import pandas as pd


def apply_vol_target(
    weights: pd.Series,
    cov: pd.DataFrame,
    config: dict,
) -> tuple[pd.Series, dict]:
    """Apply volatility targeting overlay to optimised weights.

    σ_portfolio = √(w'Σw × 252)
    scale_factor = min(σ_target / σ_portfolio, max_scale)
    w_final = w_optimized × scale_factor

    Applied AFTER optimization, not baked into the objective function.

    Parameters
    ----------
    weights : pd.Series
        Optimised portfolio weights (index = tickers).
    cov : pd.DataFrame
        N x N daily covariance matrix.
    config : dict
        Full config dict.

    Returns
    -------
    tuple[pd.Series, dict]
        (adjusted weights, info dict with scale factor and vol details).
        If vol targeting is disabled, returns original weights unchanged.
    """
    vt = config.get("vol_targeting", {})
    enabled = vt.get("enabled", False)

    if not enabled:
        return weights, {"enabled": False}

    target_vol = vt.get("target_annual_vol", 0.10)
    max_scale = vt.get("max_scale", 1.0)

    tickers = weights.index
    w = weights.reindex(tickers).values
    sigma = cov.reindex(index=tickers, columns=tickers).values

    # Annualised portfolio vol
    port_var = w @ sigma @ w * 252
    port_vol = np.sqrt(max(port_var, 0))

    # Scale factor
    if port_vol > 0:
        scale = min(target_vol / port_vol, max_scale)
    else:
        # Portfolio vol is zero — keep weights as-is
        scale = 1.0

    w_scaled = weights * scale

    # Renormalise to sum to 1 (since we might have reduced exposure)
    # Note: if scale < 1, the "cash" remainder is implicit
    # For a fully-invested constraint, we keep original weights when scale >= 1
    # and proportionally scale down when < 1 (remainder is cash)

    info = {
        "enabled": True,
        "target_vol": target_vol,
        "portfolio_vol_before": port_vol,
        "portfolio_vol_after": port_vol * scale,
        "scale_factor": scale,
        "cash_weight": 1.0 - w_scaled.sum(),
    }

    return w_scaled, info
