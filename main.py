"""
Portfolio Optimization Engine — main orchestrator.

Loads config, fetches data, runs all optimizers and backtests,
and produces results. No logic lives here — it delegates to modules.
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np

from utils.config_loader import load_config
from src.data_loader import fetch_prices, fetch_market_caps
from src.returns import log_returns, arithmetic_returns
from src.covariance import estimate_covariance
from src.expected_returns import estimate_expected_returns
from src.optimizer_mv import optimize_mean_variance
from src.optimizer_rp import optimize_risk_parity
from src.optimizer_hrp import optimize_hrp
from src.optimizer_bl import optimize_black_litterman
from src.vol_target import apply_vol_target
from src.efficient_frontier import generate_efficient_frontier
from src.backtester import run_all_backtests
from src.benchmarks import compute_all_benchmarks
from src.metrics import compute_all_metrics
from src.risk_decomposition import (
    risk_contribution,
    herfindahl_index,
    decompose_all_methods,
)


def run_pipeline() -> dict:
    """Execute the full portfolio optimization pipeline.

    Steps:
      1. Load config
      2. Fetch prices + market caps
      3. Compute returns and covariance
      4. Run all 4 optimizers (point-in-time, latest window)
      5. Compute risk decomposition
      6. Generate efficient frontier
      7. Run rolling backtests for all methods
      8. Compute benchmarks
      9. Compute metrics for all methods + benchmarks

    Returns
    -------
    dict
        Full results bundle with keys:
          config, prices, returns, covariance, cov_meta,
          weights (dict of method -> weights),
          risk_decomp, frontier, backtest_results,
          benchmark_returns, metrics_table
    """
    config = load_config()

    # --- Data ---
    tickers = config["universe"]["tickers"]
    lookback = config["data"]["lookback_years"]
    price_field = config["data"]["price_field"]

    print(f"[main] Fetching {len(tickers)} tickers, {lookback}y lookback...")
    prices = fetch_prices(tickers, lookback, price_field)
    tickers = prices.columns.tolist()  # May have dropped low-coverage tickers
    print(f"[main] Prices loaded: {prices.shape[0]} days, {len(tickers)} assets")

    # Market caps for Black-Litterman
    print("[main] Fetching market caps...")
    raw_caps = fetch_market_caps(tickers)
    bl_config = config.get("black_litterman", {})
    if raw_caps.isna().all():
        print("[main] WARNING: All market caps failed — using manual fallback weights")
        manual = bl_config.get("manual_market_weights", {})
        market_weights = pd.Series(manual).reindex(tickers).fillna(1.0 / len(tickers))
    else:
        market_weights = raw_caps.fillna(raw_caps.median())
    market_weights = market_weights / market_weights.sum()

    # --- Returns & Covariance ---
    log_ret = log_returns(prices)
    arith_ret = arithmetic_returns(prices)
    cov_method = config["covariance"]["default_method"]
    print(f"[main] Estimating covariance ({cov_method})...")
    cov, cov_meta = estimate_covariance(log_ret, cov_method, config)

    # --- Expected Returns ---
    mu_method = config["expected_returns"]["default_method"]
    mu = estimate_expected_returns(arith_ret, mu_method, config, cov, market_weights)

    # --- Point-in-time Optimizations (latest data) ---
    print("[main] Running optimizers...")
    weights = {}

    weights["Mean-Variance"] = optimize_mean_variance(mu, cov, config)
    weights["Risk Parity"] = optimize_risk_parity(cov, config)
    weights["HRP"] = optimize_hrp(cov, config)
    bl_weights, bl_info = optimize_black_litterman(cov, market_weights, config)
    weights["Black-Litterman"] = bl_weights

    # Apply vol targeting if enabled
    for name in list(weights.keys()):
        weights[name], _ = apply_vol_target(weights[name], cov, config)

    # --- Risk Decomposition ---
    print("[main] Computing risk decomposition...")
    risk_decomp = decompose_all_methods(weights, cov)
    herfindahl = {name: herfindahl_index(w) for name, w in weights.items()}

    # --- Efficient Frontier ---
    print("[main] Generating efficient frontier...")
    frontier = generate_efficient_frontier(mu, cov, config)

    # --- Rolling Backtests ---
    print("[main] Running rolling backtests (this may take a few minutes)...")
    backtest_results = run_all_backtests(prices, config, market_weights)

    # --- Benchmarks ---
    print("[main] Computing benchmarks...")
    benchmark_returns = compute_all_benchmarks(prices, cov, config)

    # --- Metrics ---
    print("[main] Computing performance metrics...")
    spy_returns = benchmark_returns.get("SPY B&H")

    metrics_table = {}
    for name, bt in backtest_results.items():
        label = {"mv": "Mean-Variance", "rp": "Risk Parity", "hrp": "HRP", "bl": "Black-Litterman"}[name]
        if len(bt["returns"]) > 0:
            metrics_table[label] = compute_all_metrics(bt["returns"], config, spy_returns)

    for name, ret in benchmark_returns.items():
        if len(ret) > 0:
            metrics_table[name] = compute_all_metrics(ret, config, spy_returns)

    print("[main] Pipeline complete.")

    return {
        "config": config,
        "prices": prices,
        "log_returns": log_ret,
        "arith_returns": arith_ret,
        "covariance": cov,
        "cov_meta": cov_meta,
        "expected_returns": mu,
        "market_weights": market_weights,
        "weights": weights,
        "bl_info": bl_info,
        "risk_decomp": risk_decomp,
        "herfindahl": herfindahl,
        "frontier": frontier,
        "backtest_results": backtest_results,
        "benchmark_returns": benchmark_returns,
        "metrics_table": metrics_table,
    }


if __name__ == "__main__":
    results = run_pipeline()

    # Print summary
    print("\n" + "=" * 60)
    print("PORTFOLIO OPTIMIZATION RESULTS")
    print("=" * 60)

    print("\n--- Current Weights ---")
    for name, w in results["weights"].items():
        print(f"\n{name}:")
        for ticker, weight in w.items():
            if weight > 0.001:
                print(f"  {ticker}: {weight:.1%}")

    print("\n--- Performance Metrics ---")
    metrics_df = pd.DataFrame(results["metrics_table"]).T
    if not metrics_df.empty:
        for col in metrics_df.columns:
            if "Ratio" in col or "CAGR" in col or "Vol" in col or "Drawdown" in col or "CVaR" in col or "Error" in col:
                metrics_df[col] = metrics_df[col].map(lambda x: f"{x:.2%}" if not pd.isna(x) else "N/A")
            else:
                metrics_df[col] = metrics_df[col].map(lambda x: f"{x:.3f}" if not pd.isna(x) else "N/A")
        print(metrics_df.to_string())

    print("\n--- Herfindahl Index (concentration) ---")
    for name, h in results["herfindahl"].items():
        print(f"  {name}: {h:.4f}")
