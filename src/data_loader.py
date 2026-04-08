"""
Data loader — fetches adjusted close prices from yfinance with caching.

Handles missing data by dropping tickers with < 80% coverage and forward-
filling remaining gaps. Caches raw downloads to data/cache/ to avoid
redundant API calls within the same session.
"""

import os
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf


CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"


def fetch_prices(
    tickers: list[str],
    lookback_years: int = 10,
    price_field: str = "Adj Close",
    cache: bool = True,
) -> pd.DataFrame:
    """Download adjusted close prices for the given tickers.

    Parameters
    ----------
    tickers : list[str]
        ETF tickers to download.
    lookback_years : int
        Number of years of history to fetch.
    price_field : str
        Column name in yfinance output (default 'Adj Close').
    cache : bool
        If True, cache to disk and reuse within the same calendar day.

    Returns
    -------
    pd.DataFrame
        Columns = tickers, index = DatetimeIndex of trading dates.
        Tickers with < 80% coverage are dropped (with a warning printed).
    """
    end = datetime.today()
    start = end - timedelta(days=lookback_years * 365)

    cache_path = _cache_path(tickers, lookback_years)
    if cache and cache_path.exists():
        cached = pd.read_parquet(cache_path)
        if cached.index.max().date() >= (end - timedelta(days=3)).date():
            return cached

    # Download from yfinance
    raw = yf.download(
        tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False,
    )

    # Handle multi-level columns from yfinance
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw[price_field].copy()
    else:
        # Single ticker case
        prices = raw[[price_field]].copy()
        prices.columns = tickers[:1]

    # Ensure all requested tickers are present
    prices = prices.reindex(columns=tickers)

    # Coverage check: drop tickers with < 80% non-null values
    coverage = prices.notna().mean()
    low_coverage = coverage[coverage < 0.80].index.tolist()
    if low_coverage:
        print(f"[data_loader] WARNING: Dropping tickers with <80% coverage: {low_coverage}")
        prices = prices.drop(columns=low_coverage)

    # Forward-fill remaining gaps, then drop any leading NaNs
    prices = prices.ffill().dropna(how="any")

    # Flatten index if needed
    prices.index = pd.DatetimeIndex(prices.index)
    prices.index.name = "date"

    if cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        prices.to_parquet(cache_path)

    return prices


def fetch_market_caps(tickers: list[str]) -> pd.Series:
    """Fetch current market capitalisation for each ticker via yfinance.

    Falls back to NaN for tickers where the data is unavailable.
    Market-cap weights are normalised by the caller (optimizer_bl).

    Returns
    -------
    pd.Series
        Index = ticker, values = market cap in USD.
    """
    caps = {}
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            # ETFs report 'totalAssets' (AUM) rather than 'marketCap'
            cap = info.get("marketCap") or info.get("totalAssets")
            caps[ticker] = float(cap) if cap else float("nan")
        except Exception:
            caps[ticker] = float("nan")
    return pd.Series(caps)


def _cache_path(tickers: list[str], lookback_years: int) -> Path:
    """Deterministic cache filename based on tickers and date."""
    key = "_".join(sorted(tickers)) + f"_{lookback_years}y"
    today = datetime.today().strftime("%Y%m%d")
    return CACHE_DIR / f"prices_{key}_{today}.parquet"
