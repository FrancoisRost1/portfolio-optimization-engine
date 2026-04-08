"""
Covariance estimation — three methods:
  1. Sample covariance (baseline only — noisy, unstable inverse)
  2. Ledoit-Wolf shrinkage (default — α-weighted blend of sample + structured target)
  3. RMT / Marchenko-Pastur cleaning (advanced — removes noise eigenvalues)
"""

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def sample_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    """Compute the sample covariance matrix.

    Σ_sample = (1 / (T-1)) × X'X where X = demeaned returns.

    Known problems: noisy eigenvalues, unstable inverse, poor out-of-sample.
    Used ONLY as a comparison baseline — never as default.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily log returns, columns = tickers.

    Returns
    -------
    pd.DataFrame
        N x N covariance matrix with ticker labels.
    """
    cov = returns.cov()
    return cov


def ledoit_wolf_covariance(returns: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Compute the Ledoit-Wolf shrinkage covariance matrix.

    Σ_LW = α × F + (1 - α) × Σ_sample
    where F = scaled identity target (μ × I, sklearn implementation),
    α = optimal shrinkage intensity (Ledoit-Wolf 2004).

    Note: sklearn's LedoitWolf shrinks toward a scaled identity matrix,
    not the constant-correlation target from the original paper.
    This is the DEFAULT covariance for all optimizers.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily log returns, columns = tickers.

    Returns
    -------
    tuple[pd.DataFrame, float]
        (N x N shrunk covariance matrix, shrinkage intensity α).
    """
    tickers = returns.columns
    lw = LedoitWolf().fit(returns.values)
    cov_matrix = pd.DataFrame(lw.covariance_, index=tickers, columns=tickers)
    shrinkage = lw.shrinkage_
    return cov_matrix, shrinkage


def rmt_covariance(returns: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    """Compute covariance with Random Matrix Theory (Marchenko-Pastur) cleaning.

    The cleaning is applied to the CORRELATION matrix (standardised), not the
    raw covariance — this is the textbook Marchenko-Pastur procedure:
      1. Standardise sample cov to correlation C
      2. Eigendecompose C
      3. Estimate noise variance σ² = median eigenvalue of C
      4. Compute MP upper bound: λ_max = σ²(1 + √(N/T))²
      5. Replace eigenvalues ≤ λ_max with their average (noise removal)
      6. Reconstruct cleaned correlation, convert back to covariance
      7. Preserve trace (total variance) by rescaling

    Parameters
    ----------
    returns : pd.DataFrame
        Daily log returns, columns = tickers.
    config : dict
        RMT configuration from config.yaml['covariance']['rmt'].

    Returns
    -------
    tuple[pd.DataFrame, dict]
        (N x N cleaned covariance matrix, info dict with cleaning details).
    """
    tickers = returns.columns
    T, N = returns.shape
    q = N / T  # ratio of assets to observations

    # Sample covariance and standard deviations
    cov_sample = returns.cov().values
    original_trace = np.trace(cov_sample)
    std_devs = np.sqrt(np.diag(cov_sample))
    std_devs = np.where(std_devs == 0, 1e-10, std_devs)

    # Standardise to correlation matrix for MP cleaning
    corr = cov_sample / np.outer(std_devs, std_devs)
    corr = np.clip(corr, -1.0, 1.0)

    # Eigendecomposition of correlation matrix
    eigenvalues, eigenvectors = np.linalg.eigh(corr)

    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Noise variance estimate: median eigenvalue of correlation matrix
    # For a standardised matrix, σ² ≈ 1 under pure noise, but we use median
    # for robustness when signal eigenvalues are present
    sigma2 = float(np.median(eigenvalues))

    # Marchenko-Pastur bounds (applied to correlation eigenvalues)
    lambda_max = sigma2 * (1 + np.sqrt(q)) ** 2
    lambda_min = sigma2 * (1 - np.sqrt(q)) ** 2

    # Identify noise eigenvalues (those within the MP band)
    noise_mask = eigenvalues <= lambda_max
    n_cleaned = int(noise_mask.sum())

    # Replace noise eigenvalues with their average
    eigenvalue_floor = config.get("eigenvalue_floor", 1e-10)
    if n_cleaned > 0:
        noise_avg = max(eigenvalues[noise_mask].mean(), eigenvalue_floor)
        eigenvalues_clean = eigenvalues.copy()
        eigenvalues_clean[noise_mask] = noise_avg
    else:
        eigenvalues_clean = eigenvalues.copy()

    # Floor any negative values
    eigenvalues_clean = np.maximum(eigenvalues_clean, eigenvalue_floor)

    # Reconstruct cleaned correlation matrix
    corr_clean = eigenvectors @ np.diag(eigenvalues_clean) @ eigenvectors.T
    corr_clean = (corr_clean + corr_clean.T) / 2  # Ensure symmetry

    # Convert back to covariance: Σ_clean = D × C_clean × D
    cov_clean = np.outer(std_devs, std_devs) * corr_clean

    # Preserve trace (total variance) if configured
    if config.get("preserve_trace", True):
        clean_trace = np.trace(cov_clean)
        if clean_trace > 0:
            cov_clean *= original_trace / clean_trace

    info = {
        "lambda_max": lambda_max,
        "lambda_min": lambda_min,
        "sigma2": sigma2,
        "n_cleaned": n_cleaned,
        "n_total": N,
        "q_ratio": q,
        "eigenvalues_raw": eigenvalues,
        "eigenvalues_clean": eigenvalues_clean,
    }

    return pd.DataFrame(cov_clean, index=tickers, columns=tickers), info


def estimate_covariance(
    returns: pd.DataFrame, method: str, config: dict
) -> tuple[pd.DataFrame, dict]:
    """Dispatch to the requested covariance estimation method.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily log returns.
    method : str
        One of 'sample', 'ledoit_wolf', 'rmt'.
    config : dict
        Full config dict (used for RMT parameters).

    Returns
    -------
    tuple[pd.DataFrame, dict]
        (covariance matrix, metadata dict).
        Metadata keys vary by method:
          - sample: {}
          - ledoit_wolf: {'shrinkage_intensity': float}
          - rmt: {'lambda_max', 'lambda_min', 'n_cleaned', ...}
    """
    if method == "sample":
        cov = sample_covariance(returns)
        return cov, {}

    elif method == "ledoit_wolf":
        cov, alpha = ledoit_wolf_covariance(returns)
        return cov, {"shrinkage_intensity": alpha}

    elif method == "rmt":
        rmt_config = config.get("covariance", {}).get("rmt", {})
        cov, info = rmt_covariance(returns, rmt_config)
        return cov, info

    else:
        raise ValueError(f"Unknown covariance method '{method}'. Use: sample, ledoit_wolf, rmt")


def covariance_to_correlation(cov: pd.DataFrame) -> pd.DataFrame:
    """Convert a covariance matrix to a correlation matrix.

    Parameters
    ----------
    cov : pd.DataFrame
        N x N covariance matrix.

    Returns
    -------
    pd.DataFrame
        N x N correlation matrix.
    """
    std = np.sqrt(np.diag(cov.values))
    # Avoid division by zero
    std = np.where(std == 0, 1e-10, std)
    corr = cov.values / np.outer(std, std)
    # Clip to [-1, 1] for numerical safety
    corr = np.clip(corr, -1.0, 1.0)
    return pd.DataFrame(corr, index=cov.index, columns=cov.columns)
