"""
Hierarchical Risk Parity (HRP), Lopez de Prado (2016).

Three steps:
  1. Cluster assets by correlation distance using hierarchical clustering.
  2. Quasi-diagonalise: reorder the covariance matrix by dendrogram leaf order.
  3. Recursive bisection: allocate inverse-variance weights within each branch.

No optimizer needed, purely algorithmic, no matrix inversion. Naturally
produces diversified, stable allocations.
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

from src.constraints import clip_to_constraints


def correlation_distance(corr: pd.DataFrame) -> np.ndarray:
    """Compute correlation-based distance matrix.

    d(i,j) = sqrt(0.5 × (1 - ρ_ij))

    Parameters
    ----------
    corr : pd.DataFrame
        N x N correlation matrix.

    Returns
    -------
    np.ndarray
        Condensed distance vector (for scipy linkage).
    """
    dist = np.sqrt(0.5 * (1 - corr.values))
    # Ensure exact symmetry and zero diagonal
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2
    return squareform(dist)


def quasi_diagonalise(link: np.ndarray, n: int) -> list[int]:
    """Reorder assets by dendrogram leaf order (quasi-diagonalisation).

    Parameters
    ----------
    link : np.ndarray
        Linkage matrix from scipy.
    n : int
        Number of original assets.

    Returns
    -------
    list[int]
        Ordered indices of assets.
    """
    return leaves_list(link).tolist()


def recursive_bisection(
    cov: np.ndarray, sorted_indices: list[int]
) -> np.ndarray:
    """Allocate weights via recursive bisection on sorted assets.

    At each split, the weight is divided between the two halves
    proportionally to the inverse of their cluster variance.

    Parameters
    ----------
    cov : np.ndarray
        N x N covariance matrix (reordered is not required, we use indices).
    sorted_indices : list[int]
        Asset indices in quasi-diagonal order.

    Returns
    -------
    np.ndarray
        Weights array of length N (original index order).
    """
    n = cov.shape[0]
    weights = np.ones(n)

    # Stack of (cluster_indices) to process
    clusters = [sorted_indices]

    while clusters:
        new_clusters = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue

            mid = len(cluster) // 2
            left = cluster[:mid]
            right = cluster[mid:]

            # Cluster variance = inverse-variance portfolio variance
            var_left = _cluster_variance(cov, left)
            var_right = _cluster_variance(cov, right)

            # Allocate proportional to inverse variance
            total_inv_var = 1.0 / var_left + 1.0 / var_right
            alpha_left = (1.0 / var_left) / total_inv_var
            alpha_right = (1.0 / var_right) / total_inv_var

            # Scale weights for each sub-cluster
            for idx in left:
                weights[idx] *= alpha_left
            for idx in right:
                weights[idx] *= alpha_right

            new_clusters.append(left)
            new_clusters.append(right)

        clusters = new_clusters

    return weights


def _cluster_variance(cov: np.ndarray, indices: list[int]) -> float:
    """Compute the variance of the inverse-variance portfolio for a cluster.

    Parameters
    ----------
    cov : np.ndarray
        Full covariance matrix.
    indices : list[int]
        Indices of assets in the cluster.

    Returns
    -------
    float
        Cluster variance (scalar).
    """
    sub_cov = cov[np.ix_(indices, indices)]
    # Inverse-variance weights within the cluster
    diag = np.diag(sub_cov)
    diag = np.where(diag <= 0, 1e-10, diag)
    inv_var = 1.0 / diag
    w = inv_var / inv_var.sum()
    return float(w @ sub_cov @ w)


def optimize_hrp(
    cov: pd.DataFrame, config: dict
) -> pd.Series:
    """Run Hierarchical Risk Parity allocation.

    Parameters
    ----------
    cov : pd.DataFrame
        N x N covariance matrix (daily frequency, annualised internally).
    config : dict
        Full config dict.

    Returns
    -------
    pd.Series
        HRP weights (index = tickers), clipped to config constraints.
    """
    tickers = cov.index.tolist()
    n = len(tickers)
    sigma = cov.values * 252  # Annualise

    hrp_config = config.get("optimization", {}).get("hrp", {})
    linkage_method = hrp_config.get("linkage_method", "single")

    # Step 1: Correlation distance and clustering
    std = np.sqrt(np.diag(sigma))
    std = np.where(std == 0, 1e-10, std)
    corr = sigma / np.outer(std, std)
    corr = np.clip(corr, -1.0, 1.0)
    corr_df = pd.DataFrame(corr, index=tickers, columns=tickers)

    dist = correlation_distance(corr_df)
    link = linkage(dist, method=linkage_method)

    # Step 2: Quasi-diagonalise
    sorted_idx = quasi_diagonalise(link, n)

    # Step 3: Recursive bisection
    raw_weights = recursive_bisection(sigma, sorted_idx)

    weights = pd.Series(raw_weights, index=tickers, name="hrp_weights")

    # Post-allocation constraint enforcement (clipping + renorm)
    weights = clip_to_constraints(weights, config)

    return weights
