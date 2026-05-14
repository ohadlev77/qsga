from typing import Iterable
from numpy.typing import ArrayLike

import numpy as np

from scipy.spatial import distance
from scipy.stats import ks_2samp, wasserstein_distance


def compare_weight_distributions(
    w1: ArrayLike | Iterable[float],
    w2: ArrayLike | Iterable[float]
) -> dict[str, float]:
    """
    Compare two equally sized weight samples with basic distributional distances.

    Args:
        w1: First set of weights (array-like); flattened internally via ``ravel()``.
        w2: Second set of weights (array-like); must have the same size as ``w1``.

    Returns:
        Dictionary with:
            - ks_stat: Kolmogorov-Smirnov statistic.
            - ks_pvalue: p-value for KS test of identical distributions.
            - wasserstein: 1D earth-mover distance between samples.
            - mean_diff: Difference in sample means (w1 - w2).
            - std_diff: Difference in sample standard deviations (w1 - w2).

    Notes:
        ``numpy.ravel`` flattens the input to 1D (view when possible, copy otherwise),
        so multi-dimensional arrays are treated as simple samples.
    """

    w1 = np.asarray(w1, dtype=float).ravel()
    w2 = np.asarray(w2, dtype=float).ravel()

    if w1.size != w2.size:
        raise ValueError("Weight arrays must have the same size")

    ks_stat, ks_p = ks_2samp(w1, w2)
    wdist = wasserstein_distance(w1, w2)
    return {
        "ks_stat": float(ks_stat),
        "ks_pvalue": float(ks_p),
        "wasserstein": float(wdist),
        "mean_diff": float(w1.mean() - w2.mean()),
        "std_diff": float(w1.std(ddof=1) - w2.std(ddof=1)),
    }


def compare_hermitian_spectra_different_metrics(
    spectrum_a: np.ndarray, 
    spectrum_b: np.ndarray
) -> dict[str, float]:
    """
    Computes similarity/distance measures between eigenvalue spectra
    based on 'A Guide to Similarity Measures' (2024).
    """
    # Ensure vectors are same length and sorted (standard for spectra)
    p = np.sort(spectrum_a)
    q = np.sort(spectrum_b)
    
    # 1. Euclidean Distance (L2) 
    # Measures the physical distance between the two spectral curves.
    # Formula: sqrt(sum(|Pi - Qi|^2))
    l2_dist = np.linalg.norm(p - q)
    
    # 2. Cosine Similarity 
    # Measures the alignment of the spectral trends/shapes.
    # Formula: <P,Q> / (||P|| * ||Q||)
    cos_sim = 1 - distance.cosine(p, q)
    
    # 3. Canberra Distance 
    # Sensitive to differences in smaller eigenvalues (values near zero).
    # Formula: sum(|Pi - Qi| / (|Pi| + |Qi|))
    can_dist = distance.canberra(p, q)
    
    # 4. Jensen-Shannon Divergence (JSD) 
    # Measures the distributional difference.
    # Requires normalization to form a PDF (sum = 1) [cite: 174-176].
    p_pdf = p / np.sum(p)
    q_pdf = q / np.sum(q)
    js_div = distance.jensenshannon(p_pdf, q_pdf)**2  # Scipy returns sqrt(JSD)
    
    return {
        "euclidean_distance_l2": float(l2_dist),
        "cosine_similarity": float(cos_sim),
        "canberra_distance": float(can_dist),
        "jensen_shannon_divergence": float(js_div)
    }