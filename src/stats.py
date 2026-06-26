# ============================================================
# src/stats.py
#
# Statistical analysis functions for MALDI-MSI data.
#
# Functions:
#   - differential_abundance : find enriched peaks per region
#   - fdr_correction         : Benjamini-Hochberg FDR correction
#   - colocalization         : spatial correlation between peaks
#
# Author: Reza Rajaee
# References:
#   Bemis, Föll et al. (2023) Nature Methods 20:1883
#   Benjamini & Hochberg (1995) J. R. Stat. Soc. B 57:289
# ============================================================

import numpy as np
import pandas as pd
from scipy import stats


def fdr_correction(p_values, alpha=0.05):
    """
    Benjamini-Hochberg FDR correction for multiple testing.

    When testing thousands of m/z features simultaneously,
    we expect many false positives by chance. FDR correction
    controls the expected proportion of false discoveries.

    Reference: Benjamini & Hochberg (1995) J. R. Stat. Soc. B 57:289

    Parameters
    ----------
    p_values : np.ndarray — raw p-values
    alpha    : float — FDR threshold (default 0.05)

    Returns
    -------
    rejected  : np.ndarray bool — True if significant after correction
    p_adjusted: np.ndarray — adjusted p-values
    """
    n = len(p_values)
    if n == 0:
        return np.array([]), np.array([])

    # Sort p-values
    order     = np.argsort(p_values)
    rank      = np.empty_like(order)
    rank[order] = np.arange(1, n + 1)

    # BH adjusted p-values
    p_adjusted = p_values * n / rank
    # Enforce monotonicity (from right to left)
    p_adjusted = np.minimum.accumulate(p_adjusted[::-1])[::-1]
    p_adjusted = np.minimum(p_adjusted, 1.0)

    rejected = p_adjusted <= alpha
    return rejected, p_adjusted


def differential_abundance(spectra, labels, reference_mz,
                             alpha=0.05, verbose=True):
    """
    Find m/z peaks significantly enriched in each tissue region.

    For each cluster vs all other clusters, performs a
    Mann-Whitney U test on peak intensities. Applies
    Benjamini-Hochberg FDR correction across all tests.

    Mann-Whitney U test is used because:
      - MALDI intensities are not normally distributed
      - It is robust to outliers
      - It tests whether intensities in one group tend to be
        higher than in the other group

    Reference: Bemis, Föll et al. (2023) Nature Methods 20:1883

    Parameters
    ----------
    spectra      : np.ndarray (n_pixels, n_peaks)
    labels       : np.ndarray (n_pixels,) — cluster assignments
    reference_mz : np.ndarray (n_peaks,) — m/z values
    alpha        : float — FDR threshold

    Returns
    -------
    pd.DataFrame with columns:
        mz, cluster, fold_change, p_value, p_adjusted, significant
    """
    clusters = np.unique(labels)
    results  = []

    if verbose:
        print(f"  Testing {len(reference_mz)} peaks across "
              f"{len(clusters)} clusters...")

    for cluster in clusters:
        mask_in  = labels == cluster
        mask_out = labels != cluster

        group_in  = spectra[mask_in]
        group_out = spectra[mask_out]

        p_values     = []
        fold_changes = []

        for j in range(spectra.shape[1]):
            # Mann-Whitney U test
            u_stat, p = stats.mannwhitneyu(
                group_in[:, j], group_out[:, j],
                alternative="greater"  # test if cluster has HIGHER intensity
            )
            p_values.append(p)

            # Fold change: mean intensity in cluster vs mean in rest
            mean_in  = group_in[:, j].mean()
            mean_out = group_out[:, j].mean()
            fc = mean_in / (mean_out + 1e-10)
            fold_changes.append(fc)

        p_values     = np.array(p_values)
        fold_changes = np.array(fold_changes)

        # FDR correction
        rejected, p_adj = fdr_correction(p_values, alpha=alpha)

        for j in range(len(reference_mz)):
            results.append({
                "mz"          : reference_mz[j],
                "cluster"     : cluster,
                "fold_change" : round(fold_changes[j], 4),
                "p_value"     : p_values[j],
                "p_adjusted"  : p_adj[j],
                "significant" : rejected[j],
            })

    df = pd.DataFrame(results)

    if verbose:
        for c in clusters:
            n_sig = df[(df["cluster"] == c) & df["significant"]].shape[0]
            print(f"  Cluster {c}: {n_sig} significant peaks (FDR < {alpha})")

    return df


def colocalization(spectra, reference_mz, top_n=20):
    """
    Compute spatial colocalization between peaks.

    Two peaks are colocalized if their spatial distributions
    are correlated — i.e. they tend to appear together in
    the same pixels. This indicates they may be part of the
    same biological process or molecule class.

    Computes Pearson correlation between all pairs of peak
    ion images.

    Reference: Bemis, Föll et al. (2023) Nature Methods 20:1883

    Parameters
    ----------
    spectra      : np.ndarray (n_pixels, n_peaks)
    reference_mz : np.ndarray (n_peaks,)
    top_n        : int — number of top peaks to include

    Returns
    -------
    pd.DataFrame — correlation matrix (top_n × top_n)
    """
    # Use top_n most abundant peaks for clarity
    mean_intensity = spectra.mean(axis=0)
    top_indices    = np.argsort(mean_intensity)[-top_n:][::-1]

    top_spectra = spectra[:, top_indices]
    top_mz      = reference_mz[top_indices]

    corr_matrix = np.corrcoef(top_spectra.T)

    labels = [f"{mz:.1f}" for mz in top_mz]
    return pd.DataFrame(corr_matrix, index=labels, columns=labels)
