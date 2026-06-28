# ============================================================
# src/stats.py
#
# Statistical analysis functions for MALDI-MSI data.
#
# Functions:
#   - fdr_correction         : Benjamini-Hochberg FDR correction
#   - differential_abundance : find enriched peaks per region (one-vs-rest)
#   - colocalization         : spatial correlation between peaks
#
# Note on differential_abundance:
#   Implements one-vs-rest testing. For datasets with strongly unequal
#   region sizes, the dominant region may produce trivially significant
#   results. Use pairwise testing in the notebook (see Notebook 03) to
#   avoid this. This function is retained for completeness.
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

    When testing hundreds of m/z features simultaneously,
    we expect many false positives by chance. FDR correction
    controls the expected proportion of false discoveries.

    Implementation follows the original BH (1995) procedure:
    1. Sort p-values in ascending order
    2. Compute adjusted p-values: p_adj[i] = p[i] * n / rank[i]
    3. Enforce monotonicity (from largest rank to smallest)
    4. Map adjusted p-values back to original order

    Reference: Benjamini & Hochberg (1995) J. R. Stat. Soc. B 57:289

    Parameters
    ----------
    p_values : np.ndarray — raw p-values
    alpha    : float — FDR threshold (default 0.05)

    Returns
    -------
    rejected   : np.ndarray bool — True if significant after correction
    p_adjusted : np.ndarray — BH-adjusted p-values
    """
    n = len(p_values)
    if n == 0:
        return np.array([]), np.array([])

    # Sort p-values ascending
    order    = np.argsort(p_values)
    p_sorted = p_values[order]

    # BH adjustment in sorted order
    ranks         = np.arange(1, n + 1)
    p_adj_sorted  = p_sorted * n / ranks

    # Enforce monotonicity from largest rank to smallest
    # (ensures p_adj[i] <= p_adj[i+1] after mapping back)
    p_adj_sorted = np.minimum.accumulate(p_adj_sorted[::-1])[::-1]
    p_adj_sorted = np.minimum(p_adj_sorted, 1.0)

    # Map back to original order
    p_adjusted          = np.empty(n)
    p_adjusted[order]   = p_adj_sorted

    rejected = p_adjusted <= alpha
    return rejected, p_adjusted


def differential_abundance(spectra, labels, reference_mz,
                             alpha=0.05, verbose=True):
    """
    Find m/z peaks significantly enriched in each tissue region.

    One-vs-rest approach: for each cluster, tests each peak against
    all other pixels combined using Mann-Whitney U test.

    Note: for datasets with strongly unequal region sizes, the dominant
    region may produce trivially significant results (all peaks significant).
    In such cases, use pairwise testing instead — see Notebook 03 for
    the pairwise implementation.

    Mann-Whitney U test is used because:
      - MALDI intensities are not normally distributed
      - It is robust to outliers
      - It tests whether intensities in one group tend to be higher

    Reference: Bemis, Föll et al. (2023) Nature Methods 20:1883

    Parameters
    ----------
    spectra      : np.ndarray (n_pixels, n_peaks)
    labels       : np.ndarray (n_pixels,) — cluster assignments
    reference_mz : np.ndarray (n_peaks,) — m/z values
    alpha        : float — FDR threshold (default 0.05)
    verbose      : bool

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
            u_stat, p = stats.mannwhitneyu(
                group_in[:, j], group_out[:, j],
                alternative="greater"
            )
            p_values.append(p)

            mean_in  = group_in[:, j].mean()
            mean_out = group_out[:, j].mean()
            fc = mean_in / (mean_out + 1e-10)
            fold_changes.append(fc)

        p_values     = np.array(p_values)
        fold_changes = np.array(fold_changes)

        rejected, p_adj = fdr_correction(p_values, alpha=alpha)

        for j in range(len(reference_mz)):
            results.append({
                "mz"          : reference_mz[j],
                "cluster"     : cluster,
                "fold_change" : round(fold_changes[j], 4),
                "p_value"     : p_values[j],
                "p_adjusted"  : p_adj[j],
                "significant" : bool(rejected[j]),
            })

    df = pd.DataFrame(results)

    if verbose:
        for c in clusters:
            n_sig = df[(df["cluster"] == c) & df["significant"]].shape[0]
            print(f"  Cluster {c}: {n_sig} significant peaks (FDR < {alpha})")

    return df


def colocalization(spectra, reference_mz, top_n=15):
    """
    Compute spatial colocalization between peaks.

    Two peaks are colocalized if their spatial distributions
    are correlated — they tend to appear together in the same pixels.
    This indicates they may belong to the same lipid class, metabolic
    pathway, or cell type.

    Computes Pearson correlation between all pairs of peak ion images
    for the top_n most abundant peaks.

    Important limitation: global Pearson correlation is partly driven
    by tissue composition. If the dominant region has high intensity for
    two peaks, they will correlate simply due to co-abundance in that
    region — not necessarily because they are biologically co-regulated.
    For mechanistic interpretation, within-region colocalization is
    more appropriate.

    Reference: Bemis, Föll et al. (2023) Nature Methods 20:1883

    Parameters
    ----------
    spectra      : np.ndarray (n_pixels, n_peaks)
    reference_mz : np.ndarray (n_peaks,)
    top_n        : int — number of top abundant peaks to include (default 15)

    Returns
    -------
    pd.DataFrame — Pearson correlation matrix (top_n × top_n)
    """
    mean_intensity = spectra.mean(axis=0)
    top_indices    = np.argsort(mean_intensity)[-top_n:][::-1]

    top_spectra = spectra[:, top_indices]
    top_mz      = reference_mz[top_indices]

    corr_matrix = np.corrcoef(top_spectra.T)

    labels = [f"{mz:.1f}" for mz in top_mz]
    return pd.DataFrame(corr_matrix, index=labels, columns=labels)
