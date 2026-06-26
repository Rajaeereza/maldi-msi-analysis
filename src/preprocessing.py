# ============================================================
# src/preprocessing.py
#
# Preprocessing pipeline for centroided MALDI-MSI data.
#
# Pipeline (following Bemis, Föll et al. 2023):
#   1. TIC normalisation
#   2. Peak frequency filtering (2.5% threshold)
#   3. Rebinning to reference peaks at 10 ppm
#
# Author: Reza Rajaee
# References:
#   Bemis, Föll et al. (2023) Nature Methods 20:1883
#   Deininger et al. (2011) Anal. Chem. 84:1277
#   Römpp et al. (2010) Angew. Chem. Int. Ed. 49:3834
# ============================================================

import numpy as np
import pandas as pd


def normalise_tic(spectra):
    """
    TIC (Total Ion Current) normalisation.

    Divides each pixel spectrum by its total intensity sum.
    This corrects for pixel-to-pixel variation in total signal
    caused by differences in tissue thickness, matrix crystal
    size, or laser energy absorption.

    Reference: Deininger et al. (2011) Anal. Chem. 84:1277

    Parameters
    ----------
    spectra : np.ndarray (n_pixels, n_mz)

    Returns
    -------
    np.ndarray (n_pixels, n_mz) — TIC normalised
    """
    tic = spectra.sum(axis=1, keepdims=True)
    tic = np.where(tic == 0, 1, tic)
    return spectra / tic


def filter_peaks_frequency(spectra, mz_values, min_freq=0.025):
    """
    Keep only peaks present in at least min_freq of all pixels.

    Removes noise peaks that appear in very few pixels.
    The remaining peaks become the reference peak list used
    for rebinning.

    min_freq=0.025 means keep peaks present in ≥2.5% of pixels.
    This matches the threshold used in the thesis analysis.

    Reference: Bemis, Föll et al. (2023) Nature Methods 20:1883

    Parameters
    ----------
    spectra   : np.ndarray (n_pixels, n_mz)
    mz_values : np.ndarray (n_mz,)
    min_freq  : float — minimum fraction of pixels (default 0.025)

    Returns
    -------
    filtered_spectra : np.ndarray (n_pixels, n_peaks)
    reference_mz     : np.ndarray (n_peaks,) — reference peak list
    """
    # Frequency = fraction of pixels where this m/z has intensity > 0
    frequency  = (spectra > 0).mean(axis=0)
    keep       = frequency >= min_freq

    filtered_spectra = spectra[:, keep]
    reference_mz     = mz_values[keep]

    return filtered_spectra, reference_mz


def rebin_to_reference(spectra, mz_values, reference_mz,
                        tolerance_ppm=10.0):
    """
    Rebin pixel spectra to a common set of reference m/z values.

    Even in centroided data, the same molecule may be detected
    at slightly different m/z positions in different pixels due
    to instrument calibration variation. This function aligns
    all pixels to the same reference m/z grid by assigning each
    detected peak to the nearest reference peak within tolerance.

    Uses 10 ppm tolerance matching the mass accuracy of the
    Orbitrap instrument used to acquire PXD001283.

    Reference: Bemis, Föll et al. (2023) Nature Methods 20:1883
               — Cardinal v3 uses 10 ppm binning for this dataset

    Parameters
    ----------
    spectra       : np.ndarray (n_pixels, n_mz)
    mz_values     : np.ndarray (n_mz,) — current m/z axis
    reference_mz  : np.ndarray (n_ref,) — reference peak positions
    tolerance_ppm : float — matching tolerance (default 10 ppm)

    Returns
    -------
    rebinned : np.ndarray (n_pixels, n_ref) — aligned feature matrix
    """
    n_pixels = spectra.shape[0]
    n_ref    = len(reference_mz)
    rebinned = np.zeros((n_pixels, n_ref), dtype=np.float32)

    for j, ref_mz in enumerate(reference_mz):
        # Find all m/z values within tolerance of this reference peak
        ppm_diff = np.abs(mz_values - ref_mz) / ref_mz * 1e6
        within   = ppm_diff <= tolerance_ppm

        if within.any():
            # Sum intensities within the tolerance window
            rebinned[:, j] = spectra[:, within].sum(axis=1)

    return rebinned


def run_preprocessing(spectra, mz_values,
                       min_freq=0.025,
                       tolerance_ppm=10.0,
                       verbose=True):
    """
    Run the complete preprocessing pipeline.

    Steps:
      1. TIC normalisation
      2. Peak frequency filtering (≥2.5% of pixels)
      3. Rebinning to reference peaks at 10 ppm

    Parameters
    ----------
    spectra       : np.ndarray (n_pixels, n_mz) — raw intensities
    mz_values     : np.ndarray (n_mz,) — m/z axis
    min_freq      : float — frequency filter threshold (default 0.025)
    tolerance_ppm : float — rebinning tolerance (default 10 ppm)
    verbose       : bool

    Returns
    -------
    dict with keys:
        spectra_norm     : TIC normalised (before filtering)
        spectra_preprocessed : final feature matrix
        reference_mz     : reference peak m/z values
        n_peaks          : number of peaks after filtering
    """
    if verbose:
        print("── Preprocessing pipeline ───────────────────────────")
        print(f"  Input: {spectra.shape[0]} pixels × {spectra.shape[1]} m/z values")

    # Step 1 — TIC normalisation
    if verbose:
        print("\n  [1] TIC normalisation")
        print("      Divides each spectrum by its total intensity.")
        print("      Corrects for pixel-level technical variation.")
    spectra_norm = normalise_tic(spectra)

    # Step 2 — Peak frequency filtering
    if verbose:
        print(f"\n  [2] Peak frequency filtering (threshold: {min_freq*100:.1f}%)")
        print(f"      Keeps only peaks present in ≥{min_freq*100:.1f}% of pixels.")
        print("      Removes noise peaks that appear in very few pixels.")
    filtered, reference_mz = filter_peaks_frequency(
        spectra_norm, mz_values, min_freq)
    if verbose:
        print(f"      Peaks retained: {filtered.shape[1]} / {len(mz_values)}")

    # Step 3 — Rebinning
    if verbose:
        print(f"\n  [3] Rebinning to reference peaks ({tolerance_ppm} ppm)")
        print("      Aligns m/z values across pixels to common reference.")
        print(f"      Using {tolerance_ppm} ppm — matches Orbitrap mass accuracy.")
    rebinned = rebin_to_reference(
        spectra_norm, mz_values, reference_mz, tolerance_ppm)
    if verbose:
        print(f"      Final matrix: {rebinned.shape[0]} × {rebinned.shape[1]}")
        print("\n  ✓ Preprocessing complete.")

    return {
        "spectra_norm"        : spectra_norm,
        "spectra_preprocessed": rebinned,
        "reference_mz"        : reference_mz,
        "n_peaks"             : rebinned.shape[1],
    }
