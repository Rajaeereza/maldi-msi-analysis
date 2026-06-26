# ============================================================
# src/preprocessing.py
#
# Preprocessing pipeline for centroided MALDI-MSI data.
#
# Pipeline (following thesis methodology):
#   1. Rebinning at 10 ppm  — align m/z values across pixels
#   2. TIC normalisation    — correct pixel-level variation
#   3. Frequency filtering  — keep peaks in >= 2.5% of pixels
#   4. Rebinning to reference peaks at 10 ppm
#
# Author: Reza Rajaee
# References:
#   Bemis, Föll et al. (2023) Nature Methods 20:1883
#   Deininger et al. (2011) Anal. Chem. 84:1277
#   Römpp et al. (2010) Angew. Chem. Int. Ed. 49:3834
# ============================================================

import numpy as np
import pandas as pd


def rebin_at_resolution(spectra, mz_values, tolerance_ppm=10.0):
    """
    Step 1 — Initial rebinning at 10 ppm.

    Even in centroided data, the same molecule may be detected
    at slightly different m/z positions in different pixels due
    to instrument calibration variation. This groups m/z values
    within 10 ppm tolerance into common bins, creating a
    consistent m/z axis across all pixels.

    Uses 10 ppm matching the Orbitrap mass accuracy of this instrument.

    Reference: Bemis, Föll et al. (2023) Nature Methods 20:1883
               Cardinal v3 uses 10 ppm binning for PXD001283

    Parameters
    ----------
    spectra       : np.ndarray (n_pixels, n_mz)
    mz_values     : np.ndarray (n_mz,)
    tolerance_ppm : float — binning tolerance (default 10 ppm)

    Returns
    -------
    rebinned  : np.ndarray (n_pixels, n_bins)
    bin_mz    : np.ndarray (n_bins,)
    """
    if len(mz_values) == 0:
        return spectra, mz_values

    # Group m/z values into bins
    bins = []
    cur_bin = [0]

    for j in range(1, len(mz_values)):
        ppm = abs(mz_values[j] - mz_values[cur_bin[-1]]) / mz_values[cur_bin[-1]] * 1e6
        if ppm <= tolerance_ppm:
            cur_bin.append(j)
        else:
            bins.append(cur_bin)
            cur_bin = [j]
    bins.append(cur_bin)

    # Representative m/z per bin = median
    bin_mz   = np.array([np.median(mz_values[b]) for b in bins])

    # Sum intensities within each bin
    rebinned = np.zeros((spectra.shape[0], len(bins)), dtype=np.float32)
    for k, b in enumerate(bins):
        rebinned[:, k] = spectra[:, b].sum(axis=1)

    return rebinned, bin_mz


def normalise_tic(spectra):
    """
    Step 2 — TIC (Total Ion Current) normalisation.

    Divides each pixel spectrum by its total intensity sum.
    Corrects for pixel-to-pixel variation in total signal caused
    by differences in tissue thickness, matrix crystal size,
    or laser energy absorption.

    After TIC normalisation, each spectrum sums to 1.0, making
    intensities comparable across pixels.

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
    Step 3 — Peak frequency filtering at 2.5%.

    Keeps only peaks present (intensity > 0) in at least
    min_freq fraction of all pixels. Removes noise peaks
    that appear in very few pixels and are unlikely to
    represent real molecular signals.

    The remaining peaks become the reference peak list
    used for the final rebinning step.

    min_freq=0.025 means keep peaks in >= 2.5% of pixels.

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
    frequency        : np.ndarray (n_mz,) — frequency per peak
    """
    frequency = (spectra > 0).mean(axis=0)
    keep      = frequency >= min_freq

    return spectra[:, keep], mz_values[keep], frequency


def rebin_to_reference(spectra, mz_values, reference_mz,
                        tolerance_ppm=10.0):
    """
    Step 4 — Rebinning to reference peaks.

    For each reference peak, sums all intensities within the
    tolerance window from each pixel. This produces the final
    consistent feature matrix where every pixel has the same
    set of m/z features.

    Reference: Bemis, Föll et al. (2023) Nature Methods 20:1883

    Parameters
    ----------
    spectra       : np.ndarray (n_pixels, n_mz)
    mz_values     : np.ndarray (n_mz,)
    reference_mz  : np.ndarray (n_ref,)
    tolerance_ppm : float — default 10 ppm

    Returns
    -------
    rebinned : np.ndarray (n_pixels, n_ref)
    """
    n_pixels = spectra.shape[0]
    n_ref    = len(reference_mz)
    rebinned = np.zeros((n_pixels, n_ref), dtype=np.float32)

    for j, ref_mz in enumerate(reference_mz):
        ppm_diff = np.abs(mz_values - ref_mz) / ref_mz * 1e6
        within   = ppm_diff <= tolerance_ppm
        if within.any():
            rebinned[:, j] = spectra[:, within].sum(axis=1)

    return rebinned


def run_preprocessing(spectra, mz_values,
                       tolerance_ppm=10.0,
                       min_freq=0.025,
                       verbose=True):
    """
    Run the complete preprocessing pipeline.

    Steps:
      1. Rebinning at 10 ppm
      2. TIC normalisation
      3. Frequency filtering at 2.5%
      4. Rebinning to reference peaks at 10 ppm

    Parameters
    ----------
    spectra       : np.ndarray (n_pixels, n_mz) — raw intensities
    mz_values     : np.ndarray (n_mz,) — m/z axis
    tolerance_ppm : float — rebinning tolerance (default 10 ppm)
    min_freq      : float — frequency filter (default 0.025 = 2.5%)
    verbose       : bool

    Returns
    -------
    dict with keys:
        spectra_binned        : after step 1 rebinning
        spectra_norm          : after step 2 normalisation
        spectra_preprocessed  : final feature matrix (step 4)
        reference_mz          : reference peak m/z values
        bin_mz                : m/z axis after step 1
        frequency             : peak frequency before filtering
        n_peaks               : number of peaks in final matrix
    """
    if verbose:
        print("── Preprocessing pipeline ───────────────────────────")
        print(f"  Input: {spectra.shape[0]} pixels x {spectra.shape[1]} m/z values")

    # Step 1 — Rebinning at 10 ppm
    if verbose:
        print(f"\n  [1] Rebinning at {tolerance_ppm} ppm")
        print("      Groups nearby m/z values into common bins.")
        print(f"      Ensures same m/z axis across all pixels.")
    spectra_binned, bin_mz = rebin_at_resolution(
        spectra, mz_values, tolerance_ppm)
    if verbose:
        print(f"      m/z bins after rebinning: {len(bin_mz)}")

    # Step 2 — TIC normalisation
    if verbose:
        print("\n  [2] TIC normalisation")
        print("      Divides each spectrum by its total intensity.")
        print("      Corrects for pixel-level technical variation.")
    spectra_norm = normalise_tic(spectra_binned)

    # Step 3 — Frequency filtering
    if verbose:
        print(f"\n  [3] Frequency filtering (threshold: {min_freq*100:.1f}%)")
        print(f"      Keeps peaks present in >= {min_freq*100:.1f}% of pixels.")
        print("      Removes noise peaks that appear in very few pixels.")
    filtered, reference_mz, frequency = filter_peaks_frequency(
        spectra_norm, bin_mz, min_freq)
    if verbose:
        print(f"      Peaks retained: {filtered.shape[1]} / {len(bin_mz)}")

    # Step 4 — Rebinning to reference peaks
    if verbose:
        print(f"\n  [4] Rebinning to reference peaks ({tolerance_ppm} ppm)")
        print("      Assigns each pixel intensity to nearest reference peak.")
    spectra_pp = rebin_to_reference(
        spectra_norm, bin_mz, reference_mz, tolerance_ppm)
    if verbose:
        print(f"\n  Done. Final matrix: {spectra_pp.shape[0]} x {spectra_pp.shape[1]}")

    return {
        "spectra_binned"       : spectra_binned,
        "spectra_norm"         : spectra_norm,
        "spectra_preprocessed" : spectra_pp,
        "reference_mz"         : reference_mz,
        "bin_mz"               : bin_mz,
        "frequency"            : frequency,
        "n_peaks"              : spectra_pp.shape[1],
    }
