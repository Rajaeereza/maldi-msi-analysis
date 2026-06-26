# ============================================================
# src/load.py
#
# Functions for loading MALDI-MSI data from imzML format.
#
# Author: Reza Rajaee
# Reference: Römpp et al. (2010) Angew. Chem. Int. Ed. 49:3834
# ============================================================

import numpy as np
import pandas as pd
from pyimzml.ImzMLParser import ImzMLParser
from pathlib import Path


def load_imzml(filepath, mz_min=400, mz_max=1000, verbose=True):
    """
    Load a MALDI-MSI dataset from an imzML file.

    The imzML format stores MALDI-MSI data as two files:
      - .imzML : XML metadata (pixel coordinates, m/z values)
      - .ibd   : binary intensity data

    For the mouse bladder dataset (PXD001283), data is already
    centroided — peaks are pre-detected by the instrument,
    baseline is removed, and spectra are clean.

    Parameters
    ----------
    filepath : str — path to the .imzML file
    mz_min   : float — minimum m/z value to keep (default 400 Da)
    mz_max   : float — maximum m/z value to keep (default 1000 Da)
    verbose  : bool — print progress

    Returns
    -------
    dict with keys:
        spectra     : np.ndarray (n_pixels, n_mz) — intensity matrix
        mz_values   : np.ndarray (n_mz,) — m/z axis
        coordinates : pd.DataFrame — x, y pixel coordinates
        n_pixels    : int
        n_mz        : int
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    parser = ImzMLParser(filepath)
    n_pixels = len(parser.coordinates)

    if verbose:
        print(f"Loading: {filepath}")
        print(f"  Pixels: {n_pixels}")

    # Load first spectrum to get m/z axis
    mz_first, _ = parser.getspectrum(0)
    mz_first     = np.array(mz_first)

    # Apply m/z range filter
    mz_mask  = (mz_first >= mz_min) & (mz_first <= mz_max)
    mz_axis  = mz_first[mz_mask]
    n_mz     = mz_mask.sum()

    if verbose:
        print(f"  m/z range: {mz_min}–{mz_max} Da")
        print(f"  m/z values after range filter: {n_mz}")

    # Load all spectra
    spectra = np.zeros((n_pixels, n_mz), dtype=np.float32)
    coords  = []

    for i in range(n_pixels):
        mz_i, int_i = parser.getspectrum(i)
        mz_i  = np.array(mz_i)
        int_i = np.array(int_i, dtype=np.float32)

        # Apply same m/z range filter
        mask_i = (mz_i >= mz_min) & (mz_i <= mz_max)
        int_filtered = int_i[mask_i]

        # Handle case where pixel has different m/z length
        if len(int_filtered) == n_mz:
            spectra[i] = int_filtered
        else:
            # Interpolate to common m/z axis
            spectra[i] = np.interp(mz_axis, mz_i[mask_i], int_filtered,
                                    left=0, right=0)

        coords.append(parser.coordinates[i][:2])  # x, y only

        if verbose and (i + 1) % 5000 == 0:
            print(f"  Loaded {i+1}/{n_pixels} pixels...")

    coordinates = pd.DataFrame(coords, columns=["x", "y"])

    if verbose:
        print(f"  Done. Matrix shape: {spectra.shape}")
        print(f"  Memory: {spectra.nbytes / 1e6:.1f} MB")

    return {
        "spectra"     : spectra,
        "mz_values"   : mz_axis,
        "coordinates" : coordinates,
        "n_pixels"    : n_pixels,
        "n_mz"        : n_mz,
    }


def reconstruct_image(values, coordinates):
    """
    Reconstruct a 2D spatial image from flat pixel values.

    Parameters
    ----------
    values      : np.ndarray (n_pixels,) — one value per pixel
    coordinates : pd.DataFrame — x, y pixel coordinates

    Returns
    -------
    np.ndarray (height, width) — 2D image, NaN where no pixel
    """
    x = coordinates["x"].values.astype(int) - coordinates["x"].min()
    y = coordinates["y"].values.astype(int) - coordinates["y"].min()

    image = np.full((y.max() + 1, x.max() + 1), np.nan)
    image[y, x] = values
    return image
