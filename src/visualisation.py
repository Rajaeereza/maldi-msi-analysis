# ============================================================
# src/visualisation.py
#
# Simple plotting functions for MALDI-MSI data.
#
# Author: Reza Rajaee
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

plt.rcParams.update({
    "figure.dpi"      : 130,
    "font.size"       : 10,
    "axes.titlesize"  : 11,
    "axes.spines.top" : False,
    "axes.spines.right": False,
})

CLUSTER_COLOURS = [
    "#2d6a9f", "#e07b39", "#3a9e6f",
    "#c0392b", "#8e44ad", "#f0c330",
    "#16a085", "#7f8c8d"
]


def plot_image(image, title="", cmap="viridis",
               colorbar_label="", ax=None, show=True):
    """Plot a 2D spatial image (TIC map, ion image, segmentation map)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))

    im = ax.imshow(image, cmap=cmap,
                   interpolation="nearest", aspect="equal")
    plt.colorbar(im, ax=ax, label=colorbar_label, shrink=0.8)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")

    if show:
        plt.tight_layout()
        plt.show()
    return ax


def plot_spectrum(mz, intensities, title="",
                  colour="#2d6a9f", zoom=None, ax=None, show=True):
    """Plot a single mass spectrum."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 3.5))

    if zoom:
        mask = (mz >= zoom[0]) & (mz <= zoom[1])
        mz_plot  = mz[mask]
        int_plot = intensities[mask]
    else:
        mz_plot  = mz
        int_plot = intensities

    ax.vlines(mz_plot, ymin=0, ymax=int_plot,
              linewidth=0.6, color=colour, alpha=0.8)
    ax.set_xlabel("m/z (Da)")
    ax.set_ylabel("Intensity")
    ax.set_title(title, fontsize=10)

    if show:
        plt.tight_layout()
        plt.show()
    return ax


def plot_segmentation(labels, coordinates, title="",
                       ax=None, show=True):
    """Plot a pixel segmentation map coloured by cluster label."""
    from src.load import reconstruct_image

    x = coordinates["x"].values.astype(int) - coordinates["x"].min()
    y = coordinates["y"].values.astype(int) - coordinates["y"].min()
    seg_map = np.full((y.max()+1, x.max()+1), np.nan)
    seg_map[y, x] = labels

    n_clusters = len(np.unique(labels[~np.isnan(labels)]))
    cmap   = mcolors.ListedColormap(CLUSTER_COLOURS[:n_clusters])
    bounds = np.arange(-0.5, n_clusters + 0.5, 1)
    norm   = mcolors.BoundaryNorm(bounds, cmap.N)

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))

    ax.imshow(seg_map, cmap=cmap, norm=norm,
              interpolation="nearest", aspect="equal")

    patches = [mpatches.Patch(color=CLUSTER_COLOURS[i],
                               label=f"Region {i}")
               for i in range(n_clusters)]
    ax.legend(handles=patches, loc="upper right", fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")

    if show:
        plt.tight_layout()
        plt.show()
    return ax
