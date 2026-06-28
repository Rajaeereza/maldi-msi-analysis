# MALDI-MSI Analysis — Mouse Urinary Bladder

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

This repository demonstrates a complete MALDI-MSI data analysis pipeline
on a public benchmark dataset — from raw data loading to statistical
interpretation of tissue regions. It is part of a private tutoring
material collection.

---

## Dataset

Mouse urinary bladder — Römpp et al. (2010) *Angew. Chem. Int. Ed.* 49:3834  
PRIDE: [PXD001283](https://www.ebi.ac.uk/pride/archive/projects/PXD001283)  
Instrument: AP-SMALDI LTQ Orbitrap, 10 µm spatial resolution  
Format: Centroided imzML — 34,840 pixels × ~153 peaks after preprocessing

---

## Analysis pipeline

**Notebook 01 — Data Exploration and Preprocessing**
- What is MALDI-MSI data and how it is structured
- TIC map and mean spectrum
- Pixel-level comparison (centre vs periphery)
- Four-step preprocessing: 10 ppm rebinning → TIC normalisation →
  2.5% frequency filtering → rebinning to reference peaks
- Step-by-step preprocessing visualisation with diagnostic figures
- Top peaks per lipid class (LPC, PE, PC, TAG)
- PCA before and after preprocessing (scree plot + spatial colouring)
- t-SNE before and after preprocessing

**Notebook 02 — Tissue Segmentation**
- Choosing k: elbow curve + silhouette score + biological evidence
- K-means, GMM (tied covariance), Spectral Clustering (n_neighbors=50)
- Pairwise ARI comparison with Hungarian label alignment
- Agreement map — where all three methods consistently agree
- Pairwise disagreement maps
- Difference spectrum at tissue boundaries
- PCA and t-SNE coloured by cluster assignments

**Notebook 03 — Statistical Analysis**
- Pairwise Mann-Whitney U tests (not one-vs-rest) per region pair
- Benjamini-Hochberg FDR correction
- Marker deduplication (100 ppm minimum spacing)
- Volcano plots — one per region vs most contrasting partner
- Top 3 marker ion images per region
- Mean spectra per region with peak annotation
- Global colocalization analysis (top 15 peaks, Pearson correlation)

---

## Key methodological decisions

- **No StandardScaler before clustering or PCA** — TIC normalisation
  already makes intensities comparable; scaling would equalise noise
  and signal
- **PCA input for clustering** — first 10 PCs used as clustering input,
  not raw 153-dimensional feature matrix
- **Pairwise differential abundance** — avoids trivially significant
  results for the dominant tissue region (64.6% of pixels)
- **Hungarian algorithm for label alignment** — required before any
  pixel-level agreement comparison between segmentation methods

---

## How to run

1. Open in GitHub Codespaces
2. Install dependencies:
```bash
pip install pyimzml scikit-learn seaborn matplotlib pandas scipy numpy -q
```
3. Download dataset (see `data/README.md`) — ~815 MB
4. Run notebooks **in order**: 01 → 02 → 03
   - Notebook 02 loads results saved by Notebook 01
   - Notebook 03 loads results saved by Notebook 02

---

## Repository structure

```
src/
  load.py           — imzML loading and image reconstruction
  preprocessing.py  — 4-step preprocessing pipeline
  visualisation.py  — plotting functions and assign_lipid_class
  stats.py          — fdr_correction, differential_abundance, colocalization

notebooks/
  01_data_exploration_preprocessing.ipynb
  02_segmentation.ipynb
  03_statistical_analysis.ipynb

results/
  figures/          — all generated figures (tracked by git)
  tables/           — CSV output tables (tracked by git)
```

---

## References

1. Römpp et al. (2010) *Angew. Chem. Int. Ed.* 49:3834 — dataset
2. Bemis, Föll et al. (2023) *Nature Methods* 20:1883 — preprocessing pipeline
3. Deininger et al. (2011) *Anal. Chem.* 84:1277 — TIC normalisation
4. Alexandrov et al. (2010) *J. Proteome Res.* 9:6535 — K-means for MSI
5. Benjamini & Hochberg (1995) *J. R. Stat. Soc. B* 57:289 — FDR correction
6. Mann & Whitney (1947) *Ann. Math. Stat.* 18:50 — Mann-Whitney U test
7. Hubert & Arabie (1985) *J. Classification* 2:193 — ARI
8. Von Luxburg (2007) *Stat. Comput.* 17:395 — spectral clustering
9. Hsu & Turk (2009) *J. Chromatogr. B* 877:2714 — lipid identification by mass

---

## Author

Reza Rajaee — private tutoring material
