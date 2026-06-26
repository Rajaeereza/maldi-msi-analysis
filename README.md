# MALDI-MSI Analysis — Mouse Urinary Bladder

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

This repository is part of a private tutoring material collection.
It demonstrates a complete MALDI-MSI data analysis pipeline on a
public benchmark dataset — from raw data loading to statistical
interpretation of tissue regions.

---

## Dataset

Mouse urinary bladder — Römpp et al. (2010) *Angew. Chem. Int. Ed.* 49:3834
PRIDE: [PXD001283](https://www.ebi.ac.uk/pride/archive/projects/PXD001283)

---

## Analysis pipeline

**Notebook 01 — Data Exploration and Preprocessing**
- What is MALDI-MSI data
- Ion images, TIC map, pixel spectra
- TIC normalisation → 2.5% frequency filter → 10 ppm rebinning
- Top 5 peaks and lipid class assignment
- PCA before and after preprocessing
- t-SNE visualisation

**Notebook 02 — Tissue Segmentation**
- Choosing k: elbow curve + silhouette score + biological evidence
- K-means, GMM, Spectral Clustering
- Comparing segmentation maps (ARI)
- Disagreement analysis at tissue boundaries
- PCA/t-SNE coloured by cluster assignments

**Notebook 03 — Statistical Analysis**
- Differential abundance: Mann-Whitney U test per region
- Benjamini-Hochberg FDR correction
- Volcano plots
- Marker ion images per region
- Colocalization analysis

---

## How to run

1. Open in GitHub Codespaces
2. Install dependencies:
```bash
pip install pyimzml scikit-learn seaborn matplotlib pandas scipy numpy -q
```
3. Download dataset (see `data/README.md`)
4. Run notebooks 01 → 02 → 03 in order

---

## References

1. Römpp et al. (2010) *Angew. Chem. Int. Ed.* 49:3834 — dataset
2. Bemis, Föll et al. (2023) *Nature Methods* 20:1883 — preprocessing pipeline
3. Deininger et al. (2011) *Anal. Chem.* 84:1277 — normalisation
4. Alexandrov et al. (2010) *J. Proteome Res.* 9:6535 — K-means for MSI
5. Benjamini & Hochberg (1995) *J. R. Stat. Soc. B* 57:289 — FDR correction
6. Hubert & Arabie (1985) *J. Classification* 2:193 — ARI
7. Von Luxburg (2007) *Stat. Comput.* 17:395 — spectral clustering

---

## Author

Reza Rajaee — private tutoring material
