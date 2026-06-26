# Dataset Download Instructions

## Mouse Urinary Bladder Dataset

**Reference:** Römpp A, et al. (2010) Histology by mass spectrometry.
*Angewandte Chemie International Edition* 49:3834.

**PRIDE accession:** PXD001283

## Download

In the Codespaces terminal:

```bash
wget "https://ftp.pride.ebi.ac.uk/pride/data/archive/2014/11/PXD001283/HR2MSImouseurinarybladderS096.imzML" \
     -O data/mouse_bladder.imzML

wget "https://ftp.pride.ebi.ac.uk/pride/data/archive/2014/11/PXD001283/HR2MSImouseurinarybladderS096.ibd" \
     -O data/mouse_bladder.ibd
```

The ibd file is ~815 MB — download takes a few minutes.
