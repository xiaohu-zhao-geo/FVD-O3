# FVD-O3

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![Status: Under Review](https://img.shields.io/badge/status-under%20review-lightgrey.svg)

A Python implementation of the **FVD-O3** workflow for factorial variance decomposition of simulated near-surface ozone responses to anthropogenic precursor emission changes in AerChemMIP.

---

## Installation

```bash
git clone https://github.com/xiaohu-zhao-geo/FVD-O3.git
cd FVD-O3
pip install -r requirements.txt
```

### Requirements

- Python >= 3.9
- numpy >= 1.24
- scipy >= 1.10
- pandas >= 2.0
- xarray >= 2023.1
- netCDF4 >= 1.6
- geopandas >= 0.13
- shapely >= 2.0

---

## Quick Start

Run the three steps in order:

```bash
python preprocess.py
python analyse.py
python regrid_sensitivity.py
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Citation

Citation information will be provided upon formal publication of the associated research paper.
