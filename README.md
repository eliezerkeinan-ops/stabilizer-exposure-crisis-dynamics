# Thresholded Stabilizer--Exposure Scaling and Two-Mode Crisis Response in Economic Systems

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20686861.svg)](https://doi.org/10.5281/zenodo.20686861)

This repository contains the manuscript source, processed data, code, and generated figures for reproducing the analysis in:

**Thresholded Stabilizer--Exposure Scaling and Two-Mode Crisis Response in Economic Systems**

Public repository: https://github.com/eliezerkeinan-ops/stabilizer-exposure-crisis-dynamics

## Contents

- `main.tex` - LaTeX manuscript source with line numbers.
- `main.pdf` - compiled manuscript preview.
- `figures/` - manuscript figures and supplementary threshold-sensitivity figures.
- `data/` - processed reproducibility datasets used by the manuscript, figures, and tables.
- `code/` - Python scripts that document raw WDI processing, regenerate figures, and reproduce threshold-sensitivity outputs.
- `supplementary/` - conceptual GIF animations and their parameter/trajectory files.
- `data_sources.md` - source databases, variable definitions, and transformation notes.
- `requirements.txt` - Python dependencies.

## Reproducing the computational outputs

From the repository root:

```bash
pip install -r requirements.txt
# optional raw-WDI processing documentation; requires internet access
python code/00_build_dynamic_recovery_metrics.py

python code/01_make_toy_oscillator_figures.py
python code/02_make_phase_portrait_and_regime_map.py
python code/03_make_empirical_scaling_figures.py
python code/04_make_growth_forcing_figures.py
python code/hc_threshold_sensitivity.py
```

Scripts 01--04 and `hc_threshold_sensitivity.py` regenerate figures into `figures/` and derived reproducibility tables into `data/`. Script 00 documents the raw WDI processing used to construct the dynamic depth-duration variables; it fetches public WDI data and writes intermediate outputs under `output/`.

## Recompiling the manuscript

```bash
latexmk -pdf main.tex
```

## Data availability

All processed data and author-generated code required to reproduce the figures and tables are publicly available at Zenodo:

https://doi.org/10.5281/zenodo.20686861

The development repository is available at:

https://github.com/eliezerkeinan-ops/stabilizer-exposure-crisis-dynamics

## Scope note

The empirical CSV files are processed reproducibility datasets, not raw database exports. The raw sources are OECD SOCX, BIS Total Credit, and World Bank WDI, as documented in `data_sources.md` and the manuscript bibliography.

The toy oscillator is transparent and non-calibrated. It illustrates the role-based reduced representation, threshold activation, phase portrait, and regime map; it is not a fitted country simulation.


## Canonical stabilizer-to-fragility coordinate

All manuscript correlations and Z1-based figures use the canonical `Z1_trueH_Hc*` columns from `data/stabilizer_exposure_dataset_OECDsigma_BISH.csv`. Derived tables such as `country_archetypes_table.csv`, `Fu_country_table.csv`, and `dynamic_recovery_crisis_countries.csv` include these canonical columns as well. Legacy `Z1_Hc*` aliases are retained only for backwards compatibility and mirror the canonical values in this package.
