# Thresholded Stabilizer--Exposure Scaling and Two-Mode Crisis Response in Economic Systems

PLOS Complex Systems submission/reproducibility package, v1.4.

## Contents

- `main.tex` — LaTeX manuscript source with line numbers.
- `main.pdf` — compiled manuscript preview.
- `figures/` — manuscript figures and supplementary Hc-sensitivity figures.
- `data/` — processed reproducibility datasets used by the manuscript, figures, and tables.
- `code/` — Python scripts that regenerate figures and Hc sensitivity outputs from processed data.
- `supplementary/` — conceptual GIF animations and their parameter/trajectory files.
- `data_sources.md` — source databases, variable definitions, and transformation notes.
- `submission_text/` — cover letter and statements for the PLOS submission form.
- `requirements.txt` — Python dependencies.

## Reproducing the computational outputs

From the package root:

```bash
pip install -r requirements.txt
python code/01_make_toy_oscillator_figures.py
python code/02_make_phase_portrait_and_regime_map.py
python code/03_make_empirical_scaling_figures.py
python code/04_make_growth_forcing_figures.py
python code/hc_threshold_sensitivity.py
```

The scripts regenerate figures into `figures/` and derived reproducibility tables into `data/`.

## Recompiling the manuscript

```bash
latexmk -pdf main.tex
```

## Scope note

The empirical CSV files are processed reproducibility datasets, not raw database exports. The raw sources are OECD SOCX, BIS Total Credit, and World Bank WDI, as documented in `data_sources.md` and the manuscript bibliography.

The toy oscillator is transparent and non-calibrated. It illustrates the normal form, threshold activation, phase portrait, and regime map; it is not a fitted country simulation.
