# Figure-generation and robustness scripts

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20686861.svg)](https://doi.org/10.5281/zenodo.20686861)
Run these scripts from the package root.

```bash
python code/01_make_toy_oscillator_figures.py
python code/02_make_phase_portrait_and_regime_map.py
python code/03_make_empirical_scaling_figures.py
python code/04_make_growth_forcing_figures.py
python code/hc_threshold_sensitivity.py
```

The first four scripts regenerate the toy-model, phase-portrait, threshold-map, empirical-scaling, and growth-forcing figures into `figures/`.

`hc_threshold_sensitivity.py` regenerates:

- `figures/figS1_hc_sensitivity_R2009.png`
- `figures/figS2_hc_sensitivity_DC.png`
- `data/hc_sensitivity_table.csv`
- `data/hc_sensitivity_leave_one_out.csv`

`05_make_supplementary_movies.py` documents the pre-rendered supplementary movies; GIF frame regeneration is intentionally not part of the default lightweight workflow.


Canonical Z1: scripts 03 and 04 validate and use `Z1_trueH_Hc60` explicitly. The shorter `Z1_Hc60` aliases are retained for compatibility but should not be used as independent sources.
