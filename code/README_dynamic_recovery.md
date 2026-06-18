# Dynamic Recovery Extension

This script adds the time axis to the stabilizer-to-exposure pilot.

It tests whether pre-crisis

\[
Z_1 = \frac{\Sigma}{X + I + (H-60)_+}
\]

is associated with post-crisis recovery geometry:

- \(D_C\): consumption depth-duration below pre-crisis trend
- \(D_Y\): GDP depth-duration below pre-crisis trend
- \(B_T=(D_Y-D_C)/D_Y\): dynamic buffering index
- \(T_C\): years until household consumption per capita returns to 2007 level

## Required local input

Put this file next to the script or under `data/`:

```text
stabilizer_exposure_dataset_OECDsigma_BISH.csv
```

This is the file we already created from:
- OECD SOCX for \(\Sigma\)
- BIS household debt for \(H\)
- World Bank WDI for \(X,I,R\)

## Run

```bash
python -m pip install pandas numpy matplotlib requests
python build_dynamic_recovery_metrics.py
```

## Outputs

```text
output/dynamic_recovery_dataset.csv
output/dynamic_recovery_correlation_summary.csv
output/D_C_vs_Z1.png
output/B_T_vs_Z1.png
output/T_C_vs_Z1.png
output/D_C_vs_Sigma.png
```

## Trend convention

For each country, the script fits a log-linear trend to real per-capita levels over 2000-2007, then computes post-2008 gaps against that trend.

This is intentionally a simple pilot, not a final econometric specification.
