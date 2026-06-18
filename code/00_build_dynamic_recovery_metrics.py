#!/usr/bin/env python3
"""
Dynamic recovery extension for the stabilizer-to-exposure model.

Purpose
-------
This script adds the time axis:
    pre-crisis Z1  ->  post-crisis recovery geometry

It fetches World Bank WDI real-level time series for 2000-2015:
    GDP real levels
    household consumption real levels
    population

Then it computes:
    D_C: household-consumption depth-duration below pre-crisis trend, 2008-2015
    D_Y: GDP depth-duration below pre-crisis trend, 2008-2015
    T_C_level: years until consumption per capita returns to 2007 level
    T_Y_level: years until GDP per capita returns to 2007 level
    B_T = (D_Y - D_C) / D_Y, dynamic buffering index

It merges these dynamic measures with a pre-existing exposure dataset containing:
    Sigma_socx_gdp
    H_household_debt_gdp
    exports_gdp
    gfcf_gdp
    Z0 / Z1 if already computed; canonical Z1_trueH_Hc* columns are preserved and mirrored to Z1_Hc* aliases.

Expected input
--------------
Place this file in the same folder as:
    stabilizer_exposure_dataset_OECDsigma_BISH.csv

or put that file under:
    data/stabilizer_exposure_dataset_OECDsigma_BISH.csv

Run:
    python build_dynamic_recovery_metrics.py

Outputs
-------
output/dynamic_recovery_dataset.csv
output/dynamic_recovery_correlation_summary.csv
output/D_C_vs_Z1.png
output/B_T_vs_Z1.png
output/T_C_vs_Z1.png
output/D_C_vs_Sigma.png

Notes
-----
This is a pilot script. The trend is a log-linear pre-crisis trend fitted over 2000-2007
to per-capita real levels. This is intentionally simple and transparent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Dict
import sys
import math

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt


COUNTRIES = {
    "FRA": "France",
    "DEU": "Germany",
    "SWE": "Sweden",
    "DNK": "Denmark",
    "GBR": "United Kingdom",
    "USA": "United States",
    "IRL": "Ireland",
    "ESP": "Spain",
    "ITA": "Italy",
    "FIN": "Finland",
    "CAN": "Canada",
    "AUS": "Australia",
    "JPN": "Japan",
    "NLD": "Netherlands",
    "BEL": "Belgium",
    "AUT": "Austria",
    "CHE": "Switzerland",
    "KOR": "Korea",
}

WDI_INDICATORS = {
    "gdp_real": "NY.GDP.MKTP.KN",        # GDP, constant LCU
    "cons_real": "NE.CON.PRVT.KN",      # Households and NPISHs final consumption expenditure, constant LCU
    "pop": "SP.POP.TOTL",               # Population, total
    "unemployment": "SL.UEM.TOTL.ZS",   # Unemployment, total (% labor force), optional diagnostic
}


def fetch_wdi_indicator(codes: Iterable[str], indicator: str, start: int, end: int) -> pd.DataFrame:
    code_string = ";".join(codes)
    url = f"https://api.worldbank.org/v2/country/{code_string}/indicator/{indicator}"
    params = {"format": "json", "per_page": 20000, "date": f"{start}:{end}"}
    response = requests.get(url, params=params, timeout=90)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list) or len(payload) < 2:
        raise RuntimeError(f"Unexpected WDI response for {indicator}: {payload!r}")

    rows = []
    for item in payload[1]:
        rows.append({
            "code": item.get("countryiso3code"),
            "country_wb": item.get("country", {}).get("value"),
            "year": int(item.get("date")),
            "value": item.get("value"),
        })
    return pd.DataFrame(rows)


def fetch_wdi_panel(start: int = 2000, end: int = 2015) -> pd.DataFrame:
    frames = []
    for name, indicator in WDI_INDICATORS.items():
        print(f"Fetching WDI {name}: {indicator}")
        d = fetch_wdi_indicator(COUNTRIES.keys(), indicator, start, end)
        d["variable"] = name
        frames.append(d[["code", "country_wb", "year", "variable", "value"]])

    long = pd.concat(frames, ignore_index=True)
    wide = (
        long.pivot_table(index=["code", "country_wb", "year"], columns="variable", values="value", aggfunc="first")
        .reset_index()
    )
    wide["country"] = wide["code"].map(COUNTRIES).fillna(wide["country_wb"])
    return wide


def find_exposure_file() -> Path:
    candidates = [
        Path("stabilizer_exposure_dataset_OECDsigma_BISH.csv"),
        Path("data") / "stabilizer_exposure_dataset_OECDsigma_BISH.csv",
        Path("/mnt/data/stabilizer_exposure_dataset_OECDsigma_BISH.csv"),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find stabilizer_exposure_dataset_OECDsigma_BISH.csv. "
        "Place it next to this script or under ./data/."
    )


def safe_corr(x: pd.Series, y: pd.Series) -> float:
    d = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < 3:
        return float("nan")
    return float(d["x"].corr(d["y"]))


def fit_loglinear_trend(pre: pd.DataFrame, year_col: str, value_col: str):
    """
    Fit log(value) = a*year + b over pre-crisis years.
    Returns a function trend(year).
    """
    clean = pre[[year_col, value_col]].dropna()
    clean = clean[clean[value_col] > 0]
    if len(clean) < 4:
        return None
    coef = np.polyfit(clean[year_col].to_numpy(), np.log(clean[value_col].to_numpy()), 1)

    def trend(year):
        return float(np.exp(coef[0] * year + coef[1]))

    return trend


def first_recovery_year(post: pd.DataFrame, value_col: str, threshold: float) -> float:
    """
    Return first post-2007 year where value >= threshold.
    If already true in 2008, returns 1 year. If no recovery by end, returns NaN.
    """
    for _, row in post.sort_values("year").iterrows():
        val = row.get(value_col)
        if pd.notna(val) and val >= threshold:
            return float(row["year"] - 2007)
    return float("nan")


def compute_country_metrics(g: pd.DataFrame) -> Dict[str, float]:
    g = g.sort_values("year").copy()

    # Per capita real series
    g["gdp_pc"] = g["gdp_real"] / g["pop"]
    g["cons_pc"] = g["cons_real"] / g["pop"]

    pre = g[(g["year"] >= 2000) & (g["year"] <= 2007)]
    post = g[(g["year"] >= 2008) & (g["year"] <= 2015)]

    if len(pre) < 4 or len(post) < 1:
        return {}

    out = {"code": g["code"].iloc[0], "country": g["country"].iloc[0]}

    for var, label in [("cons_pc", "C"), ("gdp_pc", "Y")]:
        trend_fun = fit_loglinear_trend(pre, "year", var)
        base2007 = g.loc[g["year"] == 2007, var]
        if trend_fun is None or base2007.empty or pd.isna(base2007.iloc[0]):
            continue

        base2007 = float(base2007.iloc[0])
        p = post[["year", var]].copy()
        p[f"{var}_trend"] = p["year"].apply(trend_fun)
        p[f"{var}_gap_frac"] = np.maximum(0.0, (p[f"{var}_trend"] - p[var]) / p[f"{var}_trend"])

        out[f"D_{label}_frac_2008_2015"] = float(p[f"{var}_gap_frac"].sum())
        out[f"max_gap_{label}_frac_2008_2015"] = float(p[f"{var}_gap_frac"].max())
        out[f"T_{label}_level_years"] = first_recovery_year(p, var, base2007)

        # Time to trend recovery: first year where gap fraction is zero or negative before clipping.
        p["raw_gap"] = (p[f"{var}_trend"] - p[var]) / p[f"{var}_trend"]
        trend_recovered = p[p["raw_gap"] <= 0]
        out[f"T_{label}_trend_years"] = (
            float(trend_recovered["year"].min() - 2007) if len(trend_recovered) else float("nan")
        )

    # Dynamic buffer index: how much smaller consumption depth-duration is relative to GDP depth-duration
    dy = out.get("D_Y_frac_2008_2015")
    dc = out.get("D_C_frac_2008_2015")
    if dy is not None and pd.notna(dy) and dy > 0 and dc is not None and pd.notna(dc):
        out["B_T_depth_buffer"] = (dy - dc) / dy
    else:
        out["B_T_depth_buffer"] = float("nan")

    # Unemployment shock diagnostic
    if "unemployment" in g.columns:
        pre_u = pre["unemployment"].mean()
        post_u = post["unemployment"].max()
        if pd.notna(pre_u) and pd.notna(post_u):
            out["U_max_minus_pre_mean"] = float(post_u - pre_u)

    return out


def compute_dynamic_metrics(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for code, g in panel.groupby("code"):
        metrics = compute_country_metrics(g)
        if metrics:
            rows.append(metrics)
    return pd.DataFrame(rows)


def ensure_z_columns(df: pd.DataFrame, hcs=(50, 60, 70, 80, 90, 100)) -> pd.DataFrame:
    """
    Ensure Z0 and Z1 columns exist.

    Canonical manuscript correlations use the `Z1_trueH_Hc*` columns from
    stabilizer_exposure_dataset_OECDsigma_BISH.csv. If those columns are
    present, preserve them and mirror them into the backwards-compatible
    `Z1_Hc*` aliases used by older plotting notebooks. Only compute Z1 from
    Sigma/H directly when the canonical columns are absent.
    """
    required = ["Sigma_socx_gdp", "exports_gdp", "gfcf_gdp", "H_household_debt_gdp"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns needed for Z: {missing}")

    df = df.copy()
    if "Z0_trueH" in df.columns:
        df["Z0"] = df["Z0_trueH"]
    else:
        df["Z0"] = df["Sigma_socx_gdp"] / (df["exports_gdp"] + df["H_household_debt_gdp"] + df["gfcf_gdp"])

    for hc in hcs:
        canonical = f"Z1_trueH_Hc{hc}"
        alias = f"Z1_Hc{hc}"
        if canonical in df.columns:
            df[alias] = df[canonical]
        else:
            df[canonical] = df["Sigma_socx_gdp"] / (
                df["exports_gdp"] + df["gfcf_gdp"] + np.maximum(0.0, df["H_household_debt_gdp"] - hc)
            )
            df[alias] = df[canonical]
    return df


def scatter(df: pd.DataFrame, x: str, y: str, out: Path, title: str, xlabel: str, ylabel: str):
    d = df[[x, y, "code"]].replace([np.inf, -np.inf], np.nan).dropna()
    plt.figure(figsize=(8.4, 5.8))
    plt.scatter(d[x], d[y], s=82)
    for _, row in d.iterrows():
        plt.annotate(row["code"], (row[x], row[y]), textcoords="offset points", xytext=(5, 5), fontsize=8.5)
    if len(d) >= 3:
        coef = np.polyfit(d[x], d[y], 1)
        lx = np.linspace(d[x].min(), d[x].max(), 100)
        plt.plot(lx, coef[0] * lx + coef[1], linewidth=1)
        r = d[x].corr(d[y])
        title = f"{title} (r={r:.2f})"
    plt.axhline(0, linewidth=1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close()


def main():
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    exposure_path = find_exposure_file()
    print(f"Reading exposure dataset: {exposure_path}")
    exposure = pd.read_csv(exposure_path)
    exposure = ensure_z_columns(exposure)

    panel = fetch_wdi_panel(2000, 2015)
    panel.to_csv(output_dir / "wdi_real_levels_panel_2000_2015.csv", index=False)

    dyn = compute_dynamic_metrics(panel)
    dyn.to_csv(output_dir / "dynamic_recovery_metrics_only.csv", index=False)

    merged = exposure.merge(dyn, on=["code", "country"], how="left")

    # Useful crisis subset: negative 2009 GDP growth, with dynamic measures.
    if "gdp_growth_2009" in merged.columns:
        crisis = merged[merged["gdp_growth_2009"] < 0].copy()
    else:
        crisis = merged.copy()

    merged.to_csv(output_dir / "dynamic_recovery_dataset.csv", index=False)
    crisis.to_csv(output_dir / "dynamic_recovery_crisis_countries.csv", index=False)

    # Correlation summary
    model_cols = ["Sigma_socx_gdp", "Z0", "Z1_Hc50", "Z1_Hc60", "Z1_Hc70", "Z1_Hc80", "Z1_Hc90", "Z1_Hc100"]
    outcome_cols = [
        "R_buffer_2009",
        "D_C_frac_2008_2015",
        "D_Y_frac_2008_2015",
        "B_T_depth_buffer",
        "T_C_level_years",
        "T_Y_level_years",
        "max_gap_C_frac_2008_2015",
        "max_gap_Y_frac_2008_2015",
        "U_max_minus_pre_mean",
    ]

    rows = []
    for m in model_cols:
        for o in outcome_cols:
            if m in crisis.columns and o in crisis.columns:
                rows.append({"model": m, "outcome": o, "corr": safe_corr(crisis[m], crisis[o])})
    corr_df = pd.DataFrame(rows)
    corr_df.to_csv(output_dir / "dynamic_recovery_correlation_summary.csv", index=False)

    # Main plots
    scatter(
        crisis, "Z1_Hc60", "D_C_frac_2008_2015", output_dir / "D_C_vs_Z1.png",
        "Consumption depth-duration vs stabilizer-to-exposure ratio",
        "Z1 = Sigma / (X + I + max(0,H-60))",
        "D_C: summed fractional consumption gap below trend, 2008-2015",
    )
    scatter(
        crisis, "Z1_Hc60", "B_T_depth_buffer", output_dir / "B_T_vs_Z1.png",
        "Dynamic buffering index vs stabilizer-to-exposure ratio",
        "Z1 = Sigma / (X + I + max(0,H-60))",
        "B_T = (D_Y - D_C) / D_Y",
    )
    scatter(
        crisis, "Z1_Hc60", "T_C_level_years", output_dir / "T_C_vs_Z1.png",
        "Consumption recovery time vs stabilizer-to-exposure ratio",
        "Z1 = Sigma / (X + I + max(0,H-60))",
        "T_C: years after 2007 to regain 2007 consumption per capita",
    )
    scatter(
        crisis, "Sigma_socx_gdp", "D_C_frac_2008_2015", output_dir / "D_C_vs_Sigma.png",
        "Consumption depth-duration vs stabilizer size",
        "Sigma = OECD public social expenditure / GDP, 2008 (%)",
        "D_C: summed fractional consumption gap below trend, 2008-2015",
    )

    print("Done.")
    print("Main outputs:")
    print(output_dir / "dynamic_recovery_dataset.csv")
    print(output_dir / "dynamic_recovery_correlation_summary.csv")
    print(output_dir / "D_C_vs_Z1.png")
    print(output_dir / "B_T_vs_Z1.png")
    print(output_dir / "T_C_vs_Z1.png")


if __name__ == "__main__":
    main()
