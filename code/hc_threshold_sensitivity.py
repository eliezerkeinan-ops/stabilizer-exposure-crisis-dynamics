"""
Hc threshold sensitivity analysis for the PLOS Complex Systems submission.

Run from the package root:
    python code/hc_threshold_sensitivity.py

Required input files:
    data/stabilizer_exposure_dataset_OECDsigma_BISH.csv
    data/Fu_country_table.csv

Outputs:
    figures/figS1_hc_sensitivity_R2009.png
    figures/figS2_hc_sensitivity_DC.png
    data/hc_sensitivity_table.csv
    data/hc_sensitivity_leave_one_out.csv
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIG = ROOT / "figures"
SE_FILE = DATA / "stabilizer_exposure_dataset_OECDsigma_BISH.csv"
FU_FILE = DATA / "Fu_country_table.csv"

CORE = ["CAN", "DEU", "DNK", "ESP", "FIN", "FRA", "GBR", "IRL", "ITA", "JPN", "NLD", "SWE", "USA"]
HC_VALS = [50, 60, 70]
HC_COLS = [f"Z1_trueH_Hc{h}" for h in HC_VALS]
OUTCOMES = {"R2009": "R_buffer_2009", "DC": "D_C_frac_2008_2015"}


def require_columns(df: pd.DataFrame, cols: Iterable[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def merge_outcomes(se: pd.DataFrame, fu: pd.DataFrame) -> pd.DataFrame:
    outcome_cols = ["code", "D_C_frac_2008_2015", "R_buffer_2009"]
    require_columns(fu, outcome_cols, "Fu_country_table.csv")
    se_clean = se.drop(columns=[c for c in outcome_cols[1:] if c in se.columns], errors="ignore")
    return se_clean.merge(fu[outcome_cols], on="code", how="left", validate="one_to_one")


def pearson_with_ci(x: pd.Series, y: pd.Series) -> dict[str, float | int]:
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    n = len(d)
    if n < 4:
        raise ValueError(f"Need at least 4 complete observations for Fisher CI, got n={n}")
    r, p = stats.pearsonr(d["x"], d["y"])
    r_clip = float(np.clip(r, -0.999999999, 0.999999999))
    z = np.arctanh(r_clip)
    se_z = 1.0 / np.sqrt(n - 3)
    ci_lo = np.tanh(z - 1.96 * se_z)
    ci_hi = np.tanh(z + 1.96 * se_z)
    return {"r": float(r), "p": float(p), "n": int(n), "ci_lo": float(ci_lo), "ci_hi": float(ci_hi)}


def spearman_corr(x: pd.Series, y: pd.Series) -> dict[str, float | int]:
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    rho, p = stats.spearmanr(d["x"], d["y"])
    return {"rho": float(rho), "p": float(p), "n": int(len(d))}


def leave_one_out(x: pd.Series, y: pd.Series, codes: pd.Series, outcome: str, hc: int) -> pd.DataFrame:
    d = pd.DataFrame({"x": x, "y": y, "code": codes}).dropna()
    rows = []
    for dropped in d["code"]:
        dd = d[d["code"] != dropped]
        r, p = stats.pearsonr(dd["x"], dd["y"])
        rho, ps = stats.spearmanr(dd["x"], dd["y"])
        rows.append({
            "Hc_%": hc,
            "outcome": outcome,
            "dropped": dropped,
            "pearson_r": r,
            "pearson_p": p,
            "spearman_rho": rho,
            "spearman_p": ps,
            "n": len(dd),
        })
    return pd.DataFrame(rows)


def scatter_panel(ax, x, y, codes, hc, r, n):
    ax.scatter(x, y, s=90, alpha=0.85, edgecolor="black", linewidth=0.5, zorder=3)
    for xi, yi, code in zip(x, y, codes):
        ax.annotate(code, (xi, yi), textcoords="offset points", xytext=(4, 4), fontsize=8)
    if len(x) >= 2 and np.nanstd(x) > 0:
        coef = np.polyfit(x, y, 1)
        xx = np.linspace(np.nanmin(x), np.nanmax(x), 100)
        ax.plot(xx, coef[0] * xx + coef[1], lw=1.2, color="gray", zorder=2)
    ax.axhline(np.nanmedian(y), ls="--", lw=0.8, color="gray", alpha=0.6)
    ax.axvline(np.nanmedian(x), ls="--", lw=0.8, color="gray", alpha=0.6)
    ax.set_title(rf"$H_c$ = {hc}%   r = {r:.3f}, n = {n}", fontsize=11)
    ax.grid(True, alpha=0.3)


def make_sensitivity_figure(core_df: pd.DataFrame, outcome_col: str, outname: str, ylabel: str, title: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    for ax, hc, col in zip(axes, HC_VALS, HC_COLS):
        d = core_df.dropna(subset=[col, outcome_col])
        pc = pearson_with_ci(d[col], d[outcome_col])
        scatter_panel(ax, d[col].values, d[outcome_col].values, d["code"].values, hc, pc["r"], pc["n"])
        ax.set_xlabel(rf"$Z_1$ ($H_c$ = {hc}%)", fontsize=10)
    axes[0].set_ylabel(ylabel, fontsize=10)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG / outname, dpi=200)
    plt.close(fig)
    print(f"Saved {FIG / outname}")


def main() -> None:
    if not SE_FILE.exists():
        raise FileNotFoundError(SE_FILE)
    if not FU_FILE.exists():
        raise FileNotFoundError(FU_FILE)

    se = pd.read_csv(SE_FILE)
    fu = pd.read_csv(FU_FILE)
    require_columns(se, ["code"] + HC_COLS, "stabilizer_exposure_dataset_OECDsigma_BISH.csv")
    df = merge_outcomes(se, fu)
    core_df = df[df["code"].isin(CORE)].copy()
    present = sorted(core_df["code"].dropna().unique().tolist())
    missing_core = [c for c in CORE if c not in present]
    if len(core_df) != len(CORE) or missing_core:
        raise ValueError(f"Expected {len(CORE)} core countries; got {len(core_df)}. Missing: {missing_core}")
    require_columns(core_df, HC_COLS + ["R_buffer_2009", "D_C_frac_2008_2015"], "merged core data")

    make_sensitivity_figure(
        core_df,
        "R_buffer_2009",
        "figS1_hc_sensitivity_R2009.png",
        r"$R_{2009}=g_C^{real}-g_Y^{real}$ (pp)",
        r"Sensitivity to $H_c$: $Z_1$ vs relative insulation $R_{2009}$",
    )
    make_sensitivity_figure(
        core_df,
        "D_C_frac_2008_2015",
        "figS2_hc_sensitivity_DC.png",
        r"$D_C$: consumption depth-duration",
        r"Sensitivity to $H_c$: $Z_1$ vs crisis depth-duration $D_C$",
    )

    rows = []
    loo_all = []
    for hc, col in zip(HC_VALS, HC_COLS):
        row = {"Hc_%": hc, "n": len(core_df.dropna(subset=[col, "R_buffer_2009"]))}
        for outcome_name, outcome_col in OUTCOMES.items():
            p = pearson_with_ci(core_df[col], core_df[outcome_col])
            s = spearman_corr(core_df[col], core_df[outcome_col])
            loo = leave_one_out(core_df[col], core_df[outcome_col], core_df["code"], outcome_name, hc)
            loo_all.append(loo)
            row.update({
                f"pearson_{outcome_name}": round(p["r"], 3),
                f"pearson_p_{outcome_name}": round(p["p"], 3),
                f"CI95_lo_{outcome_name}": round(p["ci_lo"], 3),
                f"CI95_hi_{outcome_name}": round(p["ci_hi"], 3),
                f"spearman_{outcome_name}": round(s["rho"], 3),
                f"spearman_p_{outcome_name}": round(s["p"], 3),
                f"LOO_min_pearson_{outcome_name}": round(float(loo["pearson_r"].min()), 3),
                f"LOO_max_pearson_{outcome_name}": round(float(loo["pearson_r"].max()), 3),
                f"LOO_sign_preserved_{outcome_name}": bool((loo["pearson_r"] > 0).all()) if outcome_name == "R2009" else bool((loo["pearson_r"] < 0).all()),
            })
        rows.append(row)

    table = pd.DataFrame(rows)
    table.to_csv(DATA / "hc_sensitivity_table.csv", index=False)
    pd.concat(loo_all, ignore_index=True).to_csv(DATA / "hc_sensitivity_leave_one_out.csv", index=False)
    print("\n=== Hc Sensitivity Summary ===")
    print(table.to_string(index=False))
    print("\nDone.")


if __name__ == "__main__":
    main()
