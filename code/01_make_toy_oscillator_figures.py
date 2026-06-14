"""Reproduce toy oscillator figures for Stable Inertial Mass manuscript.
Run from the Overleaf package root: python code/01_make_toy_oscillator_figures.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
DATA = ROOT / "data"
FIG.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)

T, dt = 45.0, 0.01
t = np.arange(0, T + dt, dt)
regimes = [("fragile-fast", 0.15), ("balanced", 0.23), ("stable-sluggish", 0.31)]

m0, alpha = 1.0, 4.0
c0, beta = 0.25, 3.0
kx, xc, kappa = 1.0, 1.12, 0.75
cu, lam = 0.35, 0.50
Ax, tx, sigx = 1.22, 10.0, 1.0
Au, tu, sigu = 0.74, 7.0, 1.5
Fx = Ax * np.exp(-0.5 * ((t - tx) / sigx) ** 2)
Fu = Au * np.exp(-0.5 * ((t - tu) / sigu) ** 2)

def simulate(S: float):
    m = m0 + alpha * S
    cx = c0 + beta * S
    x = np.zeros_like(t); v = np.zeros_like(t)
    u = np.zeros_like(t); w = np.zeros_like(t)
    ax = np.zeros_like(t); au = np.zeros_like(t); D = np.zeros_like(t)
    for i in range(len(t)-1):
        nonlinear = kappa * max(0.0, x[i] - xc)
        ax[i] = (Fx[i] + nonlinear - cx*v[i] - kx*x[i]) / m
        au[i] = (Fu[i] - lam*max(0.0, x[i]) - cu*w[i]) / m
        v[i+1] = v[i] + ax[i]*dt
        x[i+1] = x[i] + v[i+1]*dt
        w[i+1] = w[i] + au[i]*dt
        u[i+1] = u[i] + w[i+1]*dt
        D[i+1] = D[i] + max(0.0, x[i+1])*dt
    ax[-1], au[-1] = ax[-2], au[-2]
    return {"S":S,"m":m,"cx":cx,"x":x,"v":v,"u":u,"w":w,"ax":ax,"au":au,"D":D,
            "xmax":float(x.max()),"D_x":float(np.trapezoid(np.maximum(0,x),t)),
            "threshold_area":float(np.trapezoid(np.maximum(0,x-xc),t)),
            "max_growth_accel":float(au.max()),"crosses_threshold":bool(x.max()>xc)}

sims = {name: simulate(S) for name, S in regimes}
S_grid = np.linspace(0.08, 0.38, 121)
metrics = pd.DataFrame([{**{"S":S}, **{k:simulate(S)[k] for k in ["m","cx","xmax","D_x","threshold_area","max_growth_accel","crosses_threshold"]}} for S in S_grid])
metrics.to_csv(DATA/"toy_v2_scan_metrics.csv", index=False)
pd.DataFrame([{"regime":n,"S":s["S"],"m(S)":s["m"],"c_x(S)":s["cx"],"x_max":s["xmax"],"x_c":xc,"D_x":s["D_x"],"threshold_area":s["threshold_area"],"max_growth_accel":s["max_growth_accel"],"crosses_threshold":s["crosses_threshold"]} for n,s in sims.items()]).to_csv(DATA/"toy_v2_regime_summary.csv", index=False)

fig, axes = plt.subplots(1,3,figsize=(18,5.3))
for name, sim in sims.items():
    axes[0].plot(t, sim["x"], label=f"{name}, S={sim['S']:.2f}")
axes[0].axhline(xc, linestyle="--", linewidth=1, label="$x_c$")
axes[0].set_xlabel("time"); axes[0].set_ylabel("crisis displacement $x(t)$"); axes[0].set_title("Crisis coordinate"); axes[0].grid(True, alpha=.3); axes[0].legend(fontsize=8)
for name, sim in sims.items():
    axes[1].plot(t, sim["u"], label=f"{name}, S={sim['S']:.2f}")
axes[1].set_xlabel("time"); axes[1].set_ylabel("growth coordinate $u(t)$"); axes[1].set_title("Growth coordinate"); axes[1].grid(True, alpha=.3)
axes[2].plot(metrics["max_growth_accel"], metrics["D_x"], marker=".", linewidth=1)
for name, sim in sims.items():
    axes[2].scatter([sim["max_growth_accel"]], [sim["D_x"]], s=90)
    axes[2].annotate(name, (sim["max_growth_accel"], sim["D_x"]), textcoords="offset points", xytext=(5,5), fontsize=8)
axes[2].set_xlabel("maximum growth acceleration"); axes[2].set_ylabel("crisis depth-duration $D_x$"); axes[2].set_title("Trade-off frontier"); axes[2].grid(True, alpha=.3)
fig.tight_layout(); fig.savefig(FIG/"fig1_toy_trajectories_tradeoff.png", dpi=240); plt.close(fig)

fig, axes = plt.subplots(1,3,figsize=(18,5.3))
axes[0].plot(metrics["S"], metrics["xmax"], marker="."); axes[0].axhline(xc, linestyle="--", linewidth=1)
axes[0].set_xlabel("stable reservoir $S$"); axes[0].set_ylabel("$x_{max}$"); axes[0].set_title("Threshold amplitude"); axes[0].grid(True, alpha=.3)
axes[1].plot(metrics["S"], metrics["D_x"], marker=".", label="$D_x$"); axes[1].plot(metrics["S"], metrics["threshold_area"], marker=".", label="$\\int[x-x_c]_+dt$")
axes[1].set_xlabel("stable reservoir $S$"); axes[1].set_ylabel("area"); axes[1].set_title("Depth-duration"); axes[1].grid(True, alpha=.3); axes[1].legend()
axes[2].plot(metrics["S"], metrics["max_growth_accel"], marker=".")
axes[2].set_xlabel("stable reservoir $S$"); axes[2].set_ylabel("max growth acceleration"); axes[2].set_title("Growth responsiveness"); axes[2].grid(True, alpha=.3)
fig.tight_layout(); fig.savefig(FIG/"fig2_stable_mass_scan.png", dpi=240); plt.close(fig)
print("Wrote toy figures and toy CSVs")
