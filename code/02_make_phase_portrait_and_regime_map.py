"""Reproduce phase portrait and threshold-regime map.
Run from package root: python code/02_make_phase_portrait_and_regime_map.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT/"figures"; DATA=ROOT/"data"
FIG.mkdir(exist_ok=True); DATA.mkdir(exist_ok=True)

T, dt = 45.0, 0.025
t = np.arange(0,T+dt,dt)
regimes=[("fragile-fast",0.15),("balanced",0.23),("stable-sluggish",0.31)]
m0, alpha = 1.0, 4.0
c0, beta = 0.25, 3.0
kx, xc, kappa = 1.0, 1.12, 0.75
cu, lam = 0.35, 0.50
Ax0, tx, sigx = 1.22, 10.0, 1.0
Au, tu, sigu = 0.74, 7.0, 1.5
Fu = Au*np.exp(-0.5*((t-tu)/sigu)**2)

def simulate(S, Ax=Ax0, nonlinear=True):
    Fx=Ax*np.exp(-0.5*((t-tx)/sigx)**2)
    m=m0+alpha*S; cx=c0+beta*S
    x=np.zeros_like(t); v=np.zeros_like(t); u=np.zeros_like(t); w=np.zeros_like(t)
    for i in range(len(t)-1):
        nl=kappa*max(0.0,x[i]-xc) if nonlinear else 0.0
        ax=(Fx[i]+nl-cx*v[i]-kx*x[i])/m
        au=(Fu[i]-lam*max(0,x[i])-cu*w[i])/m
        v[i+1]=v[i]+ax*dt; x[i+1]=x[i]+v[i+1]*dt
        w[i+1]=w[i]+au*dt; u[i+1]=u[i]+w[i+1]*dt
    return x,v,u,w

def xmax(S, Ax=Ax0, nonlinear=True):
    return simulate(S, Ax=Ax, nonlinear=nonlinear)[0].max()

# Linear response summary
Scrit=brentq(lambda S: xmax(S,Ax0,False)-xc, 0.08, 0.38)
summary=[]
for name,S in regimes:
    m=m0+alpha*S; cx=c0+beta*S
    zeta=cx/(2*np.sqrt(kx*m)); omega=np.sqrt(kx/m)
    disc=cx*cx-4*m*kx
    summary.append({"regime":name,"S":S,"m":m,"c_x":cx,"zeta_x":zeta,"omega_x":omega,"discriminant":disc,"xmax":xmax(S,Ax0,True)})
summary.append({"regime":"Scrit","S":Scrit,"m":m0+alpha*Scrit,"c_x":c0+beta*Scrit,"zeta_x":(c0+beta*Scrit)/(2*np.sqrt(kx*(m0+alpha*Scrit))),"omega_x":np.sqrt(kx/(m0+alpha*Scrit)),"discriminant":np.nan,"xmax":xc})
pd.DataFrame(summary).to_csv(DATA/"linear_response_threshold_summary.csv", index=False)

# Linear-response functions zeta_x(S) and omega_x(S)
Sline=np.linspace(0.08,0.38,200)
zeta=(c0+beta*Sline)/(2*np.sqrt(kx*(m0+alpha*Sline)))
omega=np.sqrt(kx/(m0+alpha*Sline))
fig,ax1=plt.subplots(figsize=(7.4,5.2))
ax1.plot(Sline,zeta,linewidth=2,label=r"$\zeta_x(S)$")
ax1.axhline(1.0,linestyle="--",linewidth=1,label="critical damping")
for name,S in regimes:
    ax1.axvline(S,linestyle=":",linewidth=1)
    ax1.annotate(name.split('-')[0],(S,(c0+beta*S)/(2*np.sqrt(kx*(m0+alpha*S)))),textcoords="offset points",xytext=(4,6),fontsize=8)
ax1.set_xlabel("stable reservoir $S$")
ax1.set_ylabel(r"damping ratio $\zeta_x$")
ax1.grid(True,alpha=.3)
ax2=ax1.twinx()
ax2.plot(Sline,omega,linewidth=2,linestyle="--",label=r"$\omega_x(S)$")
ax2.set_ylabel(r"natural frequency $\omega_x$")
lines,labels=ax1.get_legend_handles_labels(); lines2,labels2=ax2.get_legend_handles_labels()
ax1.legend(lines+lines2,labels+labels2,loc="center right",fontsize=8)
ax1.set_title("Linear-response quantities in the crisis mode")
fig.tight_layout(); fig.savefig(FIG/"fig10_linear_response_functions.png",dpi=240); fig.savefig(FIG/"fig10_linear_response_functions.pdf"); plt.close(fig)

# Phase portrait
fig,ax=plt.subplots(figsize=(8.5,6.2))
for name,S in regimes:
    x,v,_,_=simulate(S,Ax0,True)
    ax.plot(x,v,label=f"{name}, S={S:.2f}",linewidth=1.7)
    ax.scatter([x[np.argmax(x)]],[v[np.argmax(x)]],s=35)
ax.axvline(xc, linestyle="--", linewidth=1, label="$x_c$")
ax.axhline(0, linewidth=.8)
ax.set_xlabel("crisis displacement $x$"); ax.set_ylabel("velocity $\\dot{x}$")
ax.set_title("Crisis-mode phase portrait")
ax.grid(True,alpha=.3); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG/"fig8_phase_portrait.png",dpi=240); fig.savefig(FIG/"fig8_phase_portrait.pdf"); plt.close(fig)

# Regime map in (S, Ax)
Svals=np.linspace(0.08,0.38,71); Avals=np.linspace(0.75,1.65,71)
rows=[]; Z=np.empty((len(Avals),len(Svals)))
for i,A in enumerate(Avals):
    for j,S in enumerate(Svals):
        val=xmax(S,A,True)/xc
        Z[i,j]=val
        rows.append({"S":S,"A_x":A,"xmax_over_xc":val,"crosses_threshold":val>1.0})
pd.DataFrame(rows).to_csv(DATA/"threshold_regime_map.csv",index=False)
fig,ax=plt.subplots(figsize=(8.6,6.4))
cs=ax.contourf(Svals,Avals,Z,levels=np.linspace(Z.min(),Z.max(),20),cmap="viridis")
fig.colorbar(cs,ax=ax,label="$x_{max}/x_c$")
ax.contour(Svals,Avals,Z,levels=[1.0],colors="black",linewidths=2)
ax.contour(Svals,Avals,Z,levels=[0.95,1.05],colors="white",linestyles="--",linewidths=1)
for name,S in regimes:
    ax.scatter([S],[Ax0],s=95,edgecolor="black",facecolor="white")
    ax.annotate(name,(S,Ax0),textcoords="offset points",xytext=(6,6),fontsize=8,color="white")
ax.set_xlabel("stable reservoir $S$"); ax.set_ylabel("crisis forcing amplitude $A_x$")
ax.set_title("Threshold regime map")
ax.grid(True,alpha=.25)
fig.tight_layout(); fig.savefig(FIG/"fig9_threshold_regime_map.png",dpi=240); fig.savefig(FIG/"fig9_threshold_regime_map.pdf"); plt.close(fig)
print(f"Wrote phase portrait and regime map. Scrit={Scrit:.3f}")
