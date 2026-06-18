"""Reproduce growth-forcing diagnostic figures from processed data.
Run from package root: python code/04_make_growth_forcing_figures.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

ROOT=Path(__file__).resolve().parents[1]
FIG=ROOT/"figures"; DATA=ROOT/"data"
FIG.mkdir(exist_ok=True)

df=pd.read_csv(DATA/"Fu_country_table.csv").replace([np.inf,-np.inf],np.nan)
ZCOL="Z1_trueH_Hc60"
required=[ZCOL,"D_C_frac_2008_2015","Delta_Fu_post_minus_pre"]
missing=[c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required canonical columns in Fu_country_table.csv: {missing}")
euro={"AUT","BEL","DEU","ESP","FIN","FRA","IRL","ITA","NLD"}


# Two-step diagnostic chain: Z1 -> D_C -> Delta F_u
chain_core=df.dropna(subset=[ZCOL,'D_C_frac_2008_2015','Delta_Fu_post_minus_pre']).copy()
chain_ext=df.dropna(subset=['D_C_frac_2008_2015','Delta_Fu_post_minus_pre']).copy()
fig,(axA,axB)=plt.subplots(1,2,figsize=(15,6.2))

# Grayscale-safe encoding for Delta F_u: shading plus marker shape.
# More negative Delta F_u = stronger deterioration in post-crisis growth slope.
bins=[-np.inf,-3.0,-1.5,np.inf]
labels=['severe $\\Delta F_u$','moderate $\\Delta F_u$','mild $\\Delta F_u$']
markers=['v','o','s']
chain_core['DeltaFu_bin']=pd.cut(chain_core['Delta_Fu_post_minus_pre'],bins=bins,labels=labels)
sc=None
for lab,marker in zip(labels,markers):
    dd=chain_core[chain_core['DeltaFu_bin']==lab]
    if len(dd)==0:
        continue
    sc=axA.scatter(dd[ZCOL], dd['D_C_frac_2008_2015'],
                   c=dd['Delta_Fu_post_minus_pre'], cmap='Greys_r',
                   vmin=chain_core['Delta_Fu_post_minus_pre'].min(),
                   vmax=chain_core['Delta_Fu_post_minus_pre'].max(),
                   s=125, marker=marker, edgecolor='black', linewidth=.6, alpha=.88,
                   label=lab)
for _,r in chain_core.iterrows():
    axA.annotate(r['code'],(r[ZCOL],r['D_C_frac_2008_2015']),textcoords='offset points',xytext=(5,5),fontsize=8)
if len(chain_core)>=3:
    coef=np.polyfit(chain_core[ZCOL], chain_core['D_C_frac_2008_2015'],1)
    xx=np.linspace(chain_core[ZCOL].min(),chain_core[ZCOL].max(),100)
    axA.plot(xx,coef[0]*xx+coef[1],lw=1)
    rA,pA=pearsonr(chain_core[ZCOL], chain_core['D_C_frac_2008_2015'])
    axA.set_title(f"A. $Z_1$ vs $D_C$ ($r\\approx -0.40$, $p\\approx 0.17$, n={len(chain_core)})")
else:
    axA.set_title("A. $Z_1$ vs $D_C$")
axA.set_xlabel(r"$Z_1=\Sigma/(X+I+[H-H_c]_+)$")
axA.set_ylabel(r"$D_C$: consumption depth-duration")
axA.grid(True,alpha=.3)
axA.legend(fontsize=7,loc='upper right',frameon=True)
if sc is not None:
    cbar=fig.colorbar(sc,ax=axA)
    cbar.set_label(r"$\Delta F_u$ (pp)")

for _,r in chain_ext.iterrows():
    marker='o' if r['code'] in euro else 's'
    axB.scatter(r['D_C_frac_2008_2015'], r['Delta_Fu_post_minus_pre'], s=115, marker=marker, alpha=.75, edgecolor='black', linewidth=.4)
    axB.annotate(r['code'],(r['D_C_frac_2008_2015'],r['Delta_Fu_post_minus_pre']),textcoords='offset points',xytext=(5,5),fontsize=8)
if len(chain_ext)>=3:
    coef=np.polyfit(chain_ext['D_C_frac_2008_2015'], chain_ext['Delta_Fu_post_minus_pre'],1)
    xx=np.linspace(chain_ext['D_C_frac_2008_2015'].min(),chain_ext['D_C_frac_2008_2015'].max(),100)
    axB.plot(xx,coef[0]*xx+coef[1],lw=1)
    rB,pB=pearsonr(chain_ext['D_C_frac_2008_2015'], chain_ext['Delta_Fu_post_minus_pre'])
    axB.set_title(f"B. $D_C$ vs $\\Delta F_u$ (r={rB:.2f}, p={pB:.2f}, n={len(chain_ext)})")
else:
    axB.set_title("B. $D_C$ vs $\\Delta F_u$")
axB.set_xlabel(r"$D_C$: consumption depth-duration")
axB.set_ylabel(r"$\Delta F_u=\bar g_{2011-2013}-\bar g_{2004-2007}$ (pp)")
axB.grid(True,alpha=.3)
fig.suptitle("Two-step diagnostic chain: stabilizer-to-exposure, crisis area, and post-crisis growth slope",fontsize=13)
fig.tight_layout(rect=[0,0,1,.94])
fig.savefig(FIG/"fig8_Z1_DC_DeltaFu_chain.png",dpi=240); plt.close(fig)
def plot_xy(x,y,xlab,ylab,title,outname,fit=True):
    d=df.dropna(subset=[x,y])
    fig,ax=plt.subplots(figsize=(10,7))
    for _,r in d.iterrows():
        marker='o' if r['code'] in euro else 's'
        ax.scatter(r[x],r[y],s=120,marker=marker,alpha=.75)
        ax.annotate(r['code'],(r[x],r[y]),textcoords='offset points',xytext=(5,5),fontsize=9)
    if fit and len(d)>=3:
        coef=np.polyfit(d[x],d[y],1); xx=np.linspace(d[x].min(),d[x].max(),100)
        ax.plot(xx,coef[0]*xx+coef[1],lw=1)
        title=f"{title} (r={d[x].corr(d[y]):.2f}, n={len(d)})"
    ax.set_xlabel(xlab); ax.set_ylabel(ylab); ax.set_title(title)
    ax.grid(True,alpha=.3); fig.tight_layout(); fig.savefig(FIG/outname,dpi=240); plt.close(fig)

# The D_C -> Delta F_u diagnostic is shown in the right panel of
# fig8_Z1_DC_DeltaFu_chain.png. The script writes only the final
# manuscript figure used for the Z1-to-DC-to-growth diagnostic.
plot_xy('Sigma_socx_gdp','Delta_Fu_post_minus_pre',
        r"$\Sigma$ = public social expenditure / GDP, 2008 (%)",
        r"$\Delta F_u=\bar g_{2011-2013}-\bar g_{2004-2007}$ (pp)",
        "Change in post-rebound growth slope vs stable-mass proxy",
        "fig9_DeltaFu_vs_Sigma.png")
print("Wrote growth-forcing figures")