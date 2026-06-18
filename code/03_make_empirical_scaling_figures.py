"""Reproduce empirical scaling figures from processed reproducibility data.
Run from package root: python code/03_make_empirical_scaling_figures.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
FIG=ROOT/"figures"; DATA=ROOT/"data"
FIG.mkdir(exist_ok=True)

df=pd.read_csv(DATA/"country_archetypes_table.csv")
df=df.replace([np.inf,-np.inf],np.nan)
ZCOL="Z1_trueH_Hc60"
required=["archetype",ZCOL,"R_buffer_2009"]
missing=[c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required canonical columns in country_archetypes_table.csv: {missing}")
core=df[df["archetype"].notna() & df[ZCOL].notna() & df["R_buffer_2009"].notna()].copy()
# Exclude reference point from fit/median lines
fit=core[core["code"]!="ISR"].copy()
euro={"AUT","BEL","DEU","ESP","FIN","FRA","IRL","ITA","NLD"}

fig,ax=plt.subplots(figsize=(10,7))
for _,r in fit.iterrows():
    marker='o' if r['code'] in euro else 's'
    dc=r.get('D_C_frac_2008_2015',np.nan)
    size=90+750*(0 if pd.isna(dc) else dc/max(fit['D_C_frac_2008_2015'].max(),1e-12))
    ax.scatter(r[ZCOL],r['R_buffer_2009'],s=size,marker=marker,alpha=.72)
    ax.annotate(r['code'],(r[ZCOL],r['R_buffer_2009']),textcoords='offset points',xytext=(5,5),fontsize=9)
ref=core[core['code']=='ISR']
if len(ref):
    r=ref.iloc[0]
    ax.scatter(r[ZCOL],r['R_buffer_2009'],s=250,marker='*',alpha=.95)
    ax.annotate('ISR\nnon-recession',(r[ZCOL],r['R_buffer_2009']),textcoords='offset points',xytext=(8,8),fontsize=9)
ax.axhline(fit['R_buffer_2009'].median(),ls='--',lw=1); ax.axvline(fit[ZCOL].median(),ls='--',lw=1)
ax.set_xlabel(r"$Z_1=\Sigma/(X+I+[H-H_c]_+)$")
ax.set_ylabel(r"$R_{2009}=g_C^{real}-g_Y^{real}$ (pp)")
ax.set_title("Country archetypes: stabilizer-to-exposure ratio vs relative insulation")
ax.grid(True,alpha=.3); fig.tight_layout(); fig.savefig(FIG/"fig6_country_archetypes_Z1_R.png",dpi=240); plt.close(fig)

fig,ax=plt.subplots(figsize=(10,7))
d=fit.dropna(subset=[ZCOL,'D_C_frac_2008_2015'])
for _,r in d.iterrows():
    marker='o' if r['code'] in euro else 's'
    ax.scatter(r[ZCOL],r['D_C_frac_2008_2015'],s=110,marker=marker,alpha=.75)
    ax.annotate(r['code'],(r[ZCOL],r['D_C_frac_2008_2015']),textcoords='offset points',xytext=(5,5),fontsize=9)
if len(d)>=3:
    coef=np.polyfit(d[ZCOL],d['D_C_frac_2008_2015'],1)
    xx=np.linspace(d[ZCOL].min(),d[ZCOL].max(),100)
    ax.plot(xx,coef[0]*xx+coef[1],lw=1)
ax.set_xlabel(r"$Z_1$"); ax.set_ylabel(r"$D_C$: consumption depth-duration")
ax.set_title("Country archetypes on the time-axis observable")
ax.grid(True,alpha=.3); fig.tight_layout(); fig.savefig(FIG/"fig7_country_archetypes_Z1_DC.png",dpi=240); plt.close(fig)
print("Wrote empirical scaling figures")
