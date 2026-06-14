"""Generate polished supplementary GIFs for the stable-inertial-mass toy model.

Inputs expected relative to the project root:
  supplementary/movie_toy_trajectories.csv
  supplementary/movie_parameters.csv

Outputs:
  supplementary/movie_S1_physical_role_to_role_oscillator_v2.gif
  supplementary/movie_S2_economic_coordinates_v2.gif

This script is intentionally self-contained and uses only pandas, numpy, and pillow.
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SUPP = ROOT / "supplementary"
traj_path = SUPP / "movie_toy_trajectories.csv"
params_path = SUPP / "movie_parameters.csv"

traj = pd.read_csv(traj_path)
params = pd.read_csv(params_path).iloc[0].to_dict()

regimes = ["fragile-fast", "balanced", "stable-sluggish"]
colors = {
    "fragile-fast": (31, 119, 180),
    "balanced": (255, 127, 14),
    "stable-sluggish": (44, 160, 44),
}
markers = {"fragile-fast": "circle", "balanced": "diamond", "stable-sluggish": "square"}
data = {r: traj[traj["regime"] == r].reset_index(drop=True) for r in regimes}
t = data[regimes[0]]["t"].to_numpy()
x_c = float(params["x_c"])
idxs = np.linspace(0, len(t)-1, 88).astype(int)
duration_ms = 85

def load_font(size, bold=False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for c in candidates:
        try:
            return ImageFont.truetype(c, size)
        except Exception:
            pass
    return ImageFont.load_default()

title_f, sub_f, font, small, tiny = load_font(24, True), load_font(15, True), load_font(14), load_font(12), load_font(10)

def draw_marker(draw, x, y, marker, fill, size=8, outline=(0,0,0)):
    if marker == "circle":
        draw.ellipse([x-size, y-size, x+size, y+size], fill=fill, outline=outline, width=1)
    elif marker == "diamond":
        draw.polygon([(x, y-size-2), (x+size+2, y), (x, y+size+2), (x-size-2, y)], fill=fill, outline=outline)
    elif marker == "square":
        draw.rectangle([x-size, y-size, x+size, y+size], fill=fill, outline=outline)
    else:
        draw.ellipse([x-size, y-size, x+size, y+size], fill=fill, outline=outline)

def draw_line(draw, pts, fill, width=3):
    if len(pts) >= 2:
        draw.line(pts, fill=fill, width=width)

def axis_map(xv, yv, xlim, ylim, px, py, pw, ph):
    X = px + (xv - xlim[0])/(xlim[1]-xlim[0]) * pw
    Y = py + ph - (yv - ylim[0])/(ylim[1]-ylim[0]) * ph
    return X, Y

def draw_axes(draw, px, py, pw, ph, title, xlabel="", ylabel=""):
    draw.rectangle([px, py, px+pw, py+ph], outline=(160,160,160), width=1)
    if title:
        draw.text((px, py-28), title, fill=(0,0,0), font=sub_f)
    if xlabel:
        draw.text((px+pw-70, py+ph+8), xlabel, fill=(70,70,70), font=small)
    if ylabel:
        draw.text((px, py-14), ylabel, fill=(70,70,70), font=tiny)

def draw_legend(draw, x0, y0, spacing=26):
    for i, r in enumerate(regimes):
        y = y0 + i*spacing
        draw_marker(draw, x0+8, y+8, markers[r], colors[r], size=7)
        draw.text((x0+24, y), r, fill=(0,0,0), font=small)

def arrow(draw, x0,y0,x1,y1, fill, width=3):
    draw.line([(x0,y0),(x1,y1)], fill=fill, width=width)
    ang = np.arctan2(y1-y0, x1-x0)
    L = 12
    for da in [2.55, -2.55]:
        xa = x1 + L*np.cos(ang+da)
        ya = y1 + L*np.sin(ang+da)
        draw.line([(x1,y1),(xa,ya)], fill=fill, width=width)

# Movie S1
W,H = 1440, 720
margin_x, panel_gap = 70, 35
panel_w = int((W - 2*margin_x - 2*panel_gap) / 3)
panel_h, panel_y = 365, 155
xlim, ylim = (-0.15, 4.85), (-0.60, 1.48)
frames = []
for j in idxs:
    im = Image.new("RGB", (W,H), "white")
    d = ImageDraw.Draw(im)
    d.text((50, 32), "Movie S1: role-to-role physical oscillator analogy", fill=(0,0,0), font=title_f)
    d.text((50, 70), "Horizontal: growth coordinate u(t). Vertical: transverse crisis displacement x(t). Red line: crisis threshold.", fill=(50,50,50), font=font)
    for p, r in enumerate(regimes):
        df = data[r]
        S, m, c = df["S"].iloc[0], df["m"].iloc[0], df["c_x"].iloc[0]
        px, py = margin_x + p*(panel_w+panel_gap), panel_y
        d.text((px, py-62), r, fill=(0,0,0), font=sub_f)
        d.text((px, py-39), f"S={S:.2f}   m={m:.2f}   c={c:.2f}", fill=(60,60,60), font=small)
        draw_axes(d, px, py, panel_w, panel_h, "", "u", "x")
        _, y0 = axis_map(0, 0, xlim, ylim, px, py, panel_w, panel_h)
        _, yc = axis_map(0, x_c, xlim, ylim, px, py, panel_w, panel_h)
        d.line([(px, y0), (px+panel_w, y0)], fill=(200,200,200), width=1)
        d.line([(px, yc), (px+panel_w, yc)], fill=(210,0,0), width=2)
        d.text((px+8, yc-19), "x_c", fill=(180,0,0), font=small)
        pts = [axis_map(df["u"].iloc[q], df["x"].iloc[q], xlim, ylim, px, py, panel_w, panel_h) for q in range(j+1)]
        draw_line(d, pts, colors[r], 3)
        X,Y = axis_map(df["u"].iloc[j], df["x"].iloc[j], xlim, ylim, px, py, panel_w, panel_h)
        draw_marker(d, X, Y, markers[r], colors[r], size=int(8+24*S))
        Fu, Fx = df["F_u"].iloc[j], df["F_x"].iloc[j]
        if Fu > 0.03:
            arrow(d, X, Y, X+75*Fu, Y, (0,120,0), 3)
            d.text((X+75*Fu+4, Y-16), "F_u", fill=(0,100,0), font=small)
        if Fx > 0.03:
            arrow(d, X, Y, X, Y-60*Fx, (180,0,0), 3)
            d.text((X+6, Y-60*Fx-15), "F_x", fill=(170,0,0), font=small)
        crossed = "yes" if df["x"].iloc[:j+1].max() > x_c else "no"
        d.text((px, py+panel_h+24), f"t={df['t'].iloc[j]:.1f}", fill=(0,0,0), font=small)
        d.text((px+80, py+panel_h+24), f"threshold crossed: {crossed}", fill=(0,0,0), font=small)
    d.text((50, H-45), "Concept animation only: finite-pulse response; no second attractor is included.", fill=(70,70,70), font=small)
    frames.append(im)
frames[0].save(SUPP/"movie_S1_physical_role_to_role_oscillator_v2.gif", save_all=True, append_images=frames[1:], duration=duration_ms, loop=0, optimize=False)

# Movie S2
W,H = 1440, 960
left, right_legend_x, plot_w, plot_h, plot_gap, top = 95, 1120, 930, 165, 62, 130
panels = {k:(left, top+i*(plot_h+plot_gap)) for i,k in enumerate(["x","u","D","phase"])}
x_ylim, u_ylim = (-0.58, 1.43), (-0.15, 4.75)
D_ylim = (0, max(data[r]["D_x_cumulative"].iloc[-1] for r in regimes)*1.08)
acc_ylim = (0.23, 0.34)
frames = []
for j in idxs:
    im = Image.new("RGB", (W,H), "white")
    d = ImageDraw.Draw(im)
    d.text((55, 28), "Movie S2: economic coordinates of the same toy dynamics", fill=(0,0,0), font=title_f)
    d.text((55, 66), "Regimes are overlaid on shared axes for direct visual comparison.", fill=(50,50,50), font=font)
    draw_legend(d, right_legend_x, 42)
    draw_axes(d, *panels["x"], plot_w, plot_h, "A. Crisis coordinate  x(t)", "time", "x")
    draw_axes(d, *panels["u"], plot_w, plot_h, "B. Growth coordinate  u(t)", "time", "u")
    draw_axes(d, *panels["D"], plot_w, plot_h, "C. Accumulated crisis area  D_x(t)=∫x_+(t)dt", "time", "D_x")
    draw_axes(d, *panels["phase"], plot_w, plot_h, "D. Phase trace: accumulated crisis area vs max growth acceleration", "max accel", "D_x")
    px,py = panels["x"]
    _, ythr = axis_map(0, x_c, (t.min(),t.max()), x_ylim, px, py, plot_w, plot_h)
    d.line([(px,ythr),(px+plot_w,ythr)], fill=(210,0,0), width=2)
    d.text((px+8, ythr-19), "x_c", fill=(180,0,0), font=small)
    for r in regimes:
        df, col = data[r], colors[r]
        for colname, key, ylim_now in [("x", "x", x_ylim), ("u", "u", u_ylim), ("D_x_cumulative", "D", D_ylim)]:
            pts = [axis_map(df["t"].iloc[q], df[colname].iloc[q], (t.min(),t.max()), ylim_now, *panels[key], plot_w, plot_h) for q in range(j+1)]
            draw_line(d, pts, col, 3)
            X,Y = axis_map(df["t"].iloc[j], df[colname].iloc[j], (t.min(),t.max()), ylim_now, *panels[key], plot_w, plot_h)
            draw_marker(d, X, Y, markers[r], col, size=6)
        hist_pts = []
        for q in range(0, j+1, max(1, len(t)//120)):
            hist_pts.append(axis_map(np.max(df["u_accel"].iloc[:q+1]), df["D_x_cumulative"].iloc[q], acc_ylim, D_ylim, *panels["phase"], plot_w, plot_h))
        draw_line(d, hist_pts, col, 2)
        X,Y = axis_map(np.max(df["u_accel"].iloc[:j+1]), df["D_x_cumulative"].iloc[j], acc_ylim, D_ylim, *panels["phase"], plot_w, plot_h)
        draw_marker(d, X,Y, markers[r], col, size=7)
    for key in ["x","u","D"]:
        px,py = panels[key]
        Xt,_ = axis_map(t[j], 0, (t.min(),t.max()), (0,1), px, py, plot_w, plot_h)
        d.line([(Xt,py),(Xt,py+plot_h)], fill=(130,130,130), width=1)
    d.text((right_legend_x, 145), f"time t = {t[j]:.1f}", fill=(0,0,0), font=font)
    d.text((right_legend_x, 190), "Takeaway:", fill=(0,0,0), font=sub_f)
    d.multiline_text((right_legend_x, 218), "Low S is fast but\nthreshold-prone.\nHigh S suppresses\nx(t) but reduces\nacceleration.", fill=(50,50,50), font=small, spacing=4)
    d.text((55, H-38), "Same trajectory data as Movie S1. Toy model only; not a fitted country simulation.", fill=(70,70,70), font=small)
    frames.append(im)
frames[0].save(SUPP/"movie_S2_economic_coordinates_v2.gif", save_all=True, append_images=frames[1:], duration=duration_ms, loop=0, optimize=False)

print("Wrote polished supplementary movies to", SUPP)
