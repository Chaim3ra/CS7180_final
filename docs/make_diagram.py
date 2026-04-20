"""Generate docs/model_architecture.png from configs/experiment.yaml values."""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT  = os.path.join(ROOT, "docs", "model_architecture.png")

# ── Hyperparameters (from configs/experiment.yaml) ──────────────────────────
T         = 96     # seq_len
D         = 128    # d_model
NHEAD     = 4
N_LAYERS  = 2
FFN       = 256
M_HID     = 64
FORECAST  = 4
HEAD_HID  = 128

# ── Palette ─────────────────────────────────────────────────────────────────
CW, LW = "#1565C0", "#DDEEFF"    # weather   blue
CG, LG = "#2E7D32", "#DFFCE6"    # gen       green
CM, LM = "#BF360C", "#FFF0E0"    # metadata  orange
CF, LF = "#4A148C", "#F0E6FF"    # fusion    purple
CH, LH = "#880E4F", "#FFE0EE"    # head      pink-red
CO      = "#263238"               # output    dark slate
GRAY    = "#78909C"
BGRAY   = "#546E7A"

# ── Figure setup ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(15, 21))
ax.set_xlim(0, 15)
ax.set_ylim(0, 21)
ax.axis("off")
fig.patch.set_facecolor("white")
ax.set_facecolor("white")


# ── Drawing helpers ─────────────────────────────────────────────────────────

def rbox(cx, cy, w, h, fc, ec, lw=2.0, zorder=3, rad=0.12):
    p = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle=f"round,pad={rad}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder,
    )
    ax.add_patch(p)


def txt(cx, cy, s, fs=9, c="black", fw="normal", ha="center", va="center",
        ls=1.5, zorder=5, fam="sans-serif"):
    ax.text(cx, cy, s, fontsize=fs, color=c, fontweight=fw,
            ha=ha, va=va, zorder=zorder, linespacing=ls,
            multialignment="center", fontfamily=fam)


def dim(cx, cy, s):
    ax.text(cx, cy, s, fontsize=7.5, color=BGRAY, style="italic",
            ha="center", va="center", fontfamily="monospace", zorder=6,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.5))


def arr(x1, y1, x2, y2, c=GRAY, lw=1.6, ms=13):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color=c, lw=lw, mutation_scale=ms),
        zorder=2,
    )


def elbow(x1, y1, x2, y2, c=GRAY, lw=1.6):
    """Arrow that goes vertically then horizontally (corner at (x1, y2))."""
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="->", color=c, lw=lw, mutation_scale=13,
            connectionstyle="angle,angleA=90,angleB=0",
        ),
        zorder=2,
    )


def hrule(cx, cy, w, c=GRAY):
    ax.plot([cx - w / 2, cx + w / 2], [cy, cy], color=c, lw=0.8, zorder=4)


# ── Stream x-centres ────────────────────────────────────────────────────────
WX, GX, MX = 2.8, 7.5, 12.5
FX = (WX + GX) / 2  # fusion midpoint  ≈ 5.15
CX = 7.5            # concat / head / output

# ============================================================
# TITLE
# ============================================================
txt(7.5, 20.4, "SolarForecastModel — Multi-Modal Solar Generation Forecasting",
    fs=14, fw="bold", c="#1A237E")
txt(7.5, 19.85,
    "Train: Texas + California   |   Zero-Shot Transfer Test: New York",
    fs=9.5, c=GRAY)

# ============================================================
# ROW A — INPUTS
# ============================================================
Y_IN = 18.6
IN_H, IN_W = 1.5, 3.9

for cx, ec, lc, title, detail in [
    (WX, CW, LW, "Weather Time-Series",
     f"GHI · DNI · DHI\nTemp · Wind · Humidity"),
    (GX, CG, LG, "Generation History",
     f"Past kWh readings\n(15-min intervals)"),
    (MX, CM, LM, "Site Metadata",
     "lat · lon · tilt\nazimuth · cap · elev"),
]:
    rbox(cx, Y_IN, IN_W, IN_H, lc, ec)
    txt(cx, Y_IN + 0.32, title, fs=9, fw="bold", c=ec)
    hrule(cx, Y_IN + 0.04, IN_W - 0.3, ec)
    txt(cx, Y_IN - 0.32, detail, fs=8, c="#333333")

dim(WX, Y_IN - 0.92, f"(B, {T}, 6)")
dim(GX, Y_IN - 0.92, f"(B, {T}, 1)")
dim(MX, Y_IN - 0.92, "(B, 6)")

# ============================================================
# ROW B — ENCODERS
# ============================================================
Y_ENC = 14.45
ENC_H, ENC_W = 4.0, 4.1

# Weather
rbox(WX, Y_ENC, ENC_W, ENC_H, LW, CW, lw=2.2)
txt(WX, Y_ENC + 1.65, "Weather Encoder", fs=10, fw="bold", c=CW)
hrule(WX, Y_ENC + 1.30, ENC_W - 0.4, CW)
txt(WX, Y_ENC + 0.75,
    f"Input Proj  6 → {D}\n+ Sinusoidal Pos. Enc.",
    fs=8.5, c="#222")
hrule(WX, Y_ENC + 0.18, ENC_W - 0.4, "#BBBBBB")
txt(WX, Y_ENC - 0.65,
    f"{N_LAYERS}× TransformerEncoderLayer\n"
    f"  ├─ MHA  ({NHEAD} heads, d = {D})\n"
    f"  ├─ FFN  {D}→{FFN}→{D}\n"
    f"  └─ LayerNorm + Dropout",
    fs=8, c="#333", ls=1.6)

# Generation
rbox(GX, Y_ENC, ENC_W, ENC_H, LG, CG, lw=2.2)
txt(GX, Y_ENC + 1.65, "Generation Encoder", fs=10, fw="bold", c=CG)
hrule(GX, Y_ENC + 1.30, ENC_W - 0.4, CG)
txt(GX, Y_ENC + 0.75,
    f"Input Proj  1 → {D}\n+ Sinusoidal Pos. Enc.",
    fs=8.5, c="#222")
hrule(GX, Y_ENC + 0.18, ENC_W - 0.4, "#BBBBBB")
txt(GX, Y_ENC - 0.65,
    f"{N_LAYERS}× TransformerEncoderLayer\n"
    f"  ├─ MHA  ({NHEAD} heads, d = {D})\n"
    f"  ├─ FFN  {D}→{FFN}→{D}\n"
    f"  └─ LayerNorm + Dropout",
    fs=8, c="#333", ls=1.6)

# Metadata (shorter — MLP has fewer layers)
META_ENC_H = 3.0
rbox(MX, Y_ENC + 0.5, ENC_W, META_ENC_H, LM, CM, lw=2.2)
txt(MX, Y_ENC + 0.5 + 1.1, "Metadata Encoder (MLP)", fs=10, fw="bold", c=CM)
hrule(MX, Y_ENC + 0.5 + 0.75, ENC_W - 0.4, CM)
txt(MX, Y_ENC + 0.5 - 0.35,
    f"Linear  6 → {M_HID}\nLayerNorm + ReLU\n"
    f"Linear  {M_HID} → {D}\nLayerNorm + ReLU",
    fs=8.5, c="#333", ls=1.6)

# ── Input → encoder arrows ──
ENC_TOP = Y_ENC + ENC_H / 2
for cx in (WX, GX):
    arr(cx, Y_IN - IN_H / 2, cx, ENC_TOP)
META_ENC_TOP = Y_ENC + 0.5 + META_ENC_H / 2
arr(MX, Y_IN - IN_H / 2, MX, META_ENC_TOP)

# ── Encoder output shape labels ──
ENC_BOT = Y_ENC - ENC_H / 2
META_ENC_BOT = Y_ENC + 0.5 - META_ENC_H / 2
dim(WX, ENC_BOT - 0.32, f"(B, {T}, {D})")
dim(GX, ENC_BOT - 0.32, f"(B, {T}, {D})")
dim(MX, META_ENC_BOT - 0.32, f"(B, {D})")

# ============================================================
# ROW C — CROSS-ATTENTION FUSION
# ============================================================
Y_FUS = 9.80
FUS_H, FUS_W = 2.5, 8.2

rbox(FX, Y_FUS, FUS_W, FUS_H, LF, CF, lw=2.5)
txt(FX, Y_FUS + 0.85, "Cross-Attention Fusion", fs=11, fw="bold", c=CF)
hrule(FX, Y_FUS + 0.50, FUS_W - 0.5, CF)
txt(FX, Y_FUS - 0.05,
    f"Query  ←  Generation Encoder   (B, {T}, {D})\n"
    f"Key, Value  ←  Weather Encoder  (B, {T}, {D})",
    fs=9, c="#333", ls=1.55)
txt(FX, Y_FUS - 0.80,
    f"+ Residual Connection  +  LayerNorm  →  Mean Pool over T",
    fs=8.5, c="#555")

FUS_TOP = Y_FUS + FUS_H / 2
FUS_BOT = Y_FUS - FUS_H / 2

# Weather → fusion (K, V label)
arr(WX, ENC_BOT, WX, FUS_TOP, c=CW)
ax.text(WX - 0.45, (ENC_BOT + FUS_TOP) / 2,
        "K, V", fontsize=8.5, color=CF, fontweight="bold",
        ha="right", va="center", zorder=6)

# Generation → fusion (Q label)
arr(GX, ENC_BOT, GX, FUS_TOP, c=CG)
ax.text(GX + 0.2, (ENC_BOT + FUS_TOP) / 2,
        "Q", fontsize=8.5, color=CF, fontweight="bold",
        ha="left", va="center", zorder=6)

dim(FX, FUS_BOT - 0.32, f"(B, {D})")

# ============================================================
# ROW D — CONCATENATE
# ============================================================
Y_CAT = 7.95
CAT_H, CAT_W = 1.1, 9.0

rbox(CX, Y_CAT, CAT_W, CAT_H, "#ECEFF1", BGRAY, lw=2.0)
txt(CX, Y_CAT + 0.18, "Concatenate", fs=10, fw="bold", c="#37474F")
txt(CX, Y_CAT - 0.22,
    f"Fused (B, {D})  ⊕  Metadata (B, {D})",
    fs=8.5, c="#555")

CAT_TOP = Y_CAT + CAT_H / 2
CAT_BOT = Y_CAT - CAT_H / 2

# Fusion → concat (straight down from FX to CX via elbow)
arr(FX, FUS_BOT, FX, CAT_TOP + 0.05, c=CF)

# Metadata encoder → concat (elbow: down then left)
elbow(MX, META_ENC_BOT, CX + CAT_W / 2 - 0.25, Y_CAT, c=CM, lw=1.6)

dim(CX, CAT_BOT - 0.32, f"(B, {2 * D})")

# ============================================================
# ROW E — REGRESSION HEAD
# ============================================================
Y_HEAD = 5.95
HEAD_H, HEAD_W = 2.2, 6.2

arr(CX, CAT_BOT, CX, Y_HEAD + HEAD_H / 2, c=CO)

rbox(CX, Y_HEAD, HEAD_W, HEAD_H, LH, CH, lw=2.2)
txt(CX, Y_HEAD + 0.75, "Regression Head (MLP)", fs=11, fw="bold", c=CH)
hrule(CX, Y_HEAD + 0.40, HEAD_W - 0.5, CH)
txt(CX, Y_HEAD - 0.22,
    f"Linear  {2*D} → {HEAD_HID}  +  ReLU  +  Dropout(0.1)\n"
    f"Linear  {HEAD_HID} → {FORECAST}",
    fs=9.5, c="#333", ls=1.6)

HEAD_BOT = Y_HEAD - HEAD_H / 2

# ============================================================
# ROW F — OUTPUT
# ============================================================
Y_OUT = 3.95
OUT_H, OUT_W = 1.35, 7.0

arr(CX, HEAD_BOT, CX, Y_OUT + OUT_H / 2, c=CO)

rbox(CX, Y_OUT, OUT_W, OUT_H, CO, CO, lw=2.0)
txt(CX, Y_OUT + 0.25, "Forecast Output", fs=11, fw="bold", c="white")
txt(CX, Y_OUT - 0.22,
    f"(B, {FORECAST})  —  kWh per 15-min step  ·  Horizon: {FORECAST} steps (1 h ahead)",
    fs=9, c="#BBDEFB")

# ============================================================
# FOOTER
# ============================================================
txt(7.5, 2.85,
    f"640K parameters  ·  AdamW (lr=1e-4)  ·  Cosine LR  ·  MSE loss  ·  PyTorch Lightning",
    fs=9, c=GRAY)
txt(7.5, 2.38,
    f"B = batch size  ·  T = {T} (24 h context at 15-min resolution)  ·  d = {D}  ·  dropout = 0.1",
    fs=8.5, c=GRAY, fam="monospace")

# ============================================================
# LEGEND
# ============================================================
legend_items = [
    (CW, "Weather stream"),
    (CG, "Generation stream"),
    (CM, "Metadata stream"),
    (CF, "Cross-attention fusion"),
    (CH, "Regression head"),
]
y_leg = 1.6
x0 = 0.7
for i, (c, lbl) in enumerate(legend_items):
    xi = x0 + i * 2.85
    ax.plot([xi, xi + 0.45], [y_leg, y_leg], color=c, lw=4, solid_capstyle="round", zorder=5)
    txt(xi + 0.6, y_leg, lbl, fs=8, c="#333", ha="left", zorder=5)

ax.plot([0.2, 14.8], [1.15, 1.15], color="#E0E0E0", lw=0.8)
txt(7.5, 0.7, "CS7180 Applied Deep Learning — Final Project  ·  Cunningham · Demidov · Agarwal",
    fs=8.5, c="#AAAAAA")

plt.savefig(OUT, dpi=160, bbox_inches="tight", facecolor="white")
print(f"Saved: {OUT}")
