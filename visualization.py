import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── Data ──────────────────────────────────────────────────────────────────────
models = [
    "Advection\n(Tier 1)",
    "PC-RF\n(Tier 3)",
    "LSTM\n(vanilla)",
    "LSTM\n(phys-residual)",
    "LSTM\n(phys-res+loss)",
    "Transformer\n(phys-residual)",
]

mae_lat  = [0.05097, 0.04022, 0.03497, 0.03429, 0.03427, 0.03576]
mae_lon  = [0.07267, 0.05906, 0.05311, 0.05254, 0.05273, 0.05433]
mean_geo = [8.721,   6.859,   6.068,   5.880,   5.885,   8.570]
med_geo  = [6.610,   5.147,   4.491,   4.280,   4.276,   5.032]

COLORS      = ["#7F77DD", "#1D9E75", "#378ADD", "#D85A30", "#C0472A", "#BA7517"]
COLORS_LITE = ["#AFA9EC", "#5DCAA5", "#85B7EB", "#F0997B", "#E07060", "#EF9F27"]
TEXT   = "#e8edf2"
MUTE   = "#9aaabb"
ACCENT = "#F0997B"

x     = np.arange(len(models)) * 0.78   # compress group spacing
width = 0.34          # slightly narrower to fit 6 groups cleanly

# font size constants — change here to tune globally
FS_TICK   = 15        # x/y tick labels
FS_VAL    = 13        # value labels on bars
FS_ANNOT  = 13        # bracket % and arrow text
FS_YLABEL = 13        # axis ylabel
FS_TITLE  = 16        # subplot titles
FS_LEGEND = 13        # all legends
FS_SUPT   = 16        # figure suptitle

def style_ax(ax):
    ax.set_facecolor("none")
    ax.tick_params(colors=MUTE, labelsize=FS_TICK)
    ax.spines[:].set_visible(False)
    ax.yaxis.grid(True, color=(1, 1, 1, 0.10), linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 9), facecolor="none")
gs  = GridSpec(2, 1, figure=fig,
               hspace=0.38,
               left=0.07, right=0.97,
               top=0.91, bottom=0.13)

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
style_ax(ax1)
style_ax(ax2)

# ── Chart 1: Geodesic Error ───────────────────────────────────────────────────
b1 = ax1.bar(x - width / 2, mean_geo, width,
             color=COLORS, alpha=0.92, zorder=3, linewidth=0)
b2 = ax1.bar(x + width / 2, med_geo,  width,
             color=COLORS_LITE, alpha=0.80, zorder=3,
             linewidth=1.4, edgecolor=COLORS)

for rect, val in zip(list(b1) + list(b2), mean_geo + med_geo):
    ax1.text(rect.get_x() + rect.get_width() / 2,
             rect.get_height() + 0.18,
             f"{val:.2f}", ha="center", va="bottom",
             fontsize=FS_VAL, color=TEXT, fontweight="bold")

# bracket annotations: Advection→PC-RF, PC-RF→LSTM vanilla, vanilla→phys-residual
for (i, j), yoff in zip([(0, 1), (1, 2), (2, 3)], [1.8, 1.4, 1.0]):
    pct = (mean_geo[i] - mean_geo[j]) / mean_geo[i] * 100
    xf  = x[i] - width / 2
    xt  = x[j] - width / 2
    yl  = max(mean_geo[i], mean_geo[j]) + yoff
    ax1.annotate("", xy=(xt, yl), xytext=(xf, yl),
                 arrowprops=dict(arrowstyle="-", color=ACCENT, lw=1.1))
    for xp in [xf, xt]:
        ax1.plot([xp, xp], [yl - 0.28, yl], color=ACCENT, lw=1.1)
    ax1.text((xf + xt) / 2, yl + 0.12, f"−{pct:.0f}%",
             ha="center", va="bottom",
             fontsize=FS_ANNOT, color=ACCENT, fontstyle="italic")

ax1.set_xticks(x)
ax1.set_xticklabels(models, color=TEXT, fontsize=FS_TICK)
ax1.set_ylabel("km", color=MUTE, fontsize=FS_YLABEL)
ax1.set_ylim(0, 13.0)
ax1.set_title("Geodesic Error  (↓ lower is better)",
              color=TEXT, fontsize=FS_TITLE, fontweight="bold", pad=10)

pm = mpatches.Patch(facecolor="#aaa", label="Mean geo (km)")
pd = mpatches.Patch(facecolor="#aaa", edgecolor="#aaa",
                    linewidth=1.4, label="Median geo (km)", alpha=0.65)
ax1.legend(handles=[pm, pd], fontsize=FS_LEGEND,
           frameon=False, labelcolor=MUTE, loc="upper right")

# ── Chart 2: MAE lat / lon ────────────────────────────────────────────────────
b3 = ax2.bar(x - width / 2, mae_lat, width,
             color=COLORS, alpha=0.92, zorder=3, linewidth=0)
b4 = ax2.bar(x + width / 2, mae_lon, width,
             color=COLORS_LITE, alpha=0.80, zorder=3,
             linewidth=1.4, edgecolor=COLORS)

for rect, val in zip(list(b3) + list(b4), mae_lat + mae_lon):
    ax2.text(rect.get_x() + rect.get_width() / 2,
             rect.get_height() + 0.0007,
             f"{val:.4f}", ha="center", va="bottom",
             fontsize=FS_VAL - 0.5, color=TEXT, fontweight="bold")

ax2.set_xticks(x)
ax2.set_xticklabels(models, color=TEXT, fontsize=FS_TICK)
ax2.set_ylabel("degrees (°)", color=MUTE, fontsize=FS_YLABEL)
ax2.set_ylim(0, 0.098)
ax2.set_title("MAE lat / lon  (↓ lower is better)",
              color=TEXT, fontsize=FS_TITLE, fontweight="bold", pad=10)

ax2.annotate("lon error > lat error\nacross all models",
             xy=(5 + width / 2, mae_lon[5]),
             xytext=(3.2, 0.083),
             fontsize=FS_ANNOT - 1, color=ACCENT, fontstyle="italic",
             arrowprops=dict(arrowstyle="->", color=ACCENT,
                             lw=0.9, connectionstyle="arc3,rad=0.25"))

pl = mpatches.Patch(facecolor="#aaa", label="MAE lat (°)")
pn = mpatches.Patch(facecolor="#aaa", edgecolor="#aaa",
                    linewidth=1.4, label="MAE lon (°)", alpha=0.65)
ax2.legend(handles=[pl, pn], fontsize=FS_LEGEND,
           frameon=False, labelcolor=MUTE, loc="upper right")

# ── Shared model color legend ─────────────────────────────────────────────────
patches = [mpatches.Patch(facecolor=c, label=m.replace("\n", " "))
           for c, m in zip(COLORS, models)]
fig.legend(handles=patches, loc="lower center", ncol=6,
           fontsize=FS_LEGEND, frameon=False, labelcolor=TEXT,
           bbox_to_anchor=(0.52, 0.01))

fig.suptitle("Model Performance Comparison",
             color=TEXT, fontsize=FS_SUPT, fontweight="bold", y=0.97)

plt.savefig("model_comparison.png", dpi=180,
            bbox_inches="tight", transparent=True)
print("Saved: model_comparison.png")
plt.show()