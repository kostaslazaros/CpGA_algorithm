"""
Three publication-quality figures for the GA feature-selection paper.

Output: ./figures_v2/  (300 dpi PNG)
"""

from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.lines as mlines

# ---------------------------------------------------------------------------
# Publication style
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.linewidth": 0.9,
    "axes.edgecolor": "#333333",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.9,
    "ytick.major.width": 0.9,
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "axes.labelcolor": "#222222",
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
})

# ---------------------------------------------------------------------------
# Data (as provided by the user's run)
# ---------------------------------------------------------------------------
DATA = [
    # (display_name, comb, contrast, genes, compress, 1-compress,
    #  cpg/gene,
    #  knn, svm, lr, dt, nb,
    #  category)
    ("CpGA",         0.662, 1.1456,  81, 0.388, 0.612, 2.58, 0.530, 0.816, 0.778, 0.667, 0.503, "proposed"),
    ("relieff",      0.520, 1.3742, 149, 0.713, 0.287, 1.40, 0.648, 0.775, 0.786, 0.671, 0.572, "statistical"),
    ("chi2",         0.482, 1.3724, 154, 0.737, 0.263, 1.36, 0.594, 0.785, 0.776, 0.647, 0.590, "statistical"),
    ("extratrees",   0.480, 1.1731, 172, 0.823, 0.177, 1.22, 0.813, 0.876, 0.874, 0.715, 0.697, "embedded"),
    ("mrmr",         0.464, 1.3560, 154, 0.737, 0.263, 1.36, 0.630, 0.794, 0.776, 0.671, 0.484, "statistical"),
    ("anova_f",      0.462, 1.3560, 154, 0.737, 0.263, 1.36, 0.630, 0.794, 0.776, 0.664, 0.484, "statistical"),
    ("fisher_score", 0.462, 1.3560, 154, 0.737, 0.263, 1.36, 0.630, 0.794, 0.776, 0.664, 0.484, "statistical"),
    ("forward_lr",   0.415, 1.3550, 158, 0.756, 0.244, 1.32, 0.610, 0.785, 0.760, 0.625, 0.463, "wrapper"),
    ("mutual_info",  0.397, 1.2150, 172, 0.823, 0.177, 1.22, 0.686, 0.817, 0.795, 0.723, 0.591, "statistical"),
    ("l1_logistic",  0.374, 0.9145, 180, 0.861, 0.139, 1.16, 0.846, 0.902, 0.931, 0.628, 0.826, "embedded"),
    ("variance",     0.341, 1.2442, 178, 0.852, 0.148, 1.17, 0.563, 0.787, 0.766, 0.724, 0.617, "statistical"),
    ("adaboost",     0.262, 0.8668, 191, 0.914, 0.086, 1.09, 0.763, 0.866, 0.876, 0.690, 0.778, "embedded"),
    ("xgboost",      0.231, 0.9072, 195, 0.933, 0.067, 1.07, 0.748, 0.850, 0.840, 0.689, 0.731, "embedded"),
    ("lightgbm",     0.217, 0.8515, 194, 0.928, 0.072, 1.08, 0.679, 0.870, 0.877, 0.691, 0.767, "embedded"),
    ("rfe_svm",      0.211, 0.8164, 199, 0.952, 0.048, 1.05, 0.691, 0.852, 0.949, 0.756, 0.745, "wrapper"),
]

# Pretty labels (no underscores) for axis tick labels in fig 2
PRETTY = {
    "CpGA":         "CpGA",
    "relieff":      "ReliefF",
    "chi2":         "Chi$^2$",
    "extratrees":   "ExtraTrees",
    "mrmr":         "mRMR",
    "anova_f":      "ANOVA-F",
    "fisher_score": "Fisher Score",
    "forward_lr":   "Forward LR",
    "mutual_info":  "Mutual Info",
    "l1_logistic":  "L1 LR",
    "variance":     "Variance",
    "adaboost":     "AdaBoost",
    "xgboost":      "XGBoost",
    "lightgbm":     "LightGBM",
    "rfe_svm":      "RFE-SVM",
}

methods       = [r[0] for r in DATA]
combined      = np.array([r[1] for r in DATA])
contrast      = np.array([r[2] for r in DATA])
compactness   = np.array([r[5] for r in DATA])
cpg_per_gene  = np.array([r[6] for r in DATA])
categories    = [r[12] for r in DATA]
GA_IDX = methods.index("CpGA")

OUT = Path("figures_v2")
OUT.mkdir(exist_ok=True)


def _save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)
    print(f"  -> {name}.png")


# ===========================================================================
# FIGURE 1 — Combined score ranking
# Distinct colour family from Fig 4: deep teal vs. coral.
# ===========================================================================
def figure_1_combined_ranking():
    HIGHLIGHT_1 = "#0F4C75"   # deep teal-blue, for CpGA
    NEUTRAL_1   = "#BBBBBB"   # mid grey, for baselines

    order = np.argsort(combined)
    sorted_methods = [methods[i] for i in order]
    sorted_scores  = combined[order]
    colors = [HIGHLIGHT_1 if m == "CpGA" else NEUTRAL_1 for m in sorted_methods]

    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    bars = ax.barh(sorted_methods, sorted_scores, color=colors,
                   edgecolor="white", linewidth=0.6, height=0.72)
    for bar, val in zip(bars, sorted_scores):
        ax.text(val + 0.012, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="left",
                fontsize=9.5, color="#333")

    # CpGA y-tick label in bold and accent colour
    for label in ax.get_yticklabels():
        if label.get_text() == "CpGA":
            label.set_fontweight("bold")
            label.set_color(HIGHLIGHT_1)

    ax.set_xlabel("Combined Score")
    ax.set_xlim(0, max(sorted_scores) * 1.15)
    ax.tick_params(axis="y", which="both", length=0)
    ax.set_axisbelow(True)
    ax.grid(axis="x", alpha=0.25, linestyle="--", linewidth=0.7)

    ax.legend(handles=[
        Patch(facecolor=HIGHLIGHT_1, label="CpGA (proposed)"),
        Patch(facecolor=NEUTRAL_1,   label="Baseline"),
    ], loc="lower right", frameon=False)

    ax.set_title("Combined score ranking across feature-selection methods",
                 loc="left", pad=10, fontweight="bold")
    fig.tight_layout()
    _save(fig, "fig1_combined_ranking")


# ===========================================================================
# FIGURE 2 — Trade-off (compactness vs. contrast),
# coloured by method category, fixed dot size, pretty labels.
# ===========================================================================
def figure_2_tradeoff():
    # Category-coded palette (ColorBrewer, colourblind-safe)
    CAT_COLOR = {
        "proposed":    "#D55E00",   # vermillion
        "statistical": "#0072B2",   # blue
        "wrapper":     "#009E73",   # green
        "embedded":    "#CC79A7",   # pink
    }
    DOT_SIZE = 170
    GA_SIZE  = 600

    fig, ax = plt.subplots(figsize=(8.4, 6.4))

    # All baselines first
    for i, m in enumerate(methods):
        if m == "CpGA":
            continue
        ax.scatter(compactness[i], contrast[i], s=DOT_SIZE,
                   c=CAT_COLOR[categories[i]],
                   edgecolor="#333", linewidth=0.7, alpha=0.85, zorder=3)

    # CpGA on top, star marker
    ax.scatter(compactness[GA_IDX], contrast[GA_IDX], s=GA_SIZE,
               c=CAT_COLOR["proposed"], edgecolor="black",
               linewidth=1.4, marker="*", zorder=5)

    # Labels (pretty names, no underscores)
    label_offsets = {
        "CpGA":         ( 0.012,  0.022),
        "relieff":      ( 0.012,  0.012),
        "chi2":         ( 0.012,  0.014),
        "extratrees":   ( 0.012,  0.012),
        "mrmr":         (-0.060, -0.030),
        "anova_f":      ( 0.012, -0.025),
        "fisher_score": (-0.080, -0.015),
        "forward_lr":   ( 0.012,  0.005),
        "mutual_info":  ( 0.012,  0.014),
        "l1_logistic":  ( 0.012, -0.018),
        "variance":     ( 0.012,  0.012),
        "adaboost":     ( 0.012,  0.014),
        "xgboost":      (-0.060,  0.014),
        "lightgbm":     ( 0.012, -0.022),
        "rfe_svm":      (-0.060, -0.020),
    }
    for i, m in enumerate(methods):
        dx, dy = label_offsets[m]
        is_ga = (m == "CpGA")
        ax.text(compactness[i] + dx, contrast[i] + dy, PRETTY[m],
                fontsize=11 if is_ga else 9.5,
                fontweight="bold" if is_ga else "normal",
                color=CAT_COLOR["proposed"] if is_ga else "#333")

    ax.set_xlabel("Gene Compactness")
    ax.set_ylabel("Mean Contrast Score")
    ax.set_xlim(0.0, 0.72)
    ax.set_ylim(0.78, 1.50)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.7, zorder=0)

    handles = [
        mlines.Line2D([], [], marker="*", linestyle="None",
                      markerfacecolor=CAT_COLOR["proposed"],
                      markeredgecolor="black",
                      markersize=18, label="CpGA (proposed)"),
        mlines.Line2D([], [], marker="o", linestyle="None",
                      markerfacecolor=CAT_COLOR["statistical"],
                      markeredgecolor="#333",
                      markersize=11, label="Statistical"),
        mlines.Line2D([], [], marker="o", linestyle="None",
                      markerfacecolor=CAT_COLOR["wrapper"],
                      markeredgecolor="#333",
                      markersize=11, label="Wrapper-based"),
        mlines.Line2D([], [], marker="o", linestyle="None",
                      markerfacecolor=CAT_COLOR["embedded"],
                      markeredgecolor="#333",
                      markersize=11, label="Embedded"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False)

    ax.set_title("Feature Selection Methods Performance Comparison",
                 loc="left", pad=10, fontweight="bold")
    fig.tight_layout()
    _save(fig, "fig2_tradeoff")


# ===========================================================================
# FIGURE 4 — Average CpGs per unique gene
# Distinct colour family from Fig 1: warm coral/orange.
# ===========================================================================
def figure_4_cpgs_per_gene():
    HIGHLIGHT_4 = "#C2185B"   # crimson/pink — distinct from Fig 1 teal
    NEUTRAL_4   = "#F4A261"   # warm sand-orange — distinct from Fig 1 grey

    order = np.argsort(cpg_per_gene)
    sorted_methods = [methods[i] for i in order]
    sorted_vals    = cpg_per_gene[order]
    colors = [HIGHLIGHT_4 if m == "CpGA" else NEUTRAL_4 for m in sorted_methods]

    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    bars = ax.barh(sorted_methods, sorted_vals, color=colors,
                   edgecolor="white", linewidth=0.6, height=0.72)
    for bar, v in zip(bars, sorted_vals):
        ax.text(v + 0.04, bar.get_y() + bar.get_height() / 2,
                f"{v:.2f}", va="center", ha="left",
                fontsize=9.5, color="#333")

    # CpGA y-tick label in bold and accent colour
    for label in ax.get_yticklabels():
        if label.get_text() == "CpGA":
            label.set_fontweight("bold")
            label.set_color(HIGHLIGHT_4)

    # NOTE: vertical reference line at x=1 removed per user request

    ax.set_xlabel("Average CpGs per Unique Gene")
    ax.set_xlim(0.95, max(sorted_vals) * 1.12)
    ax.tick_params(axis="y", length=0)
    ax.set_axisbelow(True)
    ax.grid(axis="x", alpha=0.25, linestyle="--", linewidth=0.7)
    ax.legend(handles=[
        Patch(facecolor=HIGHLIGHT_4, label="CpGA (proposed)"),
        Patch(facecolor=NEUTRAL_4,   label="Baseline"),
    ], loc="lower right", frameon=False)
    ax.set_title("Per-gene CpG redundancy",
                 loc="left", pad=10, fontweight="bold")
    fig.tight_layout()
    _save(fig, "fig4_cpgs_per_gene")


if __name__ == "__main__":
    print("Generating figures...")
    figure_1_combined_ranking()
    figure_2_tradeoff()
    figure_4_cpgs_per_gene()
    print(f"\nSaved to {OUT.resolve()}/")
