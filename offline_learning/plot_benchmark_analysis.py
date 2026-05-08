"""
Generate publication-quality figures from benchmark_comparison_results.json.
Usage: python offline_learning/plot_benchmark_analysis.py [--save-dir DIR]
"""
import argparse
import json
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

OPPONENTS = ["RandomPlayer", "RaisedPlayer", "RulesBasedPlayer", "MCTSPlayer (no VM)"]
OPP_LABELS = ["Random", "Raised", "Rules", "MCTS\n(no VM)"]

# Display order for models (by architecture then target)
MODEL_ORDER = [
    "linear/chip_delta", "linear/rollout", "linear/winner", "linear/mixed",
    "mlp/chip_delta",    "mlp/rollout",    "mlp/winner",    "mlp/mixed",
    "transformer/chip_delta", "transformer/rollout", "transformer/winner", "transformer/mixed",
]
MODEL_LABELS = [m.replace("/", "\n") for m in MODEL_ORDER]

TARGET_COLORS = {
    "chip_delta": "#4c72b0",
    "rollout":    "#dd8452",
    "winner":     "#55a868",
    "mixed":      "#c44e52",
}
ARCH_HATCHES = {
    "linear":      "",
    "mlp":         "//",
    "transformer": "xx",
}

BEST_MODEL = "linear/mixed"


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)["models"]


def extract_stats(models: dict) -> dict[str, dict]:
    stats = {}
    for name, data in models.items():
        if name == "root":
            continue
        opps = data["opponents"]
        wr = {opp: opps[opp]["win_rate"] for opp in OPPONENTS}
        cd = {opp: opps[opp]["mean_chip_delta"] for opp in OPPONENTS}
        ci = {opp: opps[opp]["win_rate_ci_95"] for opp in OPPONENTS}
        mean_wr = np.mean(list(wr.values()))
        mean_cd = np.mean(list(cd.values()))
        stats[name] = dict(win_rate=wr, chip_delta=cd, ci=ci,
                           mean_wr=mean_wr, mean_cd=mean_cd)
    return stats


# ── Figure 1: Win-rate heatmap ────────────────────────────────────────────────
def plot_winrate_heatmap(stats: dict, save_path: str | None = None):
    data = np.array([[stats[m]["win_rate"][opp] for opp in OPPONENTS]
                     for m in MODEL_ORDER]) * 100

    cmap = LinearSegmentedColormap.from_list(
        "wr", ["#d73027", "#ffffbf", "#1a9850"], N=256)

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=0, vmax=100)

    ax.set_xticks(range(len(OPPONENTS)))
    ax.set_xticklabels(OPP_LABELS)
    ax.set_yticks(range(len(MODEL_ORDER)))
    ax.set_yticklabels(MODEL_LABELS, fontsize=7.5)

    for i, m in enumerate(MODEL_ORDER):
        for j, opp in enumerate(OPPONENTS):
            v = data[i, j]
            color = "white" if v < 30 or v > 70 else "black"
            weight = "bold" if m == BEST_MODEL else "normal"
            ax.text(j, i, f"{v:.0f}%", ha="center", va="center",
                    fontsize=7, color=color, fontweight=weight)

    # Highlight best model row
    best_row = MODEL_ORDER.index(BEST_MODEL)
    for spine in ["top", "bottom", "left", "right"]:
        ax.add_patch(plt.Rectangle(
            (-0.5, best_row - 0.5), len(OPPONENTS), 1,
            fill=False, edgecolor="#c44e52", linewidth=2, zorder=3))

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Win rate (%)", rotation=270, labelpad=12)

    # Architecture dividers
    ax.axhline(3.5, color="white", linewidth=1.5)
    ax.axhline(7.5, color="white", linewidth=1.5)
    ax.text(-0.75, 1.5, "Linear", rotation=90, va="center", ha="right",
            fontsize=7, color="#333")
    ax.text(-0.75, 5.5, "MLP", rotation=90, va="center", ha="right",
            fontsize=7, color="#333")
    ax.text(-0.75, 9.5, "Transformer", rotation=90, va="center", ha="right",
            fontsize=7, color="#333")

    ax.set_title("Win Rate by Model and Opponent (20 games × 100 rounds)", pad=8)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ── Figure 2: Mean chip delta bar chart ───────────────────────────────────────
def plot_mean_chip_delta(stats: dict, save_path: str | None = None):
    models = MODEL_ORDER
    mean_cds = [stats[m]["mean_cd"] for m in models]

    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    x = np.arange(len(models))
    colors = [TARGET_COLORS[m.split("/")[1]] for m in models]
    hatches = [ARCH_HATCHES[m.split("/")[0]] for m in models]

    bars = ax.bar(x, mean_cds, color=colors, hatch=hatches,
                  edgecolor="white", linewidth=0.6, zorder=2)

    # Highlight best
    best_idx = models.index(BEST_MODEL)
    bars[best_idx].set_edgecolor("#c44e52")
    bars[best_idx].set_linewidth(2.5)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("/", "\n") for m in models],
                       fontsize=6.5, rotation=0)
    ax.set_ylabel("Mean chip delta (chips)")
    ax.set_title("Mean Chip Delta Across All Opponents")
    ax.yaxis.grid(True, alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    # Annotate best bar
    ax.annotate("★ best\n(−48)",
                xy=(best_idx, mean_cds[best_idx]),
                xytext=(best_idx, mean_cds[best_idx] + 60),
                ha="center", fontsize=7, color="#c44e52", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#c44e52", lw=1.2))

    # Legend: targets
    target_patches = [mpatches.Patch(color=c, label=t)
                      for t, c in TARGET_COLORS.items()]
    # Legend: architectures
    arch_patches = [mpatches.Patch(facecolor="#aaa", hatch=h, edgecolor="white",
                                   label=a)
                    for a, h in ARCH_HATCHES.items()]
    l1 = ax.legend(handles=target_patches, title="Target",
                   loc="lower right", ncol=2, fontsize=7, title_fontsize=7)
    ax.add_artist(l1)
    ax.legend(handles=arch_patches, title="Arch",
              loc="lower left", ncol=3, fontsize=7, title_fontsize=7)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ── Figure 3: Per-opponent win rate, grouped by target ────────────────────────
def plot_target_comparison(stats: dict, save_path: str | None = None):
    targets = ["chip_delta", "rollout", "winner", "mixed"]
    archs = ["linear", "mlp", "transformer"]

    fig, axes = plt.subplots(1, 4, figsize=(7, 2.5), sharey=True)
    x = np.arange(len(OPPONENTS))
    width = 0.25

    for ax, target in zip(axes, targets):
        for i, arch in enumerate(archs):
            key = f"{arch}/{target}"
            wrs = [stats[key]["win_rate"][opp] * 100 for opp in OPPONENTS]
            offset = (i - 1) * width
            ax.bar(x + offset, wrs, width, label=arch,
                   color=["#4c72b0", "#dd8452", "#55a868"][i],
                   alpha=0.85, zorder=2)

        ax.set_title(target, fontweight="bold" if target == "mixed" else "normal",
                     color="#c44e52" if target == "mixed" else "black")
        ax.set_xticks(x)
        ax.set_xticklabels(OPP_LABELS, fontsize=7)
        ax.axhline(50, color="gray", linewidth=0.7, linestyle="--", zorder=1)
        ax.yaxis.grid(True, alpha=0.3, zorder=0)
        ax.set_axisbelow(True)
        if ax == axes[0]:
            ax.set_ylabel("Win rate (%)")

    axes[-1].legend(title="Arch", fontsize=7, title_fontsize=7,
                    loc="upper right")
    fig.suptitle("Win Rate by Training Target and Opponent",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ── Figure 4: RaisedPlayer chip delta — key discriminator ────────────────────
def plot_raised_chip_delta(stats: dict, save_path: str | None = None):
    opp = "RaisedPlayer"
    models = MODEL_ORDER
    cds = [stats[m]["chip_delta"][opp] for m in models]
    colors = [TARGET_COLORS[m.split("/")[1]] for m in models]
    hatches = [ARCH_HATCHES[m.split("/")[0]] for m in models]

    fig, ax = plt.subplots(figsize=(6.5, 2.5))
    x = np.arange(len(models))
    bars = ax.bar(x, cds, color=colors, hatch=hatches,
                  edgecolor="white", linewidth=0.6, zorder=2)

    best_idx = models.index(BEST_MODEL)
    bars[best_idx].set_edgecolor("#c44e52")
    bars[best_idx].set_linewidth(2.5)

    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("/", "\n") for m in models], fontsize=6.5)
    ax.set_ylabel("Mean chip delta vs RaisedPlayer")
    ax.set_title("Chip Delta vs RaisedPlayer — Only linear/mixed achieves positive EV")
    ax.yaxis.grid(True, alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    ax.annotate("+21\n★",
                xy=(best_idx, cds[best_idx]),
                xytext=(best_idx, cds[best_idx] + 100),
                ha="center", fontsize=8, color="#c44e52", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#c44e52", lw=1.2))

    target_patches = [mpatches.Patch(color=c, label=t)
                      for t, c in TARGET_COLORS.items()]
    ax.legend(handles=target_patches, title="Target",
              loc="lower right", ncol=2, fontsize=7, title_fontsize=7)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        default="offline_learning/models/benchmark_comparison_results.json",
        help="Path to benchmark_comparison_results.json",
    )
    parser.add_argument(
        "--save-dir",
        default="offline_learning/models/figures",
        help="Directory to save figures (empty string = interactive display)",
    )
    args = parser.parse_args()

    models = load_results(args.results)
    stats = extract_stats(models)

    save_dir = args.save_dir
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        p = lambda name: os.path.join(save_dir, name)
    else:
        p = lambda name: None

    plot_winrate_heatmap(stats, p("fig1_winrate_heatmap.png"))
    plot_mean_chip_delta(stats, p("fig2_mean_chip_delta.png"))
    plot_target_comparison(stats, p("fig3_target_comparison.png"))
    plot_raised_chip_delta(stats, p("fig4_raised_chip_delta.png"))

    if save_dir:
        print(f"\nAll figures saved to {save_dir}/")


if __name__ == "__main__":
    main()
