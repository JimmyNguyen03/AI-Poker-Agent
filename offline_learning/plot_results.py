"""
Generate report-quality plots from benchmark_comparison_results.json.

Usage (from repo root):
    python offline_learning/plot_results.py
    python offline_learning/plot_results.py --results path/to/benchmark_comparison_results.json
    python offline_learning/plot_results.py --out offline_learning/models/report_plots

Saves five figures to --out directory.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.size":          10,
    "axes.titlesize":     13,
    "axes.titleweight":   "bold",
    "axes.titlepad":      12,
    "axes.labelsize":     11,
    "axes.labelcolor":    "#222222",
    "axes.edgecolor":     "#CCCCCC",
    "axes.linewidth":     0.8,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "xtick.color":        "#555555",
    "ytick.color":        "#555555",
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "grid.color":         "#E8E8E8",
    "grid.linewidth":     0.8,
    "legend.framealpha":  0.9,
    "legend.edgecolor":   "#DDDDDD",
    "legend.fontsize":    9,
    "figure.facecolor":   "white",
    "savefig.facecolor":  "white",
    "savefig.dpi":        180,
})

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
_ROOT            = Path(__file__).resolve().parent.parent
_DEFAULT_RESULTS = _ROOT / "offline_learning" / "models" / "benchmark_comparison_results.json"
_DEFAULT_OUT     = _ROOT / "offline_learning" / "models" / "report_plots"

OPPONENTS  = ["RandomPlayer", "RaisedPlayer", "MCTSPlayer (no VM)"]
OPP_LABELS = ["Random", "Raised", "MCTS"]
TARGETS    = ["chip_delta", "rollout", "mixed", "winner"]
ARCHS      = ["linear", "mlp", "transformer"]
MODEL_ORDER = [f"{a}/{t}" for a in ARCHS for t in TARGETS]

# Refined palette
TARGET_COLORS = {
    "chip_delta": "#C0392B",   # assertive red — aggressive play style
    "rollout":    "#2166AC",   # deep blue    — calibrated play style
    "mixed":      "#E08728",   # amber        — blended signal
    "winner":     "#3B9E6E",   # forest green — binary outcome signal
}
ARCH_COLORS = {
    "linear":      "#2166AC",
    "mlp":         "#C0392B",
    "transformer": "#3B9E6E",
}
OPP_COLORS = ["#4575B4", "#C0392B", "#3B9E6E"]

ARCH_MARKERS = {"linear": "o", "mlp": "s", "transformer": "^"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load(path: Path) -> dict:
    with open(path) as f:
        raw = json.load(f)
    out = {}
    for label, entry in raw["models"].items():
        arch, target = label.split("/")
        opps = {}
        for opp, stats in entry["opponents"].items():
            opps[opp] = {
                "win_rate":   stats["win_rate"],
                "chip_delta": stats["mean_chip_delta"],
                "ci":         stats["win_rate_ci_95"],
            }
        out[label] = {"arch": arch, "target": target, "opponents": opps}
    return out


def _arch_band(ax, labels: list[str], n_cols: int) -> None:
    """Shade alternating architecture blocks on a heatmap y-axis."""
    for i, arch in enumerate(ARCHS):
        y0 = i * len(TARGETS) - 0.5
        y1 = y0 + len(TARGETS)
        color = ARCH_COLORS[arch]
        ax.axhspan(y0, y1, xmin=-0.05, xmax=0,
                   clip_on=False, color=color, alpha=0.18, linewidth=0)
        ax.text(-0.22, (y0 + y1) / 2, arch, transform=ax.get_yaxis_transform(),
                color=color, fontsize=9, fontweight="bold",
                va="center", ha="right", rotation=90)


def _colored_yticks(ax, labels: list[str]) -> None:
    for tick, label in zip(ax.get_yticklabels(), labels):
        arch = label.get_text().split("/")[0]
        tick.set_color(ARCH_COLORS.get(arch, "#222222"))


def _save(fig, out: Path, name: str) -> None:
    fig.savefig(out / name, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {name}")


# ---------------------------------------------------------------------------
# Plot 1 — Win-rate heatmap
# ---------------------------------------------------------------------------
def plot_winrate_heatmap(data: dict, out: Path) -> None:
    labels = [m for m in MODEL_ORDER if m in data]
    matrix = np.array([[data[m]["opponents"][opp]["win_rate"]   for opp in OPPONENTS] for m in labels])
    ci_mat = np.array([[data[m]["opponents"][opp]["ci"]          for opp in OPPONENTS] for m in labels])

    fig, ax = plt.subplots(figsize=(6.5, 8.5))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(OPP_LABELS)))
    ax.set_xticklabels(OPP_LABELS, fontsize=10, fontweight="bold")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    _colored_yticks(ax, ax.get_yticklabels())

    for r in range(len(labels)):
        for c in range(len(OPPONENTS)):
            wr  = matrix[r, c]
            ci  = ci_mat[r, c]
            ink = "white" if (wr < 0.32 or wr > 0.72) else "#111111"
            ax.text(c, r, f"{wr:.0%}", ha="center", va="center",
                    fontsize=9, fontweight="bold", color=ink)
            ax.text(c, r + 0.28, f"±{ci:.0%}", ha="center", va="center",
                    fontsize=6.5, color=ink, alpha=0.8)

    for sep in [3.5, 7.5]:
        ax.axhline(sep, color="#555555", linewidth=1.2, linestyle="--", alpha=0.5)

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Win Rate", fontsize=9)
    cbar.ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    ax.set_title("Win Rate — All Models vs All Opponents", pad=14)
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)

    _arch_band(ax, labels, len(OPPONENTS))
    plt.tight_layout()
    _save(fig, out, "1_winrate_heatmap.png")


# ---------------------------------------------------------------------------
# Plot 2 — Aggressive vs Calibrated scatter
# ---------------------------------------------------------------------------
def plot_aggression_scatter(data: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6))

    # Quadrant shading
    ax.axvspan(0.5, 1.05, 0, 0.5 / 0.75, color="#FFF3F3", zorder=0)   # high random, low raised
    ax.axvspan(0.5, 1.05, 0.5 / 0.75, 1,  color="#F0FFF4", zorder=0)  # high random, high raised
    ax.axvspan(0,   0.5,  0, 1,            color="#F8F8F8", zorder=0)  # low random

    # Reference lines
    ax.axvline(0.5, color="#AAAAAA", linewidth=0.9, linestyle="--", zorder=1)
    ax.axhline(0.5, color="#AAAAAA", linewidth=0.9, linestyle="--", zorder=1)

    # Quadrant labels
    ax.text(0.97, 0.72, "Best of\nboth worlds", transform=ax.transAxes,
            fontsize=8, color="#2E7D32", ha="right", va="top", style="italic")
    ax.text(0.97, 0.05, "Aggressive\n(chips vs random)", transform=ax.transAxes,
            fontsize=8, color="#B71C1C", ha="right", va="bottom", style="italic")

    plotted: list[tuple[float, float, str, str]] = []

    for label, info in data.items():
        x = info["opponents"]["RandomPlayer"]["win_rate"]
        y = info["opponents"]["RaisedPlayer"]["win_rate"]
        target = info["target"]
        arch   = info["arch"]
        color  = TARGET_COLORS[target]
        marker = ARCH_MARKERS[arch]

        ax.scatter(x, y, s=110, color=color, marker=marker,
                   edgecolors="white", linewidths=1.0, zorder=4)

        short = f"{arch[0].upper()}/{target[:3]}"
        plotted.append((x, y, short, color))

    # Simple offset strategy: push label right, unless near right edge
    for (x, y, txt, color) in plotted:
        dx = 0.012 if x < 0.87 else -0.065
        dy = 0.015
        ax.annotate(txt, (x, y), xytext=(x + dx, y + dy),
                    fontsize=7.5, color=color, fontweight="bold",
                    arrowprops=None)

    ax.set_xlabel("Win Rate vs Random", fontsize=11)
    ax.set_ylabel("Win Rate vs Raised", fontsize=11)
    ax.set_xlim(0.38, 1.02)
    ax.set_ylim(-0.02, 0.72)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title("Aggressive vs Calibrated Trade-off")
    ax.grid(True, alpha=0.4)

    # Legend — targets (colour) + archs (marker)
    target_handles = [
        mpatches.Patch(color=c, label=t.replace("_", " "))
        for t, c in TARGET_COLORS.items()
    ]
    arch_handles = [
        plt.Line2D([0], [0], marker=ARCH_MARKERS[a], color="w",
                   markerfacecolor="#555555", markersize=8, label=a)
        for a in ARCHS
    ]
    leg1 = ax.legend(handles=target_handles, title="Target", loc="upper left",
                     fontsize=8, title_fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=arch_handles, title="Architecture", loc="lower right",
              fontsize=8, title_fontsize=8)

    plt.tight_layout()
    _save(fig, out, "2_aggression_scatter.png")


# ---------------------------------------------------------------------------
# Plot 3 — Win rate by target mode (avg across architectures)
# ---------------------------------------------------------------------------
def plot_target_comparison(data: dict, out: Path) -> None:
    avg = {t: {opp: [] for opp in OPPONENTS} for t in TARGETS}
    for info in data.values():
        t = info["target"]
        for opp in OPPONENTS:
            avg[t][opp].append(info["opponents"][opp]["win_rate"])
    for t in TARGETS:
        for opp in OPPONENTS:
            avg[t][opp] = float(np.mean(avg[t][opp]))

    x     = np.arange(len(TARGETS))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8.5, 5))

    for i, (opp, short, color) in enumerate(zip(OPPONENTS, OPP_LABELS, OPP_COLORS)):
        vals = [avg[t][opp] for t in TARGETS]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, vals, width - 0.02, label=short,
                      color=color, alpha=0.88, edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.012,
                    f"{v:.0%}", ha="center", va="bottom", fontsize=8,
                    color="#333333")

    ax.set_xticks(x)
    target_labels = [t.replace("_", "\n") for t in TARGETS]
    ax.set_xticklabels(target_labels, fontsize=10)
    ax.set_ylabel("Avg Win Rate (across architectures)")
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.axhline(0.5, color="#888888", linestyle="--", linewidth=0.9, alpha=0.6)
    ax.text(len(TARGETS) - 0.45, 0.52, "50% baseline", fontsize=8,
            color="#888888", va="bottom")
    ax.set_title("Effect of Training Target on Win Rate\n"
                 "(averaged across linear, MLP, transformer)")
    ax.legend(title="Opponent", loc="upper right")
    ax.grid(True, axis="y", alpha=0.4)
    ax.set_axisbelow(True)

    plt.tight_layout()
    _save(fig, out, "3_target_comparison.png")


# ---------------------------------------------------------------------------
# Plot 4 — MCTS win rate ranked
# ---------------------------------------------------------------------------
def plot_mcts_performance(data: dict, out: Path) -> None:
    mcts_opp = "MCTSPlayer (no VM)"
    labels    = [m for m in MODEL_ORDER if m in data]
    wr_all    = [data[m]["opponents"][mcts_opp]["win_rate"] for m in labels]
    ci_all    = [data[m]["opponents"][mcts_opp]["ci"]       for m in labels]
    arch_all  = [data[m]["arch"]                            for m in labels]

    order    = np.argsort(wr_all)            # ascending for barh (best at top)
    labels_s = [labels[i]   for i in order]
    wr_s     = [wr_all[i]   for i in order]
    ci_s     = [ci_all[i]   for i in order]
    arch_s   = [arch_all[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, 5.5))

    # Subtle arch banding (horizontal stripes)
    seen = {}
    for j, arch in enumerate(arch_s):
        if arch not in seen:
            seen[arch] = []
        seen[arch].append(j)
    for arch, idxs in seen.items():
        ax.axhspan(min(idxs) - 0.5, max(idxs) + 0.5,
                   color=ARCH_COLORS[arch], alpha=0.06, zorder=0)

    colors = [ARCH_COLORS[a] for a in arch_s]
    bars   = ax.barh(range(len(labels_s)), wr_s, color=colors,
                     alpha=0.85, edgecolor="white", linewidth=0.5, height=0.65)
    ax.errorbar(wr_s, range(len(labels_s)), xerr=ci_s,
                fmt="none", color="#333333", capsize=3, linewidth=1.0, capthick=1.0)

    for j, (v, arch) in enumerate(zip(wr_s, arch_s)):
        ax.text(v + 0.01, j, f"{v:.0%}", va="center", fontsize=8.5,
                color=ARCH_COLORS[arch], fontweight="bold")

    ax.set_yticks(range(len(labels_s)))
    ax.set_yticklabels(labels_s, fontsize=9)
    for tick, arch in zip(ax.get_yticklabels(), arch_s):
        tick.set_color(ARCH_COLORS[arch])

    ax.set_xlabel("Win Rate vs MCTS (no value model)")
    ax.set_xlim(0, 1.06)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.axvline(0.5, color="#888888", linestyle="--", linewidth=0.9, alpha=0.7)
    ax.set_title("Win Rate vs MCTS — Ranked\n"
                 "More training iterations unlocked MLP capacity")
    ax.grid(True, axis="x", alpha=0.35)
    ax.set_axisbelow(True)

    legend_handles = [
        mpatches.Patch(color=ARCH_COLORS[a], label=a) for a in ARCHS
    ]
    ax.legend(handles=legend_handles, title="Architecture",
              loc="lower right", fontsize=9)

    plt.tight_layout()
    _save(fig, out, "4_mcts_performance.png")


# ---------------------------------------------------------------------------
# Plot 5 — Mean chip delta heatmap
# ---------------------------------------------------------------------------
def plot_chipdelta_heatmap(data: dict, out: Path) -> None:
    labels = [m for m in MODEL_ORDER if m in data]
    matrix = np.array([[data[m]["opponents"][opp]["chip_delta"]
                        for opp in OPPONENTS] for m in labels])

    vabs = max(abs(matrix.min()), abs(matrix.max()))

    # Custom diverging colormap: deeper saturation at extremes
    cmap = plt.get_cmap("RdYlGn")
    fig, ax = plt.subplots(figsize=(6.5, 8.5))
    im = ax.imshow(matrix, cmap=cmap, vmin=-vabs, vmax=vabs, aspect="auto")

    ax.set_xticks(range(len(OPP_LABELS)))
    ax.set_xticklabels(OPP_LABELS, fontsize=10, fontweight="bold")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    _colored_yticks(ax, ax.get_yticklabels())

    for r in range(len(labels)):
        for c in range(len(OPPONENTS)):
            v   = matrix[r, c]
            ink = "white" if abs(v) > 550 else "#111111"
            ax.text(c, r, f"{v:+.0f}", ha="center", va="center",
                    fontsize=9, fontweight="bold", color=ink)

    for sep in [3.5, 7.5]:
        ax.axhline(sep, color="#555555", linewidth=1.2, linestyle="--", alpha=0.5)

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Mean Chip Delta (chips)", fontsize=9)
    cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x:+.0f}"))

    ax.set_title("Mean Chip Delta — All Models vs All Opponents\n"
                 "(initial stack = 1 000 chips)", pad=14)
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)

    _arch_band(ax, labels, len(OPPONENTS))
    plt.tight_layout()
    _save(fig, out, "5_chipdelta_heatmap.png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate report plots from benchmark results")
    parser.add_argument("--results", default=str(_DEFAULT_RESULTS))
    parser.add_argument("--out",     default=str(_DEFAULT_OUT))
    args = parser.parse_args()

    results_path = Path(args.results)
    out_dir      = Path(args.out)

    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        raise SystemExit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    data = load(results_path)

    print(f"Loaded {len(data)} models -> {out_dir}/\n")
    plot_winrate_heatmap(data, out_dir)
    plot_aggression_scatter(data, out_dir)
    plot_target_comparison(data, out_dir)
    plot_mcts_performance(data, out_dir)
    plot_chipdelta_heatmap(data, out_dir)
    print("\nDone.")
