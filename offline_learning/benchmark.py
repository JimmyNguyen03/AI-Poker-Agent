
from __future__ import annotations
import json
import math
import signal
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Windows: patch SIGALRM-based timeout before any engine imports.
if not hasattr(signal, "SIGALRM"):
    import pypokerengine.api.game as _game_api

    def _noop_timeout2(*_args, **_kwargs):
        def _decorate(fn):
            return fn
        return _decorate

    _game_api.timeout2 = _noop_timeout2

from pypokerengine.api.game import setup_config, start_poker


def _bench_worker(args: tuple) -> dict:
    """
    Run one benchmark game in a subprocess.

    args = (model_path, simulations, opp_spec, num_rounds, initial_stack, small_blind, hero_is_first)
      model_path — path to value model JSON (or None for untrained hero)
      opp_spec   — "random" | "raised" | "mcts_no_vm"
    """
    model_path, simulations, opp_spec, num_rounds, initial_stack, small_blind, hero_is_first = args

    _root = str(Path(__file__).resolve().parent.parent)
    if _root not in sys.path:
        sys.path.insert(0, _root)

    # Re-apply SIGALRM patch in spawned workers (module-level code re-runs on import).
    if not hasattr(signal, "SIGALRM"):
        import pypokerengine.api.game as _gapi
        def _noop(*_a, **_kw):
            return lambda fn: fn
        _gapi.timeout2 = _noop

    from pypokerengine.api.game import setup_config, start_poker
    from mcts_player import MCTSPlayer

    hero = MCTSPlayer(simulations=simulations, value_model_path=model_path)

    if opp_spec == "random":
        from randomplayer import RandomPlayer
        opp = RandomPlayer()
    elif opp_spec == "raised":
        from raise_player import RaisedPlayer
        opp = RaisedPlayer()
    elif opp_spec == "rules":
        from offline_learning.rules_player import RulesBasedPlayer
        opp = RulesBasedPlayer()
    else:  # "mcts_no_vm"
        opp = MCTSPlayer(simulations=simulations)

    config = setup_config(
        max_round=num_rounds,
        initial_stack=initial_stack,
        small_blind_amount=small_blind,
    )
    if hero_is_first:
        config.register_player(name="hero", algorithm=hero)
        config.register_player(name="opp",  algorithm=opp)
    else:
        config.register_player(name="opp",  algorithm=opp)
        config.register_player(name="hero", algorithm=hero)

    result = start_poker(config, verbose=0)
    hero_stack = next(p["stack"] for p in result["players"] if p["name"] == "hero")
    opp_stack  = next(p["stack"] for p in result["players"] if p["name"] == "opp")
    return {
        "hero_stack": hero_stack,
        "opp_stack":  opp_stack,
        "chip_delta": hero_stack - initial_stack,
        "hero_won":   hero_stack > opp_stack,
    }


def benchmark_vs(
    model_path: str | None,
    simulations: int,
    opp_spec: str,
    opp_label: str,
    n_games: int = 10,
    num_rounds: int = 50,
    initial_stack: int = 1000,
    small_blind: int = 10,
    n_workers: int = 1,
) -> dict:
    """
    Run n_games against opp_spec opponent and return aggregate statistics.

    opp_spec: "random" | "raised" | "mcts_no_vm"
    Uses ProcessPoolExecutor when n_workers > 1.
    """
    print(f"\n  vs {opp_label}: ", end="", flush=True)

    worker_args = [
        (model_path, simulations, opp_spec, num_rounds, initial_stack, small_blind, g % 2 == 0)
        for g in range(n_games)
    ]

    if n_workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        print(f"(parallel, workers={n_workers}) ", end="", flush=True)
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            game_results = list(executor.map(_bench_worker, worker_args))
    else:
        game_results = []
        for wargs in worker_args:
            res = _bench_worker(wargs)
            game_results.append(res)
            print("W" if res["hero_won"] else "L", end="", flush=True)

    chip_deltas: list[float] = []
    wins = 0
    for res in game_results:
        chip_deltas.append(float(res["chip_delta"]))
        if res["hero_won"]:
            wins += 1

    if n_workers > 1:
        # Print results after parallel batch completes
        print("".join("W" if r["hero_won"] else "L" for r in game_results), end="")

    print()

    n = len(chip_deltas)
    mean_delta = sum(chip_deltas) / n
    std_delta = math.sqrt(sum((d - mean_delta) ** 2 for d in chip_deltas) / max(n - 1, 1))
    win_rate = wins / n
    win_rate_ci = 1.96 * math.sqrt(win_rate * (1 - win_rate) / max(n, 1))

    return {
        "opponent": opp_label,
        "n_games": n,
        "num_rounds": num_rounds,
        "wins": wins,
        "losses": n - wins,
        "win_rate": win_rate,
        "win_rate_ci_95": win_rate_ci,
        "chip_deltas": chip_deltas,
        "mean_chip_delta": mean_delta,
        "std_chip_delta": std_delta,
    }


def run_benchmark(
    n_games: int = 10,
    num_rounds: int = 50,
    simulations: int = 100,
    model_path: str | None = None,
    save_path: str | None = None,
    plot_save: str | None = None,
    n_workers: int = 1,
) -> dict:
    # Resolve model path
    default_model = _ROOT / "offline_learning" / "models" / "transformer" / "rollout" / "submission_value_model.json"
    if model_path is None and default_model.exists():
        model_path = str(default_model)

    if model_path:
        print(f"Loaded trained model: {model_path}")
    else:
        print("No trained model found — benchmarking untrained MCTS with default weights.")

    opponents = [
        ("random",     "RandomPlayer"),
        ("raised",     "RaisedPlayer"),
        ("rules",      "RulesBasedPlayer"),
        ("mcts_no_vm", "MCTSPlayer (no VM)"),
    ]

    print(f"\nRunning benchmark: {n_games} games x {num_rounds} rounds each"
          + (f" (workers={n_workers})" if n_workers > 1 else ""))
    print("W=hero wins, L=hero loses\n" + "-" * 40)

    results: dict = {"opponents": {}}
    for opp_spec, label in opponents:
        stats = benchmark_vs(
            model_path=model_path,
            simulations=simulations,
            opp_spec=opp_spec,
            opp_label=label,
            n_games=n_games,
            num_rounds=num_rounds,
            n_workers=n_workers,
        )
        results["opponents"][label] = stats
        print(
            f"  {label:<22s}  "
            f"win={stats['win_rate']:.0%} (+/-{stats['win_rate_ci_95']:.0%})  "
            f"chip delta={stats['mean_chip_delta']:+.1f} +/- {stats['std_chip_delta']:.1f}"
        )

    results["config"] = {
        "n_games": n_games,
        "num_rounds": num_rounds,
        "simulations": simulations,
        "model_path": model_path,
    }

    # Save results JSON
    if save_path is None:
        save_path = str(_ROOT / "offline_learning" / "models" / "benchmark_results.json")
    _write_json_log(save_path, results, "\nResults saved")

    # Additional log artifact for easier experiment tracking.
    save_path_obj = Path(save_path)
    log_path = save_path_obj.with_name(f"{save_path_obj.stem}_log.json")
    _write_json_log(log_path, results, "Results log saved")

    # Plot
    _plot_benchmark(results, plot_save)

    return results


def _write_json_log(path: Path | str, payload: dict, label: str) -> None:
    """Write a JSON log payload to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"{label} -> {path}")


def _plot_benchmark(results: dict, save_path: str | None = None) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    opponents = list(results["opponents"].keys())
    win_rates = [results["opponents"][o]["win_rate"] for o in opponents]
    win_cis = [results["opponents"][o]["win_rate_ci_95"] for o in opponents]
    mean_deltas = [results["opponents"][o]["mean_chip_delta"] for o in opponents]
    std_deltas = [results["opponents"][o]["std_chip_delta"] for o in opponents]
    n_games = results["config"]["n_games"]

    colors = ["#4878cf", "#6acc65", "#d65f5f"]
    x = np.arange(len(opponents))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Trained MCTS Value Model vs Baselines", fontsize=14, fontweight="bold")

    # --- Left: Win rates with 95% CI error bars ---
    ax = axes[0]
    bars = ax.bar(x, win_rates, color=colors, alpha=0.82, edgecolor="white", width=0.5)
    ax.errorbar(x, win_rates, yerr=win_cis, fmt="none", color="black", capsize=6, linewidth=1.5)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="50% baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(opponents, fontsize=9)
    ax.set_ylabel("Win Rate")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Win Rate (n={n_games} games each, 95% CI)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    for bar, wr in zip(bars, win_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{wr:.0%}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # --- Right: Mean chip delta with std error bars ---
    ax2 = axes[1]
    bars2 = ax2.bar(x, mean_deltas, color=colors, alpha=0.82, edgecolor="white", width=0.5)
    ax2.errorbar(x, mean_deltas, yerr=std_deltas, fmt="none", color="black", capsize=6, linewidth=1.5)
    ax2.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax2.set_xticks(x)
    ax2.set_xticklabels(opponents, fontsize=9)
    ax2.set_ylabel("Avg Chip Delta (per game)")
    ax2.set_title(f"Average Chip Delta +/- Std (n={n_games} games)")
    ax2.grid(True, axis="y", alpha=0.3)
    for bar, d in zip(bars2, mean_deltas):
        ypos = bar.get_height() + (max(std_deltas) * 0.05 if d >= 0 else -max(std_deltas) * 0.15)
        ax2.text(bar.get_x() + bar.get_width() / 2, ypos,
                 f"{d:+.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved -> {save_path}")
    else:
        plt.show()


def _find_all_models(models_dir: Path) -> list[tuple[str, Path]]:
    """
    Scan models_dir for submission_value_model.json files.
    Returns list of (label, path) sorted by label.
    Checks the root folder, immediate subdirectories, and one level deeper
    (to handle models/<model>/<target>/submission_value_model.json).
    """
    found: list[tuple[str, Path]] = []
    root_model = models_dir / "submission_value_model.json"
    if root_model.exists():
        found.append(("root", root_model))
    for subdir in sorted(models_dir.iterdir()):
        if not subdir.is_dir():
            continue
        candidate = subdir / "submission_value_model.json"
        if candidate.exists():
            found.append((subdir.name, candidate))
        else:
            # One level deeper: models/<model>/<target>/
            for subsubdir in sorted(subdir.iterdir()):
                if subsubdir.is_dir():
                    candidate2 = subsubdir / "submission_value_model.json"
                    if candidate2.exists():
                        found.append((f"{subdir.name}/{subsubdir.name}", candidate2))
    return found


def run_all_benchmarks(
    n_games: int = 10,
    num_rounds: int = 50,
    simulations: int = 100,
    models_dir: Path | None = None,
    n_workers: int = 1,
) -> dict[str, dict]:
    """
    Discover every model in models_dir, benchmark each one, save per-model
    results, then produce a combined comparison plot.
    Returns a dict mapping model label → results dict.
    """
    if models_dir is None:
        models_dir = _ROOT / "offline_learning" / "models"
    models_dir = Path(models_dir)

    entries = _find_all_models(models_dir)
    if not entries:
        print(f"No models found under {models_dir}. Train first.")
        return {}

    print(f"Found {len(entries)} model(s): {[label for label, _ in entries]}")

    all_results: dict[str, dict] = {}
    for label, model_path in entries:
        print(f"\n{'='*60}")
        print(f"  Benchmarking: {label}  ({model_path})")
        print(f"{'='*60}")
        save_dir = model_path.parent
        results = run_benchmark(
            n_games=n_games,
            num_rounds=num_rounds,
            simulations=simulations,
            model_path=str(model_path),
            save_path=str(save_dir / "benchmark_results.json"),
            plot_save=str(save_dir / "benchmark.png"),
            n_workers=n_workers,
        )
        all_results[label] = results

    # Combined comparison across all models
    comparison_json_path = models_dir / "benchmark_comparison_results.json"
    _write_json_log(
        comparison_json_path,
        {
            "config": {
                "n_games": n_games,
                "num_rounds": num_rounds,
                "simulations": simulations,
                "models_dir": str(models_dir),
            },
            "models": all_results,
        },
        "\nComparison results JSON saved",
    )
    comparison_path = models_dir / "benchmark_comparison.png"
    _plot_comparison(all_results, comparison_path)
    return all_results


def _plot_comparison(all_results: dict[str, dict], save_path: Path) -> None:
    """Grouped bar chart comparing win rates and chip deltas across model types."""
    import matplotlib.pyplot as plt
    import numpy as np

    if not all_results:
        return

    labels = list(all_results.keys())
    # Collect all opponent names (should be identical across models)
    opponents = list(next(iter(all_results.values()))["opponents"].keys())

    x = np.arange(len(opponents))
    n_models = len(labels)
    width = 0.8 / n_models
    palette = plt.cm.tab10.colors

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Model Comparison vs Baselines", fontsize=14, fontweight="bold")

    for ax, metric, ylabel, title in [
        (axes[0], "win_rate",        "Win Rate",              "Win Rate by Model"),
        (axes[1], "mean_chip_delta", "Avg Chip Delta",        "Chip Delta by Model"),
    ]:
        for i, label in enumerate(labels):
            vals = [all_results[label]["opponents"][o][metric] for o in opponents]
            errs = [
                all_results[label]["opponents"][o].get(
                    "win_rate_ci_95" if metric == "win_rate" else "std_chip_delta", 0
                )
                for o in opponents
            ]
            offset = (i - n_models / 2 + 0.5) * width
            bars = ax.bar(x + offset, vals, width=width * 0.9,
                          label=label, color=palette[i % 10], alpha=0.85)
            ax.errorbar(x + offset, vals, yerr=errs,
                        fmt="none", color="black", capsize=3, linewidth=1)

        if metric == "win_rate":
            ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
            ax.set_ylim(0, 1.05)
        else:
            ax.axhline(0, color="gray", linestyle="--", linewidth=1)

        ax.set_xticks(x)
        ax.set_xticklabels(opponents, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nComparison plot saved -> {save_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark trained MCTS value model vs baselines")
    parser.add_argument("--games", type=int, default=10, help="Games per opponent")
    parser.add_argument("--rounds", type=int, default=50, help="Rounds per game")
    parser.add_argument("--simulations", type=int, default=200, help="MCTS simulations per action")
    parser.add_argument("--model", type=str, default=None, help="Path to a specific value model JSON")
    parser.add_argument("--all", action="store_true",
                        help="Benchmark every model found in models/ and produce a comparison plot")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel game workers per opponent (default: 1 = sequential). "
                             "Each worker plays one game; set to n_games for max throughput.")
    parser.add_argument(
        "--save",
        type=str,
        default=str(_ROOT / "offline_learning" / "models" / "transformer" / "rollout" / "benchmark_results.json"),
    )
    parser.add_argument(
        "--plot-save",
        type=str,
        default=str(_ROOT / "offline_learning" / "models" / "transformer" / "rollout" / "benchmark.png"),
        help="Save plot to file (default: benchmark.png). Pass '' to show interactively.",
    )
    args = parser.parse_args()

    if args.all:
        run_all_benchmarks(
            n_games=args.games,
            num_rounds=args.rounds,
            simulations=args.simulations,
            n_workers=args.workers,
        )
    else:
        run_benchmark(
            n_games=args.games,
            num_rounds=args.rounds,
            simulations=args.simulations,
            model_path=args.model,
            save_path=args.save,
            plot_save=args.plot_save or None,
            n_workers=args.workers,
        )
