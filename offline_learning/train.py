"""
Offline training: iterative self-play + SGD on the linear value model.

Usage (from repo root):
    python offline_learning/train.py
    python offline_learning/train.py --model mlp
    python offline_learning/train.py --model transformer
    python offline_learning/train.py --model all        # trains all three to separate subfolders
    python offline_learning/train.py --iterations 5 --games-per-iter 10 --rounds-per-game 100 --epochs 10 --lr 0.005
"""

from __future__ import annotations
import json
import math
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from offline_learning.value_model import ValueModel

_MODEL_TYPES = ("linear", "mlp", "transformer")


def make_model(model_type: str):
    """Instantiate a fresh model of the requested type."""
    if model_type == "mlp":
        from offline_learning.mlp_value_model import MLPValueModel
        return MLPValueModel()
    if model_type == "transformer":
        from offline_learning.transformer_value_model import TransformerValueModel
        return TransformerValueModel()
    return ValueModel()


# ------------------------------------------------------------------
# Core training step
# ------------------------------------------------------------------

def train(
    data: list[tuple[dict, float]],
    model,
    lr: float = 0.01,
    epochs: int = 10,
) -> tuple[object, list[dict]]:
    """
    SGD on MSE. Calls model.update(features, target, lr) for each sample.

    Returns (updated_model, epoch_metrics) where epoch_metrics is a list of
    {"mse": float, "std": float} dicts — one per epoch — for plotting.
    """
    if not data:
        print("  [train] no data — skipping")
        return model, []

    epoch_metrics: list[dict] = []

    for epoch in range(1, epochs + 1):
        random.shuffle(data)
        squared_errors: list[float] = []

        for features, target in data:
            sq_err = model.update(features, target, lr)
            squared_errors.append(sq_err)

        n = len(squared_errors)
        mse = sum(squared_errors) / n
        variance = sum((e - mse) ** 2 for e in squared_errors) / max(n - 1, 1)
        std = math.sqrt(variance)
        epoch_metrics.append({"mse": mse, "std": std})
        print(f"  epoch {epoch:2d}/{epochs}  MSE={mse:.4f}  std={std:.4f}  n={n}")

    return model, epoch_metrics


# ------------------------------------------------------------------
# Iterative self-play loop
# ------------------------------------------------------------------

def run_iterative_self_play(
    model_type: str = "linear",
    iterations: int = 3,
    games_per_iter: int = 5,
    rounds_per_game: int = 50,
    epochs_per_iter: int = 5,
    lr: float = 0.01,
    output_path: str | None = None,
) -> object:
    """
    Main training loop.

    Each iteration:
      1. Play games cycling p2 across: RandomPlayer, RaisedPlayer, frozen past model.
      2. Collect street-discounted (features, target) pairs from p1 only.
      3. Run SGD for epochs_per_iter passes.
      4. Snapshot trained model into frozen_pool for future iterations.
      5. Repeat.

    Saves the trained model to output_path and a training log to the same
    directory as training_log.json (used by plot_training.py).
    """
    from offline_learning.self_play import run_self_play_episode, DataCollectingPlayer
    from randomplayer import RandomPlayer
    from raise_player import RaisedPlayer

    model = make_model(model_type)
    # Accumulated frozen snapshots: list of (iteration_number, model).
    # Games cycle across three opponent slots:
    #   slot 0 → RandomPlayer
    #   slot 1 → RaisedPlayer
    #   slot 2 → frozen past version (falls back to RandomPlayer until pool is non-empty)
    frozen_pool: list[tuple[int, object]] = []

    log: dict = {
        "config": {
            "model_type": model_type,
            "iterations": iterations,
            "games_per_iter": games_per_iter,
            "rounds_per_game": rounds_per_game,
            "epochs_per_iter": epochs_per_iter,
            "lr": lr,
        },
        "iterations": [],
    }

    print(
        f"[{model_type}] Starting iterative self-play: "
        f"{iterations} iters x {games_per_iter} games x {rounds_per_game} rounds"
    )

    for iteration in range(1, iterations + 1):
        print(f"\n=== Iteration {iteration}/{iterations} ===")

        all_data: list[tuple[dict, float]] = []
        for g in range(1, games_per_iter + 1):
            slot = (g - 1) % 3
            if slot == 0:
                opp = RandomPlayer()
                opp_label = "RandomPlayer"
            elif slot == 1:
                opp = RaisedPlayer()
                opp_label = "RaisedPlayer"
            else:
                if frozen_pool:
                    iter_n, frozen_model = random.choice(frozen_pool)
                    opp = DataCollectingPlayer(value_model=frozen_model)
                    opp_label = f"frozen iter {iter_n}"
                else:
                    opp = RandomPlayer()
                    opp_label = "RandomPlayer (pool empty)"

            print(f"  game {g}/{games_per_iter} vs {opp_label}...", end=" ", flush=True)
            episode_data = run_self_play_episode(
                num_rounds=rounds_per_game, value_model=model, opp_player=opp
            )
            all_data.extend(episode_data)
            print(f"{len(episode_data)} examples")

        print(f"  total examples this iteration: {len(all_data)}")
        model, epoch_metrics = train(all_data, model, lr=lr, epochs=epochs_per_iter)

        # Snapshot the trained model for use as a future frozen opponent.
        # The snapshot is a fresh model loaded from a temp save to ensure it's a true copy.
        frozen_pool.append((iteration, _clone_model(model, model_type)))

        log["iterations"].append({
            "iteration": iteration,
            "n_examples": len(all_data),
            "epoch_metrics": epoch_metrics,
        })

    # Save model
    if output_path:
        model.save(output_path)
        print(f"\nModel saved -> {output_path}")

        log_path = Path(output_path).parent / "training_log.json"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)
        print(f"Training log saved -> {log_path}")

    return model


def _clone_model(model, model_type: str):
    """Return an independent copy of a model for use as a frozen opponent."""
    import io, json as _json
    buf = io.StringIO()
    # Reuse the save/load round-trip via a string buffer.
    # We patch save to use the buffer instead of a file path.
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as tmp:
        tmp_path = tmp.name
    try:
        model.save(tmp_path)
        from offline_learning.value_model import load_value_model
        return load_value_model(tmp_path)
    finally:
        os.unlink(tmp_path)


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Iterative self-play training for the poker value model")
    parser.add_argument("--model", type=str, default="linear",
                        choices=[*_MODEL_TYPES, "all"],
                        help="Model architecture to train. 'all' trains all three to separate subfolders.")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--games-per-iter", type=int, default=5)
    parser.add_argument("--rounds-per-game", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument(
        "--output",
        type=str,
        default=str(_ROOT / "offline_learning" / "models" / "submission_value_model.json"),
        help="Output path. For --model all, each type is saved to a subfolder of this file's parent.",
    )
    args = parser.parse_args()

    targets = list(_MODEL_TYPES) if args.model == "all" else [args.model]

    for model_type in targets:
        if len(targets) > 1:
            # Save each model type to its own subfolder.
            base_dir = Path(args.output).parent
            fname    = Path(args.output).name
            out_path = str(base_dir / model_type / fname)
            print(f"\n{'='*60}")
            print(f"  Training model: {model_type}  →  {out_path}")
            print(f"{'='*60}")
        else:
            out_path = args.output

        model = run_iterative_self_play(
            model_type=model_type,
            iterations=args.iterations,
            games_per_iter=args.games_per_iter,
            rounds_per_game=args.rounds_per_game,
            epochs_per_iter=args.epochs,
            lr=args.lr,
            output_path=out_path,
        )

        if model_type == "linear":
            print("\nFinal weights:")
            for k, v in sorted(model.weights.items()):
                print(f"  {k:<30s} {v:+.4f}")
        else:
            print(f"\nFinal model: {model!r}")
