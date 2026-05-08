
from __future__ import annotations
import json
import math
import os
import random
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from offline_learning.value_model import ValueModel
from offline_learning.self_play import _TARGET_MODES

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


def _game_worker(args: tuple) -> list:
    """
    Run one self-play episode in a subprocess.

    args = (num_rounds, target_mode, model_path, opp_spec, frozen_path)
      model_path   — path to current value model JSON (or None for untrained)
      opp_spec     — "random" | "raised" | "rules" | "frozen"
      frozen_path  — path to frozen opponent model JSON (only used when opp_spec="frozen")
    """
    num_rounds, target_mode, model_path, opp_spec, frozen_path = args

    # Workers spawned by ProcessPoolExecutor start fresh; re-add repo root to sys.path.
    _root = str(Path(__file__).resolve().parent.parent)
    if _root not in sys.path:
        sys.path.insert(0, _root)

    from offline_learning.value_model import load_value_model
    value_model = load_value_model(model_path) if model_path else None

    if opp_spec == "random":
        from randomplayer import RandomPlayer
        opp = RandomPlayer()
    elif opp_spec == "raised":
        from raise_player import RaisedPlayer
        opp = RaisedPlayer()
    elif opp_spec == "rules":
        from offline_learning.rules_player import RulesBasedPlayer
        opp = RulesBasedPlayer()
    else:  # "frozen"
        from offline_learning.self_play import DataCollectingPlayer
        frozen_model = load_value_model(frozen_path) if frozen_path else None
        opp = DataCollectingPlayer(value_model=frozen_model)

    from offline_learning.self_play import run_self_play_episode
    return run_self_play_episode(
        num_rounds=num_rounds,
        value_model=value_model,
        opp_player=opp,
        target_mode=target_mode,
    )


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


def run_iterative_self_play(
    model_type: str = "linear",
    iterations: int = 3,
    games_per_iter: int = 5,
    rounds_per_game: int = 50,
    epochs_per_iter: int = 5,
    lr: float = 0.01,
    output_path: str | None = None,
    target_mode: str = "rollout",
    warmup_iters: int = 2,
    n_workers: int = 1,
) -> object:
    """Iterative self-play training loop."""
    model = make_model(model_type)
    # frozen_pool stores (iter_n, path) — temp files that persist until training ends.
    frozen_pool: list[tuple[int, str]] = []
    frozen_tmp_paths: list[str] = []

    log: dict = {
        "config": {
            "model_type": model_type,
            "target_mode": target_mode,
            "iterations": iterations,
            "warmup_iters": warmup_iters,
            "games_per_iter": games_per_iter,
            "rounds_per_game": rounds_per_game,
            "epochs_per_iter": epochs_per_iter,
            "lr": lr,
            "n_workers": n_workers,
        },
        "iterations": [],
    }

    print(
        f"[{model_type}|{target_mode}] Starting iterative self-play: "
        f"{iterations} iters x {games_per_iter} games x {rounds_per_game} rounds "
        f"(warmup: {warmup_iters} iters, workers: {n_workers})"
    )

    try:
        for iteration in range(1, iterations + 1):
            in_warmup = iteration <= warmup_iters
            phase_label = "warmup" if in_warmup else "focus"
            print(f"\n=== Iteration {iteration}/{iterations} [{phase_label}] ===")

            # Save current model to a temp file shared by all game workers this iteration.
            with tempfile.NamedTemporaryFile(
                suffix=".json", delete=False, mode="w", encoding="utf-8"
            ) as tmp:
                model_tmp = tmp.name
            model.save(model_tmp)

            # Build per-game specs (all primitives — picklable).
            game_specs: list[tuple[str, str | None, str]] = []  # (opp_spec, frozen_path, label)
            for g in range(1, games_per_iter + 1):
                slot = (g - 1) % 3
                if in_warmup:
                    if slot == 0:
                        game_specs.append(("random", None, "RandomPlayer"))
                    elif slot == 1:
                        game_specs.append(("raised", None, "RaisedPlayer"))
                    else:
                        game_specs.append(("rules", None, "RulesBasedPlayer"))
                else:
                    if slot == 0:
                        game_specs.append(("raised", None, "RaisedPlayer"))
                    elif slot == 1 or not frozen_pool:
                        game_specs.append(("rules", None, "RulesBasedPlayer"))
                    else:
                        iter_n, fp = random.choice(frozen_pool)
                        game_specs.append(("frozen", fp, f"frozen iter {iter_n}"))

            worker_args = [
                (rounds_per_game, target_mode, model_tmp, opp_spec, frozen_path)
                for opp_spec, frozen_path, _ in game_specs
            ]

            all_data: list[tuple[dict, float]] = []

            if n_workers > 1:
                from concurrent.futures import ProcessPoolExecutor
                print(
                    f"  launching {games_per_iter} games in parallel "
                    f"(workers={n_workers})...",
                    flush=True,
                )
                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    results = list(executor.map(_game_worker, worker_args))
                for (_, _, label), result in zip(game_specs, results):
                    print(f"  vs {label}: {len(result)} examples")
                    all_data.extend(result)
            else:
                for g_idx, ((_, _, label), wargs) in enumerate(
                    zip(game_specs, worker_args), start=1
                ):
                    print(
                        f"  game {g_idx}/{games_per_iter} vs {label}...",
                        end=" ",
                        flush=True,
                    )
                    result = _game_worker(wargs)
                    all_data.extend(result)
                    print(f"{len(result)} examples")

            os.unlink(model_tmp)

            print(f"  total examples this iteration: {len(all_data)}")
            model, epoch_metrics = train(all_data, model, lr=lr, epochs=epochs_per_iter)

            # Snapshot the trained model to a temp file for use as a frozen opponent.
            with tempfile.NamedTemporaryFile(
                suffix=".json", delete=False, mode="w", encoding="utf-8"
            ) as tmp:
                frozen_snap = tmp.name
            model.save(frozen_snap)
            frozen_pool.append((iteration, frozen_snap))
            frozen_tmp_paths.append(frozen_snap)

            log["iterations"].append({
                "iteration": iteration,
                "n_examples": len(all_data),
                "epoch_metrics": epoch_metrics,
            })

    finally:
        for p in frozen_tmp_paths:
            try:
                os.unlink(p)
            except OSError:
                pass

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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Iterative self-play training for the poker value model")
    parser.add_argument("--model", type=str, default="linear",
                        choices=[*_MODEL_TYPES, "all"],
                        help="Model architecture to train. 'all' trains all three to separate subfolders.")
    parser.add_argument("--target", type=str, default="rollout",
                        choices=[*_TARGET_MODES, "all"],
                        help=(
                            "Training target. "
                            "'rollout' (default): mean of N abstract rollouts from each child state — "
                            "card-aware and low-variance. "
                            "'chip_delta': actual chip change for the round normalised to [-1, 1]. "
                            "'winner': +1 if hero wins the round's pot, -1 otherwise. "
                            "'all': train every target mode to separate subfolders."
                        ))
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--warmup-iters", type=int, default=2,
                        help="Iterations using Random/Raised opponents before switching to RulesBasedPlayer.")
    parser.add_argument("--games-per-iter", type=int, default=5)
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel game workers per iteration (default: 1 = sequential). "
                             "Set to os.cpu_count() for maximum throughput.")
    parser.add_argument("--rounds-per-game", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument(
        "--output",
        type=str,
        default=str(_ROOT / "offline_learning" / "models" / "submission_value_model.json"),
        help="Output path. For --model/--target all, each combo is saved to a subfolder.",
    )
    args = parser.parse_args()

    model_targets = list(_MODEL_TYPES) if args.model == "all" else [args.model]
    train_targets = list(_TARGET_MODES) if args.target == "all" else [args.target]
    combos = [(m, t) for m in model_targets for t in train_targets]
    multi = len(combos) > 1

    for model_type, target_mode in combos:
        if multi:
            base_dir = Path(args.output).parent
            fname    = Path(args.output).name
            # Use subfolders like  models/mlp/chip_delta/submission_value_model.json
            # or  models/mlp/submission_value_model.json when only one target is swept.
            if len(train_targets) > 1:
                out_path = str(base_dir / model_type / target_mode / fname)
            else:
                out_path = str(base_dir / model_type / fname)
            print(f"\n{'='*60}")
            print(f"  Training model={model_type}  target={target_mode}  →  {out_path}")
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
            target_mode=target_mode,
            warmup_iters=args.warmup_iters,
            n_workers=args.workers,
        )

        if model_type == "linear":
            print("\nFinal weights:")
            for k, v in sorted(model.weights.items()):
                print(f"  {k:<30s} {v:+.4f}")
        else:
            print(f"\nFinal model: {model!r}")
