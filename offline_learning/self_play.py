import json
import shutil
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from tqdm import tqdm

# Support script execution from repo root.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Windows: pypokerengine's timeout2 uses SIGALRM, which is unavailable. Replace before game imports it.
import pypokerengine.utils.timeout_decorator as _poker_timeout  # noqa: E402

def _timeout2_noop(seconds=None, defaultretval="Blah", **kwargs):  # noqa: ARG001
    def decorate(function):
        return function

    return decorate


_poker_timeout.timeout2 = _timeout2_noop

from pypokerengine.api.game import setup_config, start_poker  # noqa: E402
from mcts.state import build_state
from mcts_player import MCTSPlayer
from randomplayer import RandomPlayer
from offline_learning.data import TARGET_SCHEMA_VERSION, DecisionSample, JsonlDatasetWriter
from offline_learning.features import mcts_rollout_leaf_value, state_to_features
from offline_learning.train_value_model import train_value_model


def _status(msg: str) -> None:
    """Log one line without breaking tqdm progress bars."""
    tqdm.write(msg)


_DEFAULT_SELF_PLAY_JSONL = "offline_learning/data/self_play.jsonl"


def _sanitize_run_name(name: str) -> str:
    n = name.strip()
    if not n:
        return "run"
    for sep in ("/", "\\"):
        n = n.replace(sep, "_")
    while ".." in n:
        n = n.replace("..", "_")
    return n or "run"


def submission_value_model_destinations(run_models_dir: Optional[Path] = None) -> Tuple[Path, Optional[Path]]:
    """
    Paths for the frozen "submit this" checkpoint.
    Standard path is what mcts_player.setup_ai and submission/custom_player load.
    Optional second path is a copy under the current run's models/ folder.
    """
    standard = Path(__file__).resolve().parent / "models" / "submission_value_model.json"
    run_copy = (run_models_dir / "submission_value_model.json") if run_models_dir is not None else None
    return standard, run_copy


def export_best_model_for_submission(best_model_path: str, run_models_dir: Optional[Path] = None) -> Dict[str, str]:
    """
    Copy the best checkpoint JSON to submission_value_model.json (repo + optional run folder).
    Returns absolute paths written under keys 'standard' and 'run_copy' (if used).
    """
    src = Path(best_model_path)
    out: Dict[str, str] = {}
    if not src.is_file():
        return out
    standard, run_dest = submission_value_model_destinations(run_models_dir)
    standard.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, standard)
    out["standard"] = str(standard.resolve())
    if run_dest is not None:
        run_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, run_dest)
        out["run_copy"] = str(run_dest.resolve())
    return out


def resolve_run_directory(output_dir: str, run_name: str = "") -> Path:
    """
    If run_name is non-empty: <output_dir>/runs/<run_name>/ (sanitized).
    Else: <output_dir>/ as given (default iterative layout stays under offline_learning/).
    """
    base = Path(output_dir)
    if run_name.strip():
        return (base / "runs" / _sanitize_run_name(run_name)).resolve()
    return base.resolve()


def _stack_by_uuid(round_state: Dict, target_uuid: str) -> Tuple[int, int]:
    seats = round_state.get("seats", [])
    hero = next((seat for seat in seats if seat.get("uuid") == target_uuid), {})
    opp = next((seat for seat in seats if seat.get("uuid") != target_uuid), {})
    return int(hero.get("stack", 0)), int(opp.get("stack", 0))


def _round_training_target(chip_delta: int, norm: int) -> float:
    """Scale chip swing into [-1, 1] for value fitting, normalized by pot size at decision time."""
    if norm <= 0:
        return 0.0
    x = float(chip_delta) / float(norm)
    return max(-1.0, min(1.0, x))


class LoggingMCTSPlayer(MCTSPlayer):
    def __init__(
        self,
        game_index: int,
        sink: List[DecisionSample],
        initial_stack: int = 1000,
        value_model_path: str = "",
        rollout_depth: int = 8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.game_index = game_index
        self.sink = sink
        self._initial_stack = max(1, int(initial_stack))
        self.value_model_path = value_model_path
        self._pending: List[Dict] = []
        self._round_index = 0
        self._hero_stack_round_start = 0

    def receive_round_start_message(self, round_count, hole_card, seats):
        self._pending = []
        self._round_index = int(round_count)
        hero_seat = next((s for s in seats if s.get("uuid") == self.uuid), None)
        self._hero_stack_round_start = int(hero_seat.get("stack", 0)) if hero_seat else 0

    def declare_action(self, valid_actions, hole_card, round_state):
        action = super().declare_action(valid_actions, hole_card, round_state)
        state = build_state(
            hero_uuid=self.uuid,
            valid_actions=valid_actions,
            hole_card=hole_card,
            round_state=round_state,
            belief_snapshot=self._belief_model.snapshot(),
        )
        self._pending.append(
            {
                "action": action,
                "legal_actions": list(state.legal_actions),
                "features": state_to_features(state),
                "hero_stack_at_decision": int(state.hero_stack),
                "rollout_aligned_target": float(mcts_rollout_leaf_value(state)),
                "metadata": {
                    "pot_main": float(state.pot_main),
                    "hero_stack": float(state.hero_stack),
                    "opp_stack": float(state.opp_stack),
                    "chosen_amount": 0.0,
                    "value_model_path": self.value_model_path,
                },
            }
        )
        return action

    def receive_round_result_message(self, winners, hand_info, round_state):
        winner_uuids = {winner.get("uuid") for winner in winners if "uuid" in winner}
        if self.uuid in winner_uuids and len(winner_uuids) == 1:
            reward = 1.0
        elif self.uuid not in winner_uuids:
            reward = -1.0
        else:
            reward = 0.0
        final_hero_stack, final_opp_stack = _stack_by_uuid(round_state, self.uuid)
        round_chip_delta = final_hero_stack - self._hero_stack_round_start
        for row in self._pending:
            hero_at_decision = int(row["hero_stack_at_decision"])
            per_decision_delta = final_hero_stack - hero_at_decision
            pot_at_decision = max(1, int(row["metadata"]["pot_main"]))
            training_target = _round_training_target(per_decision_delta, pot_at_decision)
            meta = dict(row["metadata"])
            meta["round_chip_delta"] = float(round_chip_delta)
            meta["hero_stack_round_start"] = float(self._hero_stack_round_start)
            meta["hero_stack_at_decision"] = float(hero_at_decision)
            meta["per_decision_chip_delta"] = float(per_decision_delta)
            self.sink.append(
                DecisionSample(
                    game_index=self.game_index,
                    round_index=self._round_index,
                    player_uuid=self.uuid,
                    action=row["action"],
                    legal_actions=row["legal_actions"],
                    features=row["features"],
                    terminal_reward=reward,
                    training_target=training_target,
                    rollout_aligned_target=float(row["rollout_aligned_target"]),
                    final_hero_stack=final_hero_stack,
                    final_opp_stack=final_opp_stack,
                    metadata=meta,
                )
            )


def generate_self_play_dataset(
    output_path: str,
    num_games: int = 20,
    rounds_per_game: int = 50,
    simulations: int = 250,
    initial_stack: int = 1000,
    small_blind_amount: int = 10,
    value_model_path: str = "",
    rollout_depth: int = 8,
) -> Dict[str, float]:
    rows: List[DecisionSample] = []
    for game_idx in tqdm(range(num_games), desc="Generating self-play dataset"):
        config = setup_config(
            max_round=rounds_per_game,
            initial_stack=initial_stack,
            small_blind_amount=small_blind_amount,
        )
        config.register_player(
            name="mcts_a",
            algorithm=LoggingMCTSPlayer(
                game_index=game_idx,
                sink=rows,
                initial_stack=initial_stack,
                simulations=simulations,
                value_model_path=value_model_path,
                rollout_depth=rollout_depth,
            ),
        )
        config.register_player(
            name="mcts_b",
            algorithm=LoggingMCTSPlayer(
                game_index=game_idx,
                sink=rows,
                initial_stack=initial_stack,
                simulations=simulations,
                value_model_path=value_model_path,
                rollout_depth=rollout_depth,
            ),
        )
        start_poker(config, verbose=0)

    with JsonlDatasetWriter(output_path) as writer:
        for row in rows:
            writer.write(row)

    mean_reward = sum(r.terminal_reward for r in rows) / max(1, len(rows))
    mean_training_target = sum(r.training_target for r in rows) / max(1, len(rows))
    return {
        "num_games": float(num_games),
        "rows_written": float(len(rows)),
        "mean_reward": float(mean_reward),
        "mean_training_target": float(mean_training_target),
    }


def _mcts_factory(simulations: int, value_model_path: str = "") -> Callable[[], MCTSPlayer]:
    return lambda: MCTSPlayer(simulations=simulations, value_model_path=value_model_path)


def _evaluate_matchup(
    hero_factory: Callable[[], object],
    opp_factory: Callable[[], object],
    num_games: int,
    rounds_per_game: int,
    initial_stack: int,
    small_blind_amount: int,
) -> Dict[str, float]:
    hero_final_sum = 0.0
    opp_final_sum = 0.0
    hero_wins = 0
    opp_wins = 0
    draws = 0
    for _ in tqdm(range(num_games), desc="Evaluating matchup"):
        config = setup_config(
            max_round=rounds_per_game,
            initial_stack=initial_stack,
            small_blind_amount=small_blind_amount,
        )
        config.register_player(name="hero", algorithm=hero_factory())
        config.register_player(name="opp", algorithm=opp_factory())
        result = start_poker(config, verbose=0)
        hero_stack = float(result["players"][0]["stack"])
        opp_stack = float(result["players"][1]["stack"])
        hero_final_sum += hero_stack
        opp_final_sum += opp_stack
        if hero_stack > opp_stack:
            hero_wins += 1
        elif hero_stack < opp_stack:
            opp_wins += 1
        else:
            draws += 1

    n = max(1.0, float(num_games))
    return {
        "games": float(num_games),
        "hero_avg_final_stack": hero_final_sum / n,
        "opp_avg_final_stack": opp_final_sum / n,
        "hero_win_rate": float(hero_wins) / n,
        "opp_win_rate": float(opp_wins) / n,
        "draw_rate": float(draws) / n,
        "avg_stack_delta": (hero_final_sum - opp_final_sum) / n,
    }


def _prune_replay_buffer(replay_path: str, schema: int = TARGET_SCHEMA_VERSION) -> Tuple[int, int]:
    """
    Drop jsonl rows whose target_schema != schema (legacy self-play formats).
    Rewrites the file only when at least one row is removed.
    Returns (kept_count, dropped_count).
    """
    path = Path(replay_path)
    if not path.is_file():
        return 0, 0
    kept_lines: List[str] = []
    dropped = 0
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            item = json.loads(raw)
        except json.JSONDecodeError:
            dropped += 1
            continue
        if int(item.get("target_schema", 1)) != int(schema):
            dropped += 1
            continue
        kept_lines.append(raw.strip())
    kept = len(kept_lines)
    if dropped:
        path.write_text("\n".join(kept_lines) + ("\n" if kept_lines else ""), encoding="utf-8")
    return kept, dropped


def _append_jsonl_records(source_path: str, target_path: str) -> int:
    src = Path(source_path)
    dst = Path(target_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        return 0
    raw = src.read_text(encoding="utf-8")
    if not raw.strip():
        return 0
    lines = [line for line in raw.splitlines() if line.strip()]
    with dst.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return len(lines)


def evaluate_model_against_baselines(
    model_path: str,
    baseline_model_path: str = "",
    eval_games: int = 30,
    rounds_per_game: int = 50,
    simulations: int = 250,
    initial_stack: int = 1000,
    small_blind_amount: int = 10,
) -> Dict[str, Dict[str, float]]:
    candidate = _mcts_factory(simulations=simulations, value_model_path=model_path)
    untrained_mcts = _mcts_factory(simulations=simulations, value_model_path="")
    baselines: Dict[str, Callable[[], object]] = {
        "random": lambda: RandomPlayer(),
        "untrained_mcts": untrained_mcts,
    }
    if baseline_model_path:
        baselines["previous_model"] = _mcts_factory(
            simulations=simulations,
            value_model_path=baseline_model_path,
        )

    results: Dict[str, Dict[str, float]] = {}
    for name, factory in baselines.items():
        print(f"Evaluating {name} vs candidate")
        results[name] = _evaluate_matchup(
            hero_factory=candidate,
            opp_factory=factory,
            num_games=eval_games,
            rounds_per_game=rounds_per_game,
            initial_stack=initial_stack,
            small_blind_amount=small_blind_amount,
        )
    return results


def run_iterative_rl_training(
    output_dir: str,
    iterations: int = 3,
    self_play_games: int = 20,
    eval_games: int = 30,
    rounds_per_game: int = 50,
    simulations: int = 250,
    initial_stack: int = 1000,
    small_blind_amount: int = 10,
    epochs_per_iter: int = 5,
    learning_rate: float = 0.01,
    value_model_kind: str = "linear",
    mlp_hidden_dim: int = 32,
    regression_target: str = "rollout",
    rollout_depth: int = 8,
) -> Dict[str, object]:
    """
    Explicit RL-style loop:
    1) Policy evaluation data collection via self-play with current policy/value.
    2) Reward-based value fitting on cumulative experience replay.
    3) Policy improvement by loading the new value model into MCTS.
    4) Baseline evaluation and checkpoint selection.
    """
    root = Path(output_dir)
    data_dir = root / "data"
    models_dir = root / "models"
    metrics_dir = root / "metrics"
    data_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    replay_path = str(data_dir / "replay_buffer.jsonl")
    current_best_model = ""
    iteration_summaries: List[Dict[str, object]] = []
    n_iter = max(1, iterations)

    _status(
        f"RL | iterations={n_iter} | replay={replay_path} | "
        f"games/iter={self_play_games} | eval_games/baseline={eval_games} | "
        f"value_model={value_model_kind}"
        + (f" (hidden={mlp_hidden_dim})" if value_model_kind.lower() == "mlp" else "")
        + f" | y={regression_target}"
    )

    k_prune, d_prune = _prune_replay_buffer(replay_path)
    if d_prune:
        _status(f"Replay pruned legacy rows | kept={k_prune} dropped={d_prune} | {replay_path}")

    for iter_idx in tqdm(range(1, n_iter + 1), desc="Iterative RL training"):
        _status(f"[{iter_idx}/{n_iter}] self-play | checkpoint: {current_best_model or '(none)'}")
        iter_data_path = str(data_dir / f"self_play_iter_{iter_idx}.jsonl")
        self_play_summary = generate_self_play_dataset(
            output_path=iter_data_path,
            num_games=self_play_games,
            rounds_per_game=rounds_per_game,
            simulations=simulations,
            initial_stack=initial_stack,
            small_blind_amount=small_blind_amount,
            value_model_path=current_best_model,
            rollout_depth=rollout_depth,
        )
        appended = _append_jsonl_records(iter_data_path, replay_path)
        _status(
            f"[{iter_idx}/{n_iter}] self-play done | samples_iter={int(self_play_summary['rows_written'])} | "
            f"mean_y_chip={self_play_summary['mean_training_target']:.4f} | "
            f"mean_sparse_r={self_play_summary['mean_reward']:.3f} | appended_to_replay={appended}"
        )

        candidate_model_path = str(models_dir / f"value_model_iter_{iter_idx}.json")
        candidate_loss_path = str(models_dir / f"loss_history_iter_{iter_idx}.json")
        candidate_plot_path = str(models_dir / f"loss_curve_iter_{iter_idx}.png")
        train_summary = train_value_model(
            dataset_path=replay_path,
            output_model_path=candidate_model_path,
            loss_history_path=candidate_loss_path,
            loss_plot_path=candidate_plot_path,
            epochs=epochs_per_iter,
            lr=learning_rate,
            warm_start_path=current_best_model,
            model_kind=value_model_kind,
            hidden_dim=mlp_hidden_dim,
            regression_target=regression_target,
            rollout_depth=rollout_depth,
        )
        ws = bool(train_summary.get("warm_started", 0.0))
        _status(
            f"[{iter_idx}/{n_iter}] fit done | final_loss={train_summary['final_loss']:.4f} | "
            f"warm_start_ok={ws} | out={candidate_model_path}"
        )

        baseline_results = evaluate_model_against_baselines(
            model_path=candidate_model_path,
            baseline_model_path=current_best_model,
            eval_games=eval_games,
            rounds_per_game=rounds_per_game,
            simulations=simulations,
            initial_stack=initial_stack,
            small_blind_amount=small_blind_amount,
        )
        aggregate_score = sum(v["avg_stack_delta"] for v in baseline_results.values())
        if current_best_model:
            prev_results = evaluate_model_against_baselines(
                model_path=current_best_model,
                baseline_model_path="",
                eval_games=eval_games,
                rounds_per_game=rounds_per_game,
                simulations=simulations,
                initial_stack=initial_stack,
                small_blind_amount=small_blind_amount,
            )
            prev_score = sum(v["avg_stack_delta"] for v in prev_results.values())
        else:
            prev_score = float("-inf")

        accepted_as_best = (not current_best_model) or (aggregate_score >= prev_score)
        submission_export: Dict[str, str] = {}
        if accepted_as_best:
            current_best_model = candidate_model_path
            submission_export = export_best_model_for_submission(current_best_model, models_dir)
            if submission_export:
                _status(
                    f"[{iter_idx}/{n_iter}] submission checkpoint -> {submission_export.get('standard', '')}"
                )

        prev_s = prev_score if prev_score != float("-inf") else None
        prev_part = f"{prev_s:.1f}" if prev_s is not None else "—"
        _status(
            f"[{iter_idx}/{n_iter}] eval | agg_stack_delta={aggregate_score:.1f} (prev_best={prev_part}) | "
            f"accepted={accepted_as_best} | best={current_best_model or '(none)'}"
        )

        iter_summary: Dict[str, object] = {
            "iteration": float(iter_idx),
            "self_play": self_play_summary,
            "records_appended_to_replay": float(appended),
            "train": train_summary,
            "candidate_model_path": candidate_model_path,
            "baseline_eval": baseline_results,
            "candidate_score": float(aggregate_score),
            "previous_score": float(prev_score) if prev_score != float("-inf") else None,
            "accepted_as_best": bool(accepted_as_best),
            "best_model_after_iteration": current_best_model,
            "submission_export": submission_export,
        }
        iteration_summaries.append(iter_summary)

    final_submission_export = (
        export_best_model_for_submission(current_best_model, models_dir) if current_best_model else {}
    )
    if final_submission_export:
        _status(f"Submission value model (run best) -> {final_submission_export.get('standard', '')}")

    summary = {
        "learning_method": (
            "Iterative fitted value learning with MCTS self-play. "
            "Per iteration: collect self-play experience, fit linear value model on replay buffer "
            "(warm-started from the current best checkpoint when feature dimensions match), "
            "evaluate vs fixed baselines (random, untrained MCTS, previous checkpoint), "
            "and keep the best checkpoint by aggregate stack-delta score."
        ),
        "run_directory": str(root.resolve()),
        "iterations": float(len(iteration_summaries)),
        "best_model_path": current_best_model,
        "submission_value_model_export": final_submission_export,
        "history": iteration_summaries,
    }
    summary_path = metrics_dir / "iterative_rl_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _status(f"RL finished | best_model={summary['best_model_path']} | metrics={summary_path}")
    return summary


def _parse_args():
    parser = ArgumentParser()
    parser.add_argument("--output", type=str, default=_DEFAULT_SELF_PLAY_JSONL)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--sims", type=int, default=250)
    parser.add_argument("--iterative", action="store_true")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--eval-games", type=int, default=30)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="offline_learning",
        help="Root for iterative RL (data/models/metrics). With --run-name, uses <output-dir>/runs/<name>/.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help="Optional label; writes under <output-dir>/runs/<run-name>/ so parallel runs stay separate.",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument(
        "--value-model",
        type=str,
        choices=("linear", "mlp"),
        default="linear",
        help="Architecture for iterative training checkpoints (MCTS loads JSON kind automatically).",
    )
    parser.add_argument("--mlp-hidden-dim", type=int, default=32)
    parser.add_argument(
        "--regression-target",
        type=str,
        choices=("chips", "rollout"),
        default="rollout",
        help="Supervision: per-decision chip outcome (chips) or MCTS leaf heuristic (rollout).",
    )
    parser.add_argument("--rollout-depth", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_root = resolve_run_directory(args.output_dir, args.run_name)
    if args.iterative:
        _status(f"Run directory: {run_root}")
        summary = run_iterative_rl_training(
            output_dir=str(run_root),
            iterations=args.iterations,
            self_play_games=args.games,
            eval_games=args.eval_games,
            rounds_per_game=args.rounds,
            simulations=args.sims,
            epochs_per_iter=args.epochs,
            learning_rate=args.lr,
            value_model_kind=args.value_model,
            mlp_hidden_dim=args.mlp_hidden_dim,
            regression_target=args.regression_target,
            rollout_depth=args.rollout_depth,
            initial_stack=args.initial_stack,
            small_blind_amount=args.small_blind_amount,
        )
        print("Iterative RL summary:", summary)
    else:
        out_path = args.output
        if args.run_name.strip() and out_path == _DEFAULT_SELF_PLAY_JSONL:
            out_path = str(run_root / "data" / "self_play.jsonl")
        _status(f"Self-play | games={args.games} | out={out_path} | run_dir={run_root}")
        summary = generate_self_play_dataset(
            output_path=out_path,
            num_games=args.games,
            rounds_per_game=args.rounds,
            simulations=args.sims,
            rollout_depth=args.rollout_depth,
        )
        _status(
            f"Self-play done | rows={int(summary['rows_written'])} | "
            f"mean_y_chip={summary['mean_training_target']:.4f} | mean_sparse_r={summary['mean_reward']:.3f}"
        )
        print("Self-play summary:", summary)

