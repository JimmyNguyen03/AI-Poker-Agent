import sys
import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# Support script execution from repo root.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pypokerengine.api.game import setup_config, start_poker
from mcts.state import build_state
from mcts_player import MCTSPlayer
from randomplayer import RandomPlayer
from offline_learning.data import DecisionSample, JsonlDatasetWriter
from offline_learning.features import state_to_features
from offline_learning.train_value_model import train_value_model


def _stack_by_uuid(round_state: Dict, target_uuid: str) -> Tuple[int, int]:
    seats = round_state.get("seats", [])
    hero = next((seat for seat in seats if seat.get("uuid") == target_uuid), {})
    opp = next((seat for seat in seats if seat.get("uuid") != target_uuid), {})
    return int(hero.get("stack", 0)), int(opp.get("stack", 0))


class LoggingMCTSPlayer(MCTSPlayer):
    def __init__(
        self,
        game_index: int,
        sink: List[DecisionSample],
        value_model_path: str = "",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.game_index = game_index
        self.sink = sink
        self.value_model_path = value_model_path
        self._pending: List[Dict] = []
        self._round_index = 0

    def receive_round_start_message(self, round_count, hole_card, seats):
        self._pending = []
        self._round_index = int(round_count)

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
        for row in self._pending:
            self.sink.append(
                DecisionSample(
                    game_index=self.game_index,
                    round_index=self._round_index,
                    player_uuid=self.uuid,
                    action=row["action"],
                    legal_actions=row["legal_actions"],
                    features=row["features"],
                    terminal_reward=reward,
                    final_hero_stack=final_hero_stack,
                    final_opp_stack=final_opp_stack,
                    metadata=row["metadata"],
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
) -> Dict[str, float]:
    rows: List[DecisionSample] = []
    for game_idx in range(num_games):
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
                simulations=simulations,
                value_model_path=value_model_path,
            ),
        )
        config.register_player(
            name="mcts_b",
            algorithm=LoggingMCTSPlayer(
                game_index=game_idx,
                sink=rows,
                simulations=simulations,
                value_model_path=value_model_path,
            ),
        )
        start_poker(config, verbose=0)

    with JsonlDatasetWriter(output_path) as writer:
        for row in rows:
            writer.write(row)

    mean_reward = sum(r.terminal_reward for r in rows) / max(1, len(rows))
    return {
        "num_games": float(num_games),
        "rows_written": float(len(rows)),
        "mean_reward": float(mean_reward),
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
    for _ in range(num_games):
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

    for iter_idx in range(1, max(1, iterations) + 1):
        iter_data_path = str(data_dir / f"self_play_iter_{iter_idx}.jsonl")
        self_play_summary = generate_self_play_dataset(
            output_path=iter_data_path,
            num_games=self_play_games,
            rounds_per_game=rounds_per_game,
            simulations=simulations,
            initial_stack=initial_stack,
            small_blind_amount=small_blind_amount,
            value_model_path=current_best_model,
        )
        appended = _append_jsonl_records(iter_data_path, replay_path)

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
        if accepted_as_best:
            current_best_model = candidate_model_path

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
        }
        iteration_summaries.append(iter_summary)

    summary = {
        "learning_method": (
            "Iterative fitted value learning with MCTS self-play. "
            "Per iteration: collect self-play experience, fit linear value model on replay buffer "
            "(warm-started from the current best checkpoint when feature dimensions match), "
            "evaluate vs fixed baselines (random, untrained MCTS, previous checkpoint), "
            "and keep the best checkpoint by aggregate stack-delta score."
        ),
        "iterations": float(len(iteration_summaries)),
        "best_model_path": current_best_model,
        "history": iteration_summaries,
    }
    summary_path = metrics_dir / "iterative_rl_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _parse_args():
    parser = ArgumentParser()
    parser.add_argument("--output", type=str, default="offline_learning/data/self_play.jsonl")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--sims", type=int, default=250)
    parser.add_argument("--iterative", action="store_true")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--eval-games", type=int, default=30)
    parser.add_argument("--output-dir", type=str, default="offline_learning")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.01)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.iterative:
        summary = run_iterative_rl_training(
            output_dir=args.output_dir,
            iterations=args.iterations,
            self_play_games=args.games,
            eval_games=args.eval_games,
            rounds_per_game=args.rounds,
            simulations=args.sims,
            epochs_per_iter=args.epochs,
            learning_rate=args.lr,
        )
        print("Iterative RL summary:", summary)
    else:
        summary = generate_self_play_dataset(
            output_path=args.output,
            num_games=args.games,
            rounds_per_game=args.rounds,
            simulations=args.sims,
        )
        print("Self-play summary:", summary)

