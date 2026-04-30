import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Tuple

from pypokerengine.api.game import setup_config, start_poker

# Support script execution from repo root.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcts.state import build_state
from mcts_player import MCTSPlayer
from offline_learning.data import DecisionSample, JsonlDatasetWriter
from offline_learning.features import state_to_features


def _stack_by_uuid(round_state: Dict, target_uuid: str) -> Tuple[int, int]:
    seats = round_state.get("seats", [])
    hero = next((seat for seat in seats if seat.get("uuid") == target_uuid), {})
    opp = next((seat for seat in seats if seat.get("uuid") != target_uuid), {})
    return int(hero.get("stack", 0)), int(opp.get("stack", 0))


class LoggingMCTSPlayer(MCTSPlayer):
    def __init__(self, game_index: int, sink: List[DecisionSample], **kwargs):
        super().__init__(**kwargs)
        self.game_index = game_index
        self.sink = sink
        self._pending: List[Dict] = []
        self._round_index = 0

    def receive_round_start_message(self, round_count, hole_card, seats):
        self._pending = []
        self._round_index = int(round_count)

    def declare_action(self, valid_actions, hole_card, round_state):
        action, amount = super().declare_action(valid_actions, hole_card, round_state)
        state = build_state(
            hero_uuid=self.uuid,
            valid_actions=valid_actions,
            hole_card=hole_card,
            round_state=round_state,
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
                    "chosen_amount": float(amount),
                },
            }
        )
        return action, amount

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
            algorithm=LoggingMCTSPlayer(game_index=game_idx, sink=rows, simulations=simulations),
        )
        config.register_player(
            name="mcts_b",
            algorithm=LoggingMCTSPlayer(game_index=game_idx, sink=rows, simulations=simulations),
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


def _parse_args():
    parser = ArgumentParser()
    parser.add_argument("--output", type=str, default="offline_learning/data/self_play.jsonl")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--sims", type=int, default=250)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    summary = generate_self_play_dataset(
        output_path=args.output,
        num_games=args.games,
        rounds_per_game=args.rounds,
        simulations=args.sims,
    )
    print("Self-play summary:", summary)

