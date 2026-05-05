from pathlib import Path

from pypokerengine.players import BasePokerPlayer

from mcts.belief import OpponentBeliefModel
from mcts.search import run_mcts
from mcts.state import build_state


class MCTSPlayer(BasePokerPlayer):
    """Starter MCTS agent for COMPSCI 683 poker project."""

    def __init__(
        self,
        simulations: int = 250,
        exploration: float = 1.4,
        value_model_path: str = "",
        rollout_depth: int = 8,
    ):
        super().__init__()
        self.simulations = simulations
        self.exploration = exploration
        self.rollout_depth = rollout_depth
        self.last_search_info = {}
        self._belief_model = OpponentBeliefModel()
        self._value_model = None
        if value_model_path:
            from mcts.value_model import load_value_model

            self._value_model = load_value_model(value_model_path)

    def declare_action(self, valid_actions, hole_card, round_state):
        # Build root state from callback payload.
        state = build_state(
            hero_uuid=self.uuid,
            valid_actions=valid_actions,
            hole_card=hole_card,
            round_state=round_state,
            belief_snapshot=self._belief_model.snapshot(),
        )

        # Run bounded MCTS (simulation budget can be tuned for time constraints).
        best_action, diagnostics = run_mcts(
            root_state=state,
            num_simulations=self.simulations,
            exploration=self.exploration,
            rollout_depth=self.rollout_depth,
            value_estimator=self._estimate_value if self._value_model is not None else None,
        )
        self.last_search_info = diagnostics

        # This engine expects action string only.
        return best_action

    def _estimate_value(self, state):
        from mcts.features import state_to_features

        return self._value_model.predict(state_to_features(state))

    def get_belief_snapshot(self):
        return self._belief_model.snapshot()

    def receive_game_start_message(self, game_info):
        self._belief_model = OpponentBeliefModel()

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        self._belief_model.observe(action, self.uuid)

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass


def setup_ai():
    """
    Competition / harness entry: no-arg factory.
    Ship a checkpoint next to the repo layout below; if missing, MCTS runs without a learned value head.
    """
    here = Path(__file__).resolve().parent
    # In the tournament zip, mcts_player.py lives inside submission/ next to the model.
    # In the dev repo, submission/ is a sibling directory.
    for candidate in [
        here / "submission_value_model.json",                          # zip: submission/
        here / "submission" / "submission_value_model.json",           # dev repo
        here / "offline_learning" / "models" / "submission_value_model.json",  # legacy
    ]:
        submission = candidate
        if submission.is_file():
            break
    if submission.is_file():
        return MCTSPlayer(value_model_path=str(submission.resolve()))
    return MCTSPlayer()

