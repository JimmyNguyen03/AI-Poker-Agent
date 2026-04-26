from pypokerengine.players import BasePokerPlayer

from mcts.legal_actions import to_engine_action
from mcts.search import run_mcts
from mcts.state import build_state


class MCTSPlayer(BasePokerPlayer):
    """Starter MCTS agent for COMPSCI 683 poker project."""

    def __init__(
        self,
        simulations: int = 250,
        exploration: float = 1.4,
        value_model_path: str = "",
    ):
        super().__init__()
        self.simulations = simulations
        self.exploration = exploration
        self.last_search_info = {}
        self._value_model = None
        if value_model_path:
            from offline_learning.value_model import LinearValueModel

            self._value_model = LinearValueModel.load(value_model_path)

    def declare_action(self, valid_actions, hole_card, round_state):
        # Build root state from callback payload.
        state = build_state(
            hero_uuid=self.uuid,
            valid_actions=valid_actions,
            hole_card=hole_card,
            round_state=round_state,
        )

        # Run bounded MCTS (simulation budget can be tuned for time constraints).
        best_action, diagnostics = run_mcts(
            root_state=state,
            num_simulations=self.simulations,
            exploration=self.exploration,
            rollout_depth=8,
            value_estimator=self._estimate_value if self._value_model is not None else None,
        )
        self.last_search_info = diagnostics

        action, amount = to_engine_action(best_action, valid_actions, round_state)
        if amount is None:
            amount = 0
        return action, int(amount)

    def _estimate_value(self, state):
        from offline_learning.features import state_to_features

        return self._value_model.predict(state_to_features(state))

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass


def setup_ai():
    return MCTSPlayer()

