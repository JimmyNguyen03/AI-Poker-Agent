"""
Self-play data generation for offline training.

Runs games between a DataCollectingPlayer and a fixed opponent, records
(features, discounted_target) pairs, and returns them for training.
"""

from __future__ import annotations
import random
import signal
from typing import Optional

# On Windows, signal.SIGALRM does not exist, so the engine's timeout2 decorator
# crashes on registration. Patch game.py's reference to a no-op before any engine
# imports happen. This does not modify any files — it is a runtime-only override.
if not hasattr(signal, "SIGALRM"):
    import pypokerengine.api.game as _game_api  # noqa: E402 — must precede engine use

    def _noop_timeout2(*_args, **_kwargs):
        def _decorate(fn):
            return fn
        return _decorate

    _game_api.timeout2 = _noop_timeout2

from pypokerengine.players import BasePokerPlayer
from mcts.belief import OpponentBeliefModel

# Fraction of actions taken randomly to ensure dataset diversity.
_EPSILON = 0.15

# Number of abstract rollouts used to label each training example.
# Rollouts use the same card-aware _rollout_value as MCTS, giving low-variance
# labels that directly measure hand strength + opponent belief.
_N_ROLLOUTS = 10


class DataCollectingPlayer(BasePokerPlayer):
    """
    Plays using an epsilon-greedy policy driven by the current value model.
    Records (feature_dict, target) pairs where target is the street-discounted
    normalised chip delta for the round, clamped to [-1, 1].

    Training data uses POST-ACTION child states (the same states MCTS evaluates
    at its leaves), so training and inference share the same feature distribution.
    Fold transitions produce terminal states that MCTS handles via _rollout_value,
    so fold actions are excluded from the training set.
    """

    def __init__(self, value_model=None):
        super().__init__()
        self._value_model = value_model
        self._belief_model = OpponentBeliefModel()
        # Each entry is (street, feature_dict) so we can apply street discounts.
        self._pending: list[tuple[str, dict]] = []
        self._data: list[tuple[dict, float]] = []
        self._round_start_stack: int = 0

    # ------------------------------------------------------------------
    # Engine callbacks
    # ------------------------------------------------------------------

    def declare_action(self, valid_actions, hole_card, round_state):
        from mcts.state import build_state
        from offline_learning.features import state_to_features
        from mcts.search import _apply_heuristic_transition

        state = build_state(
            hero_uuid=self.uuid,
            valid_actions=valid_actions,
            hole_card=hole_card,
            round_state=round_state,
            belief_snapshot=self._belief_model.snapshot(),
        )
        action = self._pick_action(valid_actions, state)

        # Record the POST-ACTION child state — this is exactly what MCTS evaluates
        # at its leaves, so training and inference share the same feature distribution.
        # Fold produces a terminal child handled by _rollout_value in MCTS (never
        # calls the value model), so we skip fold to keep distributions aligned.
        child_state = _apply_heuristic_transition(state, action)
        if not child_state.is_terminal():
            child_features = state_to_features(child_state)
            # Store the full state alongside features so we can run rollouts at
            # round end to produce card-aware, low-variance training labels.
            self._pending.append((child_state, child_features))

        return action

    def receive_game_start_message(self, game_info):
        self._belief_model = OpponentBeliefModel()

    def receive_round_start_message(self, round_count, hole_card, seats):
        self._pending = []
        for seat in seats:
            uuid = seat.get("uuid") if isinstance(seat, dict) else getattr(seat, "uuid", None)
            if uuid == self.uuid:
                stack = seat.get("stack", 0) if isinstance(seat, dict) else getattr(seat, "stack", 0)
                self._round_start_stack = int(stack)
                break

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        self._belief_model.observe(action, self.uuid)

    def receive_round_result_message(self, winners, hand_info, round_state):
        from mcts.node import Node
        from mcts.search import _simulate_from

        # Label each training example with the mean of N abstract rollouts from
        # the child state.  Rollouts use the card-aware _rollout_value (which
        # incorporates hero's MC win probability), so labels have genuine
        # hand-strength signal and very low variance — unlike single-game chip
        # deltas, which correlate near-zero with hand quality.
        for child_state, child_features in self._pending:
            node = Node(state=child_state)
            values = [_simulate_from(node) for _ in range(_N_ROLLOUTS)]
            label = sum(values) / _N_ROLLOUTS
            self._data.append((child_features, label))
        self._pending = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pick_action(self, valid_actions: list, state) -> str:
        """
        Choose an action by evaluating each child state — identical to how MCTS
        uses the value model (depth-1 lookahead). This keeps the data-collection
        policy consistent with inference.
        """
        from mcts.search import _apply_heuristic_transition, _rollout_value
        from offline_learning.features import state_to_features

        valid_names = [a["action"] for a in valid_actions]

        # Epsilon-greedy: explore randomly to ensure dataset diversity
        if random.random() < _EPSILON or self._value_model is None:
            return random.choice(valid_names)

        # Score each action by its child state value (mirrors MCTS leaf evaluation)
        best_action = valid_names[0]
        best_score = float("-inf")
        for action_name in valid_names:
            child = _apply_heuristic_transition(state, action_name)
            if child.is_terminal():
                score = _rollout_value(child)
            else:
                score = self._value_model.predict(state_to_features(child))
            if score > best_score:
                best_score = score
                best_action = action_name
        return best_action

    def get_data(self) -> list[tuple[dict, float]]:
        return list(self._data)


# ------------------------------------------------------------------
# Episode runner
# ------------------------------------------------------------------

def run_self_play_episode(
    num_rounds: int = 50,
    value_model=None,
    opp_player: Optional[BasePokerPlayer] = None,
) -> list[tuple[dict, float]]:
    """
    Run one game and return (features, target) pairs from p1 only.

    p1 uses the current value_model (epsilon-greedy, data-collecting).
    p2 is opp_player — pass any BasePokerPlayer (RandomPlayer, RaisedPlayer,
    a frozen DataCollectingPlayer, etc.).  Defaults to RandomPlayer.

    Only p1's data is returned so labels reflect performance against a fixed
    opponent rather than symmetric noise from an identical policy.
    """
    from pypokerengine.api.game import setup_config, start_poker

    p1 = DataCollectingPlayer(value_model=value_model)

    if opp_player is None:
        from randomplayer import RandomPlayer
        opp_player = RandomPlayer()

    config = setup_config(max_round=num_rounds, initial_stack=1000, small_blind_amount=10)
    config.register_player(name="sp1", algorithm=p1)
    config.register_player(name="sp2", algorithm=opp_player)

    start_poker(config, verbose=0)

    return p1.get_data()
