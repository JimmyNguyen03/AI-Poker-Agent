"""
Self-play data generation for offline training.

Runs games between a DataCollectingPlayer and a fixed opponent, records
(features, target) pairs, and returns them for training.

Target modes
------------
rollout   (default) — mean of N abstract rollouts from each child state.
                      Card-aware, low-variance, matches MCTS leaf evaluation.
chip_delta            — actual chip change for the round, normalised to [-1, 1]
                      by initial stack (1000), scaled by a per-street discount so
                      earlier decisions receive proportionally less credit.
winner                — +1.0/-1.0 for winning/losing the round pot, scaled by the
                      same per-street discount as chip_delta.
mixed                 — per-state blend of rollout and street-discounted chip_delta:
                          alpha * rollout  +  (1-alpha) * discount * chip_delta
                      Combines the low variance of rollouts with the ground-truth
                      signal of real outcomes.

Street discounts (preflop→river): 0.25 / 0.5 / 0.75 / 1.0
"""

from __future__ import annotations
import random
import signal
from typing import Optional

_TARGET_MODES = ("rollout", "chip_delta", "winner", "mixed")
_INITIAL_STACK = 1000  # normalisation constant for chip_delta

# Earlier streets receive less credit for the round outcome because
# more chance remains between the decision and the final result.
_STREET_DISCOUNT: dict[str, float] = {
    "preflop": 0.25,
    "flop":    0.50,
    "turn":    0.75,
    "river":   1.00,
    "showdown": 1.00,
}

# Weight on the rollout component in mixed mode (remainder goes to chip_delta).
_MIXED_ALPHA = 0.6

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
_N_ROLLOUTS = 5


class DataCollectingPlayer(BasePokerPlayer):
    """
    Plays using an epsilon-greedy policy driven by the current value model.
    Records (feature_dict, target) pairs labelled according to target_mode.

    Training data uses POST-ACTION child states (the same states MCTS evaluates
    at its leaves), so training and inference share the same feature distribution.
    Fold transitions produce terminal states that MCTS handles via _rollout_value,
    so fold actions are excluded from the training set.
    """

    def __init__(self, value_model=None, target_mode: str = "rollout"):
        super().__init__()
        if target_mode not in _TARGET_MODES:
            raise ValueError(f"target_mode must be one of {_TARGET_MODES}, got {target_mode!r}")
        self._value_model = value_model
        self._target_mode = target_mode
        self._belief_model = OpponentBeliefModel()
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

        # Record the POST-ACTION child state for non-fold actions.
        # For fold actions we also record training data: the fold value is
        # deterministic (no rollout needed) so examples go directly to _data.
        child_state = _apply_heuristic_transition(state, action)
        if child_state.is_terminal():
            if child_state.hero_folded:
                child_features = state_to_features(child_state)
                fold_val = self._fold_label(pre_state=state, fold_child=child_state)
                self._data.append((child_features, max(-1.0, min(1.0, fold_val))))
        else:
            child_features = state_to_features(child_state)
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
        if not self._pending:
            return

        if self._target_mode == "rollout":
            from mcts.node import Node
            from mcts.search import _simulate_from
            for child_state, child_features in self._pending:
                node = Node(state=child_state)
                values = [_simulate_from(node) for _ in range(_N_ROLLOUTS)]
                self._data.append((child_features, sum(values) / _N_ROLLOUTS))

        elif self._target_mode == "mixed":
            from mcts.node import Node
            from mcts.search import _simulate_from
            cd = self._chip_delta(round_state)
            for child_state, child_features in self._pending:
                node = Node(state=child_state)
                values = [_simulate_from(node) for _ in range(_N_ROLLOUTS)]
                rollout = sum(values) / _N_ROLLOUTS
                disc = _STREET_DISCOUNT.get(child_state.street, 1.0)
                blended = _MIXED_ALPHA * rollout + (1.0 - _MIXED_ALPHA) * disc * cd
                self._data.append((child_features, max(-1.0, min(1.0, blended))))

        else:
            # chip_delta / winner — label is per-state (street discount varies)
            hero_won = any(
                (w.get("uuid") if isinstance(w, dict) else getattr(w, "uuid", None)) == self.uuid
                for w in winners
            )
            cd = self._chip_delta(round_state)
            for child_state, child_features in self._pending:
                disc = _STREET_DISCOUNT.get(child_state.street, 1.0)
                if self._target_mode == "chip_delta":
                    label = max(-1.0, min(1.0, disc * cd))
                else:  # winner
                    label = disc * (1.0 if hero_won else -1.0)
                self._data.append((child_features, label))

        self._pending = []

    def _fold_label(self, pre_state, fold_child) -> float:
        """
        Immediate label for a hero fold. No round-end data needed — outcome is deterministic.
        Uses pre_state.street for discount and pre_state.hero_stack for chip delta.
        """
        from mcts.search import _rollout_value
        disc = _STREET_DISCOUNT.get(pre_state.street, 1.0)
        rollout = _rollout_value(fold_child)  # now chip-loss-aware, not fixed 0.0
        if self._target_mode == "rollout":
            return rollout
        cd = (pre_state.hero_stack - self._round_start_stack) / _INITIAL_STACK
        if self._target_mode == "chip_delta":
            return disc * cd
        if self._target_mode == "winner":
            return disc * -1.0
        # mixed
        return _MIXED_ALPHA * rollout + (1.0 - _MIXED_ALPHA) * disc * cd

    def _chip_delta(self, round_state) -> float:
        """Chip change this round normalised by initial stack, in [-1, 1]. Not yet discounted."""
        final_stack = self._round_start_stack
        for seat in round_state.get("seats", []):
            uuid = seat.get("uuid") if isinstance(seat, dict) else getattr(seat, "uuid", None)
            if uuid == self.uuid:
                final_stack = int(
                    seat.get("stack", self._round_start_stack) if isinstance(seat, dict)
                    else getattr(seat, "stack", self._round_start_stack)
                )
                break
        return max(-1.0, min(1.0, (final_stack - self._round_start_stack) / _INITIAL_STACK))

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
    target_mode: str = "rollout",
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

    p1 = DataCollectingPlayer(value_model=value_model, target_mode=target_mode)

    if opp_player is None:
        from randomplayer import RandomPlayer
        opp_player = RandomPlayer()

    config = setup_config(max_round=num_rounds, initial_stack=1000, small_blind_amount=10)
    config.register_player(name="sp1", algorithm=p1)
    config.register_player(name="sp2", algorithm=opp_player)

    start_poker(config, verbose=0)

    return p1.get_data()
