"""
Simple rules-based poker player for use as a training opponent.

Decision logic (in order):
  1. 10% random deviation — keeps training data diverse, prevents exploitation.
  2. win_rate >= 0.68  → raise (strong hand, value bet); call if raise unavailable.
  3. win_rate >= 0.50  → call (medium hand — getting in cheaply).
  4. win_rate >= 0.35  → call only if pot odds justify it (win_rate > pot_odds).
  5. win_rate <  0.35  → fold; check for free if call_amount == 0.

Win rate is estimated via a lightweight MC simulation (25 rollouts) using the
same pypokerengine estimator used by the heuristics module, so it is
card-aware but fast enough not to slow down self-play training.
"""
from __future__ import annotations
import random

from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.card_utils import estimate_hole_card_win_rate, gen_cards

_SIMS = 10          # MC rollouts per decision — balance speed vs accuracy
_EPSILON = 0.10     # random deviation probability


class RulesBasedPlayer(BasePokerPlayer):
    """
    A calibrated training opponent that reasons from hand strength and pot odds.
    Significantly harder than RandomPlayer or RaisedPlayer while remaining
    deterministic-ish (so learning signal is stable).
    """

    def declare_action(self, valid_actions, hole_card, round_state):
        from mcts.state import _compute_bet_amounts

        fold_action  = valid_actions[0]
        call_action  = valid_actions[1]
        can_raise    = len(valid_actions) == 3
        raise_action = valid_actions[2] if can_raise else None

        call_amount, _ = _compute_bet_amounts(self.uuid, round_state)

        if random.random() < _EPSILON:
            return random.choice([a["action"] for a in valid_actions])

        community = gen_cards(round_state.get("community_card", []))
        hole      = gen_cards(hole_card)
        win_rate  = estimate_hole_card_win_rate(_SIMS, 2, hole, community)

        pot_main = (round_state.get("pot") or {}).get("main", {})
        pot      = pot_main.get("amount", 0) if isinstance(pot_main, dict) else 0
        pot_odds = call_amount / (pot + call_amount + 1e-9) if call_amount > 0 else 0.0

        if win_rate >= 0.68:
            return raise_action["action"] if can_raise else call_action["action"]

        if win_rate >= 0.50:
            return call_action["action"]

        if win_rate >= 0.35 and win_rate > pot_odds:
            return call_action["action"]

        # Weak hand: check for free, otherwise fold
        return call_action["action"] if call_amount == 0 else fold_action["action"]

    def receive_game_start_message(self, game_info):       pass
    def receive_round_start_message(self, round_count, hole_card, seats): pass
    def receive_street_start_message(self, street, round_state):          pass
    def receive_game_update_message(self, action, round_state):           pass
    def receive_round_result_message(self, winners, hand_info, round_state): pass


def setup_ai():
    return RulesBasedPlayer()
