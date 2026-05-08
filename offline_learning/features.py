"""
Convert a PokerState to a flat feature dict for the linear value model.

Uses MC win-rate from the heuristics module so the model can distinguish
strong hands from weak ones. Results are cached by (hole_card, community_card,
street) so the expensive MC call fires at most once per street per round.
"""

from __future__ import annotations
from functools import lru_cache
from mcts.state import PokerState

_STREET_IDX = {"preflop": 0, "flop": 1, "turn": 2, "river": 3, "showdown": 4}

# More sims reduce MC variance so rollout labels and features use similar estimates.
_WIN_RATE_SIMS = 20

# Canonical street key per community-card count: ensures features and rollout both
# hit the same LRU cache entry when they have the same hole + community cards.
_COMMUNITY_STREET = {0: "preflop", 3: "flop", 4: "turn", 5: "river"}


@lru_cache(maxsize=2048)
def _hand_strength(hole_card: tuple, community_card: tuple, street: str) -> tuple:
    """Return (win_rate, hand_strength_norm, hand_group_norm). Cached by key."""
    from heuristics.abstraction_heuristics import group_hand_strength
    g = group_hand_strength(
        list(hole_card),
        {"community_card": list(community_card), "street": street},
        win_rate_simulations=_WIN_RATE_SIMS,
    )
    return g.win_rate, g.hand_score / 8.0, g.group_strength / 5.0


def state_to_features(state: PokerState) -> dict[str, float]:
    """Return a feature dict for the linear value model."""
    street_idx = _STREET_IDX.get(state.street, 0)
    total_stack = max(state.hero_stack + state.opp_stack, 1)
    sb = state.small_blind_amount

    # Bet pricing: real engine amounts when available, fixed-limit heuristic otherwise.
    inc = 2 * sb if state.street in ("preflop", "flop") else 4 * sb
    call_amount = state.actual_call_amount if state.actual_call_amount > 0 else max(1, inc // 2)
    raise_amount = state.actual_raise_amount if state.actual_raise_amount > 0 else inc

    # Real hand strength via MC win-rate (cached — at most one MC call per street).
    if state.hole_card:
        wr_street = _COMMUNITY_STREET.get(len(state.community_card), "preflop")
        win_rate, hand_strength_norm, hand_group_norm = _hand_strength(
            state.hole_card, state.community_card, wr_street
        )
    else:
        win_rate = 0.5
        hand_strength_norm = 0.0
        hand_group_norm = 0.2

    return {
        # Hand strength (the critical missing signal)
        "win_rate": win_rate,
        "hand_strength_norm": hand_strength_norm,
        "hand_group_norm": hand_group_norm,
        # Bet pricing
        "pot_odds": call_amount / max(state.pot_main + call_amount, 1),
        "call_price_stack_fraction": call_amount / max(state.hero_stack, 1),
        "raise_price_stack_fraction": raise_amount / max(state.hero_stack, 1),
        # Chip position
        "stack_advantage": (state.hero_stack - state.opp_stack) / total_stack,
        "hero_stack_ratio": state.hero_stack / total_stack,
        # Game progression
        "street_progress": street_idx / 4.0,
        "street_is_preflop": float(state.street == "preflop"),
        "street_is_flop": float(state.street == "flop"),
        "street_is_turn": float(state.street == "turn"),
        "street_is_river": float(state.street == "river"),
        # Opponent beliefs
        "opp_raise_rate": state.opp_raise_rate,
        "opp_fold_rate": state.opp_fold_rate,
        "opp_call_rate": state.opp_call_rate,
        "opp_aggression": state.opp_raise_rate - state.opp_fold_rate,
        # Positional
        "hero_is_next": float(state.hero_is_next),
    }
