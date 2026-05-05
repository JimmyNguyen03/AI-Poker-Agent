"""
Convert a PokerState to a flat feature dict for the value model.

Only depends on pypokerengine (whitelisted by the tournament portal) and the
Python standard library — no heuristics or offline_learning imports needed.
"""

from __future__ import annotations
from functools import lru_cache
from mcts.state import PokerState

_STREET_IDX = {"preflop": 0, "flop": 1, "turn": 2, "river": 3, "showdown": 4}

_WIN_RATE_SIMS = 50

_COMMUNITY_STREET = {0: "preflop", 3: "flop", 4: "turn", 5: "river"}

_HAND_TO_SCORE = {
    "HIGHCARD": 0, "ONEPAIR": 1, "TWOPAIR": 2, "THREECARD": 3,
    "STRAIGHT": 4, "FLASH": 5, "FULLHOUSE": 6, "FOURCARD": 7, "STRAIGHTFLASH": 8,
}


@lru_cache(maxsize=2048)
def _hand_strength(hole_card: tuple, community_card: tuple, street: str) -> tuple:
    """Return (win_rate, hand_strength_norm, hand_group_norm). Cached by key."""
    from pypokerengine.utils.card_utils import estimate_hole_card_win_rate, gen_cards
    from pypokerengine.engine.hand_evaluator import HandEvaluator

    hole = gen_cards(list(hole_card))
    community = gen_cards(list(community_card)) if community_card else []

    win_rate = estimate_hole_card_win_rate(_WIN_RATE_SIMS, 2, hole, community)
    hand_name = HandEvaluator.gen_hand_rank_info(hole, community)["hand"]["strength"]
    hand_score = _HAND_TO_SCORE.get(hand_name, 0)
    group_strength = next((g for t, g in [(0.80, 5), (0.65, 4), (0.50, 3), (0.35, 2)] if win_rate >= t), 1)

    return win_rate, hand_score / 8.0, group_strength / 5.0


def state_to_features(state: PokerState) -> dict[str, float]:
    """Return a feature dict for the linear value model."""
    street_idx = _STREET_IDX.get(state.street, 0)
    total_stack = max(state.hero_stack + state.opp_stack, 1)
    sb = state.small_blind_amount

    inc = 2 * sb if state.street in ("preflop", "flop") else 4 * sb
    call_amount = state.actual_call_amount if state.actual_call_amount > 0 else max(1, inc // 2)
    raise_amount = state.actual_raise_amount if state.actual_raise_amount > 0 else inc

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
        "win_rate": win_rate,
        "hand_strength_norm": hand_strength_norm,
        "hand_group_norm": hand_group_norm,
        "pot_odds": call_amount / max(state.pot_main + call_amount, 1),
        "call_price_stack_fraction": call_amount / max(state.hero_stack, 1),
        "raise_price_stack_fraction": raise_amount / max(state.hero_stack, 1),
        "stack_advantage": (state.hero_stack - state.opp_stack) / total_stack,
        "hero_stack_ratio": state.hero_stack / total_stack,
        "street_progress": street_idx / 4.0,
        "street_is_preflop": float(state.street == "preflop"),
        "street_is_flop": float(state.street == "flop"),
        "street_is_turn": float(state.street == "turn"),
        "street_is_river": float(state.street == "river"),
        "opp_raise_rate": state.opp_raise_rate,
        "opp_fold_rate": state.opp_fold_rate,
        "opp_call_rate": state.opp_call_rate,
        "opp_aggression": state.opp_raise_rate - state.opp_fold_rate,
        "hero_is_next": float(state.hero_is_next),
    }
