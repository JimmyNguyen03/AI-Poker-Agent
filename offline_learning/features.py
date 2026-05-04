from typing import List

from heuristics.abstraction_heuristics import build_cutoff_abstraction
from heuristics.hand_features import MODEL_FEATURE_KEYS
from mcts.state import PokerState


def state_to_features(state: PokerState) -> List[float]:
    """Feature vector aligned with heuristics.hand_features.MODEL_FEATURE_KEYS for value training."""
    round_state = state.round_state_raw
    cutoff = build_cutoff_abstraction(
        hero_uuid=state.hero_uuid,
        hole_card=list(state.hole_card),
        round_state=round_state,
        valid_actions=None,
        win_rate_simulations=None,
    )

    total_stack = max(1.0, float(state.hero_stack + state.opp_stack))
    street_progress = {
        "preflop": 0.0,
        "flop": 1.0 / 3.0,
        "turn": 2.0 / 3.0,
        "river": 1.0,
        "showdown": 1.0,
    }.get(state.street, 0.0)

    feature_dict = {
        "win_rate": float(cutoff.features.get("win_rate", 0.0)),
        "hand_strength_norm": float(cutoff.hand_group.hand_score) / 8.0,
        "hand_group_norm": float(cutoff.hand_group.group_strength) / 5.0,
        "pot_odds": float(cutoff.features.get("pot_odds", 0.0)),
        "call_price_stack_fraction": float(cutoff.call_group.amount_over_stack),
        "raise_price_stack_fraction": float(cutoff.raise_group.amount_over_stack),
        "hero_stack_ratio": float(state.hero_stack) / total_stack,
        "stack_advantage": float(cutoff.features.get("stack_advantage", 0.0)),
        "in_position": 1.0 if state.hero_is_next else 0.0,
        "street_progress": street_progress,
        "board_danger": 0.0,  # Not implemented in heuristics yet.
        "opp_raise_rate": float(state.opp_raise_rate),
        "opp_aggression": float(state.opp_raise_rate),
    }
    return [float(feature_dict.get(key, 0.0)) for key in MODEL_FEATURE_KEYS]
