from typing import Any, Dict, List

from heuristics.abstraction_heuristics import build_cutoff_abstraction
from heuristics.hand_features import MODEL_FEATURE_KEYS
from mcts.state import PokerState


def mcts_rollout_leaf_value(state: PokerState) -> float:
    """
    Mirror of mcts.search._rollout_value (heuristic leaf prior in [-1, 1]).
    Keep in sync if groupmates change search rollouts.
    """
    if state.street == "showdown":
        if state.hero_stack > state.opp_stack:
            return 1.0
        if state.hero_stack < state.opp_stack:
            return -1.0
        return 0.0
    stack_delta = state.hero_stack - state.opp_stack
    norm = max(state.hero_stack + state.opp_stack, 1)
    base_value = stack_delta / norm
    belief_adjustment = 0.35 * (0.5 - state.opp_strength_estimate)
    v = base_value + belief_adjustment
    return max(-1.0, min(1.0, v))


def _round_state_aligned_with_poker_state(state: PokerState) -> Dict[str, Any]:
    """
    Heuristics read stacks/pot/street/board from round_state; MCTS updates PokerState only.
    Overlay PokerState onto a copy so training and MCTS value calls see one consistent snapshot.
    """
    raw = state.round_state_raw
    aligned: Dict[str, Any] = dict(raw)
    aligned["street"] = state.street
    aligned["community_card"] = list(state.community_card)

    new_seats: List[Dict[str, Any]] = []
    for seat in raw.get("seats", []):
        s = dict(seat)
        uid = s.get("uuid")
        if uid == state.hero_uuid:
            s["stack"] = state.hero_stack
        elif uid and uid != state.hero_uuid:
            s["stack"] = state.opp_stack
        new_seats.append(s)
    aligned["seats"] = new_seats

    pot = raw.get("pot", {})
    if isinstance(pot, dict):
        main = pot.get("main", {})
        if isinstance(main, dict):
            new_main = dict(main)
            new_main["amount"] = state.pot_main
            aligned["pot"] = {**pot, "main": new_main}
        else:
            aligned["pot"] = {**pot, "main": {"amount": state.pot_main}}
    else:
        aligned["pot"] = {"main": {"amount": state.pot_main}}

    return aligned


def state_to_features(state: PokerState) -> List[float]:
    """Feature vector aligned with heuristics.hand_features.MODEL_FEATURE_KEYS for value training."""
    round_state = _round_state_aligned_with_poker_state(state)
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
