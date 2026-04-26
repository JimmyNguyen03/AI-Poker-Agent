from typing import List

from mcts.state import PokerState


STREET_INDEX = {
    "preflop": 0.0,
    "flop": 1.0,
    "turn": 2.0,
    "river": 3.0,
    "showdown": 4.0,
}

ACTION_SET = ("fold", "call", "raise")


def state_to_features(state: PokerState) -> List[float]:
    """Small, stable feature vector for placeholder training."""
    total_stack = max(1.0, float(state.hero_stack + state.opp_stack))
    return [
        float(state.hero_stack) / total_stack,
        float(state.opp_stack) / total_stack,
        float(state.pot_main) / total_stack,
        STREET_INDEX.get(state.street, 0.0) / 4.0,
        float(len(state.community_card)) / 5.0,
        1.0 if "fold" in state.legal_actions else 0.0,
        1.0 if "call" in state.legal_actions else 0.0,
        1.0 if "raise" in state.legal_actions else 0.0,
    ]

