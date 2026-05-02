from typing import Optional

STREET_TO_INDEX = {
    "preflop": 0,
    "flop": 1,
    "turn": 2,
    "river": 3,
    "showdown": 4,
}
HAND_TO_SCORE = {
    "HIGHCARD": 0,
    "ONEPAIR": 1,
    "TWOPAIR": 2,
    "THREECARD": 3,
    "STRAIGHT": 4,
    "FLASH": 5,
    "FULLHOUSE": 6,
    "FOURCARD": 7,
    "STRAIGHTFLASH": 8,
}
FEATURE_WEIGHTS: dict[str, float] = {
    "win_rate": 1.60,
    "hand_strength_norm": 0.45,
    "hand_group_norm": 0.35,
    "pot_odds": 0.40,
    "call_price_stack_fraction": -0.55,
    "raise_price_stack_fraction": -0.25,
    "hero_stack_ratio": 0.35,
    "stack_advantage": 0.55,
    "in_position": 0.15,
    "street_progress": 0.10,
    "board_danger": -0.10,
    "opp_raise_rate": -0.20,
    "opp_aggression": -0.20,
    "bias": 0,
}

class HandStrengthGroup:
    def __init__(self, group_strength: int, hand: str, hand_score: int, win_rate: float):
        self.group_strength = group_strength
        self.hand = hand
        self.hand_score = hand_score
        self.win_rate = win_rate
        
class BetSizeGroup:
    def __init__(self, action: str, amount: int, bet_size_group: int,
                 amount_over_pot: float, amount_over_stack: float):
        self.action = action
        self.amount = amount
        self.bet_size_group = bet_size_group
        self.amount_over_pot = amount_over_pot
        self.amount_over_stack = amount_over_stack
        
class CutoffAbstraction:
    def __init__(self,hand_group: HandStrengthGroup, call_group: BetSizeGroup,
                 raise_group: BetSizeGroup, features: dict[str, float], heuristic_value: float):
        self.hand_group = hand_group
        self.call_group = call_group
        self.raise_group = raise_group
        self.features = features
        self.heuristic_value = heuristic_value
        
class LinearCutoffEvaluator:
    def __init__(self, weights: Optional[dict[str, float]]):
        self.weights = dict(FEATURE_WEIGHTS)
        self.weights.update(weights or {})

    def score(self, features: dict[str, float]) -> float:
        s = sum(self.weights.get(n, 0) * float(v) for n, v in features.items())
        s += self.weights.get("bias", 0)
        return max(-1.0, min(1.0, s))
