import random
from typing import Any, Callable, Dict, List, Optional, Tuple

from .node import Node
from .state import PokerState


def _limit_raise_increment_chips(small_blind: int, street: str) -> int:
    """Match fixed-limit engine / abstraction_heuristics: preflop+flop 2*sb, turn+river 4*sb."""
    if street in ("preflop", "flop"):
        return 2 * small_blind
    if street in ("turn", "river"):
        return 4 * small_blind
    return 0


def _board_prefix_for_street(round_state_raw: Dict[str, Any], street: str) -> Tuple[str, ...]:
    """
    When the abstract model advances street, use board cards from the same engine snapshot
    (root round_state_raw) so rollouts do not play the turn with a preflop-only board.
    """
    raw_cards = list(round_state_raw.get("community_card", []))
    want = {"preflop": 0, "flop": 3, "turn": 4, "river": 5, "showdown": len(raw_cards)}.get(
        street, len(raw_cards)
    )
    if want > len(raw_cards):
        want = len(raw_cards)
    return tuple(raw_cards[:want])


def _apply_heuristic_transition(state: PokerState, action: str) -> PokerState:
    hero_stack = state.hero_stack
    opp_stack = state.opp_stack
    pot = state.pot_main
    sb = state.small_blind_amount
    street_order = ["preflop", "flop", "turn", "river", "showdown"]
    idx = street_order.index(state.street) if state.street in street_order else 0
    inc = _limit_raise_increment_chips(sb, state.street)

    if action == "fold":
        # Folding loses current investment opportunity.
        hero_stack = max(0, hero_stack - sb)
        street = "showdown"
        community = _board_prefix_for_street(state.round_state_raw, street)
    elif action == "call":
        # Symmetric toy model: call costs half a min-raise increment each (matches prior sb vs 2*sb split).
        pay = max(1, inc // 2) if inc else sb
        hero_stack = max(0, hero_stack - pay)
        opp_stack = max(0, opp_stack - pay)
        pot += 2 * pay
        street = street_order[min(idx + 1, len(street_order) - 1)]
        community = _board_prefix_for_street(state.round_state_raw, street)
    else:  # raise
        pay = inc if inc else 2 * sb
        hero_stack = max(0, hero_stack - pay)
        opp_stack = max(0, opp_stack - pay)
        pot += 2 * pay
        street = street_order[min(idx + 1, len(street_order) - 1)]
        community = _board_prefix_for_street(state.round_state_raw, street)

    next_legal = ("fold", "call", "raise") if street != "showdown" else tuple()
    return PokerState(
        hero_uuid=state.hero_uuid,
        hole_card=state.hole_card,
        street=street,
        next_player=1 - state.next_player,
        small_blind_amount=state.small_blind_amount,
        hero_is_next=not state.hero_is_next,
        community_card=community,
        pot_main=pot,
        hero_stack=hero_stack,
        opp_stack=opp_stack,
        legal_actions=tuple(next_legal),
        opp_fold_rate=state.opp_fold_rate,
        opp_call_rate=state.opp_call_rate,
        opp_raise_rate=state.opp_raise_rate,
        opp_strength_estimate=state.opp_strength_estimate,
        round_state_raw=state.round_state_raw,
    )


def _rollout_value(state: PokerState) -> float:
    """Simple value estimate in [-1, 1], from hero perspective."""
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
    return max(-1.0, min(1.0, base_value + belief_adjustment))


def _sample_rollout_action(state: PokerState) -> str:
    legal = list(state.legal_actions)
    if len(legal) == 1:
        return legal[0]

    if state.hero_is_next:
        raise_weight = 0.8 + (1.2 * state.opp_fold_rate)
        call_weight = 1.0 + state.opp_call_rate
        fold_weight = max(0.15, 1.0 - 0.5 * state.opp_fold_rate)
    else:
        raise_weight = 0.8 + (2.0 * state.opp_raise_rate)
        call_weight = 0.8 + (1.8 * state.opp_call_rate)
        fold_weight = max(0.15, 0.8 + (2.0 * state.opp_fold_rate))

    weights = []
    for action in legal:
        if action == "raise":
            weights.append(raise_weight)
        elif action == "call":
            weights.append(call_weight)
        else:
            weights.append(fold_weight)
    return random.choices(legal, weights=weights, k=1)[0]


def _simulate_from(
    node: Node,
    rollout_depth: int = 8,
    value_estimator: Optional[Callable[[PokerState], float]] = None,
) -> float:
    state = node.state
    depth = 0
    while not state.is_terminal() and depth < rollout_depth and state.legal_actions:
        action = _sample_rollout_action(state)
        state = _apply_heuristic_transition(state, action)
        depth += 1
    if value_estimator is not None and not state.is_terminal():
        return max(-1.0, min(1.0, float(value_estimator(state))))
    return _rollout_value(state)


def _expand(node: Node) -> Node:
    action = node.untried_actions.pop(random.randrange(len(node.untried_actions)))
    child_state = _apply_heuristic_transition(node.state, action)
    child = Node(state=child_state, parent=node, action_from_parent=action)
    node.children[action] = child
    return child


def run_mcts(
    root_state: PokerState,
    num_simulations: int = 250,
    exploration: float = 1.4,
    rollout_depth: int = 8,
    value_estimator: Optional[Callable[[PokerState], float]] = None,
) -> Tuple[str, Dict[str, Dict[str, float]]]:
    root = Node(state=root_state)
    if not root.state.legal_actions:
        return "fold", {}

    for _ in range(num_simulations):
        node = root

        # Selection
        while node.is_fully_expanded() and node.children:
            node = node.best_child_uct(exploration)

        # Expansion
        if not node.state.is_terminal() and node.untried_actions:
            node = _expand(node)

        # Simulation
        reward = _simulate_from(node, rollout_depth=rollout_depth, value_estimator=value_estimator)

        # Backpropagation
        while node is not None:
            node.visits += 1
            node.value_sum += reward
            node = node.parent

    if not root.children:
        return random.choice(list(root.state.legal_actions)), {}

    # By lecture convention: pick action with highest visit count.
    best_action, best_child = max(root.children.items(), key=lambda kv: kv[1].visits)
    diagnostics = {
        action: {"visits": child.visits, "mean_value": child.mean_value}
        for action, child in root.children.items()
    }
    return best_action, diagnostics

