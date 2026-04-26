import random
from typing import Callable, Dict, List, Optional, Tuple

from .node import Node
from .state import PokerState


def _apply_heuristic_transition(state: PokerState, action: str) -> PokerState:
    """Cheap transition model for starter MCTS.

    This does not fully emulate hidden cards; it approximates immediate effects
    and is intended as a scaffold you can later replace with Emulator rollouts.
    """
    hero_stack = state.hero_stack
    opp_stack = state.opp_stack
    pot = state.pot_main
    sb = state.small_blind_amount
    street_order = ["preflop", "flop", "turn", "river", "showdown"]
    idx = street_order.index(state.street) if state.street in street_order else 0

    if action == "fold":
        # Folding loses current investment opportunity.
        hero_stack = max(0, hero_stack - sb)
        street = "showdown"
    elif action == "call":
        call_cost = sb if state.street in ("preflop", "flop") else 2 * sb
        hero_stack = max(0, hero_stack - call_cost)
        opp_stack = max(0, opp_stack - call_cost)
        pot += 2 * call_cost
        street = street_order[min(idx + 1, len(street_order) - 1)]
    else:  # raise
        raise_cost = 2 * sb if state.street in ("preflop", "flop") else 4 * sb
        hero_stack = max(0, hero_stack - raise_cost)
        opp_stack = max(0, opp_stack - raise_cost)
        pot += 2 * raise_cost
        street = street_order[min(idx + 1, len(street_order) - 1)]

    next_legal = ("fold", "call", "raise") if street != "showdown" else tuple()
    return PokerState(
        hero_uuid=state.hero_uuid,
        hole_card=state.hole_card,
        street=street,
        next_player=1 - state.next_player,
        small_blind_amount=state.small_blind_amount,
        community_card=state.community_card,
        pot_main=pot,
        hero_stack=hero_stack,
        opp_stack=opp_stack,
        legal_actions=tuple(next_legal),
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
    return max(-1.0, min(1.0, stack_delta / norm))


def _simulate_from(
    node: Node,
    rollout_depth: int = 8,
    value_estimator: Optional[Callable[[PokerState], float]] = None,
) -> float:
    state = node.state
    depth = 0
    while not state.is_terminal() and depth < rollout_depth and state.legal_actions:
        action = random.choice(list(state.legal_actions))
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

