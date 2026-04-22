# MCTS Starter Component (COMPSCI 683)

This folder contains a starter implementation for the project task:

- Monte Carlo Tree Search (selection, expansion, simulation, backpropagation)
- game state representation
- legal-action generation/normalization

## Files

- `mcts_player.py`: `BasePokerPlayer` implementation using MCTS in `declare_action`
- `mcts/state.py`: `PokerState` dataclass and `build_state(...)`
- `mcts/legal_actions.py`: action-name extraction and `(action, amount)` conversion
- `mcts/node.py`: MCTS node and UCT child selection
- `mcts/search.py`: bounded MCTS with rollout policy and action selection

## Why this structure

The professor instructions require that the key project code is driven by
`declare_action()` without modifying engine internals. This implementation keeps
all custom logic in your player and helper modules.

## Quick integration in team repo

1. Copy `mcts_player.py` and `mcts/` into your project root.
2. In your local test script, register this player:

```python
from mcts_player import MCTSPlayer
config.register_player(name="team_agent", algorithm=MCTSPlayer(simulations=250))
```

1. Tune `simulations` (e.g. 100, 250, 500) to fit action time constraints.

## Current limitations (expected for starter)

- Rollout transition in `mcts/search.py` is heuristic (fast and deterministic),
not a full hidden-card simulation.
- This is ideal for milestone progress and team integration. You can later swap
`_apply_heuristic_transition(...)` with `Emulator`-based transitions.

## Suggested next upgrade

Replace rollout transition with emulator-driven playouts and add a stronger
cutoff evaluation using your teammate's EHV/abstraction features.