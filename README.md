# AI Poker Agent — COMPSCI 683

MCTS poker agent with an offline-trained value model for rollout guidance.

## Setup

```bash
conda create -n CompSci683 python=3.10
conda activate CompSci683
pip install PyPokerEngine matplotlib numpy
```

## Layout


| Area                       | Role                                                                                                         |
| -------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `mcts_player.py`           | Competition entry: `setup_ai()`, `MCTSPlayer`                                                                |
| `mcts/`                    | UCT search, state, belief model, legal actions                                                               |
| `heuristics/`              | Hand features and abstraction helpers                                                                        |
| `offline_learning/`        | Features, linear / MLP / transformer value models, self-play, `train.py`, `benchmark.py`, `plot_training.py` |
| `offline_learning/models/` | Checkpoints (gitignored); `submission_value_model.json` is auto-loaded by `mcts_player.py`                   |


## Train

From repo root:

```bash
python offline_learning/train.py --model linear    # or mlp, transformer, all
```

Useful flags (defaults in parentheses): `--iterations` (3), `--games-per-iter` (5), `--rounds-per-game` (50), `--epochs` (5), `--lr` (0.01), `--output` (`models/submission_value_model.json`). Neural models often do better with `--lr 0.001`.

Example stronger run:

```bash
python offline_learning/train.py --model all --iterations 5 --games-per-iter 10 --rounds-per-game 100 --epochs 10 --lr 0.001
```

**Training loop (short):** Each iteration mixes opponents (random, raise-heavy, frozen past checkpoint). Labels discount earlier streets (preflop → river) so early decisions are not fully blamed on river luck. Only the learning side’s trajectories are stored.

**Models:** Linear regression on 18 features (clipped). MLP 18→32→16→1 and a small single-layer transformer over 18 tokens; both use NumPy + Adam in this codebase.

**Features:** MC win-rate, normalized strength/grouping, pot odds and stack fractions, street progress, opponent fold/call/raise stats, position flag — see `offline_learning/features.py` for the full list.

## Use the trained player

`mcts_player.py` loads `offline_learning/models/submission_value_model.json` when present; JSON `model_type` selects linear / mlp / transformer.

```python
from mcts_player import setup_ai
player = setup_ai()

from mcts_player import MCTSPlayer
player = MCTSPlayer(simulations=250, value_model_path="offline_learning/models/mlp/submission_value_model.json")
```

## Benchmark & plots

```bash
python offline_learning/benchmark.py
python offline_learning/benchmark.py --games 20 --rounds 100 --model offline_learning/models/mlp/submission_value_model.json
```

Flags: `--games` (10), `--rounds` (50), `--simulations` (100), `--model`, `--save`, `--plot-save`.

```bash
python offline_learning/plot_training.py
python offline_learning/plot_training.py --log offline_learning/models/mlp/training_log.json --save offline_learning/models/mlp/training_curve.png
```

## Experiments

Log team runs in `offline_learning/EXPERIMENTS_LOG.md`. Quick grid: `python offline_learning/run_quick_experiments.py`; scale `--games` / `--rounds` for tighter benchmarks.