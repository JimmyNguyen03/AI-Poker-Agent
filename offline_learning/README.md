# offline_learning

**Jackson Michaels** — iterative self-play data generation, linear value model, and SGD-based training.

The trained model is automatically picked up by `mcts_player.py` when
`offline_learning/models/submission_value_model.json` exists, replacing the
default rollout heuristic at MCTS leaf nodes.

---

## Quick start

All commands run from the **repo root** (`AI-Poker-Agent/`).

### Train the model

```bash
# Default run (3 iterations, 5 games, 50 rounds, 5 epochs)
python offline_learning/train.py

# Quick smoke test (~1 min)
python offline_learning/train.py --iterations 2 --games-per-iter 3 --rounds-per-game 20 --epochs 3

# Longer training for submission (~20-40 min depending on hardware)
python offline_learning/train.py --iterations 5 --games-per-iter 10 --rounds-per-game 100 --epochs 10 --lr 0.005

# Custom output path
python offline_learning/train.py --output offline_learning/models/my_model.json
```

**Output:** `offline_learning/models/submission_value_model.json` + `training_log.json`

---

### Plot training curves

```bash
# Show plot interactively
python offline_learning/plot_training.py

# Save to PNG for report
python offline_learning/plot_training.py --save offline_learning/models/training_curve.png

# Use a different log file
python offline_learning/plot_training.py --log offline_learning/models/training_log.json
```

**Plots:** MSE per epoch with ±std shading, and examples collected per iteration.

---

### Run benchmarks against baselines

```bash
# Default (10 games x 50 rounds per opponent, 100 MCTS sims)
python offline_learning/benchmark.py

# More games for tighter confidence intervals
python offline_learning/benchmark.py --games 20 --rounds 100

# Faster run for quick checks
python offline_learning/benchmark.py --games 5 --rounds 30 --simulations 50

# Save plot to a specific path
python offline_learning/benchmark.py --plot-save offline_learning/models/benchmark.png

# Show plot interactively instead of saving
python offline_learning/benchmark.py --plot-save ""
```

**Opponents:** `RandomPlayer`, `RaisedPlayer`, `MCTSPlayer` (untrained, no value model).

**Output:** `benchmark_results.json` + `benchmark.png`

---

## Recommended full pipeline (for report)

```bash
# Step 1 — train
python offline_learning/train.py --iterations 5 --games-per-iter 10 --rounds-per-game 100 --epochs 10 --lr 0.005

# Step 2 — plot training curves
python offline_learning/plot_training.py --save offline_learning/models/training_curve.png

# Step 3 — benchmark
python offline_learning/benchmark.py --games 20 --rounds 100 --plot-save offline_learning/models/benchmark.png
```

---

## Files

| File | Purpose |
|---|---|
| `features.py` | `state_to_features(PokerState) -> dict` — 13 fast features, no MC |
| `value_model.py` | `ValueModel` (linear, clipped to [-1,1]), `load_value_model(path)` |
| `self_play.py` | `DataCollectingPlayer`, `run_self_play_episode()` — epsilon-greedy self-play |
| `train.py` | `train()` (SGD), `run_iterative_self_play()`, CLI |
| `plot_training.py` | MSE curves + data-volume bar chart from `training_log.json` |
| `benchmark.py` | Win-rate + chip-delta bars with error against 3 baselines |
| `models/` | Saved model JSON, training log, benchmark results, plots |

---

## How it works

1. **Features** — 13 features from `PokerState`: stack advantage, pot fraction,
   street dummies, opponent belief rates (fold/call/raise), aggression estimate.
   No Monte Carlo simulation so the value estimator is fast enough for MCTS rollouts.

2. **Value model** — linear `f(x) = w·x + b` clipped to `[-1, 1]`.
   Weights saved as plain JSON for easy inspection.

3. **Self-play data** — two `DataCollectingPlayer` agents play `num_rounds` rounds.
   Policy: epsilon-greedy (15% random, else value-model-guided).
   Labels: normalised chip delta per round (`+/-10 big blinds = +/-1.0`).

4. **Training** — SGD on MSE. Gradient:
   `w_i -= 2 * lr * (pred - target) * x_i`

5. **Iteration** — collect data with current model → train → better policy → repeat.
   Later iterations collect more examples because fewer all-ins eliminate a player early.

---

## CLI flags reference

### `train.py`
| Flag | Default | Description |
|---|---|---|
| `--iterations` | 3 | Self-play/train cycles |
| `--games-per-iter` | 5 | Games per iteration |
| `--rounds-per-game` | 50 | Rounds per game |
| `--epochs` | 5 | SGD epochs per iteration |
| `--lr` | 0.01 | Learning rate |
| `--output` | `models/submission_value_model.json` | Model output path |

### `benchmark.py`
| Flag | Default | Description |
|---|---|---|
| `--games` | 10 | Games per opponent |
| `--rounds` | 50 | Rounds per game |
| `--simulations` | 100 | MCTS simulations per action |
| `--model` | auto-detect | Path to value model JSON |
| `--save` | `models/benchmark_results.json` | Results JSON path |
| `--plot-save` | `models/benchmark.png` | Plot output path (`""` = interactive) |
