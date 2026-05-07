# offline_learning

**Jackson Michaels** — iterative self-play data generation, value models (linear / MLP / transformer), and SGD-based training.

The trained model is automatically picked up by `mcts_player.py` when
`offline_learning/models/submission_value_model.json` exists, replacing the
default rollout heuristic at MCTS leaf nodes.

---

## Quick start

All commands run from the **repo root** (`AI-Poker-Agent/`).

### Train the model

```bash
# Default run — linear model, rollout labels (3 iters, 5 games, 50 rounds, 5 epochs)
python offline_learning/train.py

# Quick smoke test (~1 min)
python offline_learning/train.py --iterations 2 --games-per-iter 3 --rounds-per-game 20 --epochs 3

# Longer training for submission (~20-40 min depending on hardware)
python offline_learning/train.py --iterations 5 --games-per-iter 10 --rounds-per-game 100 --epochs 10 --lr 0.005

# Train a specific model architecture
python offline_learning/train.py --model mlp
python offline_learning/train.py --model transformer

# Train all architectures to separate subfolders
python offline_learning/train.py --model all

# Use a different training target
python offline_learning/train.py --target chip_delta
python offline_learning/train.py --target winner
python offline_learning/train.py --target mixed

# Full grid: 3 model types × 3 target modes → 9 subfolders
python offline_learning/train.py --model all --target all

# Custom output path (single model only)
python offline_learning/train.py --output offline_learning/models/my_model.json
```

**Output:** `offline_learning/models/submission_value_model.json` + `training_log.json`

For `--model all` or `--target all`, each combination is saved to its own subfolder:
- Single target sweep: `models/<model>/submission_value_model.json`
- Full grid: `models/<model>/<target>/submission_value_model.json`

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
# Default (10 games x 50 rounds per opponent, 100 MCTS sims) — uses root model
python offline_learning/benchmark.py

# Benchmark all discovered models (root + every subfolder) in one run
python offline_learning/benchmark.py --all

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

**Output:** `benchmark_results.json` + `benchmark.png` per model, plus a combined
`benchmark_comparison_results.json` when using `--all`.

---

## Recommended full pipeline (for report)

```bash
# Step 1 — train all model types and target modes
python offline_learning/train.py --model all --target all \
    --iterations 5 --games-per-iter 10 --rounds-per-game 100 --epochs 10 --lr 0.005

# Step 2 — benchmark every discovered model in one pass
python offline_learning/benchmark.py --all --games 20 --rounds 100

# Step 3 — plot training curves for any individual model
python offline_learning/plot_training.py \
    --log offline_learning/models/mlp/rollout/training_log.json \
    --save offline_learning/models/mlp/rollout/training_curve.png
```

---

## Files

| File | Purpose |
|---|---|
| `features.py` | `state_to_features(PokerState) -> dict` — 18 features including MC win-rate |
| `value_model.py` | `ValueModel` (linear), `load_value_model(path)` factory |
| `mlp_value_model.py` | `MLPValueModel` — two-hidden-layer MLP (18→ReLU(32)→ReLU(16)→clip) |
| `transformer_value_model.py` | `TransformerValueModel` — single-layer self-attention over feature tokens |
| `self_play.py` | `DataCollectingPlayer`, `run_self_play_episode()` — epsilon-greedy self-play |
| `train.py` | `train()` (SGD/Adam), `run_iterative_self_play()`, CLI |
| `plot_training.py` | MSE curves + data-volume bar chart from `training_log.json` |
| `benchmark.py` | Win-rate + chip-delta bars with error against 3 baselines |
| `models/` | Saved model JSON, training log, benchmark results, plots |

---

## How it works

1. **Features** — 18 features from `PokerState`: MC win-rate, hand strength, pot odds,
   stack fractions, street dummies, opponent belief rates (fold/call/raise), aggression.

2. **Model architectures**
   - `linear` — dot-product `f(x) = w·x + b`, clipped to `[-1, 1]`. Fast; weights are
     human-readable JSON.
   - `mlp` — two hidden layers (ReLU) trained with Adam + L2 regularisation.
   - `transformer` — single-head self-attention over feature tokens, mean-pooled, linear
     output. Trained with Adam + L2 regularisation.
   All architectures export the same `predict(features) / save(path) / load(path)` interface.

3. **Training targets** (`--target`)
   | Mode | Label source | Variance | Notes |
   |---|---|---|---|
   | `rollout` *(default)* | Mean of N abstract rollouts from each child state | Low | Card-aware; matches MCTS leaf evaluation distribution |
   | `chip_delta` | Actual chip change for the round, normalised by initial stack (1000), scaled by street discount | Medium | True outcome signal; preflop decisions get 25% credit, river 100% |
   | `winner` | `+1.0/-1.0` for winning/losing the round pot, scaled by street discount | Medium | Binary signal scaled by street so preflop folds aren't over-penalised |
   | `mixed` | `0.6 × rollout + 0.4 × discount × chip_delta` per state | Low–Medium | Best of both: rollout keeps variance low, chip_delta grounds labels in real outcomes |

   **Street discounts** applied to `chip_delta`, `winner`, and the chip_delta component of `mixed`:
   preflop=0.25 · flop=0.50 · turn=0.75 · river=1.00

4. **Self-play** — hero uses an epsilon-greedy policy (15% random, else value-model-guided
   depth-1 lookahead). Data is collected from POST-ACTION child states — the same states
   MCTS evaluates at its leaves — keeping training and inference distributions aligned.
   Fold transitions are excluded (they are terminal; MCTS handles them via `_rollout_value`).

5. **Training** — SGD on MSE (linear) or Adam on MSE (MLP / transformer).

6. **Iteration** — collect data with current model → train → better policy → repeat.
   Frozen snapshots of past models are used as opponents so the agent learns to beat
   its own earlier behaviour, not just random baselines.

---

## CLI flags reference

### `train.py`

| Flag | Default | Description |
|---|---|---|
| `--model` | `linear` | Architecture: `linear`, `mlp`, `transformer`, or `all` |
| `--target` | `rollout` | Training target: `rollout`, `chip_delta`, `winner`, `mixed`, or `all` |
| `--iterations` | 3 | Self-play/train cycles |
| `--games-per-iter` | 5 | Games per iteration |
| `--rounds-per-game` | 50 | Rounds per game |
| `--epochs` | 5 | SGD/Adam epochs per iteration |
| `--lr` | 0.01 | Learning rate |
| `--output` | `models/submission_value_model.json` | Output path (subfolders used for `all`) |

### `benchmark.py`

| Flag | Default | Description |
|---|---|---|
| `--all` | off | Discover and benchmark every model under `models/` |
| `--games` | 10 | Games per opponent |
| `--rounds` | 50 | Rounds per game |
| `--simulations` | 100 | MCTS simulations per action |
| `--model` | auto-detect | Path to a specific value model JSON |
| `--save` | `models/benchmark_results.json` | Results JSON path |
| `--plot-save` | `models/benchmark.png` | Plot output path (`""` = interactive) |
