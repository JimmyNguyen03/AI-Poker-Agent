# AI Poker Agent — COMPSCI 683

MCTS poker agent with an offline-trained value model used to guide tree search.

---

## Setup

```bash
conda create -n CompSci683 python=3.10
conda activate CompSci683
pip install PyPokerEngine matplotlib numpy
```

---

## Project structure

```
AI-Poker-Agent/
├── mcts_player.py              # Competition entry point — MCTSPlayer + setup_ai()
├── randomplayer.py             # Baseline: random action each turn
├── raise_player.py             # Baseline: always raise (or call)
│
├── mcts/
│   ├── search.py               # run_mcts() — UCT tree search
│   ├── state.py                # PokerState dataclass + build_state()
│   ├── node.py                 # MCTS tree node (UCB1)
│   ├── belief.py               # OpponentBeliefModel (fold/call/raise rates)
│   └── legal_actions.py
│
├── heuristics/
│   ├── hand_features.py        # Feature definitions, FEATURE_WEIGHTS, LinearCutoffEvaluator
│   └── abstraction_heuristics.py  # group_hand_strength(), get_cutoff_state_features()
│
└── offline_learning/
    ├── features.py             # state_to_features() — 18 features, MC win-rate cached
    ├── value_model.py          # Linear model + FEATURE_KEYS + load_value_model() factory
    ├── mlp_value_model.py      # MLP: 18 → ReLU(32) → ReLU(16) → clip(1)
    ├── transformer_value_model.py  # Transformer: 18 tokens, d=16, single-head attention
    ├── self_play.py            # DataCollectingPlayer, run_self_play_episode()
    ├── train.py                # CLI: iterative self-play + SGD training
    ├── benchmark.py            # CLI: win-rate + chip-delta vs baselines
    ├── plot_training.py        # CLI: MSE curves from training_log.json
    └── models/                 # Saved checkpoints (gitignored)
        ├── submission_value_model.json   # auto-detected by mcts_player.py
        ├── linear/
        ├── mlp/
        └── transformer/
```

---

## Training a value model

All commands run from the **repo root**.

### Train one model type

```bash
# Linear (fast, interpretable)
python offline_learning/train.py --model linear

# MLP (18 → 32 → 16 → 1)
python offline_learning/train.py --model mlp

# Transformer (18 tokens, d=16, single-head attention)
python offline_learning/train.py --model transformer
```

### Train all three at once (saves to separate subfolders)

```bash
python offline_learning/train.py --model all
# saves to: models/linear/, models/mlp/, models/transformer/
```

### Full training options

| Flag | Default | Description |
|---|---|---|
| `--model` | `linear` | `linear`, `mlp`, `transformer`, or `all` |
| `--iterations` | `3` | Self-play/train cycles |
| `--games-per-iter` | `5` | Games per iteration |
| `--rounds-per-game` | `50` | Rounds per game |
| `--epochs` | `5` | SGD epochs per iteration |
| `--lr` | `0.01` | Learning rate |
| `--output` | `models/submission_value_model.json` | Checkpoint path |

### Recommended full training run

```bash
# Neural models (mlp, transformer) converge better with lr=0.001
python offline_learning/train.py \
    --model all \
    --iterations 5 \
    --games-per-iter 10 \
    --rounds-per-game 100 \
    --epochs 10 \
    --lr 0.001
```

---

## How training works

Each iteration cycles through three opponent types per game:

| Slot | Opponent | Purpose |
|---|---|---|
| 0 | `RandomPlayer` | Baseline — teaches basic hand selection |
| 1 | `RaisedPlayer` | Aggression — teaches folding to large bets |
| 2 | Frozen past model | Self-play — exploits weaknesses a fixed player can't |

Labels are **street-discounted**: preflop decisions get 40% of the round's chip delta, flop 60%, turn 80%, river 100%. This reduces the noise from attributing later-street luck to early decisions.

Only the learning player's data is collected. Symmetric self-play (both players sharing the same weights) causes gradient cancellation and was avoided.

---

## Value model architectures

### Linear (`value_model.py`)

```
score = Σ w_i * x_i + bias,  clipped to [-1, 1]
```

18 features → scalar. Fast, interpretable. Trained with analytical SGD gradient.

### MLP (`mlp_value_model.py`)

```
z1 = ReLU(W1 x + b1)    [32 units]
z2 = ReLU(W2 z1 + b2)   [16 units]
out = clip(W3 z2 + b3, -1, 1)
```

He-initialized. Backpropagation in NumPy. Adam optimizer (β₁=0.9, β₂=0.999), gradient clipping (norm=1.0), L2 regularization (λ=1e-4).

### Transformer (`transformer_value_model.py`)

```
H[i]     = value[i] * E_scale[i] + E_bias[i]   # token embedding
Q,K,V    = H @ W_Q/K/V                          # projections
attn     = softmax(Q K^T / √d)                  # (18×18) attention
context  = attn @ V                              # attended tokens
out      = clip(W_out · mean(context), -1, 1)
```

Single layer, single head, d=16. Full backprop (including softmax Jacobian) in NumPy. Adam optimizer, gradient clipping (norm=1.0), L2 regularization (λ=1e-4).

---

## Features (`offline_learning/features.py`)

| Feature | Description |
|---|---|
| `win_rate` | Monte Carlo win probability (10 simulations, cached per street) |
| `hand_strength_norm` | Hand rank 0–8 normalised to [0, 1] |
| `hand_group_norm` | Win-rate bucket 1–5 normalised to [0, 1] |
| `pot_odds` | Call amount / (pot + call amount) |
| `call_price_stack_fraction` | Call cost as fraction of hero stack |
| `raise_price_stack_fraction` | Raise cost as fraction of hero stack |
| `stack_advantage` | (hero − opp) / total chips |
| `hero_stack_ratio` | Hero stack / total chips |
| `street_progress` | Street index [0–1] |
| `street_is_*` | One-hot street dummies |
| `opp_raise_rate` | Opponent's raise frequency this game |
| `opp_fold_rate` | Opponent's fold frequency |
| `opp_call_rate` | Opponent's call frequency |
| `opp_aggression` | raise_rate − fold_rate |
| `hero_is_next` | 1 if hero acts next |

---

## Using a trained model with the MCTS player

`mcts_player.py` auto-loads `offline_learning/models/submission_value_model.json` if it exists.
Any of the three model types is detected automatically from the JSON's `model_type` field.

```python
# Competition entry
from mcts_player import setup_ai
player = setup_ai()

# Manual — point to any checkpoint
from mcts_player import MCTSPlayer
player = MCTSPlayer(simulations=250, value_model_path="offline_learning/models/mlp/submission_value_model.json")
```

---

## Benchmarking

```bash
# Default (10 games × 50 rounds vs RandomPlayer, RaisedPlayer, untrained MCTS)
python offline_learning/benchmark.py

# More games for tighter confidence intervals
python offline_learning/benchmark.py --games 20 --rounds 100

# Benchmark a specific checkpoint
python offline_learning/benchmark.py --model offline_learning/models/mlp/submission_value_model.json
```

| Flag | Default | Description |
|---|---|---|
| `--games` | `10` | Games per opponent |
| `--rounds` | `50` | Rounds per game |
| `--simulations` | `100` | MCTS simulations per action |
| `--model` | auto-detect | Path to value model JSON |
| `--save` | `models/benchmark_results.json` | Results JSON |
| `--plot-save` | `models/benchmark.png` | Plot path (`""` = interactive) |

---

## Plot training curves

```bash
python offline_learning/plot_training.py
python offline_learning/plot_training.py --save offline_learning/models/training_curve.png
python offline_learning/plot_training.py --log offline_learning/models/mlp/training_log.json
```

---

## Recommended full pipeline

```bash
# Train all three models
python offline_learning/train.py --model all --iterations 5 --games-per-iter 10 --rounds-per-game 100 --epochs 10 --lr 0.005

# Plot training curves for each
python offline_learning/plot_training.py --log offline_learning/models/linear/training_log.json --save offline_learning/models/linear/training_curve.png
python offline_learning/plot_training.py --log offline_learning/models/mlp/training_log.json    --save offline_learning/models/mlp/training_curve.png
python offline_learning/plot_training.py --log offline_learning/models/transformer/training_log.json --save offline_learning/models/transformer/training_curve.png

# Benchmark each
python offline_learning/benchmark.py --model offline_learning/models/linear/submission_value_model.json      --plot-save offline_learning/models/linear/benchmark.png
python offline_learning/benchmark.py --model offline_learning/models/mlp/submission_value_model.json         --plot-save offline_learning/models/mlp/benchmark.png
python offline_learning/benchmark.py --model offline_learning/models/transformer/submission_value_model.json --plot-save offline_learning/models/transformer/benchmark.png
```

---

## Contributors

| Component | Author |
|---|---|
| MCTS search, belief model, state abstraction | — |
| Hand features, abstraction heuristics | — |
| Offline learning (self-play, linear/MLP/transformer, training loop) | Jackson Michaels |
