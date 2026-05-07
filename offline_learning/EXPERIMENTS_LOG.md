# Experiment log (hyperparameter & baseline runs)

Use this file in meetings and in the Overleaf report.  
Artifacts (`.json` checkpoints, `.png` plots) are gitignored by project policy; re-run commands below to regenerate.

## 2026-05-06 — Quick hyperparameter sweep (local, small budget)

**Goal (per team):** run experiments with different hyperparameters and compare vs `RandomPlayer`, `RaisedPlayer`, and untrained `MCTSPlayer`.

**Training setup (all runs):**

| Field | Value |
|-------|--------|
| Iterations | 2 |
| Games / iter | 3 |
| Rounds / game | 40 |
| Initial stack / blinds | default benchmark (`1000` / `10`) |

**Benchmark setup (all runs):**

| Field | Value |
|-------|--------|
| Games / opponent | 6 |
| Rounds / game | 35 |
| MCTS simulations / action | 40 |

**Results (win rate ± approximate 95% CI from script):**

| Run ID | Model | Train `lr` | Train `epochs` | vs Random | vs Raised | vs MCTS (no value model) |
|--------|--------|------------|----------------|-----------|-----------|-------------------------|
| A | linear | 0.01 | 5 | **67%** (±38%) | **50%** (±40%) | 17% (±30%) |
| B | linear | 0.005 | 5 | 50% (±40%) | 50% (±40%) | **33%** (±38%) |
| C | mlp | 0.001 | 8 | **83%** (±30%) | 50% (±40%) | **83%** (±30%) |

**Takeaways for discussion**

1. **Sample size is small** (`n=6` per opponent): treat differences as directional, not definitive. For report-quality numbers, use `--games 20` (or more) as in `README.md`.
2. **Lower linear LR (0.005)** in this seed/budget did not beat **0.01** vs random; **MLP + lr=0.001** looked strongest on this micro-benchmark (especially vs random and vs no-VM MCTS).
3. **vs RaisedPlayer** stayed near coin-flip for all three — aligns with Quackson’s note that aggression is still a pain point; likely needs **more training iterations**, **stronger abstraction / tree prior**, or **belief-model tuning**, not just LR.

**Reproduce Run A (linear, lr=0.01)**

```bash
python offline_learning/train.py --model linear --iterations 2 --games-per-iter 3 --rounds-per-game 40 --epochs 5 --lr 0.01 --output offline_learning/models/experiments/exp_linear_lr001.json
python offline_learning/benchmark.py --model offline_learning/models/experiments/exp_linear_lr001.json --games 6 --rounds 35 --simulations 40 --save offline_learning/models/experiments/bench_linear_lr001.json --plot-save offline_learning/models/experiments/bench_linear_lr001.png
```

**Reproduce Run B (linear, lr=0.005)**

```bash
python offline_learning/train.py --model linear --iterations 2 --games-per-iter 3 --rounds-per-game 40 --epochs 5 --lr 0.005 --output offline_learning/models/experiments/exp_linear_lr0005.json
python offline_learning/benchmark.py --model offline_learning/models/experiments/exp_linear_lr0005.json --games 6 --rounds 35 --simulations 40 --save offline_learning/models/experiments/bench_linear_lr0005.json --plot-save offline_learning/models/experiments/bench_linear_lr0005.png
```

**Reproduce Run C (mlp, lr=0.001)**

```bash
python offline_learning/train.py --model mlp --iterations 2 --games-per-iter 3 --rounds-per-game 40 --epochs 8 --lr 0.001 --output offline_learning/models/experiments/exp_mlp_lr0001.json
python offline_learning/benchmark.py --model offline_learning/models/experiments/exp_mlp_lr0001.json --games 6 --rounds 35 --simulations 40 --save offline_learning/models/experiments/bench_mlp_lr0001.json --plot-save offline_learning/models/experiments/bench_mlp_lr0001.png
```

**Recommended next experiments (higher signal)**

```bash
# Longer training (README-style)
python offline_learning/train.py --model mlp --iterations 5 --games-per-iter 10 --rounds-per-game 100 --epochs 10 --lr 0.001 --output offline_learning/models/experiments/exp_mlp_fullish.json

# Tighter benchmark
python offline_learning/benchmark.py --model offline_learning/models/experiments/exp_mlp_fullish.json --games 20 --rounds 100 --simulations 100 --save offline_learning/models/experiments/bench_mlp_fullish.json --plot-save offline_learning/models/experiments/bench_mlp_fullish.png
```
