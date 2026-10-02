# Run guide

Operational reference for running, evaluating and reproducing GFlowAlpha experiments. For the
scientific rationale see [methodology.md](methodology.md); for per-experiment purpose and paper
mapping see [experiments.md](experiments.md).

## 1. Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[pysr,gp,dev]"        # core + symbolic regression + GP + test tools
# optional extras:
#   ,alphagen   external RL baselines (gymnasium, stable-baselines3, sb3-contrib)
#   ,llm        LLM constraint generation (openai)
#   ,akshare    CSI 1000 constituents in the fetcher (Baostock lacks a CSI 1000 interface)
```

- **Python ≥ 3.13.**
- **PySR / Julia.** The first PySR run initializes a Julia backend under `.venv/julia_env`
  (network required, slow). If PySR is unavailable the engine silently skips symbolic regression
  and the pipeline still runs (all trials fall back to the hard-floor reward).
- **Running modules.** Entry points use absolute imports rooted at `alpha`. Either activate the
  editable install (above) or set `PYTHONPATH=src`. The `src/alpha/cli/*.sh` launchers set
  `PYTHONPATH` for you. Prefer `python -m alpha.experiments...` over path invocation.

## 2. Quick smoke test

Always validate a configuration cheaply before a full run:

```bash
python -m alpha.experiments.transform_pysr_primary \
    --data data/warmup/csi500_daily_2020-06-30_to_2026-06-30.parquet \
    --trials 3 --time 10
```

## 3. Primary mining

```bash
# Default-equivalent full run (CSI 500, warmup split)
python -m alpha.experiments.transform_pysr_primary \
    --data data/warmup/csi500_daily_2020-06-30_to_2026-06-30.parquet \
    --market zs500 --pool buildin --mode warmup --trials 50 --time 150 --seed 42

# HS 300
python -m alpha.experiments.transform_pysr_primary \
    --market hs300 --data data/warmup/hs300_daily_2020-06-30_to_2026-06-30.parquet

# Alpha158 engineered pool
python -m alpha.experiments.transform_pysr_primary \
    --data data/warmup/csi500_daily_2020-06-30_to_2026-06-30.parquet --pool alpha158

# Timing mode
python -m alpha.experiments.transform_pysr_primary \
    --data data/warmup/csi500_daily_2020-06-30_to_2026-06-30.parquet --run_mode timing
```

Launcher: `./src/alpha/cli/run.sh` (honors `DATA`, `POOL`, `HORIZON` env vars and passes
through all flags).

### Primary CLI arguments

| Flag | Default | Meaning |
|---|---|---|
| `--data` | repo `data/warmup/hs300_...parquet` | input parquet/csv panel |
| `--trials` | `50` | maximum search trials |
| `--time` | `150` | time budget in minutes |
| `--market` | `zs500` | `zs500` (CSI 500) or `hs300` |
| `--pool` | `alpha158` | `buildin` (semantic registry) or `alpha158` |
| `--mode` | `warmup` | `warmup` / `cold` / `ratio` / `month` split |
| `--run_mode` | `cross_sectional` | `cross_sectional` or `timing` |
| `--anchor` | `data_start` | `month`-mode anchor (`first_07_01` / `data_start`) |
| `--horizon` | `1` | forward horizon `T+N` |
| `--seed` | `42` | global seed (multi-seed robustness) |

> Note: the argparse default for `--pool` is `alpha158`, while `Config.FEATURE_POOL = 'buildin'`;
> pass `--pool` explicitly for reproducible setups.

## 4. Outputs and output location

Runs write a self-contained directory named
`factor_output_academic_v81/run_<YYYYmmdd_HHMMSS>_s<seed>/` containing:

- `registry_academic.json` — factor definitions + metrics + gate status;
- `fm_summary_report.txt` — production-ready factor table;
- `mining.log` — full trial trace;
- audit JSONL and backtest figures (where applicable).

`Config.OUTPUT_DIR` defaults to `factor_output_academic_v81` **relative to the current working
directory**. Historical artifacts are archived under `outputs/` (`academic_v81/`, `pysr/`,
`legacy/`). To unify new results, run from the repository root or set `Config.OUTPUT_DIR` to
`outputs/<tag>` in the entry script.

## 5. Ablations

2×2 main table and supplementary ablations (see [experiments.md §2–3](experiments.md#2-core-ablations-22-main-table)):

| # | Script | Removes |
|---|---|---|
| 7 | `transform_pysr_ablation7_no_mlqc_gates.py` | all MLQC gates + reward shaping |
| 8 | `transform_pysr_ablation8_no_gflownet.py` | GFlowNet (random selection) |
| 9 | `transform_pysr_ablation9_random_no_mlqc.py` | GFlowNet + MLQC |
| 5 | `transform_pysr_ablation5_no_constraints.py` | all priors (no-prior lower bound) |
| 6 | `transform_pysr_ablation6_no_mlqc.py` | reward shaping only |
| 10 | `transform_pysr_ablation10_rl_mlqcv.py` | continuous shaping → discrete MLQC reward |
| – | `transform_pysr_ablation_no_diversity.py` | diversity control |
| – | `transform_pysr_ablation_gp_no_gflownet.py` | GFlowNet + PySR (GP instead) |

```bash
python -m alpha.experiments.transform_pysr_ablation7_no_mlqc_gates --data data/warmup/csi500_daily_2020-06-30_to_2026-06-30.parquet
# batch launchers (the argument is the module suffix, not a bare number):
./src/alpha/cli/run_ablation.sh 8_no_gflownet          # run_ablation.sh/all ships 8 + 9
./src/alpha/cli/run_ablation_2.sh 7_no_mlqc_gates      # run_ablation_2.sh ships 7 + 9
./src/alpha/cli/run_ablation_no_diversity.sh
```

## 6. Baselines

```bash
# GP (gplearn)
python -m alpha.experiments.transform_gp_baseline \
    --data data/warmup/csi500_daily_2020-06-30_to_2026-06-30.parquet
python -m alpha.experiments.transform_gp_baseline_aligned \
    --data data/warmup/csi500_daily_2020-06-30_to_2026-06-30.parquet
# launcher: ./src/alpha/cli/run_baseline.sh gp | gp-aligned

# External RL baselines (require the [alphagen] extra)
python -m alpha.experiments.transform_alphagen_baseline \
    --data data/warmup/csi500_daily_2020-06-30_to_2026-06-30.parquet --steps 20000 --pool 20
python -m alpha.experiments.transform_alphaqcm_baseline \
    --data data/warmup/csi500_daily_2020-06-30_to_2026-06-30.parquet --steps 20000 --pool 20
python -m alpha.experiments.transform_alphasage_gfn_baseline \
    --data data/warmup/csi500_daily_2020-06-30_to_2026-06-30.parquet \
    --encoder gnn --episodes 10000 --pool-capacity 50
# smoke variants: --steps 300 --pool 3 --time 3   (AlphaSAGE: --encoder lstm --episodes 300 --pool-capacity 3)
# launchers: ./src/alpha/cli/run_alphagen_baseline.sh [full|smoke] | run_alphaqcm_baseline.sh [full|smoke]

# Black-box ML baselines
python -m alpha.experiments.transform_lgbm_baseline \
    --data data/warmup/csi500_daily_2020-06-30_to_2026-06-30.parquet --seeds 3
python -m alpha.experiments.transform_gru_baseline \
    --data data/warmup/csi500_daily_2020-06-30_to_2026-06-30.parquet
# launchers: ./src/alpha/cli/run_lgbm_baseline.sh | run_gru_baseline.sh
```

## 7. Walk-forward robustness

```bash
# Inspect window boundaries without mining
python -m alpha.experiments.transform_pysr_walkforward --list-windows

# Full run (default splices data/csi500_daily_2016-06-30_to_2021-06-30.parquet and
# data/csi500_daily_2021-06-30_to_2026-06-30.parquet; 36:12:12 months, ~6 windows)
python -m alpha.experiments.transform_pysr_walkforward --trials 60 --time 80

# Subsets / smoke
python -m alpha.experiments.transform_pysr_walkforward --windows 1,3
python -m alpha.experiments.transform_pysr_walkforward --market hs300 --max-windows 2

# Window-family batch scripts
./src/alpha/cli/run_walkforward.sh
./src/alpha/cli/run_warmup_window.sh     # every data/warmup/*.parquet, --mode warmup
./src/alpha/cli/run_cold_window.sh       # every data/qfq/*.parquet, --mode cold
```

## 8. Evaluation

All evaluation tools take a registry produced by any mining/baseline run.

```bash
REG=factor_output_academic_v81/run_<ts>_s42/registry_academic.json

# Single-factor FM + long–short / long-only backtest
python -m alpha.evaluation.single_factor_test --registry "$REG" --data data/warmup/csi500_daily_2020-06-30_to_2026-06-30.parquet
python -m alpha.evaluation.single_factor_test --registry "$REG" --long-only --compare-cost --cost-bps 30

# Stability
python -m alpha.evaluation.check_time_stability --registry "$REG" --windows 4
python -m alpha.evaluation.transform_pysr_stablity_check --registry "$REG" --win-months 4 --stride-months 2

# Comparison vs. traditional / other registries
python -m alpha.evaluation.compare_benchmark --registry "$REG" --top-n 10

# Diversity and complementarity
python -m alpha.evaluation.factor_correlation_heatmap --registry "$REG"
python -m alpha.evaluation.combine_factor_eval --registry "$REG" --topk 5 --n-boot 2000
python -m alpha.evaluation.validate_factor_combo

# Orthogonality vs. FF5 + liquidity
python -m alpha.evaluation.fetch_ff5_liq
python -m alpha.evaluation.build_factor_panel --registry "$REG" --data data/warmup/csi500_daily_2020-06-30_to_2026-06-30.parquet
python -m alpha.evaluation.factor_to_ls_returns
python -m alpha.evaluation.orthogonality_check
# pipeline launcher: ./src/alpha/evaluation/run_orthogonality_pipeline.sh

# Reporting / diagnostics
python -m alpha.analysis.regen_fm_report --registry "$REG"
python -m alpha.analysis.decode_gp_formulas --data data/warmup/csi500_daily_2020-06-30_to_2026-06-30.parquet

# Multi-tool launcher
./src/alpha/cli/run_eval.sh single_factor --registry "$REG"
./src/alpha/cli/run_eval.sh compare --registry "$REG"
./src/alpha/cli/run_long_only_eval.sh "$REG"
```

## 9. Data

`data/` holds parquet panels (gitignored). Three refinements are provided for different split
regimes:

```
data/warmup/   # 6-year panels used with --mode warmup (12-month look-back)
data/qfq/      # forward-adjusted 5-year panels used with --mode cold
data/hfq/      # backward-adjusted panels
```

Example files:
`data/warmup/csi500_daily_2020-06-30_to_2026-06-30.parquet`,
`data/warmup/hs300_daily_2020-06-30_to_2026-06-30.parquet`,
`data/qfq/csi500_daily_2021-06-30_to_2026-06-30.parquet`.

Fetch new data with `src/alpha/data/data_fetcher.py` (Baostock). **Always pass `--data`
explicitly** — some scripts retain stale defaults. Expected columns: `date`, `symbol`, `open`,
`high`, `low`, `close`, `volume`, `amount` (optional `turn`, `pctChg`).

## 10. Command-launcher reference

| Launcher | Wraps |
|---|---|
| `run.sh` | primary mining |
| `run_ablation.sh <suffix|all>` / `run_ablation_2.sh` | ablation batches (argument is the module suffix, e.g. `8_no_gflownet`) |
| `run_ablation_no_diversity.sh` | diversity ablation |
| `run_baseline.sh gp\|gp-aligned` | GP baselines |
| `run_gp_baseline.sh` | GP baseline (alternate) |
| `run_alphagen_baseline.sh [full\|smoke]` | AlphaGen (PPO) |
| `run_alphaqcm_baseline.sh [full\|smoke]` | AlphaQCM (IQN) |
| `run_lgbm_baseline.sh` / `run_gru_baseline.sh` | black-box ML baselines |
| `run_walkforward.sh` | walk-forward |
| `run_warmup_window.sh` / `run_cold_window.sh` | per-panel window batches |
| `run_eval.sh <tool>` / `run_long_only_eval.sh` | evaluation tools |

## 11. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ModuleNotFoundError: alpha...` | set `PYTHONPATH=src` or reinstall with `pip install -e .` |
| PySR/Julia errors | first run needs network to build `.venv/julia_env`; otherwise symbolic regression is skipped by design |
| Very slow / memory heavy | reduce `--trials`/`--time`, use `--market hs300`, or run with a smaller panel |
| No factors registered | relax nothing ad hoc — verify the split/warmup setting and inspect `mining.log` gate reasons |
| External RL baseline import errors | install the `[alphagen]` extra |
| Outputs not where expected | `Config.OUTPUT_DIR` is relative to CWD; run from the repo root or set it explicitly |
