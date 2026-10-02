# AGENTS.md

## What this repo is

High-frequency (daily 5-min) quantitative factor mining system for Chinese A-shares (CSI500 / HS300). Uses **GFlowNet + PySR symbolic regression** to discover alpha factors from OHLCV microstructure data, with Fama-MacBeth evaluation, single-factor backtesting and stability checks.

## Project structure

```
src/alpha/                  ← all real code lives here (absolute imports: `from alpha.x import ...`)
  config.py                 ← Config singleton: splits, gates, reward weights, seeds, market presets
  mining/                   ← mining engine: preprocessor, gflownet, pysr_engine, orchestrator,
  │                             fm_regression, neutralize, registry, safe_ops, screener,
  │                             stats_validator, walkforward, audit_logger
  features/                 ← atomic feature registries (UltimateDaily / LargeCap / Timing / Alpha158)
  data/                     ← data_loader (clean/winsorize), data_fetcher (baostock), index generation
  experiments/              ← runnable entry points:
  │   transform_pysr_primary.py         ← **primary pipeline** (run this)
  │   transform_pysr_ablation5-10_*.py  ← ablations (2×2 main: 7=no MLQC, 8=no GFlowNet, 9=random+no MLQC)
  │   transform_pysr_ablation_gp_no_gflownet.py, transform_pysr_ablation_no_diversity.py
  │   transform_gp_baseline(_aligned).py, transform_{alphagen,alphaqcm,alphasage_gfn}_baseline.py
  │   transform_{lgbm,gru}_baseline.py, transform_pysr_walkforward.py
  evaluation/               ← backtester, single_factor_test, check_time_stability,
  │                             transform_pysr_stablity_check, compare_benchmark, factor_compare_1,
  │                             combine_factor_eval, validate_factor_combo, factor_correlation_heatmap,
  │                             build_factor_panel, factor_to_ls_returns, orthogonality_check, fetch_ff5_liq
  analysis/                 ← plot_factor_correlation_heatmap, decode_gp_formulas,
  │                             diagnose_single_features, regen_fm_report, p5, decode/analysis helpers
  cli/                      ← thin bash launchers (run.sh, run_ablation*.sh, run_*_baseline.sh, run_eval.sh)
src/vendor/                 ← vendored external baselines (alphagen, alpha_gfn, fqf_iqn_qrdqn, gan)
data/                       ← parquet market data (gitignored): warmup/ qfq/ hfq/
outputs/                    ← UNIFIED experiment output root
  run_<timestamp>_*/        ← current runs
  academic_v81/             ← historical factor_output_academic_v81 runs (run_* / ablation*)
  pysr/                     ← old paper/PYSR/outputs runs
  legacy/                   ← archived outputs: factor_output/, walkforward, comparisons, logs
docs/                       ← architecture.md, methodology.md, experiments.md, run_guide.md
```

Note: `src/alpha/pre_timing_backup/`, per-file `* copy*.py`, and
`mining/orchestrator_before_leakfix.py` are archived snapshots, not part of the current pipeline.

## Key files and which to run

- **`src/alpha/experiments/transform_pysr_primary.py`** — The industrial-grade pipeline. Run this for production factor mining.
- **`src/alpha/experiments/transform_pysr_ablation*.py`** — Ablation studies. 2×2 main table (GFlowNet whole × MLQC whole): 7=no MLQC, 8=no GFlowNet, 9=random+no MLQC (plus primary = 4 cells). Supplementary: 5=no constraints (no-prior lower bound), gp=no gflownet.
- **Baselines** — `transform_{gp_baseline,_gp_baseline_aligned,alphagen,alphaqcm,alphasage_gfn,lgbm,gru}_baseline.py`.
- **Evaluation** — `evaluation/single_factor_test.py`, `check_time_stability.py`, `transform_pysr_stablity_check.py`, `compare_benchmark.py`, `combine_factor_eval.py`, `orthogonality_check.py`.
- **Docs** — `docs/architecture.md` (layout & data flow), `docs/methodology.md` (paper-style method), `docs/experiments.md` (experiment catalogue), `docs/run_guide.md` (commands).

## Environment

- Python 3.13, venv at `.venv/`
- `pyproject.toml` exists (editable install: `pip install -e ".[pysr,gp,dev]"`). Key deps:
  - `pandas`, `numpy`, `torch`, `scipy`, `scikit-learn`, `statsmodels`, `sympy`, `lightgbm`, `juliacall`, `baostock`
  - `pysr` (PySRRegressor — optional, `pip install pysr`)
  - `openai` (optional, for LLM constraint generator)
  - `akshare` (optional, for CSI1000 constituents in `data_fetcher.py` — Baostock has no CSI1000 interface)
- PySR first run initializes a Julia backend under `.venv/julia_env` (network required).

## Running

```bash
# Activate venv
source .venv/bin/activate

# Preferred: run as modules (absolute imports)
PYTHONPATH=src python -m alpha.experiments.transform_pysr_primary --data data/warmup/csi500_daily_2020-06-30_to_2026-06-30.parquet

# Tune trials / time budget / market
python -m alpha.experiments.transform_pysr_primary --data data/warmup/csi500_daily_2020-06-30_to_2026-06-30.parquet --trials 60 --time 80 --market zs500

# Ablations / baselines / evaluation
python -m alpha.experiments.transform_pysr_ablation9_random_no_mlqc --data data/...
python -m alpha.experiments.transform_gp_baseline --data data/...
python -m alpha.evaluation.single_factor_test --registry outputs/academic_v81/run_xxx/registry_academic.json
```

- All mining entry scripts take `--data` (parquet/csv), `--trials`, `--time`; most accept `--market`.
- `src/alpha/cli/*.sh` are thin launchers that set `PYTHONPATH=src` for you.
- Output lands relative to the CWD via `Config.OUTPUT_DIR`. Entry points override it with a timestamped
  `factor_output_academic_v81/run_<ts>_s<seed>/` directory. To unify the result root, run from the repo
  root or set `Config.OUTPUT_DIR` to `outputs/<tag>` in the entry script.

## Data format

Expected parquet/csv columns:
- `date`, `symbol`, `open`, `high`, `low`, `close`, `volume`, `amount` (optional `turn`, `pctChg`)
- Sorted by `symbol` then `date`
- Sample panels in `data/{warmup,qfq,hfq}/`: `csi500_daily_*.parquet` and `hs300_daily_*.parquet` (daily bars)

## Gotchas

- **No tests exist.** Verify by running a short run (`--trials 3 --time 10`) first.
- **PySR is optional.** The pipeline silently skips symbolic regression if `pysr` is not installed.
- **LLM constraints are optional.** Without a valid `LLM_API_KEY`, the pipeline falls back to safe defaults.
- **Multiple FeatureRegistry classes exist** across files (`features/feature_registry.py` and per-file copies). When modifying features, check which file you're in.
- **Hardcoded paths.** Some scripts still carry stale absolute/relative data defaults (e.g. `diagnose_single_features.py` `DATA_PATH`, argparse defaults pointing at the old `factor_mining_pipeline/...`). Always pass `--data data/...` explicitly.
- **Memory-sensitive.** The pipelines build large feature pools. Use small `--trials`/`--time` for smoke tests.
- **No CI, no lint config, no type checking** is configured.
