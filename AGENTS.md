# AGENTS.md

## What this repo is

High-frequency (daily 5-min) quantitative factor mining system for Chinese A-shares (CSI500 / HS300). Uses **GFlowNet + PySR symbolic regression** to discover alpha factors from OHLCV microstructure data, with Fama-MacBeth evaluation, single-factor backtesting and stability checks.

## Project structure

```
src/alpha/                  ← all real code lives here
  research/                 ← industrial GFlowNet + PySR pipeline (research core)
    common/                 ← mining engine: config, preprocessor, gflownet,
    │                         orchestrator, pysr_engine, fm_regression,
    │                         neutralize, registry, safe_ops, screener,
    │                         stats_validator, walkforward, audit_logger
    transform_pysr_primary.py         ← **primary/industrial pipeline** (run this)
    transform_pysr_ablation5,7-9.py   ← ablation experiments (2×2 main: 7=no MLQC, 8=no GFlowNet, 9=random+no MLQC)
    transform_pysr_ablation_gp_*.py   ← GP baseline (no gflownet)
    transform_pysr_stablity_check.py  ← rolling-window stability check
    transform_gp_baseline.py, gp_baseline.py  ← GP baseline
    backtester.py, single_factor_test.py       ← evaluation
    check_time_stability.py, compare_benchmark.py, factor_compare_1.py
    validate_factor_combo.py          ← multi-factor complementarity
    plot_factor_correlation_heatmap.py, decode_gp_formulas.py,
    diagnose_single_features.py, p5.py, load_data.py  ← analysis
  agents/                   ← legacy agent prototypes (PySR-only variants)
    factor_llm_miner.py, gflownet_study1-6.py, p1..p7.py
  data/data_fetcher.py      ← baostock data fetching
data/                       ← parquet/csv market data (gitignored)
outputs/                    ← UNIFIED experiment output root
  run_<timestamp>_*/        ← current runs (PySR smoke runs land here via OUTPUT_DIR)
  academic_v81/             ← historical factor_output_academic_v81 runs (run_* / ablation*)
  pysr/                     ← old paper/PYSR/outputs runs
  legacy/                   ← archived outputs: factor_output/, walkforward, comparisons, logs
docs/                       ← architecture.md, run_guide.md
```

`main.py` at the root is a PyCharm scaffold — ignore it.

## Key files and which to run

- **`src/alpha/experiments/transform_pysr_primary.py`** — The industrial-grade pipeline. Run this for production factor mining.
- **`src/alpha/experiments/transform_pysr_ablation*.py`** — Ablation studies. 2×2 main table (GFlowNet whole × MLQC whole): 7=no MLQC, 8=no GFlowNet, 9=random+no MLQC (plus primary = 4 cells). Supplementary: 5=no constraints (no-prior lower bound), gp=no gflownet.
- **`src/alpha/agents/factor_llm_miner.py`** — Alternate PySR pipeline with per-run timestamped output directories.

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

# Run industrial pipeline (primary)
python src/alpha/research/transform_pysr_primary.py --data data/csi500_daily_2020-07-20_to_2026-07-19.parquet

# Tune trials / time budget / market
python src/alpha/research/transform_pysr_primary.py --data data/csi500_daily_2021-06-30_to_2026-06-30.parquet --trials 60 --time 80 --market zs500

# Ablations / baselines / evaluation
python src/alpha/experiments/transform_pysr_ablation9_random_no_mlqc.py --data data/...
python src/alpha/experiments/transform_gp_baseline.py --data data/...
python src/alpha/experiments/single_factor_test.py --registry outputs/academic_v81/run_xxx/registry_academic.json
```

- All `transform_*` entry scripts take `--data` (parquet/csv), `--trials`, `--time`; most accept `--market`.
- Output lands relative to the CWD via `Config.OUTPUT_DIR`. Default dir name is `factor_output_academic_v81` (hardcoded in `common/config.py`). For a unified result root, run from the repo root and it will create `factor_output_academic_v81/` here — or set `Config.OUTPUT_DIR` to `outputs/<tag>` in the script.

## Data format

Expected parquet/csv columns:
- `datetime`, `symbol`, `close`, `open`, `high`, `low`, `volume`, `amount`, `industry_id`
- Sorted by `symbol` then `datetime`
- Sample data in `data/`: `csi500_daily_*.parquet` and `hs300_daily_*.parquet` (daily bars)

## Gotchas

- **No tests exist.** Verify by running a short run (`--trials 3 --time 10`) first.
- **PySR is optional.** The pipeline silently skips symbolic regression if `pysr` is not installed.
- **LLM constraints are optional.** Without a valid `LLM_API_KEY`, the pipeline falls back to safe defaults.
- **Multiple FeatureRegistry classes exist** across files (`common/feature_registry.py` and per-script copies). When modifying features, check which file you're in.
- **Hardcoded paths.** Some scripts still carry stale absolute/relative data defaults (e.g. `diagnose_single_features.py` `DATA_PATH`, argparse defaults pointing at the old `factor_mining_pipeline/...`). Always pass `--data data/...` explicitly.
- **Memory-sensitive.** The pipelines build large feature pools. Use small `--trials`/`--time` for smoke tests.
- **No CI, no lint config, no type checking** is configured.
