# Architecture

This document describes the repository layout, the responsibility of each module in the mining
engine, the end-to-end data flow, and the artifact contract that downstream analysis and paper
tables rely on. It reflects the current `src/alpha/` package; legacy code paths are listed in
[Legacy and archived code](#legacy-and-archived-code).

## Repository layout

```
GFlowAlpha/
├── src/
│   ├── alpha/                      # main package (absolute imports: `from alpha.x import ...`)
│   │   ├── config.py               # Config singleton: splits, gates, reward weights, seeds, market presets
│   │   ├── data/
│   │   │   ├── data_loader.py      # CSI500Loader: read, clean, winsorize, derive returns
│   │   │   ├── data_fetcher.py     # Baostock downloader (CSI500 / HS300 / CSI1000)
│   │   │   ├── generate_index_data.py
│   │   │   └── load_data.py
│   │   ├── features/
│   │   │   ├── feature_registry.py # UltimateDaily / LargeCap / Timing atomic feature builders
│   │   │   └── alpha158_registry.py# Qlib Alpha158-style engineered feature builder
│   │   ├── mining/                 # the mining engine
│   │   │   ├── preprocessor.py     # feature-pool assembly, quality gate, clustering, splits, standardization
│   │   │   ├── gflownet.py         # feature metadata, transformer policy, trajectory sampler, TB trainer
│   │   │   ├── pysr_engine.py      # PySR symbolic-regression wrapper (optional)
│   │   │   ├── orchestrator.py     # MiningOrchestrator: per-trial search loop + reward + registration
│   │   │   ├── fm_regression.py    # Fama–MacBeth (NW) + timing evaluator
│   │   │   ├── neutralize.py       # FWL residual neutralization
│   │   │   ├── registry.py         # FactorRegistry + production-ready / formatting helpers
│   │   │   ├── safe_ops.py         # SafeOps / TimeSeriesOps: numerically guarded operators
│   │   │   ├── screener.py         # MAD winsorization, IC screening helpers
│   │   │   ├── stats_validator.py  # White Reality Check and related validation
│   │   │   ├── walkforward.py      # WalkForwardSplitter (27:12:3, purge + embargo)
│   │   │   └── audit_logger.py     # JSONL audit trail of the search
│   │   ├── experiments/            # runnable entry points (primary, ablations, baselines, walk-forward)
│   │   ├── evaluation/             # backtester, single-factor test, stability, correlation, orthogonality
│   │   ├── analysis/               # reporting and diagnostics utilities
│   │   └── cli/                    # thin bash launchers wrapping the modules above
│   └── vendor/                     # vendored external baselines (see src/vendor/README.md)
├── data/                           # market panels, gitignored (warmup/ qfq/ hfq/)
├── outputs/                        # unified artifact root; historical runs + run_* directories
├── docs/                           # architecture.md · methodology.md · experiments.md · run_guide.md
└── pyproject.toml                  # editable install, optional extras, console scripts
```

The package uses **absolute imports rooted at `alpha`**. Run entry points as modules with
`PYTHONPATH=src` (the `cli/*.sh` launchers do this for you), e.g.
`python -m alpha.experiments.transform_pysr_primary`.

## Module responsibilities

### `config.py` — the single source of experimental truth

All knobs that affect comparability live in one place so that primary runs, ablations and
baselines can be held to identical protocol:

- **Splits** — `SPLIT_MODE` (`ratio` / `month` / `walkforward`), `TRAIN_RATIO`/`VAL_RATIO`, the
  monthly split constants (`TRAIN_MONTHS`, `VAL_MONTHS`, `TEST_MONTHS`, `WARMUP_MONTHS`), and the
  walk-forward constants (`WF_TRAIN_MONTHS`, `WF_VAL_MONTHS`, `WF_TEST_MONTHS`, `WF_PURGE_DAYS`,
  `WF_EMBARGO_DAYS`).
- **Feature pool** — `MIN_FEATURES`/`MAX_FEATURES`, `TARGET_FACTOR_POOL_SIZE`,
  `CLUSTER_FEATURE_POOL`, `DEDUP_NEAR_DUPLICATES`, `NaN_RATIO_CAP`.
- **Gates** — `TSTAT_THRESHOLD`, `TSTAT_STRONG`, `ICIR_MIN`, `IC_THRESHOLD`,
  `REGISTER_IC_MIN`, `REGISTER_ICIR_MIN`, `TEST_TSTAT_MIN`, `TEST_TSTAT_STRONG`,
  `USE_MONOTONICITY_GATE`, `USE_RESIDUAL_NEUTRALIZATION`, `EXCLUDE_WEAK_FROM_PRODUCTION`.
- **Reward** — `REWARD_HARD`, `REWARD_SOFT_FLOOR`, `REWARD_TOP`, `SIGN_MISMATCH_PENALTY`,
  and the per-market `REWARD_CONFIG` (slopes/thresholds/weights for `t_stat`, `icir`, `ic`).
- **PySR / GFlowNet** — `PY_SR_*` and `GFN_*` hyperparameters.
- **Seeds** — `SEED`, `PYSR_SEED`, `GP_SEED`, `LGBM_SEED`.
- **Market presets** — `MARKET_CONFIG` (`fm_min_stocks`, `stability_entrance_t`) and
  `REWARD_CONFIG` keyed by `zs500` / `hs300`.
- **Modes** — `MODE` (`cross_sectional` / `timing`), `FEATURE_POOL` (`buildin` / `alpha158`).
- **Output** — `OUTPUT_DIR` and `REGISTRY_FILE` (note: entry points typically *override*
  `OUTPUT_DIR` with a timestamped subdirectory).

### `data/data_loader.py` — `CSI500Loader`

Reads parquet/csv, normalizes the schema (`code` → `symbol`), and applies a documented cleaning
pipeline:

1. drop ST / \*ST / PT names;
2. drop suspended bars (zero volume **and** flat OHLC);
3. drop price anomalies (`high < max(open, close)` or `low > min(open, close)`);
4. drop symbols with fewer than `min_bars` (default 252) observations;
5. per-day winsorization of `pctChg` at `[0.01, 0.99]`;
6. derive `ret = pctChg / 100`.

### `features/` — atomic feature registries

`feature_registry.py` builds the raw atomic features consumed by the mining engine:

| Registry | Used when | Role |
|---|---|---|
| `UltimateDailyFeatureRegistry` | `MARKET=zs500`, `FEATURE_POOL=buildin` | standard CSI 500 semantic pool |
| `LargeCapFeatureRegistry` | `MARKET=hs300`, `FEATURE_POOL=buildin` | large-cap / HS 300 pool |
| `TimingFeatureRegistry` | `MODE=timing` | time-series timing features |
| `Alpha158FeatureRegistry` | `FEATURE_POOL=alpha158` | Qlib Alpha158-style engineered pool |

### `mining/preprocessor.py` — `DataPreprocessor`

`prepare_full_pool` is the bottleneck through which all price-based experiments pass:

1. build atomic features for the configured market/pool;
2. attach the forward-return label `close.pct_change(HORIZON).shift(-HORIZON)`;
3. **quality gate** — drop columns whose NaN ratio exceeds `NaN_RATIO_CAP` or whose variance
   is below `1e-6`;
4. **near-duplicate dedup** — union-find over Spearman correlation `> NEAR_DUPLICATE_CORR`
   (`0.95`), keeping the higher-|IC| member; negative-correlation mirror pairs are retained;
5. **clustering orthogonalization** — complete-linkage hierarchical clustering on
   `1 − |Spearman|`, one representative (highest |IC|) per cluster, target pool size
   `TARGET_FACTOR_POOL_SIZE` (observed: 85 features);
6. fill remaining NaNs with 0, then either build the walk-forward splitter or apply the
   monthly/ratio split and fit standardization **on the training split only**.

`set_window(idx)` re-points the masks for walk-forward windows; `get_subset(features)` returns
per-feature arrays for a subset (used once per trial by the orchestrator).

### `mining/gflownet.py` — generative feature selection

- `FeatureMetadataExtractor` builds a per-feature attribute matrix (semantic/statistical tags)
  that conditions the policy.
- `GFlowNetPolicy_Transformer` — a transformer over the feature-selection action sequence with
  an availability mask.
- `TrajectorySampler_Transformer` — samples feature subsets of size `[MIN_FEATURES, MAX_FEATURES]`,
  returning trajectories with forward/backward log-probabilities; supports per-feature logit
  penalties (`set_feature_penalties`) used by soft/hard diversity masking.
- `GFlowNetTrainerInternal.train_step(log_pf, log_pb, reward)` — trajectory-balance update.

### `mining/pysr_engine.py` — `PySRMiningEngine`

Wraps `PySRRegressor` with `HAS_PYSR` capability detection. If PySR is unavailable the engine
returns no result and the orchestrator treats the trial as a hard-floor reward, so the pipeline
degrades gracefully without symbolic regression. Supports `top_k` candidate selection for the
stability-selection step (see below).

### `mining/fm_regression.py` — evaluation core

- `FamaMacBethRegressor.run` performs daily cross-sectional OLS of returns on the factor,
  averages the slope series, and computes the t-statistic with Newey–West HAC standard errors
  (`NW_LAGS`). It reports coefficient, NW-SE, t, p-value, average R²/adj-R², sign ratios,
  quantile returns, long–short spread, Rank IC mean and Rank ICIR. Regressions require at least
  `fm_min_stocks` names per day (`zs500`: 30, `hs300`: 20).
- `TimingEvaluator` / `TimingResult` provide the time-series timing counterpart
  (direction accuracy, timing Sharpe, t-stat).

### `mining/neutralize.py` — FWL neutralization

`fwl_neutralize` performs, independently for each trading day, the Frisch–Waugh–Lovell
double-residualization of both the factor and the forward return against `ln(amount)` (and any
extra controls), returning residual factor/return pairs for FM evaluation. This removes the
size/liquidity exposure that would otherwise inflate apparent significance.

### `mining/registry.py` — `FactorRegistry`

Stores factors and their metrics, exposes `get_production_ready()` and the shared
`is_production_ready()` definition (non-WEAK, |t| ≥ `PRODUCT_FMT_THRESHOLD`,
|ICIR| ≥ `PRODUCT_ICIR_THRESHOLD`, |IC| ≥ `PRODUCT_IC_THRESHOLD`, ≥ 30 cross-sectional periods).
All baselines and comparison tables use this single yardstick.

### `mining/orchestrator.py` — `MiningOrchestrator`

The search loop (see [methodology](methodology.md#4-mining-orchestrator)). Per trial it:
updates dynamic masks; samples a feature subset; runs PySR with top-K stability selection;
evaluates on validation with FWL + FM; applies MLQC gates L1–L4; computes the shaped reward and
trains the GFlowNet; then runs the **test-set registration gate** and writes to the registry.
`_registration_verdict` is shared by the primary run and all ablations/baselines.

### `mining/safe_ops.py`, `screener.py`, `stats_validator.py`, `walkforward.py`, `audit_logger.py`

- `SafeOps` / `TimeSeriesOps` — protected division, safe sigmoid, and guarded time-series
  operators used inside formula evaluation.
- `FactorScreener` — MAD winsorization and IC screening utilities.
- `whites_reality_check` — White's (2000) reality check over per-day Rank-IC series with
  stationary bootstrap, guarding against data-snooping across many trials.
- `WalkForwardSplitter` — 27:12:3-month rolling windows with `purge(1 day) + embargo(2 days)`.
- `AuditLogger` — JSONL record of every trial for reviewer-facing transparency.

## End-to-end data flow

```
parquet/csv
   │  CSI500Loader.load()                         (clean, winsorize, derive ret)
   ▼
DataPreprocessor.prepare_full_pool()              (features → labels → gate → dedup → cluster → split)
   │
   ▼
MiningOrchestrator.run(max_trials, time_limit_min)
   │   ┌─────────────────────────────── per trial ───────────────────────────────┐
   │   │ GFlowNet sample τ = {features}                                          │
   │   │ PySR fit f on train → top-K stability select on val                     │
   │   │ FWL neutralize → Fama–MacBeth (NW) on val                               │
   │   │ MLQC gates L1/L2/L3/L4                                                  │
   │   │ reward R(τ) → GFlowNet train_step                                       │
   │   │ test-set registration gate (sign + |t| + |IC| + |ICIR|) → STRONG/WEAK   │
   │   └─────────────────────────────────────────────────────────────────────────┘
   ▼
FactorRegistry → registry_academic.json / fm_summary_report.txt / mining.log / audit.jsonl
   ▼
AcademicBacktester.run()                          (T+1 open, costs, turnover, long-only or L/S)
```

## Artifact contract

Every run writes a self-contained directory (primary entry points create
`factor_output_academic_v81/run_<timestamp>_s<seed>/`; the pipeline otherwise respects
`Config.OUTPUT_DIR`):

| Artifact | Produced by | Contents |
|---|---|---|
| `registry_academic.json` | `FactorRegistry` | per-factor id, formula, feature list, complexity, FM metrics, gate status |
| `fm_summary_report.txt` | `MiningOrchestrator._generate_summary_report` | human-readable production-ready factor table |
| `mining.log` | entry-point logging config | full trial-by-trial trace |
| audit JSONL | `AuditLogger` | structured search/dedup/gate events |
| comparison tables (`comparison_*.csv` / `.md`) | baselines | baseline vs. registry comparison |
| evaluation outputs | evaluation suite | stability, correlation, complementarity, orthogonality |
| backtest figures | `AcademicBacktester` / `single_factor_test.py` | NAV curves, quantile bars |

## Legacy and archived code

Kept for provenance but **not** part of the current pipeline:

- `src/alpha/pre_timing_backup/` — snapshot of the engine before the timing/cross-sectional
  split; references sibling modules of the old layout.
- `src/alpha/features/feature_registry copy.py`, `feature_registry copy 2.py` — superseded
  copies (AGENTS.md warns to check which registry file you are editing).
- `src/alpha/mining/orchestrator_before_leakfix.py`, `orchestrator copy.py` — pre-fix snapshots.
- `factor_output_academic_v81/` at the repo root and `outputs/academic_v81/` — historical run
  archives; `outputs/<timestamp>_*/` are older unified-run directories.
- `backup/`, `backups/`, `bak/`, and the root `*.tar` files — offline snapshots, excluded from
  the documented workflow.

## Extension points

- **New market / universe** — add a `MARKET_CONFIG` and `REWARD_CONFIG` entry in `config.py`,
  and (if needed) a feature builder in `features/`.
- **New feature pool** — implement a registry class in `features/` exposing a
  `build_*_factors(df)` classmethod and branch on `FEATURE_POOL` in `preprocessor.prepare_full_pool`.
- **New ablation** — subclass `MiningOrchestrator` (or copy an existing `transform_pysr_ablation*.py`)
  and override only the component under test, keeping the shared `_registration_verdict` and
  `AcademicBacktester` so metrics remain comparable.
- **New evaluation** — re-use `FactorRegistry`/`registry_academic.json` as the input contract so
  downstream tools remain decoupled from the mining engine.
