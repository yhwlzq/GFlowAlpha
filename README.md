# GFlowAlpha

**GFlowNet-guided symbolic regression for cross-sectional alpha discovery in Chinese A-shares.**

GFlowAlpha is a research codebase for automatically discovering interpretable, economically
meaningful return-predictive factors from daily OHLCV microstructure data on CSI 500 and
HS 300 universes. A Generative Flow Network (GFlowNet) proposes candidate subsets of atomic
features; symbolic regression (PySR) turns each candidate subset into a closed-form formula;
a Multi-Layer Quality Control (MLQC) stack then neutralizes, evaluates and gates each formula
before it can enter the factor registry. Every discovered factor is evaluated with a single,
auditable protocol — FWL residual neutralization against `ln(amount)`, Fama–MacBeth
cross-sectional regressions with Newey–West standard errors, and T+1-open monthly portfolio
backtests with transaction costs.

The repository ships the full experimental apparatus: a primary mining pipeline, a GFlowNet ×
MLQC 2×2 ablation matrix, internal and external baselines, walk-forward robustness runs, and an
evaluation suite covering stability, correlation, complementarity and orthogonality diagnostics.
The design goal is **reproducibility at journal standard**: unified splits, unified gates,
unified metrics, explicit seeds, and machine-readable outputs for every experiment.

## Research question

> Can a diversity-seeking generative policy (GFlowNet) learn *which* atomic features to combine
> so that symbolic regression yields a larger, more diverse, and more stable pool of
> out-of-sample-significant alpha factors than random selection, genetic programming, or
> reinforcement-learning alpha generators — and does explicit quality control (MLQC) explain
> that gain?

## Method at a glance

```
daily OHLCV parquet
    │
    ▼
CSI500Loader                       clean: drop ST / suspended / price anomalies, min 252 bars, winsorize
    │
    ▼
DataPreprocessor.prepare_full_pool feature construction → NaN/variance gate → near-duplicate dedup
    │                              → hierarchical-clustering orthogonalization → train/val/test split
    ▼
MiningOrchestrator.run  (per trial)
    ├─ GFlowNet samples a feature subset τ (min 3, max 5 features)
    ├─ PySR fits closed-form f on train; top-K stability selection on val
    ├─ FWL neutralize f against ln(amount)   →  Fama–MacBeth (Newey–West) on val
    ├─ MLQC gates: L1 single-feature-linearity · L2 |t| · L3 monotonicity · L4 structural dedup
    ├─ shaped reward  R(τ) = floor + (top−floor)·σ(t)ᵂᵗ·σ(ICIR)ᵂⁱʳ·σ(IC)ᵂⁱᶜ  (× diversity / sign / similarity)
    └─ out-of-sample registration gate on test: sign-consistency + |t| + |IC| + |ICIR| → STRONG / WEAK
    │
    ▼
FactorRegistry  →  registry_academic.json  +  fm_summary_report.txt  +  mining.log
    │
    ▼
AcademicBacktester (T+1 open, costs, turnover)  ·  stability / correlation / orthogonality suite
```

Full method, notation and equations: [docs/methodology.md](docs/methodology.md).
Component map and data flow: [docs/architecture.md](docs/architecture.md).
Per-experiment usage and paper mapping: [docs/experiments.md](docs/experiments.md).
Command reference: [docs/run_guide.md](docs/run_guide.md).

## Key features

- **Generative feature selection.** A transformer-based GFlowNet policy samples feature subsets
  with a trajectory-balance objective, trained online on the realized quality reward of every
  trial — not a fixed heuristic.
- **Interpretable formulas.** PySR yields closed-form expressions; structural fingerprints,
  canonical signatures and effective-feature counts make dedup and complexity control explicit.
- **Multi-Layer Quality Control (MLQC).** Four independent gates (L1 linearity, L2 t-stat,
  L3 monotonicity, L4 structural dedup) plus sign-consistency and test-set registration gates.
- **Statistically honest evaluation.** FWL neutralization, Fama–MacBeth with Newey–West,
  per-market minimum cross-section sizes, White Reality Check, sub-period robustness, and a
  distinct out-of-sample registration gate.
- **Diversity control.** Reward multipliers, soft/hard feature masking and Jaccard-style
  similarity penalties prevent feature collapse.
- **Reproducible by construction.** Global seed control (`SEED`, `PYSR_SEED`, `GP_SEED`,
  `LGBM_SEED`), deterministic splits with purge/embargo, and JSONL audit logs.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[pysr,gp,dev]"          # add ,alphagen for RL baselines, ,llm for LLM constraints

# Smoke test (few trials, short budget)
python src/alpha/experiments/transform_pysr_primary.py \
    --data data/warmup/csi500_daily_2020-06-30_to_2026-06-30.parquet --trials 3 --time 10

# Full mining run
python src/alpha/experiments/transform_pysr_primary.py \
    --data data/warmup/csi500_daily_2020-06-30_to_2026-06-30.parquet --trials 50 --time 150
```

Results are written under `factor_output_academic_v81/run_<timestamp>_s<seed>/`
(`registry_academic.json`, `fm_summary_report.txt`, `mining.log`, backtest figures).

## Repository map

```
src/alpha/
  config.py        global Config: splits, gates, reward weights, seeds, market presets
  data/            baostock fetch, cleaning/loading, index generation
  features/        atomic feature registries (UltimateDaily / LargeCap / Timing / Alpha158)
  mining/          mining engine: preprocessor, gflownet, pysr_engine, fm_regression,
                   neutralize, registry, safe_ops, screener, stats_validator,
                   walkforward, audit_logger, orchestrator
  experiments/     runnable entry points: primary, ablations, baselines, walk-forward
  evaluation/      backtester, single-factor test, stability, correlation, orthogonality,
                   complementarity, factor-panel export
  analysis/        reporting / diagnostics utilities (report regen, formula decode, heatmaps)
  cli/             thin bash launchers for every experiment
src/vendor/        vendored external baselines (AlphaGen, AlphaSAGE GFlowNet, AlphaQCM/IQN, GAN)
data/              market data, gitignored (warmup/ qfq/ hfq/ variants)
outputs/           unified artifact root (historical archives + run_* dirs)
docs/              architecture.md · methodology.md · experiments.md · run_guide.md
```

## Documentation

| Document | Contents |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Package layout, module responsibilities, end-to-end data flow, artifact contract |
| [docs/methodology.md](docs/methodology.md) | Paper-style method: notation, generator, reward, MLQC, evaluation, reproducibility |
| [docs/experiments.md](docs/experiments.md) | Every experiment: question, design, protocol, command, outputs, paper mapping |
| [docs/run_guide.md](docs/run_guide.md) | Install, environment, command reference, data inventory, output layout |

## Data

Expected columns: `date`, `symbol`, `open`, `high`, `low`, `close`, `volume`, `amount`
(optional `turn`, `pctChg`). Files are parquet/csv, sorted by `symbol` then `date`.
Sample panels ship under `data/{warmup,qfq,hfq}/`; new data can be fetched with
`src/alpha/data/data_fetcher.py` (Baostock). Always pass `--data data/...` explicitly.

## License and attribution

External baselines under `src/vendor/` are vendored from their upstream projects under their
original licenses (see `src/vendor/README.md` and bundled license files); the evaluation
pipeline around them is original to this repository.
