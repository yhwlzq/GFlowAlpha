# Experiment catalogue

This is the reference catalogue for every runnable experiment in GFlowAlpha. Each entry states
the **research question**, the **design**, the **protocol**, the exact **command**, the
**outputs**, the **paper mapping** (which table/figure/claim it supports), and the **expected
result**. Commands assume the repository root as the working directory and an activated
`.venv` (see [run_guide.md](run_guide.md)).

Notation: `DATA` denotes a panel path, e.g.
`data/warmup/csi500_daily_2020-06-30_to_2026-06-30.parquet`. All entry points accept `--data`;
most accept `--market zs500|hs300`. Use `--trials 3 --time 10` for smoke tests.

The 2×2 design that anchors the main results is the cross-product of **GFlowNet guidance**
(present/absent) and **MLQC quality control** (present/absent):

| Cell | GFlowNet | MLQC | Script |
|---|---|---|---|
| Full (primary) | ✓ | ✓ | `transform_pysr_primary.py` |
| w/o MLQC | ✓ | ✗ | `transform_pysr_ablation7_no_mlqc_gates.py` |
| w/o GFlowNet | ✗ | ✓ | `transform_pysr_ablation8_no_gflownet.py` |
| w/o both (random + no MLQC) | ✗ | ✗ | `transform_pysr_ablation9_random_no_mlqc.py` |

---

## 1. Primary pipeline

### 1.1 `transform_pysr_primary.py` — GFlowNet + PySR + MLQC (main method)

- **Question.** Does the full GFlowNet-guided symbolic search under MLQC produce a pool of
  out-of-sample-significant, economically plausible alpha factors?
- **Design.** The complete method of [methodology §4](methodology.md#4-mining-orchestrator):
  GFlowNet feature-subset sampling, PySR refinement with top-K stability selection, FWL
  neutralization, Fama–MacBeth on validation, MLQC gates L1–L4, shaped reward, test-set
  registration gate.
- **Protocol.** `--market zs500`, `--pool buildin` (or `alpha158`), `--mode warmup`
  (36:12:12 months after a 12-month look-back), `--horizon 1`, `--seed 42`.
- **Command.**
  ```bash
  python -m alpha.experiments.transform_pysr_primary \
      --data data/warmup/csi500_daily_2020-06-30_to_2026-06-30.parquet \
      --market zs500 --pool buildin --mode warmup --trials 50 --time 150 --seed 42
  # launcher equivalent: ./src/alpha/cli/run.sh
  ```
- **Outputs.** `factor_output_academic_v81/run_<ts>_s42/` with `registry_academic.json`,
  `fm_summary_report.txt`, `mining.log`, backtest figures.
- **Paper mapping.** Method section; Table 1 (discovered-factor characteristics — count, |t|,
  Rank IC, ICIR, complexity); representative formulas; the "Full" column of the 2×2 table.
- **Expected result.** The largest and strongest registry among the 2×2 cells: highest count of
  production-ready factors and the best median |t|/|ICIR|, with the GFlowNet contributing
  discovery and MLQC contributing precision.

---

## 2. Core ablations (2×2 main table)

### 2.1 `transform_pysr_ablation7_no_mlqc_gates.py` — w/o MLQC

- **Question.** How much of the primary result is attributable to MLQC?
- **Design.** Identical to primary except that **all** MLQC components are removed: no reward
  shaping (raw IC reward), no t-stat/monotonicity/single-feature-linearity gates, and no
  structural dedup. Every PySR formula that can be evaluated is registered.
- **Protocol.** As primary; held identical seed/splits/market.
- **Command.**
  ```bash
  python -m alpha.experiments.transform_pysr_ablation7_no_mlqc_gates \
      --data data/warmup/csi500_daily_2020-06-30_to_2026-06-30.parquet --trials 50 --time 150
  # launcher: ./src/alpha/cli/run_ablation_2.sh 7_no_mlqc_gates
  ```
- **Outputs.** Registry containing the **full, ungated** factor list (note: `get_production_ready()`
  still post-filters using the shared thresholds; analysis should read the raw registry list).
- **Paper mapping.** Table 2 (2×2 ablation), "w/o MLQC" cell; supports the claim that quality
  control converts raw symbolic output into registrable factors.
- **Expected result.** Many more candidates but a much lower share passing the registration /
  production criteria; higher duplicate/trivial-formula rate.

### 2.2 `transform_pysr_ablation8_no_gflownet.py` — w/o GFlowNet

- **Question.** Does learned generative feature selection beat random feature selection?
- **Design.** Identical to primary except feature selection is **uniformly random** instead of
  GFlowNet-sampled; no policy training. PySR, FM, FWL, MLQC, dedup and out-of-sample evaluation
  are unchanged.
- **Protocol.** As primary; consider multiple seeds to average out random-selection variance.
- **Command.**
  ```bash
  python -m alpha.experiments.transform_pysr_ablation8_no_gflownet \
      --data data/warmup/csi500_daily_2020-06-30_to_2026-06-30.parquet --trials 50 --time 150
  # launcher: ./src/alpha/cli/run_ablation.sh 8_no_gflownet
  ```
- **Outputs.** Timestamped run directory with registry + report.
- **Paper mapping.** Table 2, "w/o GFlowNet" cell; primary evidence for the value of the
  generative selector.
- **Expected result.** Fewer registrable factors and lower diversity than the full method at a
  matched trial budget; the gap is the GFlowNet's contribution.

### 2.3 `transform_pysr_ablation9_random_no_mlqc.py` — w/o GFlowNet + w/o MLQC

- **Question.** What is the lower bound when *both* guidance and quality control are removed?
- **Design.** Combines 2.1 and 2.2: random feature selection **and** no MLQC gates/reward shaping.
- **Protocol.** As primary.
- **Command.**
  ```bash
  python -m alpha.experiments.transform_pysr_ablation9_random_no_mlqc \
      --data data/warmup/csi500_daily_2020-06-30_to_2026-06-30.parquet --trials 50 --time 150
  # run directly; not currently included in the batch launcher arrays
  ```
- **Outputs.** Timestamped run directory with registry + report.
- **Paper mapping.** Table 2, "w/o both" cell — completes the 2×2 and supports an interaction
  analysis (is the combined loss larger than the sum of the individual losses?).
- **Expected result.** The weakest pool among the four cells.

---

## 3. Supplementary ablations

### 3.1 `transform_pysr_ablation5_no_constraints.py` — no constraints (lower bound)

- **Question.** What happens with no prior knowledge whatsoever?
- **Design.** Removes feature-IC priors, GFlowNet, operator restrictions, econometric tests,
  reward shaping and dedup. Formula search uses the full unconstrained operator set
  (`square, abs, log1p, sign, inv, sqrt, cube, tanh, exp`; binary `+ - * /`).
- **Command.**
  ```bash
  python -m alpha.experiments.transform_pysr_ablation5_no_constraints \
      --data data/warmup/csi500_daily_2020-06-30_to_2026-06-30.parquet --trials 50 --time 150
  # run directly; not currently included in the batch launcher arrays
  ```
- **Outputs.** Registry + report.
- **Paper mapping.** Supplementary table; the "no-prior lower bound" reference point.
- **Expected result.** Large search space, dominated by overfit/unstable formulas; establishes
  that structural priors are necessary.

### 3.2 `transform_pysr_ablation6_no_mlqc.py` — no reward shaping

- **Question.** Is the continuous reward shaping necessary, or would a constant reward suffice?
- **Design.** Identical to primary except `_compute_reward` returns a constant `1.0`, so the
  GFlowNet receives no quality gradient. This isolates reward shaping from the MLQC gates.
- **Command.**
  ```bash
  python -m alpha.experiments.transform_pysr_ablation6_no_mlqc \
      --data data/warmup/csi500_daily_2020-06-30_to_2026-06-30.parquet --trials 50 --time 150
  ```
- **Paper mapping.** Supplementary table (reward-shaping ablation).
- **Expected result.** Weaker/diverse-by-chance sampling; fewer registrable factors than primary.

### 3.3 `transform_pysr_ablation10_rl_mlqcv.py` — RL + MLQCV (discrete gate reward)

- **Question.** Continuous reward shaping vs. a *discrete* reward built from the MLQC gate
  outcomes — which trains the policy better?
- **Design.** Replaces the continuous shaping with `R(τ) = floor + (top−floor)·Σ wᵢlᵢ / Σ wᵢ`,
  where `lᵢ ∈ {0,1}` are the four MLQC layer outcomes (L1 linearity, L2 t-stat, L3 monotonicity,
  L4 dedup). All other components (gates, FWL, diversity, sign penalty, floor) are retained.
- **Command.**
  ```bash
  python -m alpha.experiments.transform_pysr_ablation10_rl_mlqcv \
      --data data/warmup/csi500_daily_2020-06-30_to_2026-06-30.parquet --trials 50 --time 150
  ```
- **Paper mapping.** Supplementary table (reward-design comparison).
- **Expected result.** Comparable registration rate but a coarser policy gradient; used to argue
  for the continuous formulation.

### 3.4 `transform_pysr_ablation_no_diversity.py` — w/o diversity control

- **Question.** How much does explicit diversity control contribute to pool heterogeneity?
- **Design.** Primary with `USE_DYNAMIC_DIVERSITY = False`: diversity reward multiplier, soft/hard
  feature masking, and Jaccard similarity penalty are all off. GFlowNet + MLQC remain.
- **Command.**
  ```bash
  python -m alpha.experiments.transform_pysr_ablation_no_diversity \
      --data data/warmup/csi500_daily_2020-06-30_to_2026-06-30.parquet --trials 50 --time 150
  # launcher: ./src/alpha/cli/run_ablation_no_diversity.sh
  ```
- **Paper mapping.** Table/figure on factor-pool diversity (pairwise correlation, single-feature
  share, unique-formula share).
- **Expected result.** Higher pair-wise correlation and feature concentration; supports the
  claim that GFlowNet + diversity control maintain heterogeneity.

### 3.5 `transform_pysr_ablation_gp_no_gflownet.py` — GP instead of GFlowNet

- **Question.** GFlowNet-guided search vs. genetic programming with post-hoc quality control?
- **Design.** `gplearn` SymbolicRegressor evolves formulas directly on the full feature pool;
  no GFlowNet, no PySR refinement; evaluation is post-hoc FWL → FM → gates.
- **Command.**
  ```bash
  python -m alpha.experiments.transform_pysr_ablation_gp_no_gflownet \
      --data data/warmup/csi500_daily_2020-06-30_to_2026-06-30.parquet
  ```
- **Paper mapping.** Supplementary baseline table.
- **Expected result.** Blind GP search yields lower diversity/registration rate than guided search.

---

## 4. Baselines

### 4.1 GP baselines

- **`transform_gp_baseline.py`** — inline `gplearn` GP with the same data/features/FM as the
  primary pipeline; training fitness uses train-set ICIR.
- **`transform_gp_baseline_aligned.py`** — methodologically aligned variant: validation-set
  gating (`t ≥ TSTAT_THRESHOLD`), test set used only for out-of-sample reporting, FWL applied on
  both splits, and structural dedup of reported factors.
- **`gp_baseline.py`** — modular PySR-only companion.
- **Commands.**
  ```bash
  python -m alpha.experiments.transform_gp_baseline \
      --data data/warmup/csi500_daily_2020-06-30_to_2026-06-30.parquet
  python -m alpha.experiments.transform_gp_baseline_aligned \
      --data data/warmup/csi500_daily_2020-06-30_to_2026-06-30.parquet
  # launcher: ./src/alpha/cli/run_baseline.sh gp | gp-aligned
  ```
- **Paper mapping.** Baseline table; `aligned` is the fair-comparison row.
- **Expected result.** Aligned GP is a strong but less diverse baseline; used to show that the
  GFlowNet gain is not explained by the search objective alone.

### 4.2 External RL baselines (vendored, uniform evaluation)

These reuse official upstream training code under `src/vendor/`, with a qlib-free data adapter
(`alphagen_qlib`) and the **same** FWL + Fama–MacBeth + registry evaluation. Training target is
the 1-day forward return (horizon-aligned, no leakage).

| Script | Baseline | Notes |
|---|---|---|
| `transform_alphagen_baseline.py` | AlphaGen (PPO) | vendored `alphagen` AlphaPool/AlphaEnv + sb3 `MaskablePPO`; requires `[alphagen]` extra |
| `transform_alphaqcm_baseline.py` | AlphaQCM (IQN + Quantile-Corrected Moments) | vendored `fqf_iqn_qrdqn` agent |
| `transform_alphasage_gfn_baseline.py` | AlphaSAGE GFlowNet (SOTA) | vendored `alpha_gfn`; encoders `gnn` (paper core) / `lstm` / transformer |

- **Commands.**
  ```bash
  python -m alpha.experiments.transform_alphagen_baseline \
      --data data/warmup/csi500_daily_2020-06-30_to_2026-06-30.parquet --steps 20000 --pool 20
  python -m alpha.experiments.transform_alphaqcm_baseline \
      --data data/warmup/csi500_daily_2020-06-30_to_2026-06-30.parquet --steps 20000 --pool 20
  python -m alpha.experiments.transform_alphasage_gfn_baseline \
      --data data/warmup/csi500_daily_2020-06-30_to_2026-06-30.parquet \
      --encoder gnn --episodes 10000 --pool-capacity 50
  # smoke: --steps 300 --pool 3 --time 3 ; AlphaSAGE: --encoder lstm --episodes 300 --pool-capacity 3
  # launchers: ./src/alpha/cli/run_alphagen_baseline.sh | run_alphaqcm_baseline.sh [full|smoke]
  ```
- **Paper mapping.** External-baseline table; AlphaSAGE is the strongest published competitor.
- **Expected result.** Competitive single-factor quality, but lower interpretability (AlphaGen /
  AlphaQCM) or heavier compute (AlphaSAGE) than the primary pipeline.

### 4.3 Black-box machine-learning baselines

| Script | Model | Purpose |
|---|---|---|
| `transform_lgbm_baseline.py` | LightGBM (L2 on forward return) | tests whether a flexible black-box predictor beats interpretable formulas under identical splits/FM |
| `transform_gru_baseline.py` | GRU sequence model | temporal deep baseline on the same feature set |

- **Commands.**
  ```bash
  python -m alpha.experiments.transform_lgbm_baseline \
      --data data/warmup/csi500_daily_2020-06-30_to_2026-06-30.parquet --seeds 3
  python -m alpha.experiments.transform_gru_baseline \
      --data data/warmup/csi500_daily_2020-06-30_to_2026-06-30.parquet
  # launchers: ./src/alpha/cli/run_lgbm_baseline.sh | run_gru_baseline.sh
  ```
- **Outputs.** `lgbm_config.json` / `gru_config.json`, `feature_importance.csv` (LGBM),
  `fm_summary_report.txt`, comparison tables.
- **Paper mapping.** Black-box baseline rows; LGBM feature importances corroborate the economic
  relevance of discovered atoms.
- **Expected result.** Black-box models may match or exceed single-factor |IC| but lack
  interpretability and are reported as such, not as competitors to the factor pool.

---

## 5. Robustness experiments

### 5.1 `transform_pysr_walkforward.py` — rolling walk-forward (3y/1y/1y)

- **Question.** Are discovered factors stable across market regimes?
- **Design.** Two stages. **Stage A:** each window trains a fresh `MiningOrchestrator`
  (GFlowNet + PySR from scratch) on its own train/val and evaluates on its own test, backtesting
  Top-3. Default 36:12:12-month windows roll by 12 months over a spliced 2016–2026 CSI 500
  panel (six windows). **Stage B:** cross-window robustness — factors found in window `i` are
  re-evaluated on the test sets of later windows `j > i`, yielding cross-window hit rate and
  average IC/ICIR.
- **Protocol.**
  ```bash
  # list windows only
  python -m alpha.experiments.transform_pysr_walkforward --list-windows
  # full run (defaults splice two CSI 500 files)
  python -m alpha.experiments.transform_pysr_walkforward --trials 60 --time 80
  # subset of windows, or HS 300 smoke
  python -m alpha.experiments.transform_pysr_walkforward --windows 1,3
  python -m alpha.experiments.transform_pysr_walkforward --market hs300 --max-windows 2
  # launcher: ./src/alpha/cli/run_walkforward.sh
  ```
- **Outputs.** `walkforward_summary.json`, `cross_window_robustness.csv`, text report.
- **Paper mapping.** Walk-forward stability table/figure; directly supports the out-of-sample
  generalization claim.
- **Expected result.** Positive but decaying cross-window hit rate; factors discovered in one
  regime remain directionally valid in several subsequent windows.

### 5.2 Window-splitting variants

`--mode` selects the split family (`warmup` default; `cold`/`month`; `ratio`) and is a robustness
axis in itself. The CLI includes convenience launchers for panel-by-panel runs:

- `./src/alpha/cli/run_warmup_window.sh` — run every `data/warmup/*.parquet` panel with
  `--mode warmup`.
- `./src/alpha/cli/run_cold_window.sh` — run every file in `data/qfq/` with `--mode cold`.
- `./src/alpha/cli/run_ablation_2.sh` / `run_ablation.sh` — sequential ablation batches.

### 5.3 `test_body_ratio_ret2d_transforms.py` — formulation sensitivity

- **Question.** Does the algebraic form of a known interaction (`body_ratio × ret_2d`) change
  its measured significance?
- **Design.** Compares `square(body_ratio·ret_2d)`, the signed product, and
  `abs(body_ratio·ret_2d)`, each under raw and FWL-neutralized FM.
- **Command.**
  ```bash
  python -m alpha.experiments.test_body_ratio_ret2d_transforms
  ```
- **Paper mapping.** Appendix / robustness of the neutralization methodology.
- **Expected result.** Sign-preserving forms are more stable; neutralization attenuates raw
  significance, confirming the size-exposure concern.

---

## 6. Evaluation suite

All evaluation tools consume a `registry_academic.json` (or a formula directly) and share the
FWL + FM protocol, so metrics are comparable to the mining report.

| Script | Purpose | Key flags | Paper mapping |
|---|---|---|---|
| `evaluation/single_factor_test.py` | single-factor FM + quintile long–short/long-only backtest with NAV plot | `--registry` / `--formula`, `--id`, `--long-only`, `--no-fwl`, `--cost-bps`, `--compare-cost`, `--plot` | per-factor performance figures/tables |
| `evaluation/check_time_stability.py` | split test period into `N` windows, compare ICIR direction | `--registry`, `--ids`, `--windows` | sub-period stability table |
| `evaluation/transform_pysr_stablity_check.py` | rolling-window ICIR (win/stride months), mean ICIR + hit rate | `--win-months`, `--stride-months`, `--hit-threshold` | rolling-stability figure |
| `evaluation/compare_benchmark.py` | traditional single-variable factors vs. GFN-SR factors | `--registry`, `--top-n`, `--pool-size`, `--data` | benchmark comparison table |
| `evaluation/factor_compare_1.py` | traditional vs. generated factor comparison (Markdown) | `--registry`, `--data` | appendix comparison table |
| `evaluation/combine_factor_eval.py` | equal-weight combination vs. single factors; month-block bootstrap of synergy | `--topk`, `--drop-corr`, `--n-boot`, `--neutralize-multi` | 1+1>2 complementarity table |
| `evaluation/validate_factor_combo.py` | market-state (volatility / trend / dispersion) complementarity | (none) | regime-dependence table |
| `evaluation/factor_correlation_heatmap.py` / `analysis/plot_factor_correlation_heatmap.py` | factor×factor and factor×atom Spearman matrices (pooled + daily-mean) | `--registry`, `--data`, `--output` | correlation heatmaps (diversity evidence) |
| `evaluation/build_factor_panel.py` | export full factor-value panel from the registry | `--registry`, `--data`, `--fids`, `--neutralize` | intermediate for orthogonality |
| `evaluation/factor_to_ls_returns.py` | panel → equal-weight long–short daily returns | (module API) | intermediate for orthogonality |
| `evaluation/orthogonality_check.py` | regress L-S returns on FF5 + Amihud liquidity (NW HAC) | (module API) | orthogonality table |
| `evaluation/fetch_ff5_liq.py` | download FF5 factors and compute Amihud liquidity → `ff5_liq.csv` | (none) | input for orthogonality |
| `analysis/decode_gp_formulas.py` | resolve gplearn `X<idx>` placeholders to feature names | `--data` | GP interpretation appendix |
| `analysis/regen_fm_report.py` | regenerate `fm_summary_report.txt` from a registry without re-mining | `--registry`, `--title` | report regeneration |
| `analysis/diagnose_single_features.py` | single-feature diagnostics | (none) | appendix diagnostics |

**Typical pipeline.**
```bash
REG=factor_output_academic_v81/run_<ts>_s42/registry_academic.json
python -m alpha.evaluation.single_factor_test        --registry "$REG" --data data/warmup/csi500_daily_2020-06-30_to_2026-06-30.parquet
python -m alpha.evaluation.check_time_stability      --registry "$REG" --windows 4
python -m alpha.evaluation.transform_pysr_stablity_check --registry "$REG" --win-months 4 --stride-months 2
python -m alpha.evaluation.factor_correlation_heatmap --registry "$REG"
python -m alpha.evaluation.combine_factor_eval       --registry "$REG" --topk 5 --n-boot 2000
python -m alpha.evaluation.compare_benchmark         --registry "$REG" --top-n 10
# launcher for several of the above: ./src/alpha/cli/run_eval.sh <tool> --registry ...
```

---

## 7. Result reporting conventions

When assembling paper tables, apply these conventions consistently:

1. **Unit of analysis** is the factor. Report count, |FM t|, Rank IC, ICIR, complexity and gate
   status (STRONG/WEAK).
2. **Production-ready** uses the strict shared definition
   (`is_production_ready`: non-WEAK, |t| ≥ 3.0, |ICIR| ≥ 0.3, |IC| ≥ 0.03, n ≥ 30) — never a
   per-experiment threshold.
3. **Out-of-sample only.** All headline metrics come from the test split; validation is for
   search/gating.
4. **Neutralized by default.** Report FWL-neutralized FM unless explicitly contrasting raw vs.
   neutralized.
5. **Costs and turnover.** Backtests report net returns with the configured round-trip cost;
   long-only and long–short are labelled explicitly.
6. **Multiple comparisons.** Where many candidates are generated (e.g. w/o-MLQC), accompany
   headline claims with the White Reality Check.
7. **Seeds.** State seeds and, for stochastic baselines, report dispersion across seeds rather
   than a single run.
