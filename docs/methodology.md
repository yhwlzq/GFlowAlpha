# Methodology

This document specifies the GFlowAlpha method at a level of detail sufficient for reproduction
and for writing the Method section of a journal submission. All symbols, defaults and gates are
taken from `src/alpha/config.py` and the mining engine; numbers in parentheses are the shipped
defaults. See [experiments.md](experiments.md) for the experiment catalogue and
[architecture.md](architecture.md) for the code-level component map.

## 1. Problem setup and notation

We consider a cross-sectional return-prediction problem on a universe of Chinese A-shares.
Let the panel be indexed by trading day `t = 1..T` and stock `i = 1..Nₜ`. Each observation
carries a forward return

```
r_{i,t} = close_{i,t+H} / close_{i,t} − 1,          (H = 1 by default)
```

and a vector of atomic features `x_{i,t} ∈ R^P` derived from OHLCV microstructure. The goal is
to discover a small set of **closed-form** factors `f_k(x)` that predict `r` out of sample and
survive neutralization against liquidity/size.

| Symbol | Meaning |
|---|---|
| `P` | size of the post-gate atomic feature pool (observed: 85) |
| `τ` | feature subset selected by the GFlowNet, `τ ⊂ {1..P_{full}}`, `k_min ≤ |τ| ≤ k_max` |
| `f` | closed-form formula returned by PySR for subset `τ` |
| `S_fp` | set of structural fingerprints of registered formulas (exact-structure dedup) |
| `S_sig` | set of canonical signatures (feature-name-independent structure dedup) |
| `M_mask`, `H_mask` | soft / hard feature masks maintained for diversity |
| `R(τ)` | shaped reward for trajectory `τ` |

Feature-subset size is bounded by `MIN_FEATURES = 3` and `MAX_FEATURES = 5`.

## 2. Data cleaning and feature construction

`CSI500Loader` applies, in order: removal of ST/\*ST/PT names; removal of suspended bars
(zero volume **and** flat OHLC); removal of price anomalies (`high < max(open, close)` or
`low > min(open, close)`); removal of stocks with fewer than `min_bars = 252` observations;
per-day winsorization of `pctChg` at the `[1%, 99%]` quantiles; and derivation of `ret`.

Atomic features come from one of four registries (Section [architecture](architecture.md#features--atomic-feature-registries)):
`UltimateDaily` (CSI 500, default), `LargeCap` (HS 300), `Timing`, or the Qlib-style `Alpha158`.
After construction, `DataPreprocessor.prepare_full_pool` performs:

1. **Missing/variance gate.** Drop a feature if its NaN ratio exceeds `NaN_RATIO_CAP = 0.30` or
   its variance is below `1e-6`.
2. **Near-duplicate dedup.** Union-find over pairwise Spearman correlation; a *positively*
   correlated pair above `NEAR_DUPLICATE_CORR = 0.95` keeps the member with the larger |IC|.
   Negatively correlated mirror pairs (e.g. reversal vs. momentum) are intentionally retained.
3. **Clustering orthogonalization.** Complete-linkage hierarchical clustering on the distance
   `1 − |Spearman|`, taking the highest-|IC| representative of each cluster with
   `TARGET_FACTOR_POOL_SIZE` clusters. This yields the final pool `P` (85 in the shipped runs).
4. **Missing-value fill.** Remaining NaNs are set to 0.

Standardization statistics are fit **only on the training split** and then applied to validation
and test (no leakage).

## 3. Temporal splits

`SPLIT_MODE` controls the partitioning:

| Mode | Construction |
|---|---|
| `warmup` (default) | first `WARMUP_MONTHS = 12` months serve only as feature look-back; then `TRAIN_MONTHS:VAL_MONTHS:TEST_MONTHS = 36:12:12` months |
| `cold` / `month` | `36:12:12` months from the data start, no warm-up suffix |
| `ratio` | natural trading-day `TRAIN_RATIO:VAL_RATIO:TEST_RATIO = 0.6:0.2:0.2` |
| `walkforward` | rolling `WF_TRAIN_MONTHS:WF_VAL_MONTHS:WF_TEST_MONTHS = 27:12:3` months, non-overlapping test windows, with `WF_PURGE_DAYS = 1` and `WF_EMBARGO_DAYS = 2` to prevent leakage across the `H = 1` label horizon |

Validation (`val`) is used for factor candidacy, gating and reward; test (`test`) is used **only**
for the final registration gate and reported out-of-sample metrics.

## 4. Mining orchestrator

Algorithm 1 summarizes `MiningOrchestrator.run`. The GFlowNet is trained *online*: each trial's
realized quality reward immediately updates the policy, so the search shifts toward feature
subsets that yield registrable formulas.

```
Algorithm 1  GFlowNet-guided symbolic alpha mining
────────────────────────────────────────────────────────────────────────────
Input:  feature pool {x_j}, labels r, splits (tr, va, te), budget (T_max, τ_min)
Output: registry Z of (formula, gate_status)

Init:  S_fp ← ∅, S_sig ← ∅, M_mask ← ∅, H_mask ← ∅, feature_usage ← {}, π_θ
while trial < T_max and elapsed < τ_min:
    # (a) diversity state: expire masks, refresh penalties
    for f in M_mask ∪ H_mask: remaining[f] −= 1; drop on expiry
    penalty[j] ← 5.0 if j ∈ M_mask else 50.0 if j ∈ H_mask else 0.0
    if trial > 15:
        for f with appearance_ratio > 0.50: H_mask[f] ← 10

    # (b) sample a feature subset from the policy
    τ ~ π_θ;  feats ← features(τ)
    if |feats| < k_min:  train(π_θ, R_hard);  trial += 1;  continue

    # (c) symbolic regression + stability selection on the training split
    f ← PySR(feats, X_tr, y_tr)                    # top-K candidates by train score
    f ← select_stable(f, top_k=VAL_SELECT_TOP_K)   # perturbation stability on val
    if f is None or is_single_feature_linear(f):
        train(π_θ, R_hard);  trial += 1;  continue

    # (d) neutralization + Fama–MacBeth on validation
    (f̃, r̃) ← FWL(f(X_va), r_va, ln amount_va)
    stats ← FamaMacBeth(f̃, r̃, Newey–West)

    # (e) MLQC gates L1–L4
    if |stats.t| < TSTAT_THRESHOLD:            train(π_θ, R_hard); trial += 1; continue  # L2
    if not monotone(stats.quantile_returns):   train(π_θ, R_hard); trial += 1; continue  # L3
    fp ← normalize_structure(f)
    if fp ∈ S_fp:                              train(π_θ, R_hard); trial += 1; continue  # L4
    canon ← canonical_signature(f); R_dedup ← 0.3 if canon ∈ S_sig else 1.0

    # (f) shaped reward
    t_score    ← σ(k_t   · (|t|    − μ_t))
    icir_score ← σ(k_ir  · (|ICIR| − μ_ir))
    ic_score   ← σ(k_ic  · (|IC|   − μ_ic))
    R ← floor + (top − floor)·t_score^{w_t}·icir_score^{w_ir}·ic_score^{w_ic}
    if sign(t)·sign(IC) < 0 or sign(t)·sign(ICIR) < 0:  R ← 0.1·R
    if effective_feature_count(f) ≤ 1:                  R ← 0.2·R
    R ← R · diversity_mult(τ)        # 1 + 0.15·#low-usage features, capped at 1.5
    if max_feature_similarity(τ) > 0.75: R ← 0.1·R
    R ← max(R·R_dedup, floor)
    train(π_θ, R)

    # (g) out-of-sample registration gate on the test split
    stats_te ← FamaMacBeth(FWL(f(X_te), r_te, ln amount_te))
    if sign(stats_te.t)·sign(IC_te) < 0 or sign(stats_te.t)·sign(ICIR_te) < 0: reject
    if |t_te| < 2.0 or |IC_te| < 0.01 or |ICIR_te| < 0.10: reject
    gate ← STRONG if (|ICIR_te| ≥ μ_ir and |IC_te| ≥ μ_ic and |t_te| ≥ 2.5) else WEAK
    Z ← Z ∪ {(f, gate)};  S_fp ← S_fp ∪ {fp};  S_sig ← S_sig ∪ {canon}
    update masks/usage;  trial += 1

Post-process on Z: mark production_ready iff non-WEAK and |t| ≥ 3.0 and |ICIR| ≥ 0.3
                   and |IC| ≥ 0.03 and n_periods ≥ 30.
────────────────────────────────────────────────────────────────────────────
```

### 4.1 GFlowNet generator

`GFlowNetPolicy_Transformer` is a transformer over the action sequence (one action per selected
feature) conditioned on a per-feature attribute matrix built by `FeatureMetadataExtractor` and an
availability mask. `TrajectorySampler_Transformer` samples subsets with size in
`[MIN_FEATURES, MAX_FEATURES]` and returns forward/backward log-probabilities; per-feature logit
penalties implement the soft/hard diversity masks. The policy is trained by the trajectory-balance
update in `GFlowNetTrainerInternal.train_step` with the reward of Section 4.4.

### 4.2 Symbolic regression and stability selection

`PySRMiningEngine` fits `PySRRegressor` (`PY_SR_ITERATIONS`, `PY_SR_POPULATIONS`,
`PY_SR_MAXSIZE`, `PY_SR_MAXDEPTH`, `PY_SR_PARSIMONY`) on the training split and returns the top
`VAL_SELECT_TOP_K = 3` candidates ranked by the training-side score, filtered to complexity
`≤ FORMULA_MAX_CAND_CMPLX = 12`. `_select_stable_candidate` then perturbs the validation data and
keeps a formula only if it remains significant: original validation `|t| ≥ STABILITY_ENTRANCE_T = 2.0`,
at least `STABILITY_MIN_HITS = 2` perturbation states with `|t| ≥ STABILITY_T_MIN = 1.0` and
consistent sign. Single-feature linear formulas are rejected as trivial.

### 4.3 FWL neutralization and Fama–MacBeth

For each day independently, `fwl_neutralize` residualizes both `f(X)` and `r` on
`[1, ln(amount)]` (plus optional controls), removing size/liquidity exposure. `FamaMacBethRegressor`
runs daily cross-sectional OLS, averages the slope series `β_t`, and computes

```
t = mean(β_t) / SE_NW(β_t),      SE_NW from a Newey–West HAC covariance with NW_LAGS = 3,
```

requiring at least `fm_min_stocks` names per day (`zs500`: 30, `hs300`: 20) and at least 10
periods. Reported diagnostics include coefficient, NW-SE, t, p-value, average R²/adj-R²,
coefficient sign ratios, quantile returns, long–short spread, **Rank IC** mean, and **Rank ICIR**
(mean daily rank IC divided by its standard deviation).

### 4.4 Reward shaping

The reward combines three sigmoid-transformed quality scores with per-market slopes,
thresholds and weights (`REWARD_CONFIG`):

```
score_x = σ(k_x · (|x| − μ_x)),     x ∈ {t-stat, ICIR, IC}
R_raw   = REWARD_SOFT_FLOOR + (REWARD_TOP − REWARD_SOFT_FLOOR)·score_t^{w_t}·score_ICIR^{w_ir}·score_IC^{w_ic}
```

with `REWARD_SOFT_FLOOR = 0.3`, `REWARD_TOP = 10.0`, and `REWARD_HARD = 0.01` for failed trials.
Per-market values:

| Market | slope t | μ_t | slope ICIR | μ_ICIR | slope IC | μ_IC | w_t | w_ICIR | w_IC |
|---|---|---|---|---|---|---|---|---|---|
| `zs500` | 2.0 | 2.0 | 25 | 0.10 | 150 | 0.03 | 0.20 | 1.0 | 1.0 |
| `hs300` | 3.0 | 2.0 | 20 | 0.08 | 100 | 0.02 | 0.35 | 1.0 | 1.0 |

Multiplicative modifiers:

- **Sign mismatch** — if `sign(t)·sign(IC) < 0` or `sign(t)·sign(ICIR) < 0`, multiply by
  `SIGN_MISMATCH_PENALTY = 0.1`.
- **Single effective feature** — multiply by `0.2`.
- **Diversity bonus** — `min(1.5, 1 + 0.15 · #low-usage features)` where "low-usage" means usage
  `≤ max(1, 0.15·total_passed)`.
- **Similarity penalty** — multiply by `0.1` if the maximum Jaccard similarity to any registered
  formula's feature set exceeds `0.75`.
- **Canonical duplicate penalty** — multiply by `0.3` if the feature-name-independent structure
  was already registered.
- **Floor** — final reward is `max(R, REWARD_SOFT_FLOOR)`.
- **Sub-period robustness** (optional, `USE_SUBPERIOD_ROBUST_REWARD`, walk-forward runs) —
  multiply by `0.3 + 0.7·robustness`, where robustness is measured across
  `SUBPERIOD_ROBUST_SPLITS = 3` train sub-periods plus validation.

### 4.5 Multi-Layer Quality Control (MLQC)

Four gates are applied in sequence on the **validation** split; failing any gate sends the trial
to the hard floor reward and it is not considered for registration:

| Layer | Gate | Criterion |
|---|---|---|
| L1 | Single-feature linearity | formula must be non-trivial (`effective_feature_count > 1` or contain an operator) |
| L2 | Statistical significance | `|FM t| ≥ TSTAT_THRESHOLD = 2.0` |
| L3 | Monotonicity | quantile returns must be monotone (`USE_MONOTONICITY_GATE`) |
| L4 | Structural dedup | structural fingerprint not already in `S_fp` (canonical duplicates incur a soft reward penalty instead of rejection) |

### 4.6 Registration gate (out of sample)

A formula enters the registry only if, on the **test** split after FWL and FM:

1. signs are consistent: `t·IC > 0` and `t·ICIR > 0`;
2. `|t| ≥ TEST_TSTAT_MIN = 2.0`;
3. `|IC| ≥ REGISTER_IC_MIN = 0.01`;
4. `|ICIR| ≥ REGISTER_ICIR_MIN = 0.10`.

Passing formulas are labelled **STRONG** iff `|ICIR| ≥ μ_ICIR`, `|IC| ≥ μ_IC` and
`|t| ≥ TEST_TSTAT_STRONG = 2.5`; otherwise **WEAK**. This gate is implemented once
(`_registration_verdict`) and reused verbatim by primary, ablations and baselines, which is what
makes cross-experiment comparisons valid.

### 4.7 Production verdict

`is_production_ready` (used by reports and comparison tables) is stricter than registration:
non-WEAK **and** `|t| ≥ PRODUCT_FMT_THRESHOLD = 3.0`, `|ICIR| ≥ PRODUCT_ICIR_THRESHOLD = 0.3`,
`|IC| ≥ PRODUCT_IC_THRESHOLD = 0.03`, and at least 30 cross-sectional periods.
`EXCLUDE_WEAK_FROM_PRODUCTION` removes WEAK factors from production summaries.

### 4.8 Diversity control

When `USE_DYNAMIC_DIVERSITY` is on:

- a **soft mask** (penalty 5.0) is placed on the core features of each registered formula for 10
  trials;
- a **hard mask** (penalty 50.0) is placed on any feature appearing in more than 50% of trials
  after trial 15;
- the reward includes the low-usage diversity bonus and the Jaccard similarity penalty above.

Disabling these three mechanisms (`transform_pysr_ablation_no_diversity.py`) isolates their
contribution.

## 5. Portfolio backtest

`AcademicBacktester` forms monthly (rebalance `REBALANCE_FREQ_DAYS = 20`) long–short portfolios
from the cross-sectional factor ranking, using `top_quantile = 0.2` (Q5 vs. Q1). Key
methodological choices:

- **T+1 open execution** (when open/close prices are supplied): signals formed at the close of
  day `t` are executed at the open of `t+1`; the execution-day return is `close/open` (overnight
  excluded) and holding-day returns are `close/prev_close`. Limit-up (long) / limit-down (short)
  names that cannot fill at the open are dropped.
- **Transaction costs** tied to realized turnover, charged per leg as a full round trip
  (`cost_bps`, default 30 bps = 15 bps one-way).
- **Survivorship-safe handling**: suspended/missing names are reindexed to zero weight rather
  than dropped, preventing implicit weight redistribution.
- **Deterministic binning**: `rank(method='first')` before quantile cuts.
- Both **long–short** and **long-only** variants are reported; annualization uses 252 trading
  days.

## 6. Statistical validation suite

Beyond the registration gate:

- **Sub-period ICIR stability** (`check_time_stability.py`) — split the test period into `N`
  equal windows and check the sign/direction of ICIR in each.
- **Rolling-window ICIR stability** (`transform_pysr_stablity_check.py`) — sliding windows of
  `win_months` with `stride_months` overlap; reports mean ICIR and hit rate.
- **White Reality Check** (`stats_validator.whites_reality_check`) — stationary-bootstrap test
  over the per-day Rank-IC series of all candidates, controlling data-snooping across the many
  trials the search performs.
- **Cross-window robustness** (`transform_pysr_walkforward.py`, Stage B) — re-evaluates factors
  discovered in window `i` on the test sets of later windows `j > i`, reporting hit rate and
  average IC/ICIR.
- **Correlation / complementarity** — pooled and daily-mean Spearman matrices
  (`factor_correlation_heatmap.py`), combination evaluation with month-block bootstrap of the
  synergy gain (`combine_factor_eval.py`), and market-state complementarity
  (`validate_factor_combo.py`).
- **Orthogonality** — regress factor long–short returns on FF5 + Amihud liquidity with
  Newey–West HAC (`build_factor_panel.py` → `factor_to_ls_returns.py` → `orthogonality_check.py`,
  with `fetch_ff5_liq.py` to obtain the benchmarks).

## 7. Reproducibility

- **Seeds.** `SEED` (global default 42) drives `torch`, `numpy` and `random`; PySR, GP and
  LightGBM have dedicated seeds (`PYSR_SEED`, `GP_SEED`, `LGBM_SEED`). `set_global_seed` also
  pins cuDNN to deterministic mode.
- **Determinism.** Splits are calendar-anchored; binning is deterministic; standardization is
  fit on train only; test data never influences search or gating.
- **Provenance.** Each run is timestamped and self-contained (`mining.log`, `registry_academic.json`,
  `fm_summary_report.txt`, audit JSONL).
- **Environment.** Python ≥ 3.13; core dependencies `pandas`, `numpy`, `torch`, `scipy`,
  `scikit-learn`, `statsmodels`, `sympy`, `matplotlib`, `seaborn`, `lightgbm`, `baostock`.
  Optional extras: `pysr` (+ Julia via `juliacall`), `gplearn`, `alphagen`
  (`gymnasium`/`stable-baselines3`/`sb3-contrib`), `openai`, `akshare`.
- **Smoke protocol.** Verify any configuration with `--trials 3 --time 10` before launching a
  full run (see [run_guide.md](run_guide.md)).

## 8. Design rationale

- **Why a GFlowNet rather than RL-on-a-scalar?** Alpha discovery benefits from *mode diversity*:
  many distinct, individually significant formulas. GFlowNet's reward-proportional sampling
  targets a distribution over feature subsets instead of collapsing to a single high-reward
  solution, which is exactly what a factor pool needs. The discovery-vs-random contrast is the
  central ablation.
- **Why PySR on top of a learned selector?** The GFlowNet searches *which* features to combine;
  PySR solves *how* to combine them in closed form. This separation keeps formulas interpretable
  and makes the selector's contribution measurable.
- **Why MLQC as a separate, reusable layer?** A single definition of "good factor" shared by
  primary, baselines and reports is a prerequisite for credible comparisons; it is also the axis
  of the 2×2 ablation.
- **Why FWL + Fama–MacBeth?** Neutralizing `ln(amount)` prevents liquidity/size from masquerading
  as alpha, and Newey–West standard errors account for the autocorrelation induced by overlapping
  construction and monthly rebalancing.
