# AGENTS.md

## 这是什么

中证500/沪深300 日频因子挖掘系统：用 GFlowNet 约束 + PySR 符号回归从 OHLCV 微观数据发现 alpha 因子，经 Fama-MacBeth 评估、中性化与回测后写入注册表。

## 目录结构

```
src/alpha/
  config.py                ← 全局配置 (Config), OUTPUT_ROOT="outputs"
  data/                    ← data_fetcher (baostock), data_loader (CSI500Loader), load_data
  features/                ← feature_registry (原子特征池)
  mining/                  ← 核心: preprocessor/gflownet/pysr_engine/fm_regression/neutralize/orchestrator/registry/safe_ops/screener/stats_validator/walkforward/audit_logger
  evaluation/              ← backtester(AcademicBacktester)/single_factor_test/check_time_stability/compare_benchmark/factor_compare_1/validate_factor_combo/transform_pysr_stablity_check
  analysis/                ← heatmap/jaccard/diagnose/decode_gp_formulas/p5
  experiments/
    transform_pysr_primary.py   ← 主挖矿入口 (console: alpha-mine)
    ablations/                  ← 消融实验1-8 + GA/GP (10个)
    baselines/                  ← GP 基线
  cli/                      ← run.sh / run_ablation.sh / run_baseline.sh / run_eval.sh
data/                       ← parquet (gitignored)
outputs/                    ← 统一结果目录; legacy/ 存历史产物
docs/ tests/
```

## 运行

```bash
./src/alpha/cli/run.sh --debug                     # 冒烟 (3股×60日)
./src/alpha/cli/run.sh --data data/*.parquet --trials 60 --time 80
OUTPUT_ROOT=/tmp/exp ./src/alpha/cli/run.sh        # 重定向结果
pytest tests/
```

## 数据格式

parquet 列: `date, code, open, high, low, close, volume, amount, turn, pctChg`
`code` 会被 `CSI500Loader` rename 为 `symbol`。按 symbol+date 排序。

## 坑

- **类名/文件名保留历史命名**: `AcademicBacktester`, `AcademicDualEngine`, `registry_academic.json`, `transform_pysr_*` 均未改。
- **输出目录**: 各脚本用 `Config.OUTPUT_ROOT` (默认 `outputs`) 建 `run_<ts>` / `ablationX_<ts>` 子目录，可用 `OUTPUT_ROOT` 环境变量整体重定向。
- **PySR 可选**: 未装则静默跳过符号回归。Julia 后端首次运行会装 `.venv/julia_env` (较慢)。
- **顶层脚本需 juliacall 环境变量**: 直接 `python -m alpha.experiments.*` 运行即可，脚本顶部已设 `PYTHON_JULIACALL_*`。
- **无 CI/lint/typecheck**。验证方式是 `pytest tests/` + `--debug` 冒烟。
- **预处理慢**: `prepare_full_pool` 全量 ~118s/683k 行，调试时用小样本。
- **`calc_jaccard_homogeneity.py` 等分析脚本的 registry 默认路径指向 `outputs/legacy/`** 归档，必要时传 `--registry` 覆盖。
