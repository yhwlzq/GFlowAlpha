# 使用指南

## 环境准备

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[pysr,gp,dev]"   # 或 pip install -e .
```

PySR 首次运行会自动初始化 Julia 后端 (`.venv/julia_env`)，需联网。

## 主挖矿

```bash
# 冒烟 (少量 trials, 快速验证)
python src/alpha/research/transform_pysr_primary.py --data data/csi500_daily_2020-07-20_to_2026-07-19.parquet --trials 3 --time 10

# 正式运行 (默认数据, 60 trials / 80s)
python src/alpha/research/transform_pysr_primary.py --data data/csi500_daily_2020-07-20_to_2026-07-19.parquet

# 指定数据 / 参数
python src/alpha/research/transform_pysr_primary.py --data data/csi500_daily_2020-06-30_to_2026-06-30.parquet --trials 100 --time 120

# 沪深300
python src/alpha/research/transform_pysr_primary.py --market hs300 --data data/hs300_daily_2021-06-30_to_2026-06-30.parquet
```

结果默认写入 CWD 下 `factor_output_academic_v81/run_<时间戳>/`，含 `registry_academic.json` / `mining.log` / 回测图。历史结果已统一归档到 `outputs/` (见下)。

## 实验输出统一目录

所有历史实验产物已归档到仓库根 `outputs/`：

```
outputs/
├── run_<timestamp>_*/    ← 当前统一运行产物
├── academic_v81/         ← 历史 factor_output_academic_v81 运行 (run_* / ablation*)
├── pysr/                 ← 旧 paper/PYSR/outputs 运行
└── legacy/               ← 更早的归档 (factor_output/, walkforward, comparisons, logs)
```

> 注意: 脚本仍默认把新结果写到 CWD 的 `factor_output_academic_v81/`。如需归一到 `outputs/`, 可在脚本中把 `Config.OUTPUT_DIR` 设为 `outputs/<tag>`, 或运行后自行移动。

## 消融实验

2×2 主表（GFlowNet 整块 × MLQC 整块，各作一个整体移除）：

| 变体 | GFlowNet | MLQC | 脚本 |
|---|---|---|---|
| 完整 | √ | √ | `transform_pysr_primary.py` |
| 去MLQC | √ | × | `transform_pysr_ablation7_no_mlqc_gates.py` |
| 去GFlowNet | × | √ | `transform_pysr_ablation8_no_gflownet.py` |
| 无引导+无MLQC | × | × | `transform_pysr_ablation9_random_no_mlqc.py` |

补充基线（表外）：`transform_pysr_ablation5_no_constraints.py`（完全无约束，无先验下限）、`transform_pysr_ablation_gp_no_gflownet.py`（GP 替代 GFlowNet）、`transform_pysr_ablation10_rl_mlqcv.py`（RL+MQLCV，将 MLQC 四层门禁结果加权分段编码为 GFlowNet 终止奖励 R(τ)，对比连续奖励塑形）。

```bash
python src/alpha/experiments/transform_pysr_ablation7_no_mlqc_gates.py --data data/...
python src/alpha/experiments/transform_pysr_ablation8_no_gflownet.py --data data/...
python src/alpha/experiments/transform_pysr_ablation9_random_no_mlqc.py --data data/...
python src/alpha/experiments/transform_pysr_ablation5_no_constraints.py --data data/...
python src/alpha/experiments/transform_pysr_ablation_gp_no_gflownet.py --data data/...
python src/alpha/experiments/transform_pysr_ablation10_rl_mlqcv.py --data data/...
```

或批量：`./src/alpha/cli/run_ablation.sh 9` / `./src/alpha/cli/run_ablation.sh all`

## 基线

```bash
python src/alpha/research/transform_gp_baseline.py --data data/...

# AlphaGen (PPO) 外部 RL 基线 — 训练复用官方 AlphaSAGE (vendored 于 src/vendor/alphagen,
# 数据层由自研 qlib-free drop-in src/vendor/alphagen_qlib 顶替), target=前向1日收益;
# 评测统一走 FWL+Fama-MacBeth+registry。依赖: stable-baselines3/sb3-contrib/gymnasium。
python src/alpha/experiments/transform_alphagen_baseline.py --data data/... --steps 20000 --pool 20
python src/alpha/experiments/transform_alphagen_baseline.py --data data/... --steps 300 --pool 3 --time 3  # 冒烟
```

## 评估

```bash
# 单因子评估 (从注册表)
python src/alpha/research/single_factor_test.py --registry outputs/academic_v81/run_xxx/registry_academic.json

# 对比 GFN-SR vs 传统
python src/alpha/research/compare_benchmark.py --registry outputs/academic_v81/run_xxx/registry_academic.json

# 分时段稳定性 / 滚动窗口稳定性
python src/alpha/research/check_time_stability.py --registry outputs/academic_v81/run_xxx/registry_academic.json
python src/alpha/research/transform_pysr_stablity_check.py --registry outputs/academic_v81/run_xxx/registry_academic.json

# 热力图 / 因子组合 / 诊断
python src/alpha/research/plot_factor_correlation_heatmap.py --registry outputs/academic_v81/run_xxx/registry_academic.json
python src/alpha/research/validate_factor_combo.py
python src/alpha/research/diagnose_single_features.py
```

热力图输出四种矩阵 (各含 PNG/PDF 与 CSV)：
- `factor_corr_pooled` / `factor_corr_daily_mean`：因子×因子（池化 Spearman + 逐日截面秩相关均值）
- `factor_atomic_corr_pooled` / `factor_atomic_corr_daily_mean`：复合因子×所用原子特征
- 附带 `corr_report.json`（Top 相关对 + 特征共享/独占摘要）

## 数据说明

`data/` 存放 parquet (gitignored)：

```
csi500_daily_2020-06-30_to_2026-06-30.parquet
csi500_daily_2020-07-20_to_2026-07-19.parquet
csi500_daily_2021-06-30_to_2026-06-30.parquet
hs300_daily_2021-06-30_to_2026-06-30.parquet
```

可用 `src/alpha/data/data_fetcher.py` 从 baostock 抓取新数据。运行入口脚本请始终显式传 `--data data/...`（部分脚本仍带指向旧路径的默认值）。
