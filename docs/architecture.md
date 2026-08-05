# 架构说明

## 目录结构

```
src/alpha/
├── config.py              # 全局配置 (Config), 输出根目录 OUTPUT_ROOT
├── data/                  # 数据层
│   ├── data_fetcher.py    # baostock 数据抓取
│   ├── data_loader.py     # CSI500 日频数据加载/清洗
│   └── load_data.py       # 便捷加载 (CSV 版)
├── features/
│   └── feature_registry.py  # 原子特征注册表 (LargeCap / UltimateDaily)
├── mining/                # 挖掘核心 (原 common/)
│   ├── preprocessor.py    # 特征池构建 (prepare_full_pool)
│   ├── gflownet.py        # GFlowNet 采样器/训练器
│   ├── pysr_engine.py     # PySR 符号回归引擎
│   ├── fm_regression.py   # Fama-MacBeth 回归 (NW 标准误)
│   ├── neutralize.py      # FWL 中性化
│   ├── orchestrator.py    # MiningOrchestrator: 编排全流程
│   ├── registry.py        # FactorRegistry 因子注册表
│   ├── safe_ops.py        # SafeOps / TimeSeriesOps
│   ├── screener.py        # 因子筛选
│   ├── stats_validator.py # 统计校验
│   ├── walkforward.py     # 滚动窗口检验
│   └── audit_logger.py    # 审计日志 (JSONL)
├── evaluation/            # 评估层
│   ├── backtester.py      # AcademicBacktester 回测
│   ├── single_factor_test.py       # 单因子评估
│   ├── check_time_stability.py     # 分时段 ICIR 稳定性
│   ├── transform_pysr_stablity_check.py  # 滚动窗口稳定性
│   ├── compare_benchmark.py        # 传统 vs GFN-SR 对比
│   ├── factor_compare_1.py         # V8 因子对比
│   └── validate_factor_combo.py    # 多因子互补性验证
├── analysis/              # 诊断工具
│   ├── plot_factor_correlation_heatmap.py
│   ├── calc_jaccard_homogeneity.py
│   ├── diagnose_single_features.py
│   ├── decode_gp_formulas.py
│   └── p5.py
├── experiments/           # 可运行实验
│   ├── transform_pysr_primary.py   # 主挖矿入口 (alpha-mine)
│   ├── ablations/                  # 消融实验 1-8 + GA/GP
│   └── baselines/                  # GP 基线
└── cli/                   # shell 启动脚本
    ├── run.sh / run_ablation.sh / run_baseline.sh / run_eval.sh
```

## 数据流

```
parquet/csv
  → CSI500Loader (清洗/剔除ST/缩尾)
  → DataPreprocessor.prepare_full_pool (83 特征池)
  → MiningOrchestrator.run
      ├─ GFlowNet 采样候选因子结构
      ├─ PySR 符号回归拟合公式
      ├─ FM 回归 + FWL 中性化 → t-stat / ICIR / IC
      └─ 筛选入册 → FactorRegistry
  → AcademicBacktester 单因子回测
  → outputs/<run>/registry_academic.json
```

## 关键约定

- **输出统一** `outputs/` 根目录，可用 `OUTPUT_ROOT` 环境变量重定向
- **类名保留** `AcademicBacktester` / `AcademicDualEngine` 等历史命名未改动
- **PySR 可选** 未安装时静默跳过符号回归
- **Julia 环境** PySR 首次运行会在 `.venv/julia_env` 安装依赖，耗时较长
