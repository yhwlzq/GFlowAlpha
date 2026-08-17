# 架构说明

## 目录结构

```
src/alpha/
├── research/              # 研究核心 (GFlowNet + PySR 挖矿管线)
│   ├── common/            # 挖掘引擎
│   │   ├── config.py      # 全局配置 (Config), 输出目录 OUTPUT_DIR
│   │   ├── data_loader.py # CSI500 日频数据加载/清洗
│   │   ├── feature_registry.py  # 原子特征注册表 (LargeCap / UltimateDaily)
│   │   ├── preprocessor.py      # 特征池构建 (prepare_full_pool)
│   │   ├── gflownet.py          # GFlowNet 采样器/训练器
│   │   ├── pysr_engine.py       # PySR 符号回归引擎
│   │   ├── fm_regression.py     # Fama-MacBeth 回归 (NW 标准误)
│   │   ├── neutralize.py        # FWL 中性化
│   │   ├── orchestrator.py      # MiningOrchestrator: 编排全流程
│   │   ├── registry.py          # FactorRegistry 因子注册表
│   │   ├── safe_ops.py          # SafeOps / TimeSeriesOps
│   │   ├── screener.py          # 因子筛选
│   │   ├── stats_validator.py   # 统计校验
│   │   ├── walkforward.py       # 滚动窗口检验
│   │   └── audit_logger.py      # 审计日志 (JSONL)
│   ├── transform_pysr_primary.py   # 主挖矿入口
│   ├── transform_pysr_ablation*.py # 消融实验 5-8 + GA/GP
│   ├── transform_gp_baseline.py / gp_baseline.py  # GP 基线
│   ├── backtester.py              # AcademicBacktester 回测
│   ├── single_factor_test.py      # 单因子评估
│   ├── check_time_stability.py    # 分时段 ICIR 稳定性
│   ├── transform_pysr_stablity_check.py  # 滚动窗口稳定性
│   ├── compare_benchmark.py       # 传统 vs GFN-SR 对比
│   ├── factor_compare_1.py        # V8 因子对比
│   ├── validate_factor_combo.py   # 多因子互补性验证
│   ├── plot_factor_correlation_heatmap.py  # 相关热力图
│   ├── diagnose_single_features.py
│   ├── decode_gp_formulas.py
│   ├── p5.py / load_data.py
├── agents/                # 早期 agent 原型 (PySR-only)
│   ├── factor_llm_miner.py, gflownet_study1-6.py, p1..p7.py
└── data/
    └── data_fetcher.py    # baostock 数据抓取
data/                      # 行情数据 (parquet, gitignored)
outputs/                   # 统一实验输出根目录
  ├── run_<timestamp>_*/   # 当前运行产物
  ├── academic_v81/        # 历史 factor_output_academic_v81 运行
  ├── pysr/                # 旧 paper/PYSR/outputs 运行
  └── legacy/              # 旧输出归档 (factor_output, walkforward, logs)
docs/                      # 文档
```

> 说明: 代码目录保持 `research/` 自包含的平铺结构 (脚本与 `common/` 同层), 以便保留脚本间的兄弟导入 (`from common.x import ...`, `from backtester import ...`), 未改动任何代码。

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
  → factor_output_academic_v81/<run>/registry_academic.json (输出到 CWD)
```

## 关键约定

- **输出统一** 历史产物已归档到 `outputs/` (含 `academic_v81/`、`pysr/`、`legacy/`)。当前脚本仍按 `Config.OUTPUT_DIR` (默认 `factor_output_academic_v81`) 相对 CWD 落盘, 新运行建议显式设置到 `outputs/<tag>`
- **类名保留** `AcademicBacktester` / `AcademicDualEngine` 等历史命名未改动
- **PySR 可选** 未安装时静默跳过符号回归
- **Julia 环境** PySR 首次运行会在 `.venv/julia_env` 安装依赖，耗时较长
