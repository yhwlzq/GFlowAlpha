# GFlowAlpha

高频 (日频) 量化因子挖掘系统。基于 **GFlowNet + PySR 符号回归** 从 OHLCV 微观数据中发现 alpha 因子，配套 Fama-MacBeth 评估、单因子回测与稳定性检验。

## 快速开始

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[pysr,gp,dev]"

# 冒烟 (少量 trials, 快速验证)
python src/alpha/research/transform_pysr_primary.py --data data/csi500_daily_2020-07-20_to_2026-07-19.parquet --trials 3 --time 10

# 正式挖矿
python src/alpha/research/transform_pysr_primary.py --data data/csi500_daily_2020-07-20_to_2026-07-19.parquet
```

详细文档见 [docs/run_guide.md](docs/run_guide.md) 与 [docs/architecture.md](docs/architecture.md)。

## 目录

- `src/alpha/` — 主包
  - `research/` — GFlowNet + PySR 挖矿管线 (核心入口 `transform_pysr_primary.py`)
  - `agents/` — 早期 agent 原型 (PySR-only)
  - `data/` — 数据抓取 (`data_fetcher.py`)
- `data/` — 行情数据 (parquet, gitignored)
- `outputs/` — 统一实验输出根目录 (`academic_v81/`、`pysr/`、`legacy/` 为历史归档)
- `docs/` — 文档
