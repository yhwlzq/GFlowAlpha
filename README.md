# GFlowAlpha

高频 (日频) 量化因子挖掘系统。基于 **GFlowNet + PySR 符号回归** 从 OHLCV 微观数据中发现 alpha 因子，配套 Fama-MacBeth 评估、单因子回测与稳定性检验。

## 快速开始

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[pysr,gp,dev]"

# 冒烟
./src/alpha/cli/run.sh --debug

# 正式挖矿
./src/alpha/cli/run.sh --data data/csi500_daily_2020-07-20_to_2026-07-19.parquet
```

详细文档见 [docs/run_guide.md](docs/run_guide.md) 与 [docs/architecture.md](docs/architecture.md)。

## 目录

- `src/alpha/` — 主包 (数据/特征/挖掘/评估/实验/cli)
- `data/` — 行情数据 (parquet, gitignored)
- `outputs/` — 统一结果目录 (含 `legacy/` 历史归档)
- `docs/`、`tests/` — 文档与测试
