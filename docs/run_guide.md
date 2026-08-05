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
# 调试冒烟 (3 只股票 × 60 日)
./src/alpha/cli/run.sh --debug

# 正式运行 (默认数据, 60 trials)
./src/alpha/cli/run.sh

# 指定数据 / 参数
./src/alpha/cli/run.sh --data data/csi500_daily_2020-06-30_to_2026-06-30.parquet --trials 100 --time 120

# 沪深300
./src/alpha/cli/run.sh --market hs300 --data data/hs300_daily_2021-06-30_to_2026-06-30.parquet

# 重定向结果目录
OUTPUT_ROOT=/tmp/exp ./src/alpha/cli/run.sh
```

结果写入 `outputs/run_<时间戳>/`，含 `registry_academic.json` / `mining.log` / 回测图。

## 消融实验

```bash
./src/alpha/cli/run_ablation.sh 1          # 单个: 无双残差
./src/alpha/cli/run_ablation.sh all        # 全部 10 个
./src/alpha/cli/run_ablation.sh 5 --trials 30
```

## 基线

```bash
./src/alpha/cli/run_baseline.sh gp
```

## 评估

```bash
# 单因子评估 (从注册表)
./src/alpha/cli/run_eval.sh single_factor --registry outputs/run_xxx/registry_academic.json

# 对比 GFN-SR vs 传统
./src/alpha/cli/run_eval.sh compare --registry outputs/run_xxx/registry_academic.json

# 分时段稳定性 / 滚动窗口稳定性
./src/alpha/cli/run_eval.sh stability --registry outputs/run_xxx/registry_academic.json
./src/alpha/cli/run_eval.sh stability_rolling --registry outputs/run_xxx/registry_academic.json

# 热力图 / 因子组合 / 诊断
./src/alpha/cli/run_eval.sh heatmap --registry outputs/run_xxx/registry_academic.json
./src/alpha/cli/run_eval.sh combo
./src/alpha/cli/run_eval.sh diagnose
```

## 测试

```bash
pytest tests/
```

## 数据说明

`data/` 存放 parquet (gitignored)，默认引用 `csi500_daily_2020-07-20_to_2026-07-19.parquet`。
可用 `data_fetcher.py` 从 baostock 抓取新数据。
