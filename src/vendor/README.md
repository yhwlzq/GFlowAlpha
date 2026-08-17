# vendor/ — AlphaGen (PPO) 外部 RL 基线

本目录是 **AlphaSAGE** (github.com/BerkinChen/AlphaSAGE, MIT License) 的
**AlphaGen 方法** 选择性 vendored 代码 + 自研 qlib-free 适配层。仅取了训练/搜索
必需的最小子集, 统一评测仍走本仓库 `alpha.mining` 管线 (FWL + Fama-MacBeth +
registry)。

## 目录结构与来源

```
alphagen/                 ← 来自 upstream src/alphagen (未改动, 12 文件)
alphagen_qlib/            ← 自研 drop-in (零 qlib), 替代 upstream alphagen_qlib
ALPHASAGE_LICENSE         ← upstream MIT LICENSE
```

若与 upstream 同步, 重新检出:

```bash
git clone --depth 1 --filter=blob:none --sparse <AlphaSAGE> /tmp/alphasage
git -C /tmp/alphasage sparse-checkout set src/alphagen LICENSE
# 拷贝 src/alphagen 的 12 个文件到此目录(见下表), LICENSE → ALPHASAGE_LICENSE
```

> 注意: 本目录自 upstream 检出后**新增了各包目录的 `__init__.py`** (原 vendoring
> 漏掉了). `pyproject.toml` 通过
> `[tool.setuptools.packages.find] where = ["src", "src/vendor"]` 把 `alphagen` /
> `alphagen_qlib` 注册为可编辑安装的顶层包, 使 IDE 可直接解析 `from alphagen...`。
> 重新同步 upstream 时请保留这些 `__init__.py`。

### αlphagen 保留的文件 (upstream src/alphagen)

- `config.py` — 表达式语法/算子集/超参
- `data/expression.py`, `data/tokens.py`, `data/tree.py` — 表达式 AST 与解析
- `models/alpha_pool.py` — AlphaPool 表达式池 (IC reward + 扰动替换)
- `rl/env/core.py`, `rl/env/wrapper.py` — Gymnasium 环境
- `rl/policy.py` — LSTMSharedNet 策略特征提取器
- `utils/correlation.py`, `utils/pytorch_utils.py`, `utils/random.py`, `utils/__init__.py`

### 刻意未 vendored (无人引用, 已核 grep)

- `data/calculator.py`, `models/model.py`, `trade/base.py`, `trade/strategy.py`
- upstream `alphagen_qlib` (qlib 绑定) — 由本仓库自研适配层替代

## AlphaGen 数据约定 (重要)

- 张量布局 `data[T, n_features, n_stocks]`, 行=交易日(旧→新);
  前 `max_backtrack_days` 行为回溯暖身, 后 `max_future_days` 行为未来区。
- 训练 target 为前向 1 日收益 `Ref(close,-1)/close - 1` — `Ref` 求值会读到
  窗口末尾之后 1 行, 故 `max_future_days` 必须 ≥ 1 (驱动脚本 `_MAX_FUTURE = 1`)。
- 特征轴顺序固定对齐 `FeatureType`: OPEN/HIGH/LOW/CLOSE/VOLUME/VWAP;
  VWAP = amount/volume。

## 入口

`src/alpha/experiments/transform_alphagen_baseline.py`

```bash
python src/alpha/experiments/transform_alphagen_baseline.py \
    --data data/csi500_daily_...parquet --steps 20000 --pool 20
```

冒烟: `--steps 300 --pool 3 --time 3`