#!/usr/bin/env bash
# 主因子挖矿启动脚本
# 用法:
#   ./src/alpha/cli/run.sh                      # 默认数据, buildin 池
#   ./src/alpha/cli/run.sh --data data/csi500_daily_2020-06-30_to_2026-06-30.parquet --trials 60
#   ./src/alpha/cli/run.sh --pool alpha158      # 使用 Qlib Alpha158 工程化因子池
#   POOL=alpha158 OUTPUT_ROOT=/tmp/exp ./src/alpha/cli/run.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VENV_PY="$REPO_ROOT/.venv/bin/python3"
DATA="${DATA:-$REPO_ROOT/data/warmup/csi500_daily_2020-06-30_to_2026-06-30.parquet}"
POOL="${POOL:-buildin}"

ARGS=(--data "$DATA")
if [ -n "$POOL" ]; then
  ARGS+=(--pool "$POOL")
fi

if [ ! -x "$VENV_PY" ]; then
  echo "错误: 未找到 venv ($VENV_PY)。请先创建 .venv 并安装依赖。"
  exit 1
fi

cd "$REPO_ROOT"
PYTHONPATH="$REPO_ROOT/src" exec "$VENV_PY" -m alpha.experiments.transform_pysr_primary \
  "${ARGS[@]}" "$@"
