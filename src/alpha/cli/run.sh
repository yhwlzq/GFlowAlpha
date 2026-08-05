#!/usr/bin/env bash
# 主因子挖矿启动脚本
# 用法:
#   ./src/alpha/cli/run.sh                      # 默认数据
#   ./src/alpha/cli/run.sh --data data/csi500_daily_2020-06-30_to_2026-06-30.parquet --trials 60
#   OUTPUT_ROOT=/tmp/exp ./src/alpha/cli/run.sh # 重定向结果目录
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VENV_PY="$REPO_ROOT/.venv/bin/python3"
DATA="${DATA:-$REPO_ROOT/data/csi500_daily_2020-07-20_to_2026-07-19.parquet}"

if [ ! -x "$VENV_PY" ]; then
  echo "错误: 未找到 venv ($VENV_PY)。请先创建 .venv 并安装依赖。"
  exit 1
fi

cd "$REPO_ROOT"
PYTHONPATH="$REPO_ROOT/src" exec "$VENV_PY" -m alpha.experiments.transform_pysr_primary \
  --data "$DATA" "$@"
