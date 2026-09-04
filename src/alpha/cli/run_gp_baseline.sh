#!/usr/bin/env bash
# GP 基线启动脚本 (gplearn 遗传规划因子挖掘)
# 用法:
#   ./src/alpha/cli/run_gp_baseline.sh [--data ...] [--generations N] [--population N] [--jobs N] [--no-fwl]
#   DATA=... GENERATIONS=20 POPULATION=300 JOBS=2 ./src/alpha/cli/run_gp_baseline.sh
# 说明: 默认 FWL 中性化 + FM 回归 (与 GFlowAlpha 同一测试集评估口径);
#       如需 raw 对比, 加 --no-fwl。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VENV_PY="$REPO_ROOT/.venv/bin/python3"
DATA="${DATA:-$REPO_ROOT/data/warmup/csi500_daily_2020-06-30_to_2026-06-30.parquet}"
JOBS="${JOBS:-2}"

if [ ! -x "$VENV_PY" ]; then
  echo "错误: 未找到 venv ($VENV_PY)。请先创建 .venv 并安装依赖。"
  exit 1
fi

cd "$REPO_ROOT"
PYTHONPATH="$REPO_ROOT/src" exec "$VENV_PY" -m alpha.experiments.transform_gp_baseline --data "$DATA" --jobs "$JOBS" "$@"