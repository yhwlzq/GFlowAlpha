#!/usr/bin/env bash
# 无多样性控制 (w/o Diversity Control) 消融: GFlowNet + MLQC 保留, 多样性控制关闭
# 用法: ./src/alpha/cli/run_ablation_no_diversity.sh [--trials N] [--time M] [--data PATH] [--market zs500] [--pool buildin] [--mode warmup]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VENV_PY="$REPO_ROOT/.venv/bin/python3"
DATA="${DATA:-$REPO_ROOT/data/warmup/csi500_daily_2020-06-30_to_2026-06-30.parquet}"

if [ ! -x "$VENV_PY" ]; then
  echo "错误: 未找到 venv ($VENV_PY)。请先创建 .venv 并安装依赖。"
  exit 1
fi

cd "$REPO_ROOT"
PYTHONPATH="$REPO_ROOT/src" exec "$VENV_PY" -m alpha.experiments.transform_pysr_ablation_no_diversity \
  --data "$DATA" "$@"
