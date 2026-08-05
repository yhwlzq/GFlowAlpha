#!/usr/bin/env bash
# GP 基线启动脚本
# 用法:
#   ./src/alpha/cli/run_baseline.sh gp [--data ...] [--generations N] [--population N] [--time-limit M]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VENV_PY="$REPO_ROOT/.venv/bin/python3"
DATA="${DATA:-$REPO_ROOT/data/csi500_daily_2020-07-20_to_2026-07-19.parquet}"
BASELINE="${1:-gp}"; shift || true

cd "$REPO_ROOT"
case "$BASELINE" in
  gp) MODULE="alpha.experiments.baselines.transform_gp_baseline" ;;
  *) echo "未知基线: $BASELINE (可用: gp)"; exit 1 ;;
esac

PYTHONPATH="$REPO_ROOT/src" exec "$VENV_PY" -m "$MODULE" --data "$DATA" "$@"
