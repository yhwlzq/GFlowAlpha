#!/usr/bin/env bash
# 无多样性控制 (w/o Diversity Control) 消融: GFlowNet + MLQC 保留, 多样性控制关闭
# 用法: ./src/alpha/cli/run_ablation_no_diversity.sh [--trials N] [--time M] [--data PATH]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VENV_PY="$REPO_ROOT/.venv/bin/python"

DATA="${DATA:-$REPO_ROOT/data/csi500_daily_2021-06-30_to_2026-06-30.parquet}"
TRIALS=50
TIME=80

while [[ $# -gt 0 ]]; do
  case "$1" in
    --trials) TRIALS="$2"; shift 2 ;;
    --time)   TIME="$2";   shift 2 ;;
    --data)   DATA="$2";   shift 2 ;;
    *) echo "未知参数: $1"; exit 1 ;;
  esac
done

cd "$REPO_ROOT"
PYTHONPATH="$REPO_ROOT/src" "$VENV_PY" -m alpha.experiments.transform_pysr_ablation_no_diversity \
  --data "$DATA" \
  --trials "$TRIALS" \
  --time "$TIME"