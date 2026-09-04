#!/usr/bin/env bash
# 消融实验批量启动脚本
# 用法:
#   ./src/alpha/cli/run_ablation.sh <ablation号|all> [--trials N] [--time M]
#   例: ./src/alpha/cli/run_ablation.sh 6
#       ./src/alpha/cli/run_ablation.sh all --trials 30
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VENV_PY="$REPO_ROOT/.venv/bin/python3"
DATA="${DATA:-$REPO_ROOT/data/warmup/csi500_daily_2020-06-30_to_2026-06-30.parquet}"
ABLATION="${1:-all}"; shift || true

ABLATIONS=(
  #9_random_no_mlqc
  8_no_gflownet
)

cd "$REPO_ROOT"
run_one() {
  local name="$1"; shift
  # 数字前缀: transform_pysr_ablation<num>_... ; gp: transform_pysr_ablation_gp_...
  if [[ "$name" == gp_* ]]; then
    local mod="alpha.experiments.transform_pysr_ablation_${name}"
  else
    local mod="alpha.experiments.transform_pysr_ablation${name}"
  fi
  echo "===== ablation: $name ====="
  PYTHONPATH="$REPO_ROOT/src" "$VENV_PY" -m "$mod" \
    --data "$DATA" "$@"
}

if [ "$ABLATION" = "all" ]; then
  for a in "${ABLATIONS[@]}"; do run_one "$a" "$@"; done
else
  run_one "$ABLATION" "$@"
fi
