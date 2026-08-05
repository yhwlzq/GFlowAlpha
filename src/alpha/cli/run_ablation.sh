#!/usr/bin/env bash
# 消融实验批量启动脚本
# 用法:
#   ./src/alpha/cli/run_ablation.sh <ablation号|all> [--trials N] [--time M]
#   例: ./src/alpha/cli/run_ablation.sh 1
#       ./src/alpha/cli/run_ablation.sh all --trials 30
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VENV_PY="$REPO_ROOT/.venv/bin/python3"
DATA="${DATA:-$REPO_ROOT/data/csi500_daily_2020-07-20_to_2026-07-19.parquet}"
ABLATION="${1:-all}"; shift || true

ABLATIONS=(
  1_no_residual
  2_no_monotonicity
  3_no_gflownet
  4_no_dynamic_diversity
  5_no_constraints
  6_no_mlqc
  7_no_mlqc_gates
  8_no_gflownet
  ga_no_gflownet
  gp_no_gflownet
)

cd "$REPO_ROOT"
run_one() {
  local name="$1"
  echo "===== ablation: $name ====="
  PYTHONPATH="$REPO_ROOT/src" "$VENV_PY" -m alpha.experiments.ablations.transform_pysr_ablation_${name} \
    --data "$DATA" "$@"
}

if [ "$ABLATION" = "all" ]; then
  for a in "${ABLATIONS[@]}"; do run_one "$a" "$@"; done
else
  run_one "$ABLATION" "$@"
fi
