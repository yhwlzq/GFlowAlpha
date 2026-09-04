#!/usr/bin/env bash
# 评估工具启动脚本
# 用法:
#   ./src/alpha/cli/run_eval.sh single_factor --registry <path> [--id ACAD_xxx]
#   ./src/alpha/cli/run_eval.sh compare --registry <path>
#   ./src/alpha/cli/run_eval.sh stability --registry <path>
#   ./src/alpha/cli/run_eval.sh stability_rolling --registry <path>
#   ./src/alpha/cli/run_eval.sh heatmap --registry <path>
#   ./src/alpha/cli/run_eval.sh combo
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VENV_PY="$REPO_ROOT/.venv/bin/python3"
DATA="${DATA:-$REPO_ROOT/data/warmup/csi500_daily_2020-06-30_to_2026-06-30.parquet}"
TOOL="${1:-single_factor}"; shift || true

cd "$REPO_ROOT"
case "$TOOL" in
  single_factor)      MODULE="alpha.evaluation.single_factor_test" ;;
  compare)            MODULE="alpha.evaluation.compare_benchmark" ;;
  factor_compare)     MODULE="alpha.evaluation.factor_compare_1" ;;
  stability)          MODULE="alpha.evaluation.check_time_stability" ;;
  stability_rolling)  MODULE="alpha.evaluation.transform_pysr_stablity_check" ;;
  combo)              MODULE="alpha.evaluation.validate_factor_combo" ;;
  heatmap)            MODULE="alpha.analysis.plot_factor_correlation_heatmap" ;;
  diagnose)           MODULE="alpha.analysis.diagnose_single_features" ;;
  *) echo "未知工具: $TOOL"; exit 1 ;;
esac

PYTHONPATH="$REPO_ROOT/src" exec "$VENV_PY" -m "$MODULE" --data "$DATA" "$@"
