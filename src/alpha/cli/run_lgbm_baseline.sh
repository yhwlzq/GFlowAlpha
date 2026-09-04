#!/usr/bin/env bash
# LightGBM 黑盒基线启动脚本 (与符号搜索同特征/同测试集, FWL+FM 评估)
# 用法:
#   ./src/alpha/cli/run_lgbm_baseline.sh [--data ...] [--estimators N] [--lr L]
#       [--leaves N] [--seeds S] [--compare tag:path,...]
#   例: ./src/alpha/cli/run_lgbm_baseline.sh --seeds 3 --compare primary:factor_output_academic_v81/run_xxx/registry_academic.json
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VENV_PY="$REPO_ROOT/.venv/bin/python3"
DATA="${DATA:-$REPO_ROOT/data/warmup/csi500_daily_2020-06-30_to_2026-06-30.parquet}"

cd "$REPO_ROOT"
PYTHONPATH="$REPO_ROOT/src" exec "$VENV_PY" -m alpha.experiments.transform_lgbm_baseline \
  --data "$DATA" "$@"
