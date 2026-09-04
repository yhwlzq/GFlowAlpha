#!/usr/bin/env bash
# AlphaQCM (IQN+QCM) 外部 RL 基线启动脚本
# 用法:
#   ./src/alpha/cli/run_alphaqcm_baseline.sh [full|smoke] [--data ...] [--steps N] [--pool N] [--time M]
#   - full : 正式批次 (默认; --steps 20000 --pool 20)
#   - smoke: 冒烟快速检验 (--steps 300 --pool 3 --time 3)
#   其它参数 (--seed/--market/--compare/--std-lam/--skew-lam/--kurt-lam ...) 直接透传给驱动脚本。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VENV_PY="$REPO_ROOT/.venv/bin/python3"
DATA="${DATA:-$REPO_ROOT/data/warmup/csi500_daily_2020-06-30_to_2026-06-30.parquet}"
MODE="${1:-full}"; shift || true

MODULE="alpha.experiments.transform_alphaqcm_baseline"

cd "$REPO_ROOT"
case "$MODE" in
  full)  EXTRA=(--steps 20000 --pool 20) ;;
  smoke) EXTRA=(--steps 300 --pool 3 --time 3) ;;
  -*|*)  echo "未知预设: $MODE (可用: full, smoke)"; exit 1 ;;
esac

echo "===== AlphaQCM (IQN+QCM) 基线 [$MODE] ====="
PYTHONPATH="$REPO_ROOT/src" exec "$VENV_PY" -m "$MODULE" \
  --data "$DATA" "${EXTRA[@]}" "$@"
