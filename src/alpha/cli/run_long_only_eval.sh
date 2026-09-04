#!/usr/bin/env bash
# 单因子 long-only 回测 (仅做多 top20%, 不做空; L-S 仍作为参考字段)
# 用法:
#   ./src/alpha/cli/run_long_only_eval.sh <registry.json>
#   ./src/alpha/cli/run_long_only_eval.sh <registry.json> --compare-cost --cost-bps 30 --no-benchmark
# 环境变量: DATA / MARKET / MODE
#   例: DATA=data/csi500_daily_2021-06-30_to_2026-06-30.parquet \
#       ./src/alpha/cli/run_long_only_eval.sh factor_output_academic_v81/.../registry_academic.json
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VENV_PY="$REPO_ROOT/.venv/bin/python3"
DATA="${DATA:-$REPO_ROOT/data/warmup/csi500_daily_2020-06-30_to_2026-06-30.parquet}"
MARKET="${MARKET:-zs500}"
MODE="${MODE:-warmup}"

REGISTRY="${1:-}"; shift || true
if [ -z "$REGISTRY" ]; then
  echo "错误: 缺少 registry 参数。用法:"
  echo "  ./src/alpha/cli/run_long_only_eval.sh <registry.json> [选项...]"
  echo "  例: ./src/alpha/cli/run_long_only_eval.sh factor_output_academic_v81/alphagen_ppo_baseline_*/registry_academic.json"
  exit 1
fi

if [ ! -x "$VENV_PY" ]; then
  echo "错误: 未找到 venv ($VENV_PY)。请先创建 .venv 并安装依赖。"
  exit 1
fi

cd "$REPO_ROOT"
echo "===== 单因子 long-only 回测 ====="
echo "  registry: $REGISTRY"
echo "  data:     $DATA"
echo "  market:   $MARKET | mode: $MODE"
PYTHONPATH="$REPO_ROOT/src" exec "$VENV_PY" -m alpha.evaluation.single_factor_test \
  --registry "$REGISTRY" \
  --data "$DATA" \
  --long-only \
  --market "$MARKET" \
  --mode "$MODE" \
  "$@"