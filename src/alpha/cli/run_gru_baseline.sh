#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VENV_PY="$REPO_ROOT/.venv/bin/python3"
# ============================================================
# GRU 黑盒基线运行脚本
# 用法:
#   bash cli/run_gru_baseline.sh
#   DATA=data/warmup/hs300_daily_2020-06-30_to_2026-06-30.parquet SEED=123 bash cli/run_gru_baseline.sh
# ============================================================


# 可通过环境变量覆盖的参数
DATA="${DATA:-data/warmup/csi500_daily_2020-06-30_to_2026-06-30.parquet}"
MARKET="${MARKET:-zs500}"
MODE="${MODE:-warmup}"
RUN_MODE="${RUN_MODE:-cross_sectional}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-200}"
HIDDEN="${HIDDEN:-64}"
SEQ_LEN="${SEQ_LEN:-20}"
LAYERS="${LAYERS:-1}"
DROPOUT="${DROPOUT:-0.1}"
LR="${LR:-1e-3}"
ES_PATIENCE="${ES_PATIENCE:-20}"
BATCH_SIZE="${BATCH_SIZE:-1024}"


# 激活虚拟环境
echo "=========================================="
echo "GRU Baseline"
echo "  data:     $DATA"
echo "  market:   $MARKET"
echo "  mode:     $MODE"
echo "  seed:     $SEED"
echo "  epochs:   $EPOCHS"
echo "  hidden:   $HIDDEN"
echo "  seq_len:  $SEQ_LEN"
echo "=========================================="

cd "$REPO_ROOT"

exec "$VENV_PY" -m alpha.experiments.transform_gru_baseline \
    --data "$DATA" \
    --market "$MARKET" \
    --mode "$MODE" \
    --run_mode "$RUN_MODE" \
    --seed "$SEED" \
    --epochs "$EPOCHS" \
    --hidden "$HIDDEN" \
    --seq-len "$SEQ_LEN" \
    --layers "$LAYERS" \
    --dropout "$DROPOUT" \
    --lr "$LR" \
    --es-patience "$ES_PATIENCE" \
    --batch-size "$BATCH_SIZE"
