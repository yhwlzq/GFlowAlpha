#!/usr/bin/env bash
# cold 冷启动逐窗口脚本: 依次运行 data/qfq/ 下的 5 年数据文件 (07-01 锚定 36:12:12)
# 用法:
#   ./src/alpha/cli/run_cold_window.sh                     # 默认 data/qfq, 默认试数/时长
#   ./src/alpha/cli/run_cold_window.sh --trials 60 --time 180
#   COLD_DIR=/path/to/dir TRIALS=60 TIME=180 ./src/alpha/cli/run_cold_window.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VENV_PY="$REPO_ROOT/.venv/bin/python3"
DATA_DIR="${COLD_DIR:-$REPO_ROOT/data}"
TRIALS="${TRIALS:-50}"
TIME="${TIME:-120}"
POOL="${POOL:-buildin}"
MARKET="${MARKET:-zs500}"

if [ ! -x "$VENV_PY" ]; then
  echo "错误: 未找到 venv ($VENV_PY)。请先创建 .venv 并安装依赖。"
  exit 1
fi

shopt -s nullglob
FILES=("$DATA_DIR"/csi500_daily_*.parquet)
if [ ${#FILES[@]} -eq 0 ]; then
  echo "错误: 数据目录没有 parquet 文件: $DATA_DIR"
  exit 1
fi
shopt -u nullglob

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="$REPO_ROOT/outputs/run_cold_${STAMP}.log"
mkdir -p "$(dirname "$RUN_LOG")"

cd "$REPO_ROOT"
echo "== cold 冷启动逐窗口挖掘 | 目录: $DATA_DIR | $TRIALS 轮 / $TIME 分钟/窗 | 日志: $RUN_LOG" | tee -a "$RUN_LOG"

for f in "${FILES[@]}"; do
  echo "" | tee -a "$RUN_LOG"
  echo "### [$(date '+%F %T')] 窗口: ${f##*/}" | tee -a "$RUN_LOG"
  PYTHONPATH="$REPO_ROOT/src" "$VENV_PY" -m alpha.experiments.transform_pysr_primary \
      --data "$f" --mode cold --trials "$TRIALS" --time "$TIME" --pool "$POOL" --market "$MARKET" \
      2>&1 | tee -a "$RUN_LOG"
  echo "### [$(date '+%F %T')] 完成: ${f##*/} (exit=${PIPESTATUS[0]})" | tee -a "$RUN_LOG"
done

echo "" | tee -a "$RUN_LOG"
echo "== 全部 cold 窗口完成 | 共 ${#FILES[@]} 个 | 日志: $RUN_LOG" | tee -a "$RUN_LOG"