#!/usr/bin/env bash
# Walk-Forward 滚动窗口实验启动脚本 (36:12:12 月 = 3年训练/1年验证/1年测试)
# 默认数据为两份中证500 parquet (2016~2021 + 2021~2026), 自动拼接去重得到 w1~w6.
# 用法:
#   ./src/alpha/cli/run_walkforward.sh                              # 默认数据 + 默认参数
#   ./src/alpha/cli/run_walkforward.sh --list-windows               # 只查看各窗口时间段
#   ./src/alpha/cli/run_walkforward.sh --trials 60 --time 80        # 调参与时间预算
#   ./src/alpha/cli/run_walkforward.sh --market hs300 --max-windows 2  # 冒烟测试
#   ./src/alpha/cli/run_walkforward.sh --windows 1,3                # 只运行 w1 和 w3
#   DATA=data/hs300_daily_2021-06-30_to_2026-06-30.parquet ./src/alpha/cli/run_walkforward.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VENV_PY="$REPO_ROOT/.venv/bin/python3"
DATA="${DATA:-$REPO_ROOT/data/csi500_daily_2016-06-30_to_2021-06-30.parquet,$REPO_ROOT/data/csi500_daily_2021-06-30_to_2026-06-30.parquet}"

if [ ! -x "$VENV_PY" ]; then
  echo "错误: 未找到 venv ($VENV_PY)。请先创建 .venv 并安装依赖。"
  exit 1
fi

cd "$REPO_ROOT"
PYTHONPATH="$REPO_ROOT/src" exec "$VENV_PY" -m alpha.experiments.transform_pysr_walkforward \
  --data "$DATA" "$@"
