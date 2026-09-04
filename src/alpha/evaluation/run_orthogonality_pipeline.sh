#!/usr/bin/env bash
# =============================================================================
# 因子正交性检验流水线 (全自动串联)
#
#   registry(公式) → build_factor_panel → factor_to_ls_returns
#                 → fetch_ff5_liq → orthogonality_check
#
# 用法:
#   bash src/alpha/evaluation/run_orthogonality_pipeline.sh \
#       [--data data/warmup/csi500_daily_2020-06-30_to_2026-06-30.parquet] \
#       [--registry factor_output_academic_v81/run_20260831_155911/registry_academic.json] \
#       [--fids ACAD_014 ACAD_019 ACAD_028 ACAD_038 ACAD_042] \
#       [--start-date 2025-07-01] [--end-date 2026-06-30] \
#       [--drop-liq] [--nw-lag 5] [--no-neutralize] \
#       [--extra-panel /path/to/backup_panel.parquet]
#   缺省参数: data=warmup csi500, registry=run_20260831_155911,
#             样本外区间 2025-07-01 ~ 2026-06-30,
#             中性化 = 仅 ln(amount) (Config.NEUTRALIZE_CONTROLS; 与挖矿门禁同口径) <-- 默认开
#             正交回归剔除 LIQ (FF5-only, --drop-liq 默认)
#
# 说明:
#   - --fids 指定提取的因子 id; 缺省则取 registry 全部因子
#   - --start-date/--end-date 为样本外(test)区间过滤; 传空串 (--start-date "") 可回到不限区间
#   - --drop-liq 时, 正交性回归剔除 LIQ (Amihud 因子量纲过小会毒化截距);
#     保留 FF5 五因子 (MKT/SMB/HML/RMW/CMA)
#   - 默认: 建 L-S 前逐截面 FWL 中性化 (仅 ln(amount), 与挖矿门禁同口径), 剥离市值暴露后测纯信号 alpha
#   - --neutralize-multi 仍有效, 但当前 Config.NEUTRALIZE_CONTROLS 默认仅 ["amount"],
#     若改为 ["amount","turn"] 则自动切为多风险口径; --no-neutralize: 退回裸因子 (不中性化)
#   - 默认 --drop-liq: 正交回归剔除 LIQ (Amihud 量纲异常会毒化截距; 流动性溢价视为 alpha 的一部分)
#   - --extra-panel: 从备份面板补合并 registry 缺失的目标因子列
#     (例如 registry 被清理后 ACAD_038 丢失, 可用含该列的旧面板恢复)
#   - 输出落在 factor_output_academic_v81/orthogonality_<时间戳>/
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PY="$REPO_ROOT/.venv/bin/python"

# ---- 默认参数 ----
DATA="$REPO_ROOT/data/warmup/csi500_daily_2020-06-30_to_2026-06-30.parquet"
REGISTRY="$REPO_ROOT/factor_output_academic_v81/run_20260831_155911/registry_academic.json"
FIDS=()
START_DATE="2025-07-01"
END_DATE="2026-06-30"
DROP_LIQ=1
NW_LAG=5
NEUTRALIZE=1
NEUTRALIZE_MULTI=1
EXTRA_PANEL=""

usage() {
    sed -n '1,25p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data) DATA="$2"; shift 2 ;;
        --registry) REGISTRY="$2"; shift 2 ;;
        --fids) shift; while [[ $# -gt 0 && "$1" != --* ]]; do FIDS+=("$1"); shift; done ;;
        --start-date) START_DATE="$2"; shift 2 ;;
        --end-date) END_DATE="$2"; shift 2 ;;
        --drop-liq) DROP_LIQ=1; shift ;;
        --neutralize) NEUTRALIZE=1; shift ;;
        --neutralize-multi) NEUTRALIZE_MULTI=1; shift ;;
        --no-neutralize) NEUTRALIZE_MULTI=0; NEUTRALIZE=0; shift ;;
        --extra-panel) EXTRA_PANEL="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "未知参数: $1" >&2; usage ;;
    esac
done

[[ -z "$DATA" ]] && { echo "错误: 缺少 --data" >&2; usage; }
[[ -z "$REGISTRY" ]] && { echo "错误: 缺少 --registry" >&2; usage; }
[[ -f "$DATA" ]] || { echo "错误: 数据文件不存在: $DATA" >&2; exit 1; }
[[ -f "$REGISTRY" ]] || { echo "错误: registry 不存在: $REGISTRY" >&2; exit 1; }

TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$REPO_ROOT/factor_output_academic_v81/orthogonality_${TS}"
mkdir -p "$OUT_DIR"

echo "=========================================================="
echo " 数据      : $DATA"
echo " Registry  : $REGISTRY"
echo " 因子      : ${FIDS[*]:-全部}"
echo " 区间过滤  : ${START_DATE:-不限} ~ ${END_DATE:-不限}"
echo " 剔除 LIQ  : $([ "$DROP_LIQ" -eq 1 ] && echo "是" || echo "否")"
echo " 中性化    : $([ "$NEUTRALIZE_MULTI" -eq 1 ] && echo "是(仅ln(amount), 默认)" || ([ "$NEUTRALIZE" -eq 1 ] && echo "是(仅ln(amount), --neutralize)" || echo "否(裸因子, --no-neutralize)"))"
echo " NW lag    : $NW_LAG"
echo " 额外面板  : ${EXTRA_PANEL:-无}"
echo " 输出目录  : $OUT_DIR"
echo "=========================================================="

# ---- Step 1: registry → 全样本因子面板 ----
PANEL="$OUT_DIR/factor_panel.parquet"

echo ""
echo "[1/4] 构建因子面板: registry → factor_panel.parquet"
"$PY" "$SCRIPT_DIR/build_factor_panel.py" \
    --data "$DATA" \
    --registry "$REGISTRY" \
    --output "$PANEL" \
    ${FIDS[*]:+--fids "${FIDS[@]}"} \
    $([ "$NEUTRALIZE_MULTI" -eq 1 ] && echo --neutralize-multi) \
    $([ "$NEUTRALIZE" -eq 1 ] && [ "$NEUTRALIZE_MULTI" -eq 0 ] && echo --neutralize)

# 若有备份面板, 补合并 registry 缺失的目标因子列
if [[ -n "$EXTRA_PANEL" ]]; then
    [[ -f "$EXTRA_PANEL" ]] || { echo "错误: 额外面板不存在: $EXTRA_PANEL" >&2; exit 1; }
    echo "[1b] 从 $EXTRA_PANEL 补合并缺失因子..."
    "$PY" -c "
import sys
import pandas as pd
panel = pd.read_parquet('$PANEL')
extra = pd.read_parquet('$EXTRA_PANEL')
fids = ['ACAD_014','ACAD_019','ACAD_028','ACAD_038','ACAD_042']
missing = [c for c in fids if c not in panel.columns]
if missing:
    print('补合并缺失因子:', missing)
    e = extra[['date','symbol']+missing].copy()
    e['date'] = pd.to_datetime(e['date'])
    e['symbol'] = e['symbol'].astype(str)
    panel = panel.copy()
    panel['date'] = pd.to_datetime(panel['date'])
    panel['symbol'] = panel['symbol'].astype(str)
    panel = panel.merge(e, on=['date','symbol'], how='left')
    out = ['date','symbol','ret'] + fids
    panel = panel[[c for c in out if c in panel.columns]].sort_values(['symbol','date']).reset_index(drop=True)
    panel.to_parquet('$PANEL')
    print('已更新面板:', list(panel.columns))
else:
    print('无缺失目标因子, 跳过补合并')
"
fi

# ---- Step 2: 下载 FF5 + 计算 LIQ ----
FF5="$OUT_DIR/ff5_liq.csv"
echo "[2/4] 下载 FF5 + 计算 Amihud LIQ: ... --> ff5_liq.csv"
"$PY" "$SCRIPT_DIR/fetch_ff5_liq.py" \
    --panel "$DATA" \
    --output "$FF5"

# ---- Step 3: 面板 → 样本外 test 段 L-S 收益 ----
LS="$OUT_DIR/ls_returns.csv"
echo "[3/4] 因子 → 样本外 L-S 收益: ... --> ls_returns.csv"
LS_ARGS=()
if [[ -n "$START_DATE" ]]; then LS_ARGS+=(--start-date "$START_DATE"); fi
if [[ -n "$END_DATE" ]]; then LS_ARGS+=(--end-date "$END_DATE"); fi
"$PY" "$SCRIPT_DIR/factor_to_ls_returns.py" \
    --panel "$PANEL" \
    --output "$LS" \
    "${LS_ARGS[@]}"

# ---- Step 4: 正交性检验 ----
ORTH="$OUT_DIR/orthogonality_result.csv"

# 若指定了 --fids, 正交性检验也按 registry 顺序只报告这些因子
ORTH_FIDS=()
if [[ ${#FIDS[@]} -gt 0 ]]; then
    ORTH_FIDS=(--fids "${FIDS[@]}")
fi

echo "[4/4] 正交性检验: ... --> orthogonality_result.csv"
ORTH_FLAGS=()
[[ "$DROP_LIQ" -eq 1 ]] && ORTH_FLAGS+=(--drop-liq)
"$PY" "$SCRIPT_DIR/orthogonality_check.py" \
    --factors "$LS" \
    --ff5 "$FF5" \
    --nw-lag "$NW_LAG" \
    "${ORTH_FLAGS[@]}" \
    ${ORTH_FIDS[@]:+${ORTH_FIDS[@]}} \
    --output "$ORTH"

echo ""
echo "完成。结果文件:"
echo "  $PANEL"
echo "  $FF5"
echo "  $LS"
echo "  $ORTH"