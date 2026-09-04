#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 gp_baseline_all_factors.json / gp_baseline_results.json 中的公式
从 X{n} 索引翻译为真实特征名。

两种模式:
  1. 默认: 使用硬编码的初始 75 特征映射（说明见下）
  2. --load: 重新运行 DataPreprocessor 获得聚类后的精确映射（需原始数据路径）
"""

import json
import re
import os
import sys
import argparse

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ──────────────────────────────────────────────────────────
#  75 个初始特征（按 feature_registry.py 构建顺序）
# ──────────────────────────────────────────────────────────
INITIAL_FEATURES = [
    "ret_1d",          # 0
    "rev_1d",          # 1
    "ret_2d",          # 2
    "rev_2d",          # 3
    "ret_3d",          # 4
    "rev_3d",          # 5
    "ret_5d",          # 6
    "rev_5d",          # 7
    "ret_10d",         # 8
    "rev_10d",         # 9
    "ret_20d",         # 10
    "rev_20d",         # 11
    "ret_60d",         # 12
    "rev_60d",         # 13
    "mom_smooth_20d",  # 14
    "info_discrete_20d", # 15
    "vol_5d",          # 16
    "vol_10d",         # 17
    "vol_20d",         # 18
    "vol_60d",         # 19
    "skew_20d",        # 20
    "kurt_20d",        # 21
    "vol_skew_ratio",  # 22
    "max_ret_20d",     # 23
    "min_ret_20d",     # 24
    "extreme_vol_20d", # 25
    "parkinson_vol_20d", # 26
    "ivol_20d",        # 27
    "amihud_5d",       # 28
    "amihud_20d",      # 29
    "turnover_5d",     # 30
    "turnover_20d",    # 31
    "turnover_60d",    # 32
    "turnover_std_20d", # 33
    "illiq_spread",    # 34
    "volume_shock_5d", # 35
    "amount_shock_5d", # 36
    "liquidity_var",   # 37
    "clv",             # 38
    "upper_shadow",    # 39
    "lower_shadow",    # 40
    "body_ratio",      # 41
    "gap",             # 42
    "smart_money_10d", # 43
    "smart_money_20d", # 44
    "vwap_dev_5d",     # 45
    "vwap_dev_20d",    # 46
    "intraday_mom",    # 47
    "price_accel",     # 48
    "pv_corr_5d",      # 49
    "logpv_corr_5d",   # 50
    "pv_corr_10d",     # 51
    "logpv_corr_10d",  # 52
    "pv_corr_20d",     # 53
    "logpv_corr_20d",  # 54
    "obv_slope_10d",   # 55
    "money_flow_10d",  # 56
    "money_flow_20d",  # 57
    "pv_divergence",   # 58
    "bias_5d",         # 59
    "ma_cross_5d",     # 60
    "bias_10d",        # 61
    "ma_cross_10d",    # 62
    "bias_20d",        # 63
    "ma_cross_20d",    # 64
    "bias_60d",        # 65
    "ma_cross_60d",    # 66
    "rsi_6",           # 67
    "rsi_14",          # 68
    "rsi_24",          # 69
    "macd_hist",       # 70
    "macd_signal",     # 71
    "boll_width_20d",  # 72
]


def decode_formula(formula: str, feature_map: list) -> str:
    def _replace(m):
        idx = int(m.group(1))
        return feature_map[idx] if idx < len(feature_map) else m.group(0)
    return re.sub(r'X(\d+)', _replace, formula)


def _load_json(path: str) -> list:
    with open(path) as f:
        data = json.load(f)
    factors = data.get('factors', data.get('valid_factors', []))
    if not factors:
        print(f"错误: 未找到因子列表 (keys: {list(data.keys())})")
        sys.exit(1)
    return factors


def _dedup(factors: list) -> list:
    seen = set()
    out = []
    for f in factors:
        fm = f['formula']
        if fm not in seen:
            seen.add(fm)
            out.append(f)
    return out


def _print_feature_usage(factors: list, feature_map: list):
    used = set()
    for f in factors:
        for m in re.finditer(r'X(\d+)', f['formula']):
            used.add(int(m.group(1)))
    used = sorted(used)
    print(f"公式中使用到的特征索引 ({len(used)} 个):")
    for idx in used:
        name = feature_map[idx] if idx < len(feature_map) else "?"
        print(f"  X{idx:<3d} → {name}")
    print()


def _trunc(s: str, width: int = 60) -> str:
    return s if len(s) <= width else s[:width] + "..."

def _print_factors(factors: list, feature_map: list):
    unique = _dedup(factors)
    header = (f"{'因子 ID':>8s} | {'FM t-stat':>8s} | {'Rank IC':>7s} | {'ICIR':>6s} | "
              f"{'多空差(L-S)':>10s} | {'年化收益':>8s} | {'月Sharpe':>8s} | {'解码公式':>60s}")
    sep = "─" * 120

    def _row(f):
        q = f.get('quantile_monthly_rets', {})
        spread = q.get('Q5', 0) - q.get('Q1', 0) if q else 0
        sharpe = f.get('sharpe_ratio', None)
        sharpe_s = f"{sharpe:.3f}" if sharpe is not None else "N/A"
        decoded = decode_formula(f['formula'], feature_map)
        return (f"{f['id']:>8s} | {f['t_stat']:>8.3f} | {f['rank_ic']:>7.4f} | "
                f"{f['icir']:>6.4f} | {spread:>10.6f} | {f.get('annualized_return', 0)*100:>7.2f}% | "
                f"{sharpe_s:>8s} | {_trunc(decoded, 60):>60s}")

    # ── 表1: 去重因子 ──
    print(sep)
    print("  表1 — 去重因子（唯一公式）")
    print(sep)
    print(header)
    print(sep)
    for f in unique:
        print(_row(f))
    print(sep)
    print()
    for f in unique:
        decoded = decode_formula(f['formula'], feature_map)
        print(f"{f['id']} 完整公式:")
        print(f"  {decoded}")
        print()

    # ── 表2: 全部因子 ──
    print(sep)
    print(f"  表2 — 全部因子（含重复，共{len(factors)}条）")
    print(sep)
    print(header)
    print(sep)
    for f in factors:
        print(_row(f))
    print(sep)
    print()
    for f in unique:
        decoded = decode_formula(f['formula'], feature_map)
        print(f"{f['id']} 完整公式:")
        print(f"  {decoded}")
        print()


def main():
    parser = argparse.ArgumentParser(description='解码 GP 公式中的 X{n} 为特征名')
    parser.add_argument('json', nargs='?',
                        default=os.path.join(_SCRIPT_DIR, 'gp_baseline_all_factors.json'),
                        help='gp_baseline 结果 JSON 路径')
    parser.add_argument('--data', help='原始数据路径 (parquet/csv)，传入后自动运行 preprocessor 获得精确映射')
    args = parser.parse_args()

    if not os.path.exists(args.json):
        print(f"文件不存在: {args.json}")
        sys.exit(1)

    factors = _load_json(args.json)
    unique_count = len(set(f['formula'] for f in factors))

    if args.data:
        print("加载数据并运行 preprocessor 获取聚类后精确特征映射...")
        sys.path.insert(0, _SCRIPT_DIR)
        from alpha.mining.preprocessor import DataPreprocessor
        from alpha.data.data_loader import CSI500Loader
        from alpha.config import set_global_seed
        set_global_seed()
        loader = CSI500Loader(path=args.data)
        df_raw = loader.load()
        prep = DataPreprocessor()
        prep.prepare_full_pool(df_raw)
        feature_map = prep.valid_features
        source_label = "聚类后精确映射"
    else:
        feature_map = INITIAL_FEATURES
        source_label = "初始构建顺序映射"
        print("=" * 80)
        print("注意: 使用初始75特征映射（未自动运行 preprocessor）")
        print("      full_X 经聚类重排后列序与初始顺序不同。")
        print("      如需精确结果，添加 --data <数据路径>")
        print("=" * 80)
        print()

    print(f"文件: {args.json}")
    print(f"因子总数: {len(factors)}")
    print(f"唯一公式数: {unique_count}")
    print(f"特征映射来源: {source_label}")
    print()

    _print_feature_usage(factors, feature_map)
    _print_factors(factors, feature_map)


if __name__ == '__main__':
    main()
