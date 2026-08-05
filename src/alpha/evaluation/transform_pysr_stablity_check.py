#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
滚动窗口时间稳定性检验工具 (Rolling-Window ICIR Stability).

对注册表中每个因子，在测试集上按【滚动窗口】计算各子窗口的截面 Rank ICIR，
并汇总滚动均值 (Mean ICIR) 与方向胜率 (Hit Rate: ICIR>0)。

窗口划分: 每窗 win_months 个自然月, 步长 stride_months 个月 (默认 4/2, 50% 重叠).
示例: W1 (25.07-25.10) W2 (25.09-25.12) W3 (25.11-26.02) ...

用法:
  python transform_pysr_stablity_check.py --registry path/to/registry_academic.json
  python transform_pysr_stablity_check.py --registry ... --win-months 4 --stride-months 2
  python transform_pysr_stablity_check.py --registry ... --ids ACAD_014 ACAD_019
"""
import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


from alpha.config import REPO_ROOT, Config, set_global_seed
from alpha.data.data_loader import CSI500Loader
from alpha.mining.preprocessor import DataPreprocessor
from alpha.mining.safe_ops import SafeOps
from alpha.mining.neutralize import fwl_neutralize

_REPO_ROOT = REPO_ROOT
_LEGACY_REGISTRY = os.path.join(_REPO_ROOT, 'outputs', 'legacy', 'root_factor_output_academic_v81', 'run_20260720_171219', 'registry_academic.json')

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger('TimeStability')

SAFE_MATH = {
    'abs': np.abs, 'square': np.square, 'sqrt': SafeOps.safe_sqrt,
    'sign': np.sign, 'min': np.minimum, 'max': np.maximum,
    'tanh': SafeOps.safe_tanh, 'log1p': SafeOps.safe_log1p,
    'inv': SafeOps.safe_inv, 'sigmoid': SafeOps.safe_sigmoid,
    'div': SafeOps.protected_div, '/': SafeOps.protected_div,
}


def load_registry(path: str, ids: Optional[List[str]] = None) -> List[Tuple[str, str, List[str]]]:
    with open(path) as f:
        registry = json.load(f)
    factors = registry.get('factors', {})
    items = []
    for fid, info in factors.items():
        if ids and fid not in ids:
            continue
        m = info.get('metrics', info)
        formula = m.get('formula', '')
        feats = m.get('features', info.get('features', []))
        if formula and feats:
            items.append((fid, formula, feats))
    return items


def build_rolling_windows(unique_days: np.ndarray,
                          win_months: int = 4,
                          stride_months: int = 2) -> Tuple[List[np.ndarray], List[str]]:
    """按自然月构建滚动窗口 (含 50% 重叠).

    Returns:
        windows: List[np.ndarray], 每个窗口包含的交易日数组
        labels:  List[str], 如 'W1 (25.07-25.10)'
    """
    first_day = pd.Timestamp(unique_days[0])
    last_day = pd.Timestamp(unique_days[-1])
    first_month = first_day.to_period('M')
    last_month = last_day.to_period('M')

    windows, labels = [], []
    period = first_month
    idx = 0
    while True:
        win_start = period
        win_end = period + (win_months - 1)
        if win_end > last_month:
            break
        start_ts = win_start.start_time
        end_ts = win_end.end_time

        mask = (unique_days >= start_ts) & (unique_days <= end_ts)
        win_days = unique_days[mask]
        if len(win_days) == 0:
            break
        windows.append(win_days)
        s = win_start.strftime('%y.%m')
        e = win_end.strftime('%y.%m')
        labels.append(f'W{idx + 1} ({s}-{e})')
        idx += 1
        period = period + stride_months
    return windows, labels


def compute_window_icir(pred: np.ndarray, ret: np.ndarray, dates: np.ndarray,
                        window_dates: np.ndarray) -> Optional[float]:
    """单子窗口内计算截面 Rank ICIR.  无效时返回 None."""
    mask = np.isin(dates, window_dates)
    p, r = pred[mask], ret[mask]
    if len(p) < 200:
        return None
    unique_days_in = np.unique(dates[mask])
    ics = []
    for day in unique_days_in:
        dm = dates == day
        p_d, r_d = pred[dm], ret[dm]
        valid = np.isfinite(p_d) & np.isfinite(r_d)
        if valid.sum() < 30:
            continue
        ic = spearmanr(p_d[valid], r_d[valid])[0]
        if np.isfinite(ic):
            ics.append(ic)
    if len(ics) < 5:
        return None
    return float(np.mean(ics) / (np.std(ics, ddof=1) + 1e-8))


def build_window_table(prep: DataPreprocessor,
                       factors: List[Tuple[str, str, List[str]]],
                       win_months: int = 4,
                       stride_months: int = 2,
                       hit_threshold: float = 0.0) -> pd.DataFrame:
    test_mask = prep.te_mask
    test_dates = prep.full_dates[test_mask]
    test_ret = prep.full_y[test_mask]

    unique_days = np.sort(np.unique(test_dates))
    n_days = len(unique_days)
    if n_days < 200:
        logger.error(f"测试集仅 {n_days} 个交易日，不足以滚动分窗")
        return pd.DataFrame()

    windows, window_labels = build_rolling_windows(
        unique_days, win_months=win_months, stride_months=stride_months
    )
    n_windows = len(windows)
    if n_windows < 2:
        logger.error(f"滚动窗口仅 {n_windows} 个，无法做稳定性检验")
        return pd.DataFrame()

    full_label = (f"{pd.Timestamp(unique_days[0]).strftime('%y.%m')}-"
                  f"{pd.Timestamp(unique_days[-1]).strftime('%y.%m')}")
    logger.info(f"测试集 {full_label} | 窗口数: {n_windows} | "
                f"win={win_months}m stride={stride_months}m")

    rows = []
    for fid, formula, feats in factors:
        try:
            idx = [prep.feature_to_idx[f] for f in feats]
        except KeyError as e:
            logger.warning(f"  {fid}: 特征缺失 {e}")
            continue

        X_all = prep.full_X[:, idx]
        ns = {f: X_all[:, i] for i, f in enumerate(feats)}
        ns.update(SAFE_MATH)
        try:
            pred_all = eval(formula, {"__builtins__": {}}, ns)
        except Exception as e:
            logger.warning(f"  {fid}: eval 失败 ({e})")
            continue
        pred_all = np.where(np.isfinite(pred_all), pred_all, np.nan)

        pred_te = pred_all[test_mask]
        valid = np.isfinite(pred_te) & np.isfinite(test_ret)
        if valid.sum() < 500:
            logger.warning(f"  {fid}: 测试集有效样本不足 ({valid.sum()})")
            continue

        # FWL 中性化: 截面回归剔除 ln(amount) 暴露
        test_amount = prep.full_amount[test_mask]
        test_symbols = prep.full_symbols[test_mask]
        pure_pred_te, pure_ret_te = fwl_neutralize(
            pred_te[valid], test_ret[valid],
            test_dates[valid], test_symbols[valid],
            test_amount[valid]
        )
        pure_dates = test_dates[valid]

        row = {'因子': fid}
        ics = []
        for wi, w_days in enumerate(windows):
            icir = compute_window_icir(pure_pred_te, pure_ret_te, pure_dates, w_days)
            col = window_labels[wi]
            row[col] = icir if icir is not None else np.nan
            if icir is not None:
                ics.append(icir)

        valid_ics = [v for v in ics if np.isfinite(v)]
        row['滚动均值 (Mean ICIR)'] = (
            float(np.mean(valid_ics)) if valid_ics else np.nan
        )

        if len(valid_ics) >= 1:
            n_hit = sum(1 for v in valid_ics if v > hit_threshold)
            row['方向胜率 (Hit Rate: ICIR>0)'] = f'{n_hit}/{len(valid_ics)}'
        else:
            row['方向胜率 (Hit Rate: ICIR>0)'] = '—'

        full_icir = compute_window_icir(
            pure_pred_te, pure_ret_te, pure_dates, unique_days
        )
        row[f'全样本 ({full_label})'] = full_icir if full_icir is not None else np.nan

        rows.append(row)
        logger.info(f"  {fid}: {dict(row)}")

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df


def print_table(df: pd.DataFrame):
    if df.empty:
        return
    print("\n" + "=" * 120)
    print("  因子时间稳定性检验 (测试集滚动窗口 ICIR)")
    print("=" * 120)
    cols = [c for c in df.columns if c != '因子']
    fmt = "  {:<12}" + "".join("{:<22}" for _ in cols)
    header_labels = ['因子'] + cols
    print(fmt.format(*header_labels))
    print("  " + "-" * (12 + 22 * len(cols)))
    for _, row in df.iterrows():
        vals = [str(row['因子'])]
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                vals.append(f"{v:.3f}" if np.isfinite(v) else "  N/A")
            else:
                vals.append(f"  {v}")
        print(fmt.format(*vals))
    print("=" * 120)


def main():
    parser = argparse.ArgumentParser(description='测试集滚动窗口 ICIR 稳定性检验')
    parser.add_argument('--registry', type=str,
                        default=_LEGACY_REGISTRY,
                        help='注册表 JSON 路径')
    parser.add_argument('--ids', type=str, nargs='*', default=None,
                        help='因子 ID 列表 (默认全部)')
    parser.add_argument('--data', type=str,
                        default=os.path.join(_REPO_ROOT, 'data', 'csi500_daily_2020-07-20_to_2026-07-19.parquet'))
    parser.add_argument('--win-months', type=int, default=4,
                        help='滚动窗口长度 (自然月, 默认 4)')
    parser.add_argument('--stride-months', type=int, default=2,
                        help='滚动步长 (自然月, 默认 2)')
    parser.add_argument('--hit-threshold', type=float, default=0.0,
                        help='方向胜率阈值 (默认 0.0 = ICIR>0)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_global_seed(args.seed)
    reg_dir = os.path.dirname(os.path.abspath(args.registry))

    factors = load_registry(args.registry, args.ids)
    if not factors:
        logger.error("没有找到可检验的因子")
        sys.exit(1)
    logger.info(f"待检验因子: {len(factors)} 个")

    logger.info("加载数据...")
    loader = CSI500Loader(path=args.data)
    df = loader.load()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
    logger.info(f"数据: {len(df)} 行, {df['symbol'].nunique()} 只")

    prev = Config.TARGET_FACTOR_POOL_SIZE
    Config.TARGET_FACTOR_POOL_SIZE = 999
    prep = DataPreprocessor()
    prep.prepare_full_pool(df)
    Config.TARGET_FACTOR_POOL_SIZE = prev
    logger.info(f"特征: {len(prep.valid_features)} | 测试集: {prep.te_mask.sum():,} 样本")

    result_df = build_window_table(
        prep, factors,
        win_months=args.win_months,
        stride_months=args.stride_months,
        hit_threshold=args.hit_threshold,
    )
    if result_df.empty:
        logger.error("无有效结果")
        sys.exit(1)

    print_table(result_df)

    csv_path = os.path.join(reg_dir, 'time_stability_rolling.csv')
    result_df.to_csv(csv_path, index=False, float_format='%.4f')
    logger.info(f"结果保存至: {csv_path}")

    json_path = os.path.join(reg_dir, 'time_stability_rolling.json')
    result_df.to_json(json_path, orient='records', indent=2, force_ascii=False)
    logger.info(f"JSON 副本: {json_path}")


if __name__ == '__main__':
    main()
