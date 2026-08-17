#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试集分时段 ICIR 稳定性检验工具。

对注册表中每个因子，将测试集按时间等分为 N 个子窗口，
在各窗口内计算截面 Rank IC → ICIR，检查方向一致性。

用法:
  python check_time_stability.py --registry path/to/registry_academic.json
  python check_time_stability.py --registry ... --ids ACAD_014 ACAD_019
  python check_time_stability.py --registry ... --windows 4
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alpha.config import Config, set_global_seed
from alpha.data.data_loader import CSI500Loader
from alpha.mining.preprocessor import DataPreprocessor
from alpha.mining.safe_ops import SafeOps
from alpha.mining.neutralize import fwl_neutralize

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


def build_window_table(prep: DataPreprocessor, factors: List[Tuple[str, str, List[str]]],
                       n_windows: int = 3) -> pd.DataFrame:
    test_mask = prep.te_mask
    test_dates = prep.full_dates[test_mask]
    test_ret = prep.full_y[test_mask]

    unique_days = np.sort(np.unique(test_dates))
    n = len(unique_days)
    if n < n_windows * 10:
        logger.error(f"测试集仅 {n} 个交易日，不足以分 {n_windows} 段")
        return pd.DataFrame()

    cuts = [unique_days[i * n // n_windows] for i in range(1, n_windows)]
    windows = np.split(unique_days, [unique_days.searchsorted(c) for c in cuts])
    window_labels = []
    for w in windows:
        start = pd.Timestamp(w[0]).strftime('%Y.%m')
        end = pd.Timestamp(w[-1]).strftime('%Y.%m')
        window_labels.append(f'{start}–{end}')
    full_label = f"{pd.Timestamp(windows[0][0]).strftime('%Y.%m')}–{pd.Timestamp(windows[-1][-1]).strftime('%Y.%m')}"

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
            label = f'W{wi+1} ({window_labels[wi]})'
            row[label] = icir if icir is not None else np.nan
            if icir is not None:
                ics.append(icir)

        # 全样本 ICIR
        full_icir = compute_window_icir(pure_pred_te, pure_ret_te, pure_dates, unique_days)
        row[f'全样本 ({full_label})'] = full_icir if full_icir is not None else np.nan

        # 方向一致性: 所有子窗口 ICIR 同号
        valid_ics = [v for v in ics if np.isfinite(v)]
        if len(valid_ics) >= 2:
            all_same_sign = all(v * valid_ics[0] > 0 for v in valid_ics)
            row['方向一致'] = '✅' if all_same_sign else '❌'
        else:
            row['方向一致'] = '—'

        rows.append(row)
        logger.info(f"  {fid}: {dict(row)}")

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # 排序: 方向一致 ✅ 在前
    if '方向一致' in df.columns:
        df = df.sort_values('方向一致', ascending=True, key=lambda x: x.map({'✅': 0, '❌': 1, '—': 2}))
    return df


def print_table(df: pd.DataFrame):
    if df.empty:
        return
    print("\n" + "=" * 110)
    print("  因子时间稳定性检验 (测试集分窗口 ICIR)")
    print("=" * 110)
    cols = [c for c in df.columns if c != '因子']
    fmt = "  {:<14}" + "".join("{:<22}" for _ in cols)
    header_labels = ['因子'] + cols
    print(fmt.format(*header_labels))
    print("  " + "-" * (14 + 22 * len(cols)))
    for _, row in df.iterrows():
        vals = [str(row['因子'])]
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                vals.append(f"{v:.3f}" if np.isfinite(v) else "  N/A")
            else:
                vals.append(f"  {v}")
        print(fmt.format(*vals))
    print("=" * 110)


def main():
    parser = argparse.ArgumentParser(description='测试集分时段 ICIR 稳定性检验')
    parser.add_argument('--registry', type=str, default="", help='注册表 JSON 路径 (与 --formula 互斥)')
    parser.add_argument('--ids', type=str, nargs='*', default=None,
                        help='因子 ID 列表 (默认全部)')
    parser.add_argument('--data', type=str,
                        default='data/csi500_daily_2021-06-30_to_2026-06-30.parquet')
    parser.add_argument('--windows', type=int, default=3,
                        help='子窗口数量 (默认 3)')
    parser.add_argument('--seed', type=int, default=Config.SEED)
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

    result_df = build_window_table(prep, factors, n_windows=args.windows)
    if result_df.empty:
        logger.error("无有效结果")
        sys.exit(1)

    print_table(result_df)

    csv_path = os.path.join(reg_dir, 'time_stability.csv')
    result_df.to_csv(csv_path, index=False, float_format='%.4f')
    logger.info(f"结果保存至: {csv_path}")

    # 同时保存 JSON 副本
    json_path = os.path.join(reg_dir, 'time_stability.json')
    result_df.to_json(json_path, orient='records', indent=2, force_ascii=False)
    logger.info(f"JSON 副本: {json_path}")


if __name__ == '__main__':
    main()
