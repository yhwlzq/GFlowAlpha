#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
传统单变量因子 vs GFN-SR 生成因子对比评估

流程:
  1. DataPreprocessor 加载数据, 筛选候选单变量因子
  2. 训练集上 Rank IC 排序 → 取 Top N
  3. 对每个因子: FWL 双残差中性化 → FM 回归 → AcademicBacktester
  4. GFN-SR 注册表因子: 同上流程
  5. 输出对比表格 + JSON

用法:
  # 传统 top 5 + 注册表
  python compare_benchmark.py \
      --data CSI500.csv --registry registry_academic.json

  # 只看传统 top 10
  python compare_benchmark.py --data CSI500.csv --top-n 10

  # 传统 + 注册表指定 ID
  python compare_benchmark.py \
      --data CSI500.csv --registry registry.json --id ACAD_029
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alpha.config import Config, set_global_seed
from alpha.data.data_loader import CSI500Loader
from alpha.mining.preprocessor import DataPreprocessor
from alpha.mining.fm_regression import FamaMacBethRegressor, FMRegressionResult
from alpha.mining.neutralize import fwl_neutralize
from alpha.evaluation.backtester import AcademicBacktester

logger = logging.getLogger(__name__)

CANDIDATE_FEATURES = [
    'gap',
    'ret_1d', 'ret_5d', 'ret_20d',
    'rev_1d', 'rev_5d', 'rev_20d',
    'bias_5d', 'bias_20d',
    'clv',
    'vol_20d',
    'turnover_20d', 'amihud_20d',
    'pv_corr_20d',
    'rsi_14', 'macd_hist', 'boll_width_20d',
    'intraday_mom', 'price_accel',
]

SAFE_MATH = {
    'abs': np.abs,
    'square': np.square,
    'sqrt': np.sqrt,
    'sign': np.sign,
    'min': np.minimum,
    'max': np.maximum,
    'tanh': np.tanh,
    'log1p': lambda x: np.log1p(np.abs(x)) * np.sign(x),
    'inv': lambda x: np.where(np.abs(x) > 1e-8, 1.0 / x, 0.0),
    'sigmoid': lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500))),
    'div': lambda a, b: np.where(np.abs(b) > 1e-8, a / b, 0.0),
}


def _daily_mean_rank_ic(
    factor_vals: np.ndarray, ret: np.ndarray, dates: np.ndarray,
) -> float:
    """Compute mean of daily cross-sectional Spearman Rank IC (same as FamaMacBethRegressor)."""
    df = pd.DataFrame({
        'date': pd.to_datetime(dates).normalize(),
        'factor': factor_vals, 'ret': ret,
    }).dropna()
    ic_list = []
    for _, grp in df.groupby('date'):
        if len(grp) < 30:
            continue
        ic = spearmanr(grp['factor'], grp['ret'])[0]
        if np.isfinite(ic):
            ic_list.append(ic)
    return float(np.mean(ic_list)) if ic_list else 0.0


def screen_top_features(prep: DataPreprocessor, n: int = 5) -> List[str]:
    scores = []
    for feat in CANDIDATE_FEATURES:
        if feat not in prep.feature_to_idx:
            continue
        idx = prep.feature_to_idx[feat]
        fv = prep.full_X[prep.tr_mask, idx]
        ret = prep.full_y[prep.tr_mask]
        dates = prep.full_dates[prep.tr_mask]
        valid = np.isfinite(fv) & np.isfinite(ret)
        if valid.sum() < 1000:
            continue
        ic = _daily_mean_rank_ic(fv[valid], ret[valid], dates[valid])
        scores.append((abs(ic), feat))

    scores.sort(key=lambda x: x[0], reverse=True)
    selected = [feat for _, feat in scores[:n]]
    logger.info(f"训练集逐日截面 Rank IC 筛选结果 (top {n}):")
    for rank, (ic, feat) in enumerate(scores[:n], 1):
        logger.info(f"  #{rank} {feat}: |IC|={ic:.4f}")
    return selected


def evaluate_feature(
    fid: str,
    feat_name_or_formula: str,
    features: List[str],
    prep: DataPreprocessor,
    fm: FamaMacBethRegressor,
    bt: AcademicBacktester,
    is_formula: bool = False,
) -> Optional[Dict]:
    if is_formula:
        try:
            ds = prep.get_subset(features)
        except ValueError as e:
            logger.warning(f"  {fid}: {e}")
            return None
        X_all = ds['all'][0]
        ns = {f: X_all[:, i] for i, f in enumerate(features)}
        ns.update(SAFE_MATH)
        try:
            pred_all = eval(feat_name_or_formula, {"__builtins__": {}}, ns)
        except Exception as e:
            logger.warning(f"  {fid}: eval 失败 ({e})")
            return None
        pred_all = np.where(np.isfinite(pred_all), pred_all, np.nan)
    else:
        pred_all = prep.full_X[:, prep.feature_to_idx[feat_name_or_formula]].copy()

    te_mask = prep.te_mask
    pred_te = pred_all[te_mask]
    te_ret = prep.full_y[te_mask]
    te_dates = prep.full_dates[te_mask]
    te_symbols = prep.full_symbols[te_mask]
    te_amount = prep.full_amount[te_mask]
    te_open = prep.full_open[te_mask] if prep.full_open is not None else None
    te_close = prep.full_close[te_mask] if prep.full_close is not None else None

    valid = np.isfinite(pred_te) & np.isfinite(te_ret)
    if valid.sum() < 500:
        logger.warning(f"  {fid}: 测试集有效样本不足 ({valid.sum()})")
        return None

    pred_te = pred_te[valid]
    te_ret = te_ret[valid]
    te_dates = te_dates[valid]
    te_symbols = te_symbols[valid]
    te_amount = te_amount[valid]
    if te_open is not None:
        te_open = te_open[valid]
        te_close = te_close[valid]

    bt_result = bt.run(pred_te, te_ret, te_dates, te_symbols, fid,
                       open_prices=te_open, close_prices=te_close)

    fv, rv = fwl_neutralize(pred_te, te_ret, te_dates, te_symbols, te_amount)
    ok = np.isfinite(fv) & np.isfinite(rv)
    fv_fm, rv_fm, d_fm, s_fm = fv[ok], rv[ok], te_dates[ok], te_symbols[ok]
    fm_result = fm.run(fv_fm, rv_fm, d_fm, s_fm)

    return {'fid': fid, 'type': 'GFN-SR' if is_formula else '传统',
            'fm': fm_result, 'bt': bt_result}


def fmt_pval(p: float) -> str:
    if p < 0.001:
        return '<.001***'
    m = '*' if p < 0.05 else ' '
    return f'{p:.3f}{m}'


def print_table(traditional: List[Dict], gfn: List[Dict]):
    all_results = []
    for rank, r in enumerate(traditional, 1):
        all_results.append((f"{rank}", r['fid'], r['type'], r))
    for r in gfn:
        all_results.append(("", r['fid'], r['type'], r))

    print(f"\n{'=' * 135}")
    print(f"  传统单变量 vs GFN-SR 因子对比评估 | FM: FWL 中性化, 回测: raw")
    print(f"{'=' * 135}")
    print(f"{'Rank':<5} {'ID':<16} {'Type':<8} {'t-stat':>8} {'p-value':>10} "
          f"{'Coef':>9} {'NW-SE':>8} {'Rank_IC':>8} {'ICIR':>7} "
          f"{'L-S':>9} {'年化':>7} {'Sharpe':>6} {'Q1~Q5月均收益':<20}")
    print(f"{'-' * 135}")

    for rank_str, fid, ftype, r in all_results:
        fm: FMRegressionResult = r['fm']
        bt = r['bt']
        ann = bt['annualized_return'] if bt else 0.0
        shp = bt['sharpe_ratio'] if bt else 0.0
        q = bt['quantile_monthly_rets'] if bt else {}
        q_str = ' '.join(f"{q.get(f'Q{i+1}', 0)*100:.1f}%" for i in range(5))
        print(f"{rank_str:<5} {fid:<16} {ftype:<8} {fm.t_stat:>8.2f} {fmt_pval(fm.p_value):>10} "
              f"{fm.coefficient:>9.6f} {fm.std_error:>8.6f} "
              f"{fm.rank_ic_mean:>8.4f} {fm.rank_icir:>7.2f} "
              f"{fm.long_short_spread:>9.5f} "
              f"{ann*100:>6.1f}% {shp:>5.2f}  {q_str}")
    print(f"{'-' * 135}")


def main():
    parser = argparse.ArgumentParser(description='传统 vs GFN-SR 因子对比评估')
    parser.add_argument('--data', default='data/csi500_daily_2021-06-30_to_2026-06-30.parquet', help='数据路径 (CSV/Parquet)')
    parser.add_argument('--registry', type=str, default="", help='注册表 JSON 路径 (与 --formula 互斥)')
    parser.add_argument('--id', type=str, help='注册表中的因子 ID (需配合 --registry)')
    parser.add_argument('--top-n', type=int, default=5, help='筛选单变量因子个数 (默认 5)')
    parser.add_argument('--output', type=str, default=None, help='输出目录')
    parser.add_argument('--seed', type=int, default=Config.SEED, help='随机种子')
    parser.add_argument('--pool-size', type=int, default=999, help='特征聚类目标数')
    args = parser.parse_args()

    set_global_seed(args.seed)
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s | %(levelname)-7s | %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.info("加载数据...")
    loader = CSI500Loader(path=args.data)
    df = loader.load()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
    logger.info(f"数据加载: {len(df)} 行, {df['symbol'].nunique()} 只股票")

    prev = Config.TARGET_FACTOR_POOL_SIZE
    Config.TARGET_FACTOR_POOL_SIZE = args.pool_size
    logger.info("数据预处理...")
    prep = DataPreprocessor()
    prep.prepare_full_pool(df)
    Config.TARGET_FACTOR_POOL_SIZE = prev
    logger.info(f"特征数: {len(prep.valid_features)} | 测试集: {prep.te_mask.sum():,} 样本")

    top_feats = screen_top_features(prep, n=args.top_n)
    if not top_feats:
        logger.error("没有找到合格的单变量因子")
        sys.exit(1)

    fm = FamaMacBethRegressor(nw_lags=Config.NW_LAGS, n_groups=5)
    bt = AcademicBacktester(top_quantile=0.2)

    logger.info(f"\n评估 {len(top_feats)} 个传统单变量因子...")
    traditional_results = []
    for feat in top_feats:
        logger.info(f"  评估: {feat}")
        r = evaluate_feature(feat, feat, [feat], prep, fm, bt, is_formula=False)
        if r:
            traditional_results.append(r)
            logger.info(f"    ✅ t={r['fm'].t_stat:.2f}  ICIR={r['fm'].rank_icir:.2f}")

    gfn_results = []
    if args.registry:
        with open(args.registry) as f:
            registry = json.load(f)
        factors = registry.get('factors', {})
        entries = [(args.id, factors[args.id])] if args.id else list(factors.items())
        logger.info(f"\n评估 {len(entries)} 个 GFN-SR 因子...")
        for fid, info in entries:
            m = info.get('metrics', info)
            formula = m.get('formula', '')
            feats = m.get('features', info.get('features', []))
            if not formula or not feats:
                logger.warning(f"  跳过 {fid}: 缺少 formula/features")
                continue
            missing = [f for f in feats if f not in prep.feature_to_idx]
            if missing:
                logger.warning(f"  跳过 {fid}: 缺失特征 {missing}")
                continue
            logger.info(f"  评估: {fid}")
            r = evaluate_feature(fid, formula, feats, prep, fm, bt, is_formula=True)
            if r:
                gfn_results.append(r)
                logger.info(f"    ✅ t={r['fm'].t_stat:.2f}  ICIR={r['fm'].rank_icir:.2f}")

    if not traditional_results:
        logger.error("所有评估均失败")
        sys.exit(1)

    print_table(traditional_results, gfn_results)

    output_dir = args.output or f"compare_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)

    def serialize(v):
        if isinstance(v, (np.floating, float)):
            return float(v)
        if isinstance(v, (np.integer, int)):
            return int(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, FMRegressionResult):
            return v.to_dict()
        if isinstance(v, dict):
            return {k: serialize(v) for k, v in v.items()}
        return v

    out_data = {
        'traditional': [],
        'gfn': [],
        'config': {
            'top_n': args.top_n,
            'data_path': args.data,
            'registry_path': args.registry,
            'seed': args.seed,
            'timestamp': datetime.now().isoformat(),
        }
    }
    for r in traditional_results:
        bt_res = r['bt']
        out_data['traditional'].append({
            'fid': r['fid'],
            'type': r['type'],
            'fm': serialize(r['fm']),
            'backtest': serialize(bt_res) if bt_res else None,
        })
    for r in gfn_results:
        bt_res = r['bt']
        out_data['gfn'].append({
            'fid': r['fid'],
            'type': r['type'],
            'fm': serialize(r['fm']),
            'backtest': serialize(bt_res) if bt_res else None,
        })

    json_path = os.path.join(output_dir, 'comparison_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)
    logger.info(f"结果保存至: {json_path}")
    print(f"\n输出目录: {os.path.abspath(output_dir)}")


if __name__ == '__main__':
    main()
