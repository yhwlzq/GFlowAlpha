#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
body_ratio × ret_2d 三种变换对比实验 (有/无 FWL 中性化)

公式:
  1. square(body_ratio * ret_2d)   — 非线性幅度
  2. body_ratio * ret_2d           — 保留符号
  3. abs(body_ratio * ret_2d)      — 绝对值幅度

每条公式分别跑 FWL neutralized 和 raw FM 两种口径.
"""

import os, sys, logging
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from alpha.config import REPO_ROOT, Config, set_global_seed
from alpha.data.data_loader import CSI500Loader
from alpha.mining.preprocessor import DataPreprocessor
from alpha.mining.fm_regression import FamaMacBethRegressor, FMRegressionResult
from alpha.mining.neutralize import fwl_neutralize

DATA = os.path.join(REPO_ROOT, 'data', 'warmup', 'csi500_daily_2020-06-30_to_2026-06-30.parquet')

FORMULAS = [
    ('body_ratio',                'body_ratio',                  ['body_ratio']),
    ('ret_2d',                    'ret_2d',                      ['ret_2d']),
    ('square(body_ratio*ret_2d)', 'square(body_ratio * ret_2d)', ['body_ratio', 'ret_2d']),
    ('body_ratio*ret_2d',         'body_ratio * ret_2d',         ['body_ratio', 'ret_2d']),
    ('abs(body_ratio*ret_2d)',    'abs(body_ratio * ret_2d)',    ['body_ratio', 'ret_2d']),
]

SAFE_MATH = {
    'abs': np.abs, 'square': np.square, 'sqrt': lambda x: np.sqrt(np.abs(x)),
    'sign': np.sign, 'min': np.minimum, 'max': np.maximum,
    'tanh': np.tanh,
    'log1p': lambda x: np.log1p(np.abs(x)) * np.sign(x),
    'inv': lambda x: np.where(np.abs(x) > 1e-8, 1.0 / x, 0.0),
    'sigmoid': lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500))),
}


def eval_formula(formula, features, prep):
    ds = prep.get_subset(features)
    X_all = ds['all'][0]
    ns = {f: X_all[:, i] for i, f in enumerate(features)}
    ns.update(SAFE_MATH)
    pred_all = eval(formula, {"__builtins__": {}}, ns)
    return np.where(np.isfinite(pred_all), pred_all, np.nan)


def run_one(label, formula, features, prep, fm, skip_fwl):
    pred_all = eval_formula(formula, features, prep)

    te_mask = prep.te_mask
    pred_te = pred_all[te_mask]
    te_ret = prep.full_y[te_mask]
    te_dates = prep.full_dates[te_mask]
    te_symbols = prep.full_symbols[te_mask]
    te_amount = prep.full_amount[te_mask]

    valid = np.isfinite(pred_te) & np.isfinite(te_ret)
    pred_te, te_ret, te_dates, te_symbols, te_amount = (
        pred_te[valid], te_ret[valid], te_dates[valid], te_symbols[valid], te_amount[valid]
    )

    if skip_fwl:
        fv, rv = pred_te, te_ret
        d, s = te_dates, te_symbols
    else:
        fv, rv = fwl_neutralize(pred_te, te_ret, te_dates, te_symbols, te_amount)
        ok = np.isfinite(fv) & np.isfinite(rv)
        fv, rv, d, s = fv[ok], rv[ok], te_dates[ok], te_symbols[ok]

    result = fm.run(fv, rv, d, s)
    return {
        'label': label,
        'formula': formula,
        'mode': 'raw' if skip_fwl else 'FWL',
        't_stat': result.t_stat,
        'p_value': result.p_value,
        'rank_ic': result.rank_ic_mean,
        'icir': result.rank_icir,
        'coefficient': result.coefficient,
        'long_short': result.long_short_spread,
    }


def main():
    set_global_seed(42)
    logging.basicConfig(level=logging.WARNING, format='%(message)s')

    Config.MARKET = 'zs500'
    Config.FEATURE_POOL = 'buildin'
    Config.SPLIT_MODE = 'month'
    Config.SPLIT_MONTH_ANCHOR = 'data_start'
    Config.SPLIT_WARMUP = True

    print("加载数据...")
    loader = CSI500Loader(path=DATA)
    df = loader.load()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
    print(f"  {len(df)} 行, {df['symbol'].nunique()} 只股票")

    Config.TARGET_FACTOR_POOL_SIZE = 999
    prep = DataPreprocessor()
    prep.prepare_full_pool(df)
    print(f"  特征数: {len(prep.valid_features)} | 测试集: {prep.te_mask.sum():,} 样本")

    fm = FamaMacBethRegressor(nw_lags=Config.NW_LAGS, n_groups=5)

    results = []
    for tag, formula, feats in FORMULAS:
        for skip_fwl, mode_label in [(True, 'raw'), (False, 'FWL')]:
            print(f"  评估: {tag} ({mode_label})...")
            r = run_one(tag, formula, feats, prep, fm, skip_fwl)
            results.append(r)

    df_res = pd.DataFrame(results)

    print("\n" + "=" * 90)
    print("  body_ratio × ret_2d 变换对比 (CSI500, 2020-06-30 ~ 2026-06-30, warmup)")
    print("=" * 90)

    header = f"{'公式':<32} {'模式':<6} {'t-stat':>8} {'p-value':>10} {'Rank_IC':>8} {'ICIR':>7} {'L-S':>10}"
    print(header)
    print("-" * 92)
    for _, row in df_res.iterrows():
        pv = row['p_value']
        pv_str = f"{pv:.4f}{'***' if pv < 0.001 else ''}"
        print(f"{row['label']:<32} {row['mode']:<6} {row['t_stat']:>8.2f} {pv_str:>10} "
              f"{row['rank_ic']:>8.4f} {row['icir']:>7.3f} {row['long_short']:>10.5f}")
    print("-" * 92)

    print("\n  t-stat >= 2.0 为显著; ICIR >= 0.10 为可用; Rank_IC 绝对值越大越好")
    print()

    out_dir = os.path.join(Config.OUTPUT_DIR, 'body_ratio_ret2d_transform_test')
    os.makedirs(out_dir, exist_ok=True)
    df_res.to_csv(os.path.join(out_dir, 'results.csv'), index=False)
    print(f"结果已保存: {out_dir}/results.csv")


if __name__ == '__main__':
    main()
