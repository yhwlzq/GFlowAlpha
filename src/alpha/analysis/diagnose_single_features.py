#!/usr/bin/env python3
"""68个大市值特征逐一跑FM回归，不经过PySR/GFlowNet，分钟级出结果"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpha.config import Config
from alpha.features.feature_registry import LargeCapFeatureRegistry
from alpha.mining.preprocessor import DataPreprocessor
from alpha.mining.fm_regression import FamaMacBethRegressor

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', stream=sys.stdout)
logger = logging.getLogger('DiagnoseSingle')
logging.getLogger('AcademicDualEngine').disabled = True

DATA_PATH = 'data/hs300_daily_2021-06-30_to_2026-06-30.parquet'

def main():
    Config.MARKET = 'hs300'

    df = pd.read_parquet(DATA_PATH)
    df.columns = [c if c != 'code' else 'symbol' for c in df.columns]
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)

    prep = DataPreprocessor()
    prep.prepare_full_pool(df)

    fm = FamaMacBethRegressor(nw_lags=Config.NW_LAGS)
    feature_pool = Config.FINAL_FEATURE_POOL
    logger.info(f"共 {len(feature_pool)} 个特征，逐一开始 FM 回归...")

    results = []
    for i, feat in enumerate(feature_pool):
        ds = prep.get_subset([feat])
        X_val = ds['val'][0]
        y_val = ds['val'][1]
        d_val = ds['val'][2]
        s_val = ds['all'][3][prep.val_mask]
        a_val = ds['all'][4][prep.val_mask]

        factor_values = X_val[:, 0]
        res = fm.run(factor_values, y_val, d_val, s_val)

        results.append({
            'feature': feat,
            't_stat': res.t_stat,
            'p_value': res.p_value,
            'coef': res.coefficient,
            'se': res.std_error,
            'rank_ic': res.rank_ic_mean,
            'icir': res.rank_icir,
            'coef_pos_ratio': res.coef_positive_ratio,
            'coef_sig_ratio': res.coef_significant_ratio,
            'n_periods': res.n_periods,
            'avg_r2': res.avg_r_squared,
            'ls_spread': res.long_short_spread,
        })

        if (i + 1) % 20 == 0:
            logger.info(f"  已完成 {i+1}/{len(feature_pool)}")

    res_df = pd.DataFrame(results)
    res_df['abs_t'] = res_df['t_stat'].abs()
    res_df = res_df.sort_values('abs_t', ascending=False).reset_index(drop=True)

    print(f"\n{'='*105}")
    print(f"{'68 个大市值特征 → 单因子 FM 回归 (HORIZON={Config.HORIZON}, VAL集, NW_lags={Config.NW_LAGS})':^105}")
    print(f"{'='*105}")
    print(f"{'排名':>4} {'特征':<24} {'FM t-stat':>10} {'p-value':>10} {'Rank_IC':>8} {'ICIR':>7} {'正比':>6} {'显著比':>8} {'L-S':>8} {'R²':>6}")
    print(f"{'-'*105}")
    for rank, (_, row) in enumerate(res_df.iterrows(), 1):
        sig = '***' if abs(row['t_stat']) >= 3 else ('**' if abs(row['t_stat']) >= 2 else ('*' if abs(row['t_stat']) >= 1.5 else ''))
        print(f"{rank:>4} {row['feature']:<24} {row['t_stat']:>10.4f}{sig:<4} {row['p_value']:>10.4f} {row['rank_ic']:>8.4f} {row['icir']:>7.3f} {row['coef_pos_ratio']:>6.1%} {row['coef_sig_ratio']:>8.1%} {row['ls_spread']:>8.5f} {row['avg_r2']:>6.4f}")

    print(f"\n{'='*105}")
    print(f"汇总:")
    print(f"  |t| >= 2.0: {(res_df['abs_t'] >= 2.0).sum()} 个")
    print(f"  |t| >= 1.5: {(res_df['abs_t'] >= 1.5).sum()} 个")
    print(f"  |t| >= 1.0: {(res_df['abs_t'] >= 1.0).sum()} 个")
    print(f"  avg |t|: {res_df['abs_t'].mean():.4f}")
    print(f"  max |t|: {res_df['abs_t'].max():.4f} ({res_df.iloc[0]['feature']})")
    print(f"  avg ICIR: {res_df['icir'].abs().mean():.4f}")

    out_path = 'single_feature_fm_results.csv'
    res_df.to_csv(out_path, index=False)
    print(f"\n结果已保存: {out_path}")

if __name__ == '__main__':
    main()
