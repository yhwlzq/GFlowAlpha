#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多因子市场状态互补性验证
========================
两步法:
  ① 截面 Spearman 相关矩阵        → 验证因子值不重叠
  ② 市场状态分组预测力对比        → 验证经济含义互补

市场状态三维度:
  1. 过去20日波动率 (高/低)
  2. 过去20日市场趋势 (上涨/下跌)
  3. 截面离散度 (高/低)

每个子样本内计算三因子的 Rank IC 与日频多空收益差，
观察是否在不同市场状态下轮流表现更优。

用法:
    python validate_factor_combo.py

依赖:
    paper/PYSR/common/ 模块
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alpha.config import Config, set_global_seed
from alpha.data.data_loader import CSI500Loader
from alpha.mining.preprocessor import DataPreprocessor
from alpha.mining.neutralize import fwl_neutralize

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger('FactorComboValidation')

FACTORS = {
    'ACAD_029': {
        'formula': 'square(rev_1d*0.115807794)',
        'features': ['rev_1d', 'dist_52w_low', 'turnover_std_20d', 'dist_52w_high', 'price_accel'],
        'label': '平方反转(rev_1d)',
    },
    'ACAD_043': {
        'formula': 'abs(ma_cross_5d*0.019908553)',
        'features': ['dist_52w_high', 'mom_smooth_20d', 'ma_cross_5d', 'vol_skew_ratio', 'kurt_20d'],
        'label': '均线偏离(ma_cross_5d)',
    },
    'ACAD_025': {
        'formula': '(bias_5d*0.039108977)*clv',
        'features': ['bias_5d', 'rev_20d', 'macd_signal', 'clv', 'bias_60d'],
        'label': '乖离率×CLV',
    },
}

FIDS = list(FACTORS.keys())

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


# ============================================================
# 工具函数
# ============================================================

def cross_sectional_zscore(values, dates):
    df = pd.DataFrame({'date': pd.to_datetime(dates).normalize(), 'v': values})
    z = np.full(len(values), np.nan, dtype=np.float64)
    for dt, grp in df.groupby('date'):
        idx = grp.index
        m, s = grp['v'].mean(), grp['v'].std(ddof=0)
        z[idx] = (grp['v'].values - m) / (s + 1e-8)
    return z


def compute_daily_ic(pred, ret, dates):
    ics = []
    df = pd.DataFrame({
        'date': pd.to_datetime(dates).normalize(),
        'pred': pred, 'ret': ret,
    })
    for dt, grp in df.groupby('date'):
        if len(grp) < 30:
            continue
        ic = spearmanr(grp['pred'], grp['ret'])[0]
        if np.isfinite(ic):
            ics.append(ic)
    if not ics:
        return 0.0, 0.0
    ic_mean = float(np.mean(ics))
    ic_std = float(np.std(ics, ddof=1)) if len(ics) > 1 else 1e-8
    icir = ic_mean / ic_std
    return ic_mean, icir


def compute_daily_ls_spread(pred, ret, dates, top_q=0.2):
    spreads = []
    df = pd.DataFrame({
        'date': pd.to_datetime(dates).normalize(),
        'pred': pred, 'ret': ret,
    })
    for dt, grp in df.groupby('date'):
        if len(grp) < 30:
            continue
        threshold_top = grp['pred'].quantile(1 - top_q)
        threshold_bot = grp['pred'].quantile(top_q)
        long = grp.loc[grp['pred'] >= threshold_top, 'ret'].mean()
        short = grp.loc[grp['pred'] <= threshold_bot, 'ret'].mean()
        if np.isfinite(long) and np.isfinite(short):
            spreads.append(long - short)
    if not spreads:
        return 0.0, 0.0
    return float(np.mean(spreads)), float(np.std(spreads, ddof=1) / np.sqrt(len(spreads)))


# ============================================================
# 市场状态分类
# ============================================================

def compute_market_states(prep, te_mask):
    vol_idx = prep.feature_to_idx.get('vol_20d')
    ret_idx = prep.feature_to_idx.get('ret_20d')
    if vol_idx is None or ret_idx is None:
        raise ValueError("vol_20d 或 ret_20d 不在核心因子池中，无法计算市场状态")

    te_dates_norm = pd.to_datetime(prep.full_dates[te_mask]).normalize().values
    vol_vals = prep.full_X[te_mask, vol_idx]
    ret_vals = prep.full_X[te_mask, ret_idx]

    uniq_dates = np.unique(te_dates_norm)
    daily_stats = {}
    for dt in uniq_dates:
        m = te_dates_norm == dt
        daily_stats[dt] = {
            'vol_median': float(np.median(vol_vals[m])),
            'trend_mean': float(np.mean(ret_vals[m])),
            'dispersion': float(np.std(ret_vals[m], ddof=1)),
        }

    vol_threshold = float(np.median([s['vol_median'] for s in daily_stats.values()]))
    disp_threshold = float(np.median([s['dispersion'] for s in daily_stats.values()]))

    date_states = {}
    for dt, s in daily_stats.items():
        date_states[dt] = {
            'vol_state': '高波动' if s['vol_median'] >= vol_threshold else '低波动',
            'trend_state': '上涨趋势' if s['trend_mean'] >= 0 else '下跌趋势',
            'disp_state': '高离散' if s['dispersion'] >= disp_threshold else '低离散',
        }

    return date_states, daily_stats, vol_threshold, disp_threshold


def state_subset(te_dates, date_states, state_key, state_value):
    dates_norm = pd.to_datetime(te_dates).normalize().values
    return np.array([
        date_states.get(pd.Timestamp(d), {}).get(state_key) == state_value
        for d in dates_norm
    ])


# ============================================================
# 主流程
# ============================================================

def main():
    set_global_seed()

    logger.info("=" * 70)
    logger.info("  多因子市场状态互补性验证")
    logger.info("=" * 70)

    # ---- Step 0: 加载 & 预处理 ----
    logger.info("加载数据...")
    loader = CSI500Loader()
    df = loader.load()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
    logger.info(f"数据: {len(df)} 行, {df['symbol'].nunique()} 只股票")

    logger.info("预处理...")
    prep = DataPreprocessor()
    prep.prepare_full_pool(df)
    te_mask = prep.te_mask

    # ---- 构建三个因子在测试集上的预测值 ----
    factor_preds = {}
    for fid in FIDS:
        finfo = FACTORS[fid]
        ds = prep.get_subset(finfo['features'])
        X_all = ds['all'][0]
        ns = {f: X_all[:, i] for i, f in enumerate(finfo['features'])}
        ns.update(SAFE_MATH)
        pred_all = eval(finfo['formula'], {"__builtins__": {}}, ns)
        pred_all = np.where(np.isfinite(pred_all), pred_all, np.nan)
        factor_preds[fid] = pred_all[te_mask]
        logger.info(f"  {fid} ({finfo['label']}): valid={np.isfinite(factor_preds[fid]).sum():,}")

    te_ret = prep.full_y[te_mask]
    te_dates = prep.full_dates[te_mask]
    te_symbols = prep.full_symbols[te_mask]
    te_amount = prep.full_amount[te_mask]

    valid = np.isfinite(te_ret) & np.isfinite(te_amount)
    for arr in factor_preds.values():
        valid &= np.isfinite(arr)
    logger.info(f"测试集有效样本 (三因子交集): {valid.sum():,}")

    te_ret = te_ret[valid]
    te_dates = te_dates[valid]
    te_symbols = te_symbols[valid]
    te_amount = te_amount[valid]
    factor_preds = {k: v[valid] for k, v in factor_preds.items()}

    # FWL 双残差中性化 (截面回归剔除 ln amount 暴露)
    for fid in FIDS:
        fv, rv = fwl_neutralize(factor_preds[fid], te_ret, te_dates, te_symbols, te_amount)
        factor_preds[fid] = fv
        te_ret = rv

    for fid in FIDS:
        factor_preds[fid] = cross_sectional_zscore(factor_preds[fid], te_dates)

    # ============================================================
    # 【Step 1】截面 Spearman 相关性检验
    # ============================================================
    print("\n" + "=" * 70)
    print("  【Step 1】因子截面 Spearman 相关性检验")
    print("  → 验证信息不重叠：低相关 → 捕捉不同市场异象")
    print("=" * 70)

    df_corr = pd.DataFrame({'date': pd.to_datetime(te_dates).normalize()})
    for fid in FIDS:
        df_corr[fid] = factor_preds[fid]

    daily_corr_list, n_days = [], 0
    for dt, grp in df_corr.groupby('date'):
        if len(grp) < 30:
            continue
        n_days += 1
        daily_corr_list.append(grp[FIDS].corr(method='spearman').values)
    corr_mean = np.mean(daily_corr_list, axis=0)

    print(f"\n{'':>12}", end='')
    for fid in FIDS:
        print(f"{fid:>12}", end='')
    print()
    for i, fid in enumerate(FIDS):
        print(f"{fid:>12}", end='')
        for j in range(len(FIDS)):
            print(f"{corr_mean[i, j]:>12.4f}", end='')
        print()

    corr_abs = [abs(corr_mean[i, j]) for i in range(len(FIDS))
                for j in range(i + 1, len(FIDS))]
    print(f"\n  统计期数: {n_days} 个交易日")
    print(f"  平均 |ρ| = {np.mean(corr_abs):.4f}   最大 |ρ| = {np.max(corr_abs):.4f}")
    print(f"  解读: {'✅ 因子间信息重叠度低' if np.max(corr_abs) < 0.3 else '⚠️ 存在较高相关因子，需注意信息重叠'}")

    # ============================================================
    # 【Step 2】市场状态分组预测力对比
    # ============================================================
    print("\n" + "=" * 70)
    print("  【Step 2】市场状态分组预测力对比")
    print("  → 验证经济含义互补：不同状态下各因子是否轮流表现更优")
    print("=" * 70)

    # 计算市场状态
    date_states, daily_stats, vol_th, disp_th = compute_market_states(prep, te_mask)

    # 全样本基准
    print(f"\n  【基准】全样本预测力")
    print(f"  {'因子':<28} {'Rank_IC':>8} {'ICIR':>8} {'L-Spread':>10} {'LS/SE':>8}")
    print(f"  {'-'*62}")
    for fid in FIDS:
        ic_m, ic_ir = compute_daily_ic(factor_preds[fid], te_ret, te_dates)
        ls_m, ls_se = compute_daily_ls_spread(factor_preds[fid], te_ret, te_dates)
        print(f"  {FACTORS[fid]['label']:<28} {ic_m:>8.4f} {ic_ir:>8.2f} {ls_m:>10.6f} {ls_m/(ls_se+1e-8):>8.2f}")

    # 三维度状态
    dimensions = [
        ('vol_state', '波动率', {'高波动': '高波动', '低波动': '低波动'}),
        ('trend_state', '市场趋势', {'上涨趋势': '上涨趋势', '下跌趋势': '下跌趋势'}),
        ('disp_state', '截面离散度', {'高离散': '高离散', '低离散': '低离散'}),
    ]

    all_results = []

    for state_key, dim_label, state_map in dimensions:
        print(f"\n  ─── 维度: {dim_label} ───")

        header = f"  {'市场状态':<12} {'天数':>5}"
        for fid in FIDS:
            header += f"  {FACTORS[fid]['label']:>18}"
        print(header)

        sub_header = f"  {'':<12} {'':>5}"
        for _ in FIDS:
            sub_header += f"  {'IC':>7} {'LSprd':>9}"
        print(sub_header)

        for state_label in state_map.values():
            state_mask = state_subset(te_dates, date_states, state_key, state_label)
            n_state_days = len({pd.Timestamp(d).normalize()
                                for d in te_dates[state_mask]})

            row = f"  {state_label:<12} {n_state_days:>5}"

            for fid in FIDS:
                sub_ret = te_ret[state_mask]
                sub_pred = factor_preds[fid][state_mask]
                sub_dates = te_dates[state_mask]

                ic_m, ic_ir = compute_daily_ic(sub_pred, sub_ret, sub_dates)
                ls_m, ls_se = compute_daily_ls_spread(sub_pred, sub_ret, sub_dates)
                row += f"  {ic_m:>7.4f} {ls_m:>9.6f}"

                all_results.append({
                    'dimension': dim_label,
                    'state': state_label,
                    'factor': FACTORS[fid]['label'],
                    'factor_id': fid,
                    'rank_ic': ic_m,
                    'icir': ic_ir,
                    'ls_spread': ls_m,
                    'ls_tstat': ls_m / (ls_se + 1e-8),
                    'n_days': n_state_days,
                })

            print(row)

    # ---- 总结: 每个因子的"舒适区" ----
    print("\n  " + "=" * 62)
    print("  【总结】各因子的优势市场状态")
    print("  " + "=" * 62)

    df_res = pd.DataFrame(all_results)
    for fid in FIDS:
        label = FACTORS[fid]['label']
        sub = df_res[df_res['factor_id'] == fid]
        best_row = sub.loc[sub['rank_ic'].idxmax()]
        worst_row = sub.loc[sub['rank_ic'].idxmin()]

        print(f"\n  {label}:")
        print(f"    最强: {best_row['dimension']}/{best_row['state']}  "
              f"IC={best_row['rank_ic']:.4f}  LS={best_row['ls_spread']:.6f}")
        print(f"    最弱: {worst_row['dimension']}/{worst_row['state']}  "
              f"IC={worst_row['rank_ic']:.4f}  LS={worst_row['ls_spread']:.6f}")

    # 各状态下最优因子
    print(f"\n  {'='*62}")
    print(f"  【因子间对比】各状态下最优因子")
    print(f"  {'='*62}")
    for dim_label in [d[1] for d in dimensions]:
        for state_label in ['高波动', '低波动', '上涨趋势', '下跌趋势', '高离散', '低离散']:
            sub = df_res[(df_res['dimension'] == dim_label) &
                         (df_res['state'] == state_label)]
            if sub.empty:
                continue
            best = sub.loc[sub['rank_ic'].idxmax()]
            print(f"  {dim_label}/{state_label:<8} → 最优: {best['factor']:<16}  "
                  f"IC={best['rank_ic']:.4f}  LS={best['ls_spread']:.6f}")

    # ---- 保存结果 ----
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path(Config.OUTPUT_DIR) / f'state_complement_{ts}'
    out_dir.mkdir(parents=True, exist_ok=True)

    def serialize(v):
        if isinstance(v, (np.floating, float)):
            return float(v)
        if isinstance(v, (np.integer, int)):
            return int(v)
        return v

    out_data = []
    for r in all_results:
        out_data.append({k: serialize(v) for k, v in r.items()})

    meta = {
        'vol_threshold': vol_th,
        'disp_threshold': disp_th,
        'daily_stats': {str(k): v for k, v in daily_stats.items()},
        'date_states': {str(k): v for k, v in date_states.items()},
    }
    with open(out_dir / 'state_results.json', 'w', encoding='utf-8') as f:
        json.dump({'results': out_data, 'meta': meta}, f, indent=2, ensure_ascii=False)

    logger.info(f"结果保存至: {out_dir}")
    print(f"\n输出目录: {out_dir.resolve()}")


if __name__ == '__main__':
    main()
