#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
组合级 1+1>2 评估 (JFDS 轻量统计诚实版)
========================================
从 registry 取复合因子 → 逐日截面 z-score → 去重(按相关) + 取 top-K(按 |FM_t|)
→ 等权组合分 → 组合 vs 单因子的日频 L-S 收益:
  * 组合 SR 与 单因子均值 SR 的 "协同增益"
  * 月份块 bootstrap 给出协同增益 95% CI (下限>0 即 1+1>2 显著)
  * 两两截面相关矩阵 (平均/最大 |ρ|)

用法:
    python combine_factor_eval.py --registry factor_output_academic_v81/run_xxx/registry_academic.json \
        --data data/warmup/csi500_daily_2020-06-30_to_2026-06-30.parquet [--topk 5] [--drop-corr 0.98]
        [--neutralize-multi] [--n-boot 2000]
"""
import argparse
import json
import logging
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from alpha.config import Config
from alpha.data.data_loader import CSI500Loader
from alpha.mining.preprocessor import DataPreprocessor
from alpha.evaluation.build_factor_panel import eval_registry_to_panel, neutralize_factor_by_amount
from alpha.config import REPO_ROOT, Config, set_global_seed

logger = logging.getLogger('CombineFactorEval')

_REPO_ROOT = REPO_ROOT
_LEGACY_REGISTRY = os.path.join(_REPO_ROOT, 'factor_output_academic_v81', 'run_20260829_183554', 'registry_academic.json')

def cross_sectional_zscore(values: np.ndarray, dates: np.ndarray) -> np.ndarray:
    df = pd.DataFrame({'date': pd.to_datetime(dates).normalize(), 'v': values})
    z = np.full(len(values), np.nan, dtype=np.float64)
    for dt, grp in df.groupby('date'):
        idx = grp.index
        m, s = grp['v'].mean(), grp['v'].std(ddof=0)
        z[idx] = (grp['v'].values - m) / (s + 1e-12)
    return z


def daily_ls_returns(score: np.ndarray, ret: np.ndarray, dates: np.ndarray,
                     top_quantile: float = 0.2, min_stocks: int = 30) -> pd.Series:
    df = pd.DataFrame({'date': pd.to_datetime(dates).normalize(), 's': score, 'r': ret}).dropna()
    out = {}
    for dt, grp in df.groupby('date'):
        if len(grp) < min_stocks:
            continue
        n = max(int(len(grp) * top_quantile), 1)
        srt = grp.sort_values('s', ascending=False)
        out[dt] = srt['r'].head(n).mean() - srt['r'].tail(n).mean()
    return pd.Series(out, name='ls').sort_index()


def sharpe(series: pd.Series) -> float:
    r = series.dropna()
    if len(r) < 30:
        return 0.0
    sd = r.std(ddof=1)
    return float(r.mean() / (sd + 1e-12) * np.sqrt(252))


def month_block_bootstrap(combo: pd.Series, individuals: pd.Series,
                          n_boot: int = 2000, seed: int = 42) -> tuple:
    """月份块 bootstrap: 按月份整块重抽样 (保留月内自相关), 返回协同增益分布的
    (mean, p即增益<=0概率, (lo,hi)) 及中位数 SR 向量. 组合与个体对齐到同一日期集."""
    dates = combo.index
    months = dates.to_period('M').unique()
    rng = np.random.default_rng(seed)

    def sr_of(ser: pd.Series, picks: list) -> float:
        parts = [ser.loc[ser.index.to_period('M') == m] for m in picks]
        if not parts:
            return 0.0
        idx = parts[0].index
        for p in parts[1:]:
            idx = idx.union(p.index)
        sub = ser.reindex(idx)
        return sharpe(sub)

    gains = np.full(n_boot, np.nan)
    for b in range(n_boot):
        picks = list(rng.choice(months, size=len(months), replace=True))
        sr_c = sr_of(combo, picks)
        sr_i = np.mean([sr_of(ind, picks) for ind in individuals])
        gains[b] = sr_c - sr_i
    gains = gains[np.isfinite(gains)]
    mean_gain = float(np.mean(gains))
    p_le0 = float(np.mean(gains <= 0.0))
    lo, hi = float(np.percentile(gains, 2.5)), float(np.percentile(gains, 97.5))
    return mean_gain, p_le0, (lo, hi)


def main():
    parser = argparse.ArgumentParser(description='组合级 1+1>2 评估 (bootstrap 协同增益)')
    parser.add_argument('--registry', type=str, default=_LEGACY_REGISTRY, help='注册表 JSON 路径 (与 --formula 互斥)')
    parser.add_argument('--data', default='data/warmup/csi500_daily_2020-06-30_to_2026-06-30.parquet')
    parser.add_argument('--topk', type=int, default=5, help='组合用因子数 (去重后按 |FM_t| 取前 K)')
    parser.add_argument('--drop-corr', type=float, default=0.98,
                        help='两两截面 |rho| 超过该值视为重复并剔除')
    parser.add_argument('--start-date', default='2025-07-01', help='评估窗起始 (默认=主实验 test 段)')
    parser.add_argument('--end-date', default='2026-06-30', help='评估窗结束 (默认=主实验 test 段)')
    parser.add_argument('--neutralize-multi', action='store_true',
                        help='按 Config.NEUTRALIZE_CONTROLS 多风险中性化因子后再组合')
    parser.add_argument('--neutralize', action='store_true',
                        help='仅 ln(amount) 市值中性化 (旧口径)')
    parser.add_argument('--n-boot', type=int, default=2000, help='bootstrap 次数')
    parser.add_argument('--seed', type=int, default=Config.SEED)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-7s | %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join('factor_output_academic_v81', f'combine_{ts}')
    os.makedirs(out_dir, exist_ok=True)

    with open(args.registry) as f:
        registry = json.load(f)
    if not registry.get('factors'):
        logger.error('registry 无因子')
        sys.exit(1)

    logger.info('加载数据并构建全样本面板...')
    loader = CSI500Loader(path=args.data)
    df = loader.load()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
    prep = DataPreprocessor()
    prep.prepare_full_pool(df)

    panel = eval_registry_to_panel(prep, registry)
    if args.neutralize_multi:
        panel = neutralize_factor_by_amount(
            panel, prep.full_amount,
            turn=prep.full_turn if 'turn' in Config.NEUTRALIZE_CONTROLS else None)
    elif args.neutralize:
        panel = neutralize_factor_by_amount(panel, prep.full_amount)

    # 与正交性流水线同窗: 显式日期过滤 (主实验 warmup test 段), 不依赖切分模式
    d = pd.to_datetime(panel['date'])
    win = (d >= pd.Timestamp(args.start_date)) & (d <= pd.Timestamp(args.end_date))
    te = panel.loc[win].reset_index(drop=True)
    dates = pd.to_datetime(te['date']).values
    ret = te['ret'].values.astype(np.float64)
    fids = [c for c in te.columns if c not in ('date', 'symbol', 'ret')]
    logger.info(f'测试窗: {pd.Timestamp(dates.min()).date()} ~ {pd.Timestamp(dates.max()).date()} | '
                f'{len(dates)} 日 | 因子数 {len(fids)}')

    z_all = {}
    for fid in fids:
        z = cross_sectional_zscore(te[fid].values, dates)
        rho = np.corrcoef(z[~np.isnan(z)], ret[~np.isnan(z)])
        if rho.size and rho[0, 1] < 0:          # 符号对齐: 统一正 IC 方向
            z = -z
        z_all[fid] = z

    # 去重 + top-K (按 registry |FM_t|)
    fm_t = {}
    for fid, info in registry['factors'].items():
        m = info.get('metrics', info)
        fm_t[fid] = abs(m.get('FM_tstat', 0.0))
    order = sorted(fids, key=lambda f: fm_t.get(f, 0.0), reverse=True)

    keep = []
    for f in order:
        if len(keep) >= args.topk:
            break
        if any(abs(np.corrcoef(z_all[f], z_all[g])[0, 1]) > args.drop_corr for g in keep):
            continue
        keep.append(f)
    if not keep:
        logger.warning('去重后无因子, 退化为全部')
        keep = order[:args.topk]
    logger.info(f'组合因子 (K={len(keep)}): {keep}')
    logger.info(f'剔除(重复/未入选): {[f for f in fids if f not in keep]}')

    combo_z = np.nanmean(np.column_stack([z_all[f] for f in keep]), axis=1)
    combo_ls = daily_ls_returns(combo_z, ret, dates)
    ind_ls = {f: daily_ls_returns(z_all[f], ret, dates) for f in keep}

    print('\n' + '=' * 78)
    print('  组合级 1+1>2 (JFDS / 月份块 bootstrap)')
    print(f'  K={len(keep)} 去重|rho|阈值={args.drop_corr} 中性化={args.neutralize_multi or args.neutralize}')
    print('=' * 78)
    print(f"  {'因子':<12}{'SR252':>9}{'Ann':>9}{'t':>7}{'|FM_t|':>8}")
    for f in keep:
        ls = ind_ls[f]
        sr = sharpe(ls)
        ann = float((1 + ls.mean()) ** 252 - 1) if ls.mean() > -1 else -1.0
        t = float(ls.mean() / (ls.std(ddof=1) + 1e-12) * np.sqrt(len(ls)))
        print(f"  {f:<12}{sr:>9.2f}{ann:>9.1%}{t:>7.2f}{fm_t.get(f,0):>8.2f}")

    sr_combo = sharpe(combo_ls)
    sr_mean_ind = float(np.mean([sharpe(ls) for ls in ind_ls.values()]))
    synergy = sr_combo - sr_mean_ind
    print(f"\n  组合 SR         : {sr_combo:.3f}")
    print(f"  单因子均值 SR   : {sr_mean_ind:.3f}")
    print(f"  协同增益(组合−均值): {synergy:+.3f}")

    # 两两相关
    corrs = np.array([[abs(np.corrcoef(z_all[a], z_all[b])[0, 1]) for b in keep] for a in keep])
    print(f"  两两 |rho| 均值 : {np.mean(corrs[np.triu_indices(len(keep), 1)]):.3f}  "
          f"最大: {np.max(corrs[np.triu_indices(len(keep), 1)]):.3f}")

    mean_gain, p_le0, (lo, hi) = month_block_bootstrap(
        combo_ls, list(ind_ls.values()), n_boot=args.n_boot, seed=args.seed)
    print(f"\n  协同增益 bootstrap (n={args.n_boot}, 月块):")
    print(f"    均值 {mean_gain:+.3f} | 95% CI [{lo:+.3f}, {hi:+.3f}] | P(增益≤0)={p_le0:.3f}")
    print(f"    结论: {'1+1>2 显著' if lo > 0 else ('边缘/不显著' if p_le0 <= 0.1 else '未支持')}")
    print('=' * 78)

    df_out = pd.DataFrame({
        'factor': keep + ['COMBO'],
        'SR252': [sharpe(ind_ls[f]) for f in keep] + [sr_combo],
        'ann': [float((1 + ind_ls[f].mean()) ** 252 - 1) if ind_ls[f].mean() > -1 else -1.0 for f in keep]
               + [float((1 + combo_ls.mean()) ** 252 - 1) if combo_ls.mean() > -1 else -1.0],
        't': [float(ind_ls[f].mean() / (ind_ls[f].std(ddof=1) + 1e-12) * np.sqrt(len(ind_ls[f]))) for f in keep] +
             [float(combo_ls.mean() / (combo_ls.std(ddof=1) + 1e-12) * np.sqrt(len(combo_ls)))],
    })
    summary = pd.DataFrame([{
        'n_factors': len(keep), 'sr_combo': sr_combo, 'sr_mean_individual': sr_mean_ind,
        'synergy': synergy, 'bootstrap_mean_gain': mean_gain,
        'bootstrap_lo95': lo, 'bootstrap_hi95': hi, 'p_gain_le0': p_le0,
        'mean_abs_rho': float(np.mean(corrs[np.triu_indices(len(keep), 1)])),
        'max_abs_rho': float(np.max(corrs[np.triu_indices(len(keep), 1)])),
        'factors': keep, 'neutralized': args.neutralize_multi or args.neutralize,
    }])
    df_out.to_csv(os.path.join(out_dir, 'combine_metrics.csv'), index=False, encoding='utf-8-sig')
    summary.to_csv(os.path.join(out_dir, 'combine_summary.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame(corrs, index=keep, columns=keep).to_csv(
        os.path.join(out_dir, 'pairwise_corr.csv'), encoding='utf-8-sig')
    combo_ls.to_csv(os.path.join(out_dir, 'combo_ls.csv', ))
    logger.info(f'输出目录: {os.path.abspath(out_dir)}')


if __name__ == '__main__':
    main()