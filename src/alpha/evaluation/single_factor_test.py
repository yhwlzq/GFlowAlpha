#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单因子回测评估工具

 能在测试集上对任意因子公式执行:
  1. Fama-MacBeth 截面回归 (含 Newey-West 标准误)
  2. 五分位多空组合月度复利回测
  3. 累计净值曲线图 + 五分位柱状图

默认行为与挖掘管线一致:
  - FM 回归 → FWL ln(amount) 中性化后 (匹配注册表 t-stat / ICIR)
  - 多空回测 → 原始预测值 (匹配管线 "启动单因子回测")
  - 使用 --no-fwl 可禁用中性化 (对比查看 raw FM 效果)

用法:
  # 命令行指定公式
  python single_factor_test.py \
      --data CSI500_4Years_2022-06-30_to_2026-06-29.csv \
      --formula "square(rev_1d*0.115807794)" \
      --features rev_1d

  # 从注册表加载全部因子
  python single_factor_test.py \
      --data CSI500_4Years_2022-06-30_to_2026-06-29.csv \
      --registry factor_output_academic_v81/run_xxx/registry_academic.json

  # 注册表 + 指定因子 ID
  python single_factor_test.py \
      --data ... --registry ... --id ACAD_029

  # 跳过 FWL 中性化 (raw FM + raw 回测)
  python single_factor_test.py \
      --data ... --registry ... --id ACAD_029 --no-fwl

  # 画图
  python single_factor_test.py \
      --data ... --registry ... --id ACAD_029 --plot

  # 与传统因子基准对比
  python single_factor_test.py \
      --data ... --registry ... --benchmark
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


from alpha.config import REPO_ROOT, Config, set_global_seed
from alpha.data.data_loader import CSI500Loader
from alpha.mining.preprocessor import DataPreprocessor
from alpha.mining.fm_regression import FamaMacBethRegressor, FMRegressionResult
from alpha.mining.neutralize import fwl_neutralize
from alpha.mining.safe_ops import SafeOps
from alpha.evaluation.backtester import AcademicBacktester

_REPO_ROOT = REPO_ROOT
_LEGACY_REGISTRY = os.path.join(_REPO_ROOT, 'outputs', 'legacy', 'root_factor_output_academic_v81', 'run_20260720_171219', 'registry_academic.json')

logger = logging.getLogger(__name__)

SAFE_MATH = {
    'abs': np.abs,
    'square': np.square,
    'sqrt': SafeOps.safe_sqrt,
    'sign': np.sign,
    'min': np.minimum,
    'max': np.maximum,
    'tanh': SafeOps.safe_tanh,
    'log1p': SafeOps.safe_log1p,
    'inv': SafeOps.safe_inv,
    'sigmoid': SafeOps.safe_sigmoid,
    'div': SafeOps.protected_div,
    '/': SafeOps.protected_div,
}


def resolve_factors(args) -> List[Tuple[str, str, List[str]]]:
    items = []
    if args.registry:
        with open(args.registry) as f:
            registry = json.load(f)
        factors = registry.get('factors', {})
        entries = [(args.id, factors[args.id])] if args.id else list(factors.items())
        for fid, info in entries:
            m = info.get('metrics', info)
            formula = m.get('formula', '')
            feats = m.get('features', info.get('features', []))
            if formula and feats:
                items.append((fid, formula, feats))
            else:
                logger.warning(f"跳过 {fid}: 缺少 formula 或 features")
    else:
        if not args.formula or not args.features:
            raise ValueError("必须提供 --formula + --features 或 --registry")
        items.append(('manual', args.formula, args.features))
    return items


def evaluate_one(
    fid: str, formula: str, features: List[str],
    prep: DataPreprocessor, fm: FamaMacBethRegressor, bt: AcademicBacktester,
    skip_fwl: bool = False,
) -> Optional[Dict]:
    # Store the cost_bps for the report
    cost_bps = bt.cost_bps
    try:
        ds = prep.get_subset(features)
    except ValueError as e:
        logger.warning(f"  {fid}: {e}")
        return None
    X_all = ds['all'][0]
    ns = {f: X_all[:, i] for i, f in enumerate(features)}
    ns.update(SAFE_MATH)
    try:
        pred_all = eval(formula, {"__builtins__": {}}, ns)
    except Exception as e:
        logger.warning(f"  {fid}: eval 失败 ({e})")
        return None
    pred_all = np.where(np.isfinite(pred_all), pred_all, np.nan)

    te_mask = prep.te_mask
    pred_te = pred_all[te_mask]
    te_ret = prep.full_y[te_mask]
    te_dates = prep.full_dates[te_mask]
    te_symbols = prep.full_symbols[te_mask]
    te_amount = prep.full_amount[te_mask]

    valid = np.isfinite(pred_te) & np.isfinite(te_ret)
    if valid.sum() < 500:
        logger.warning(f"  {fid}: 测试集有效样本不足 ({valid.sum()})")
        return None

    pred_te = pred_te[valid]
    te_ret = te_ret[valid]
    te_dates = te_dates[valid]
    te_symbols = te_symbols[valid]
    te_amount = te_amount[valid]

    # ── Backtester: always uses raw (un-neutralized) data ──
    bt_result = bt.run(pred_te, te_ret, te_dates, te_symbols, fid)

    # ── FM regression: FWL-neutralized by default (matches registry) ──
    if not skip_fwl:
        fv, rv = fwl_neutralize(pred_te, te_ret, te_dates, te_symbols, te_amount)
        ok = np.isfinite(fv) & np.isfinite(rv)
        fv_fm, rv_fm, d_fm, s_fm = fv[ok], rv[ok], te_dates[ok], te_symbols[ok]
    else:
        fv_fm, rv_fm, d_fm, s_fm = pred_te, te_ret, te_dates, te_symbols

    fm_result = fm.run(fv_fm, rv_fm, d_fm, s_fm)

    return {
        'fid': fid, 'formula': formula,
        'fm': fm_result, 'bt': bt_result,
        'fwl_applied': not skip_fwl,
        'cost_bps': cost_bps,
    }


def fmt_pval(p: float) -> str:
    if p < 0.001:
        return '<.001***'
    m = '*' if p < 0.05 else ' '
    return f'{p:.3f}{m}'


def print_comparison_results(gross: List[Dict], net: List[Dict], cost_bps: float):
    """Print side-by-side gross vs net comparison table."""
    if not gross or not net:
        return
    print(f"\n{'=' * 125}")
    print(f"  Gross vs Net (成本 {cost_bps:.0f} bps 双边) 对比")
    print(f"{'=' * 125}")
    header = (f"{'ID':<14} {'ICIR':>7} {'t-stat':>8}"
              f" {'Ann(Gross)':>11} {'SR(Gross)':>10}"
              f" {'Ann(Net)':>10} {'SR(Net)':>9}"
              f" {'Drag(bp/m)':>10}")
    print(header)
    print(f"{'-' * 125}")
    for rg, rn in zip(gross, net):
        fm = rg['fm']
        bt_g = rg['bt']
        bt_n = rn['bt']
        ann_g = bt_g['annualized_return'] * 100 if bt_g else 0
        sr_g = bt_g['sharpe_ratio'] if bt_g else 0
        ann_n = bt_n['annualized_return'] * 100 if bt_n else 0
        sr_n = bt_n['sharpe_ratio'] if bt_n else 0
        monthly_drag_bp = (ann_g - ann_n) / 12 * 100 if bt_g and bt_n else 0
        print(f"{rg['fid']:<14} {fm.rank_icir:>7.2f} {fm.t_stat:>8.2f}"
              f" {ann_g:>10.1f}% {sr_g:>9.2f}"
              f" {ann_n:>9.1f}% {sr_n:>8.2f}"
              f" {monthly_drag_bp:>9.1f}")
    print(f"{'-' * 125}")
    print(f"  Cost model: 总 {cost_bps:.0f} bps/月, 每腿 {cost_bps/2:.0f} bps round-trip")
    print()


def print_results(results: List[Dict]):
    fwl_label = ' (FM: FWL 中性化, 回测: raw)' if results[0]['fwl_applied'] else ' (FM: raw, 回测: raw)'
    print(f"\n{'=' * 110}")
    print(f"  单因子回测评估报告{fwl_label}")
    print(f"{'=' * 110}")
    print(f"{'ID':<14} {'t-stat':>8} {'p-value':>10} {'Coef':>9} {'NW-SE':>8} "
          f"{'Rank_IC':>8} {'ICIR':>7} {'L-S':>9} {'年化':>8} {'Sharpe':>7} {'Q1~Q5 月均收益':<20}")
    print(f"{'-' * 110}")
    for r in results:
        fm: FMRegressionResult = r['fm']
        bt = r['bt']
        ann = bt['annualized_return'] if bt else 0.0
        shp = bt['sharpe_ratio'] if bt else 0.0
        q = bt['quantile_monthly_rets'] if bt else {}
        q_str = ' '.join(f"{q.get(f'Q{i+1}', 0)*100:.1f}%" for i in range(5))
        print(f"{r['fid']:<14} {fm.t_stat:>8.2f} {fmt_pval(fm.p_value):>10} "
              f"{fm.coefficient:>9.6f} {fm.std_error:>8.6f} "
              f"{fm.rank_ic_mean:>8.4f} {fm.rank_icir:>7.2f} "
              f"{fm.long_short_spread:>9.5f} "
              f"{ann*100:>7.1f}% {shp:>6.2f}  {q_str}")
    print(f"{'-' * 110}")


def plot_results(results: List[Dict], output_dir: str):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib import rcParams
    except ImportError:
        logger.warning("matplotlib 未安装，跳过画图")
        return
    rcParams['font.family'] = 'Noto Serif CJK SC'
    rcParams['axes.unicode_minus'] = False
    n = len(results)
    fig, axes = plt.subplots(n, 2, figsize=(16, 5 * n), squeeze=False)
    for i, r in enumerate(results):
        bt = r['bt']
        fm: FMRegressionResult = r['fm']
        ax1 = axes[i, 0]
        if bt and 'cumulative_ls' in bt and len(bt['cumulative_ls']) > 1:
            cum = bt['cumulative_ls']
            ax1.plot(cum, lw=1.5, color='#2E86AB')
            ax1.axhline(y=1.0, color='gray', ls='--', lw=0.6)
            ax1.set_title(f"{r['fid']} 多空累计净值 | Sharpe={bt['sharpe_ratio']:.2f} "
                          f"年化={bt['annualized_return']*100:.1f}%")
            ax1.set_ylabel('累计净值')
            ax1.grid(alpha=0.25)
        ax2 = axes[i, 1]
        if bt and 'quantile_monthly_rets' in bt:
            q = bt['quantile_monthly_rets']
            labels = [f'Q{j+1}' for j in range(5)]
            vals = [q.get(l, 0) * 100 for l in labels]
            colors = ['#C73E1D', '#F18F01', '#FFC857', '#A1C9A4', '#2E86AB']
            bars = ax2.bar(labels, vals, color=colors, edgecolor='gray', linewidth=0.5)
            ax2.axhline(y=0, color='black', lw=0.4)
            ax2.set_title(f"{r['fid']} 五分位月均收益 | L-S={fm.long_short_spread*100:.3f}%")
            ax2.set_ylabel('月均收益 (%)')
            ax2.grid(axis='y', alpha=0.25)
            for bar, v in zip(bars, vals):
                ax2.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + (0.01 if v >= 0 else -0.05),
                         f'{v:.2f}%', ha='center', fontsize=9)
    plt.suptitle('单因子回测评估', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    out = os.path.join(output_dir, 'single_factor_eval.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"图表保存至: {out}")


def run_benchmark(
    prep: DataPreprocessor, fm: FamaMacBethRegressor, bt: AcademicBacktester,
    skip_fwl: bool = False,
) -> List[Dict]:
    trad_factors = [f for f in prep.valid_features if f in prep.feature_to_idx]
    logger.info(f"基准对比: 对 {len(trad_factors)} 个传统因子跑回测...")
    results = []
    for i, feat in enumerate(trad_factors):
        r = evaluate_one(f'传统_{feat}', feat, [feat], prep, fm, bt, skip_fwl=skip_fwl)
        if r and r['bt']:
            results.append(r)
        if (i + 1) % 20 == 0:
            logger.info(f"  基准进度: {i+1}/{len(trad_factors)}")
    results.sort(key=lambda x: x['bt']['sharpe_ratio'] if x['bt'] else -999, reverse=True)
    logger.info(f"基准完成: {len(results)}/{len(trad_factors)} 个有效")
    return results


def print_benchmark_comparison(mined: List[Dict], benchmark: List[Dict]):
    if not benchmark:
        return
    n_total = len(benchmark)
    mined_sr = [r['bt']['sharpe_ratio'] for r in mined if r and r['bt']]
    all_srs = [r['bt']['sharpe_ratio'] for r in benchmark if r and r['bt']]
    all_srs_sorted = sorted(all_srs, reverse=True)

    print(f"\n{'=' * 130}")
    print(f"  挖掘因子 vs 传统因子基准对比 (按 Sharpe 排名)")
    print(f"{'=' * 130}")
    header = (f"{'ID':<14} {'IC':>7} {'ICIR':>7} {'t-stat':>8}"
              f" {'年化':>8} {'Sharpe':>7} {'LS月均':>8} {'排名':>7}")
    print(header)
    print(f"{'-' * 130}")

    def shp(r):
        return r['bt']['sharpe_ratio'] if r and r['bt'] else -999

    def ann(r):
        return r['bt']['annualized_return'] * 100 if r and r['bt'] else 0

    def ls(r):
        return r['fm'].long_short_spread * 100 if r and r['fm'] else 0

    benchmark_sorted = sorted(benchmark, key=shp, reverse=True)
    n_total = len(benchmark_sorted)

    mined_srs = [shp(r) for r in mined]

    for r in mined:
        sr = shp(r)
        rank = sum(1 for b in benchmark_sorted if shp(b) > sr) + 1
        pct = rank / n_total * 100
        fm_r = r['fm']
        bt_r = r['bt']
        print(f"{r['fid']:<14} {fm_r.rank_ic_mean:>7.4f} {fm_r.rank_icir:>7.2f} {fm_r.t_stat:>8.2f} "
              f"{ann(r):>6.1f}% {sr:>6.2f} {ls(r):>7.3f}% "
              f"{rank}/{n_total} (前{pct:.0f}%)  PySR挖掘")

    print(f"{'-' * 130}")
    for rank_i, r in enumerate(benchmark_sorted[:5]):
        fm_r = r['fm']
        print(f"{r['fid']:<14} {fm_r.rank_ic_mean:>7.4f} {fm_r.rank_icir:>7.2f} {fm_r.t_stat:>8.2f} "
              f"{ann(r):>6.1f}% {shp(r):>6.2f} {ls(r):>7.3f}%  传统 #{rank_i + 1}")

    mid_idx = n_total // 2
    r_mid = benchmark_sorted[mid_idx]
    fm_mid = r_mid['fm']
    print(f"{'-' * 130}")
    print(f"{'中位数':<14} {fm_mid.rank_ic_mean:>7.4f} {fm_mid.rank_icir:>7.2f} {fm_mid.t_stat:>8.2f} "
          f"{ann(r_mid):>6.1f}% {shp(r_mid):>6.2f} {ls(r_mid):>7.3f}%  传统中位数")

    r_last = benchmark_sorted[-1]
    fm_last = r_last['fm']
    print(f"{'末位':<14} {fm_last.rank_ic_mean:>7.4f} {fm_last.rank_icir:>7.2f} {fm_last.t_stat:>8.2f} "
          f"{ann(r_last):>6.1f}% {shp(r_last):>6.2f} {ls(r_last):>7.3f}%  传统末位")
    print(f"{'-' * 130}")
    print(f"  传统因子总数: {n_total}  |  挖掘因子 Sharpe 排名: 前 {pct:.0f}%")
    print()


def main():
    parser = argparse.ArgumentParser(description='单因子回测评估工具')
    parser.add_argument('--data', default=os.path.join(_REPO_ROOT, 'data', 'csi500_daily_2020-07-20_to_2026-07-19.parquet'), help='数据路径 (CSV/Parquet)')
    parser.add_argument('--formula', type=str, help='因子公式, 如 "square(rev_1d*0.1158)"')
    parser.add_argument('--features', type=str, nargs='+', help='因子依赖的特征名列表')
    parser.add_argument('--registry', type=str, default=_LEGACY_REGISTRY, help='注册表 JSON 路径 (与 --formula 互斥)')
    parser.add_argument('--id', type=str, help='注册表中的因子 ID (需配合 --registry)')
    parser.add_argument('--no-fwl', action='store_true', help='跳过 FWL 市值中性化 (FM 使用 raw 数据)')
    parser.add_argument('--cost-bps', type=float, default=15, help='月度双边交易成本(bps), 均摊至 long/short 两腿')
    parser.add_argument('--compare-cost', action='store_true', help='对比模式: 同时输出 gross (cost=0) 与 net (cost=cost-bps) 结果')
    parser.add_argument('--plot', action='store_true', default=True, help='生成累计净值曲线 + 五分位柱状图')
    parser.add_argument('--output', type=str, default=None, help='输出目录 (默认自动创建)')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--pool-size', type=int, default=999,
                        help='特征聚类目标数 (默认 999 即跳过聚类)')
    parser.add_argument('--benchmark', action='store_true', default=True,
                        help='与传统因子基准对比: 跑全部传统因子回测, 按 Sharpe 排名对比')
    args = parser.parse_args()

    set_global_seed(args.seed)
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s | %(levelname)-7s | %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    factors = resolve_factors(args)
    if not factors:
        logger.error("没有找到可评估的因子")
        sys.exit(1)
    logger.info(f"待评估因子数: {len(factors)}")

    logger.info("加载数据...")
    loader = CSI500Loader(path=args.data)
    df = loader.load()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
    logger.info(f"数据加载完成: {len(df)} 行, {df['symbol'].nunique()} 只股票")

    prev = Config.TARGET_FACTOR_POOL_SIZE
    Config.TARGET_FACTOR_POOL_SIZE = args.pool_size
    logger.info("数据预处理...")
    prep = DataPreprocessor()
    prep.prepare_full_pool(df)
    Config.TARGET_FACTOR_POOL_SIZE = prev
    logger.info(f"特征数: {len(prep.valid_features)} | 测试集: {prep.te_mask.sum():,} 样本")

    fm = FamaMacBethRegressor(nw_lags=Config.NW_LAGS, n_groups=5)
    use_compare = args.compare_cost and args.cost_bps > 0
    cost_gross = 0.0
    cost_net = args.cost_bps if use_compare else args.cost_bps

    if use_compare:
        bt_gross = AcademicBacktester(top_quantile=0.2, cost_bps=0.0)
        bt_net = AcademicBacktester(top_quantile=0.2, cost_bps=cost_net)
        bt_pairs = [('GROSS', 0.0, bt_gross), (f'NET_{int(cost_net)}bps', cost_net, bt_net)]
    else:
        bt = AcademicBacktester(top_quantile=0.2, cost_bps=cost_net)
        bt_pairs = [(f'cost_{int(cost_net)}bps' if cost_net > 0 else 'no_cost', cost_net, bt)]

    all_results = {label: [] for label, _, _ in bt_pairs}

    for fid, formula, features in factors:
        logger.info(f"评估: {fid}")
        missing = [f for f in features if f not in prep.feature_to_idx]
        if missing:
            logger.warning(f"  {fid}: 缺失特征 {missing}")
            continue
        for label, cost_val, bt_obj in bt_pairs:
            r = evaluate_one(fid, formula, features, prep, fm, bt_obj, skip_fwl=args.no_fwl)
            if r:
                all_results[label].append(r)
                fm_res: FMRegressionResult = r['fm']
                bt_res = r['bt']
                if bt_res:
                    logger.info(f"  [{label}] ✅ t={fm_res.t_stat:.2f}  ICIR={fm_res.rank_icir:.2f}  "
                                f"年化={bt_res['annualized_return']*100:.1f}%  "
                                f"Sharpe={bt_res['sharpe_ratio']:.2f}")
                else:
                    logger.info(f"  [{label}] ✅ t={fm_res.t_stat:.2f}  ICIR={fm_res.rank_icir:.2f}")

    # Pick the primary result set for downstream use
    primary_label = 'GROSS' if use_compare else list(all_results.keys())[0]
    results = all_results.get(primary_label, [])
    if not results:
        logger.error("所有因子评估均失败")
        sys.exit(1)

    if use_compare:
        print_comparison_results(
            all_results.get('GROSS', []),
            all_results.get(f'NET_{int(cost_net)}bps', []),
            cost_net,
        )
    else:
        print_results(results)

    if args.benchmark:
        bt_bench = AcademicBacktester(top_quantile=0.2, cost_bps=cost_net)
        benchmark_results = run_benchmark(prep, fm, bt_bench, skip_fwl=args.no_fwl)
        print_benchmark_comparison(results, benchmark_results)
        all_results['benchmark'] = benchmark_results

    output_dir = args.output or f"single_factor_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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

    for label, label_results in all_results.items():
        if not label_results:
            continue
        out_data = []
        for r in label_results:
            bt_res = r['bt']
            out_data.append({
                'fid': r['fid'],
                'formula': r['formula'],
                'fwl_applied': r['fwl_applied'],
                'cost_bps': r['cost_bps'],
                'fm': serialize(r['fm']),
                'backtest': serialize(bt_res) if bt_res else None,
            })
        json_path = os.path.join(output_dir, f'eval_results_{label.lower()}.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(out_data, f, indent=2, ensure_ascii=False)
        logger.info(f"结果保存至: {json_path}")

    if args.plot and results:
        plot_results(results, output_dir)

    print(f"\n输出目录: {os.path.abspath(output_dir)}")


if __name__ == '__main__':
    main()
