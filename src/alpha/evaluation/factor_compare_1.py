#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V8 - 因子对比评估: 传统基础因子 vs GFN-SR 生成因子
全部通过 DataPreprocessor 统一管线，确保同源可比。
输出 Markdown 表格: FM t-stat / ICIR / Rank IC / 多空收益差 / 夏普比率
"""

import sys, os, json, logging, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import rcParams
rcParams['font.family'] = 'Noto Serif CJK SC'
rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore')

from alpha.config import Config, set_global_seed
from alpha.data.data_loader import CSI500Loader
from alpha.mining.preprocessor import DataPreprocessor
from alpha.mining.fm_regression import FamaMacBethRegressor
from alpha.mining.safe_ops import TimeSeriesOps
from alpha.evaluation.backtester import AcademicBacktester

logging.getLogger().handlers.clear()
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s | %(levelname)-7s | %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('FactorComparator')


# ============================================================================
# 📊 因子对比评估器
# ============================================================================

class FactorComparator:
    def __init__(self, data_path: str, registry_path: str, debug: bool = False):
        self.data_path = data_path
        self.registry_path = registry_path
        self.debug = debug
        self.loader = CSI500Loader(path=data_path)
        self.fm = FamaMacBethRegressor(nw_lags=Config.NW_LAGS)
        self.bt = AcademicBacktester(top_quantile=0.2)
        self.prep: DataPreprocessor = None

    def run(self) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
        set_global_seed(42)

        # ---- 1. Load data ----
        logger.info("📥 加载数据...")
        df = self.loader.load()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['symbol', 'date']).reset_index(drop=True)

        # ---- 2. DataPreprocessor (bypass clustering) ----
        Config.TARGET_FACTOR_POOL_SIZE = 999
        logger.info("🔧 DataPreprocessor (跳过聚类, 保留全部特征)...")
        self.prep = DataPreprocessor()
        self.prep.prepare_full_pool(df)

        n_feat = len(self.prep.valid_features)
        logger.info(f"✅ 保留 {n_feat} 个特征 (全量)")

        # ---- 3. Test set ----
        te_mask = self.prep.te_mask
        te_ret = self.prep.full_y[te_mask].astype(np.float64)
        te_dates = self.prep.full_dates[te_mask]
        te_symbols = self.prep.full_symbols[te_mask]
        te_amount = self.prep.full_amount[te_mask].astype(np.float64)

        logger.info(f"📅 测试集: {te_mask.sum():,} 样本, "
                    f"{pd.to_datetime(te_dates).nunique()} 个交易日")

        # ---- 4. Evaluate (raw + FWL size-neutralized) ----
        trad_raw, trad_fwl = self._eval_traditional(te_ret, te_dates, te_symbols, te_amount)
        gfn_raw, gfn_fwl = self._eval_gfn(te_ret, te_dates, te_symbols, te_amount)

        return trad_raw, trad_fwl, gfn_raw, gfn_fwl

    # ------------------------------------------------------------------
    #  传统因子: 逐一取标准化后的单因子列 → FM + 回测
    # ------------------------------------------------------------------
    def _eval_traditional(self, te_ret: np.ndarray, te_dates: np.ndarray,
                          te_symbols: np.ndarray,
                          te_amount: np.ndarray) -> Tuple[List[Dict], List[Dict]]:
        logger.info("\n" + "=" * 60 + "\n📊 评估传统基础因子\n" + "=" * 60)

        feat_names = list(self.prep.valid_features)
        feat2idx = {f: i for i, f in enumerate(feat_names)}

        if self.debug:
            import random; random.seed(42)
            feat_names = random.sample(feat_names, min(5, len(feat_names)))
            logger.info(f"🔍 Debug: {len(feat_names)} 个传统因子")

        te_X = self.prep.full_X[self.prep.te_mask]

        raw_results = []
        fwl_results = []
        for col in tqdm(feat_names, desc="评估传统因子"):
            idx = feat2idx[col]
            fv = te_X[:, idx]
            r_raw = self._eval_one(fv, te_ret, te_dates, te_symbols, col)
            r_fwl = self._eval_one_fwl(fv, te_ret, te_dates, te_symbols, te_amount, col)
            if r_raw['FM_tstat'] != 0.0:
                raw_results.append(r_raw)
                fwl_results.append(r_fwl)
        return raw_results, fwl_results

    # ------------------------------------------------------------------
    #  GFN-SR: 注册表公式 → prep.get_subset() 取标准化特征 → eval → FM+回测
    # ------------------------------------------------------------------
    def _eval_gfn(self, te_ret: np.ndarray, te_dates: np.ndarray,
                  te_symbols: np.ndarray,
                  te_amount: np.ndarray) -> Tuple[List[Dict], List[Dict]]:
        logger.info("\n" + "=" * 60 + "\n📊 评估 GFN-SR 生成因子\n" + "=" * 60)

        if not os.path.exists(self.registry_path):
            logger.error(f"❌ 注册表不存在: {self.registry_path}")
            return [], []

        with open(self.registry_path) as f:
            registry = json.load(f)

        factors = registry.get('factors', {})
        logger.info(f"📦 {len(factors)} 个 GFN-SR 因子")

        if self.debug:
            import random; random.seed(42)
            keys = random.sample(list(factors.keys()), min(3, len(factors)))
            factors = {k: factors[k] for k in keys}
            logger.info(f"🔍 Debug: {len(factors)} 个 GFN-SR 因子")

        safe_math = {
            'abs': np.abs, 'square': np.square, 'sqrt': np.sqrt,
            'sign': np.sign, 'min': np.minimum, 'max': np.maximum,
            'tanh': np.tanh,
            'log1p': lambda x: np.log1p(np.abs(x)) * np.sign(x),
            'inv': lambda x: np.where(np.abs(x) > 1e-8, 1.0 / x, 0.0),
            'sigmoid': lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500))),
        }

        raw_results = []
        fwl_results = []
        for fid, info in tqdm(factors.items(), desc="评估 GFN-SR 因子"):
            m = info.get('metrics', info)
            formula = m.get('formula', '')
            feats = m.get('features', info.get('features', []))
            if not formula or not feats:
                continue

            try:
                ds = self.prep.get_subset(feats)
                X_all = ds['all'][0]
                ns = {f: X_all[:, i] for i, f in enumerate(feats)}
                ns.update(safe_math)

                pred_all = eval(formula, {"__builtins__": {}}, ns)
                pred_all = np.where(np.isfinite(pred_all), pred_all, np.nan)
                pred_te = pred_all[self.prep.te_mask]

                r_raw = self._eval_one(pred_te, te_ret, te_dates, te_symbols, fid)
                r_fwl = self._eval_one_fwl(pred_te, te_ret, te_dates, te_symbols, te_amount, fid)
                for r in (r_raw, r_fwl):
                    r['formula'] = formula
                if r_raw['FM_tstat'] != 0.0:
                    raw_results.append(r_raw)
                    fwl_results.append(r_fwl)
            except Exception as e:
                logger.warning(f"⚠️ {fid} 跳过: {e}")

        return raw_results, fwl_results

    # ------------------------------------------------------------------
    #  单因子 FM + 回测
    # ------------------------------------------------------------------
    def _eval_one(self, factor_vals: np.ndarray, ret: np.ndarray,
                  dates: np.ndarray, symbols: np.ndarray,
                  name: str) -> Dict:
        fm_result = self.fm.run(factor_vals, ret, dates, symbols)
        bt_result = self.bt.run(factor_vals, ret, dates, symbols, name)
        return {
            'name': name,
            'FM_tstat': fm_result.t_stat,
            'ICIR': fm_result.rank_icir,
            'Rank_IC': fm_result.rank_ic_mean,
            'LS_spread': fm_result.long_short_spread,
            'Sharpe': bt_result['sharpe_ratio'] if bt_result else 0.0,
        }

    # ------------------------------------------------------------------
    #  单因子 FWL Size 中性化评估 (同 V9.1 挖矿管线)
    # ------------------------------------------------------------------
    def _eval_one_fwl(self, factor_vals: np.ndarray, ret: np.ndarray,
                      dates: np.ndarray, symbols: np.ndarray,
                      amount: np.ndarray, name: str) -> Dict:
        if amount is None or np.all(amount == 0):
            return self._eval_one(factor_vals, ret, dates, symbols, name)

        ln_amt = np.log(np.maximum(amount, 1.0))
        df_n = pd.DataFrame({
            'date': pd.to_datetime(dates).normalize(),
            'pred': factor_vals, 'ret': ret, 'ln_amt': ln_amt,
            'symbol': symbols
        }).dropna()

        pure_pred = np.full(len(df_n), np.nan, dtype=np.float64)
        pure_ret = np.full(len(df_n), np.nan, dtype=np.float64)

        for dt, grp in df_n.groupby('date'):
            idx = grp.index
            if len(grp) < 30:
                pure_pred[idx] = grp['pred'].values
                pure_ret[idx] = grp['ret'].values
                continue
            X = sm.add_constant(grp['ln_amt'].values)
            try:
                pure_pred[idx] = sm.OLS(grp['pred'].values, X).fit().resid
                pure_ret[idx] = sm.OLS(grp['ret'].values, X).fit().resid
            except Exception:
                pure_pred[idx] = grp['pred'].values
                pure_ret[idx] = grp['ret'].values

        valid = np.isfinite(pure_pred) & np.isfinite(pure_ret)
        fm = self.fm.run(pure_pred[valid], pure_ret[valid],
                         df_n['date'].values[valid], df_n['symbol'].values[valid])
        bt = self.bt.run(pure_pred[valid], pure_ret[valid],
                         df_n['date'].values[valid], df_n['symbol'].values[valid],
                         f"{name}_fwl")
        return {
            'name': name,
            'FM_tstat': fm.t_stat,
            'ICIR': fm.rank_icir,
            'Rank_IC': fm.rank_ic_mean,
            'LS_spread': fm.long_short_spread,
            'Sharpe': bt['sharpe_ratio'] if bt else 0.0,
        }


# ============================================================================
# 📋 Markdown 表格生成
# ============================================================================

def _build_rows(trad: List[Dict], gfn: List[Dict]) -> List[Tuple[str, Dict]]:
    trad_sorted = sorted(trad, key=lambda x: abs(x['FM_tstat']), reverse=True)
    gfn_sorted = sorted(gfn, key=lambda x: abs(x['FM_tstat']), reverse=True)

    def _get(r, k):
        return r.get(k, 0.0) if isinstance(r, dict) else r[k]

    rows = []
    for i, r in enumerate(trad_sorted[:5], 1):
        rows.append((f'传统 Top {i} ({r.get("name","?")})', r))

    med = {}
    for k in ['FM_tstat', 'ICIR', 'Rank_IC', 'LS_spread', 'Sharpe']:
        med[k] = float(np.median([_get(r, k) for r in trad]))
    rows.append(('传统中位数', med))

    if gfn_sorted:
        rows.append(('GFN-SR (最佳)', gfn_sorted[0]))
    else:
        rows.append(('GFN-SR (最佳)', {k: 0.0 for k in ['FM_tstat', 'ICIR', 'Rank_IC', 'LS_spread', 'Sharpe']}))

    gfn_mean = {}
    for k in ['FM_tstat', 'ICIR', 'Rank_IC', 'LS_spread', 'Sharpe']:
        gfn_mean[k] = float(np.mean([_get(r, k) for r in gfn])) if gfn else 0.0
    rows.append(('GFN-SR (均值)', gfn_mean))
    return rows


def _table_lines(rows: List[Tuple[str, Dict]], title: str, note: str) -> List[str]:
    lines = [
        f"## {title}",
        "",
        f"> {note}",
        "> 排序标准: |FM t-stat| 降序  | 同一管线 (DataPreprocessor 跳过聚类)",
        "",
        "| 因子类别 | FM t-stat | ICIR | Rank IC | 多空收益差 | 夏普比率 |",
        "|----------|-----------|------|---------|------------|----------|",
    ]
    fmt = "| {:<12} | {:>8.2f} | {:>5.3f} | {:>7.4f} | {:>9.5f} | {:>7.2f} |"
    for label, r in rows:
        fmt_fm = r.get('FM_tstat', 0.0)
        fmt_icir = r.get('ICIR', 0.0)
        fmt_ric = r.get('Rank_IC', 0.0)
        fmt_ls = r.get('LS_spread', 0.0)
        fmt_sh = r.get('Sharpe', 0.0)
        lines.append(fmt.format(label, fmt_fm, fmt_icir, fmt_ric, fmt_ls, fmt_sh))
    return lines


def generate_tables(trad_raw: List[Dict], trad_fwl: List[Dict],
                    gfn_raw: List[Dict], gfn_fwl: List[Dict],
                    n_trad: int, n_gfn: int) -> str:
    raw_rows = _build_rows(trad_raw, gfn_raw)
    fwl_rows = _build_rows(trad_fwl, gfn_fwl)

    lines = []
    # Table 1: Raw
    lines += _table_lines(raw_rows,
        "因子对比评估 — 原始因子（Raw）",
        f"测试集: {n_trad} 个传统因子 vs {n_gfn} 个 GFN-SR 生成因子 | 无中性化")
    lines.append('')
    # Table 2: FWL Size-neutralized
    lines += _table_lines(fwl_rows,
        "因子对比评估 — 市值中性化（Size-Neutralized）",
        f"测试集: {n_trad} 个传统因子 vs {n_gfn} 个 GFN-SR 生成因子 | 逐截面 FWL ln(amount) 正交化")

    return '\n'.join(lines) + '\n'


# ============================================================================
# 📊 对比图生成
# ============================================================================

def generate_comparison_plots(trad_raw, trad_fwl, gfn_raw, gfn_fwl, output_dir):
    METRICS = ['FM_tstat', 'ICIR', 'Rank_IC', 'LS_spread', 'Sharpe']
    LABELS = ['FM t-stat', 'ICIR', 'Rank IC', '多空收益差', '夏普比率']

    trad_best = {k: max(trad_raw, key=lambda x: abs(x.get(k, 0))).get(k, 0) for k in METRICS}
    gfn_best = {k: max(gfn_raw, key=lambda x: abs(x.get(k, 0))).get(k, 0) for k in METRICS}
    trad_med = {k: float(np.median([r.get(k, 0) for r in trad_raw])) for k in METRICS}
    gfn_mean = {k: float(np.mean([r.get(k, 0) for r in gfn_raw])) if gfn_raw else 0 for k in METRICS}

    trad_best_f = {k: max(trad_fwl, key=lambda x: abs(x.get(k, 0))).get(k, 0) for k in METRICS}
    gfn_best_f = {k: max(gfn_fwl, key=lambda x: abs(x.get(k, 0))).get(k, 0) for k in METRICS}
    trad_med_f = {k: float(np.median([r.get(k, 0) for r in trad_fwl])) for k in METRICS}
    gfn_mean_f = {k: float(np.mean([r.get(k, 0) for r in gfn_fwl])) if gfn_fwl else 0 for k in METRICS}

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    groups = ['传统最佳', 'GFN-SR最佳', '传统中位数', 'GFN-SR均值']
    hatches = ['', '///', '', '///']

    panels = [
        ('原始因子 (Raw)', trad_best, gfn_best, trad_med, gfn_mean),
        ('市值中性化 (Size-Neutralized)', trad_best_f, gfn_best_f, trad_med_f, gfn_mean_f),
    ]
    for row, (title_prefix, trad_b, gfn_b, trad_m, gfn_m) in enumerate(panels):
        for col, (data_list, bar_title, grp_lbl) in enumerate([
            ([trad_b, gfn_b], '最佳因子对比', (groups[0], groups[1])),
            ([trad_m, gfn_m], '集中趋势对比', (groups[2], groups[3])),
        ]):
            ax = axes[row, col]
            x = np.arange(len(METRICS))
            width = 0.2
            for i, (d, label) in enumerate(zip(data_list, grp_lbl)):
                vals = [d.get(m, 0) for m in METRICS]
                ax.bar(x + i * width, vals, width, label=label,
                       color=colors[i], hatch=hatches[i], edgecolor='gray', linewidth=0.5)
            ax.set_title(f'{title_prefix} — {bar_title}', fontsize=13, fontweight='bold')
            ax.set_xticks(x + width / 2)
            ax.set_xticklabels(LABELS, fontsize=11)
            ax.axhline(y=0, color='black', linewidth=0.4)
            ax.legend(fontsize=10, loc='best')
            ax.grid(axis='y', alpha=0.3)

    plt.subplots_adjust(hspace=0.28, wspace=0.22)

    # Summary text box
    def _best_name(lst):
        return max(lst, key=lambda x: abs(x.get('FM_tstat', 0))).get('name', '?') if lst else 'N/A'

    text = (
        f"Raw: 传统最佳={_best_name(trad_raw)} | GFN-SR最佳={_best_name(gfn_raw)}\n"
        f"FWL: 传统最佳={_best_name(trad_fwl)} | GFN-SR最佳={_best_name(gfn_fwl)}\n"
        f"传统因子数={len(trad_raw)} | GFN-SR因子数={len(gfn_raw)}"
    )
    fig.text(0.5, 0.01, text, ha='center', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle('传统基础因子 vs GFN-SR 生成因子 — 全面对比', fontsize=16, fontweight='bold', y=0.98)

    out = os.path.join(output_dir, 'comparison_plot.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"✅ 对比图: {out}")


# ============================================================================
# 🚀 主入口
# ============================================================================

def main(data_path: str, registry_path: str, output_root: str = None, debug: bool = False):
    comparator = FactorComparator(data_path, registry_path, debug=debug)
    trad_raw, trad_fwl, gfn_raw, gfn_fwl = comparator.run()

    logger.info("\n" + "=" * 60 + "\n📋 生成对比表格\n" + "=" * 60)
    table = generate_tables(trad_raw, trad_fwl, gfn_raw, gfn_fwl,
                            len(trad_raw), len(gfn_raw))
    print('\n' + table)

    output_dir = os.path.join(Config.OUTPUT_ROOT,
                              f'comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(output_dir, exist_ok=True)

    out_path = os.path.join(output_dir, 'comparison_report.md')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(table)

    def _serialize(v):
        if isinstance(v, (np.floating, float)): return float(v)
        if isinstance(v, (np.integer, int)): return int(v)
        return v

    json_path = os.path.join(output_dir, 'comparison_data.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'raw': {
                'traditional': [{k: _serialize(v) for k, v in r.items()} for r in trad_raw],
                'gfn': [{k: _serialize(v) for k, v in r.items()} for r in gfn_raw],
            },
            'size_neutralized': {
                'traditional': [{k: _serialize(v) for k, v in r.items()} for r in trad_fwl],
                'gfn': [{k: _serialize(v) for k, v in r.items()} for r in gfn_fwl],
            },
            'metadata': {
                'data_path': data_path,
                'registry_path': registry_path,
                'n_traditional': len(trad_raw),
                'n_gfn': len(gfn_raw),
                'timestamp': datetime.now().isoformat(),
            }
        }, f, indent=2, ensure_ascii=False)

    logger.info(f"✅ 报告: {out_path}")
    logger.info(f"✅ 数据: {json_path}")

    generate_comparison_plots(trad_raw, trad_fwl, gfn_raw, gfn_fwl, output_dir)

    if trad_raw and gfn_raw:
        tb_r = max(trad_raw, key=lambda x: abs(x['FM_tstat']))
        gb_r = max(gfn_raw, key=lambda x: abs(x['FM_tstat']))
        tb_f = max(trad_fwl, key=lambda x: abs(x['FM_tstat']))
        gb_f = max(gfn_fwl, key=lambda x: abs(x['FM_tstat']))
        logger.info(
            f"\n📊 Raw 对比:\n"
            f"  传统最佳: {tb_r['name']}  | FM t={tb_r['FM_tstat']:.2f}  ICIR={tb_r['ICIR']:.3f}  Sharpe={tb_r['Sharpe']:.2f}\n"
            f"  GFN-SR最佳: {gb_r['name']} | FM t={gb_r['FM_tstat']:.2f}  ICIR={gb_r['ICIR']:.3f}  Sharpe={gb_r['Sharpe']:.2f}"
        )
        logger.info(
            f"\n📊 Size-Neutralized 对比:\n"
            f"  传统最佳: {tb_f['name']}  | FM t={tb_f['FM_tstat']:.2f}  ICIR={tb_f['ICIR']:.3f}  Sharpe={tb_f['Sharpe']:.2f}\n"
            f"  GFN-SR最佳: {gb_f['name']} | FM t={gb_f['FM_tstat']:.2f}  ICIR={gb_f['ICIR']:.3f}  Sharpe={gb_f['Sharpe']:.2f}"
        )


if __name__ == "__main__":
    from alpha.config import REPO_ROOT as _REPO_ROOT

    import argparse
    parser = argparse.ArgumentParser(description="传统因子 vs GFN-SR 因子对比评估")
    parser.add_argument('--data', type=str,
        default=os.path.join(_REPO_ROOT, 'data', 'csi500_daily_2020-07-20_to_2026-07-19.parquet'),
        help='日频数据路径')
    parser.add_argument('--registry', type=str,
        default=None,
        help='GFN-SR 注册表路径')
    parser.add_argument('--debug', action='store_true', help='快速验证模式')
    args = parser.parse_args()

    main(args.data, args.registry, debug=args.debug)
