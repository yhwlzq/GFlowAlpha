#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
学术级日频单因子挖掘系统 — Walk-Forward 滚动窗口实验 (36:12:12 月 = 3年训练/1年验证/1年测试).

默认窗口: train = 36 月, val = 12 月, test = 12 月 (年度调仓周期),
test 窗以 12 月为步长不重叠滚动, 窗口数由数据长度自动确定.
默认数据为两份中证500 parquet (2016-06-30~2021-06-30 与 2021-06-30~2026-06-30),
脚本会自动拼接去重 (2021-06-30 交界日仅保留一次), 形成 2016~2026 十年连续面板,
按 36:12:12 滚动恰好得到 w1~w6 六个完整窗口; 第七窗 test 已越过数据末端被自动丢弃.

流程:
  阶段 A — 逐窗口独立挖掘: 每个窗口创建全新的 MiningOrchestrator (GFlowNet+PySR 从头训练),
           仅用本窗口 train/val 挖掘、test 做样本外评估, 并回测 Top-3.
  阶段 B — 跨窗口稳健性: 把窗口 i 挖出的达标因子在后续窗口 j>i 的 test 集上重新做
           FM 评估, 计算跨窗命中率 (|t-stat|>=阈值 且符号与发现窗口一致) 与平均 IC/ICIR.
  汇总   — 输出 walkforward_summary.json / cross_window_robustness.csv / 文本报告.

支持窗口选择: --windows 使用 1-based 索引 (w1=第1窗), 支持 "1,3" 或 "1-6" 形式;
--list-windows 可只打印各窗口时间段而不挖掘. 阶段 B 的跨窗评估仅在被选窗口内进行.

因子池 (层次聚类正交化筛选) 基于全量历史构建, 相当于清洗后的候选特征全集;
各窗口仅用不同时间段训练/验证/测试, 标准化按各自窗口 train 拟合, 不跨窗口泄漏.
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["JULIA_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"
os.environ["PYTHON_JULIACALL_THREADS"] = "1"
os.environ["PYTHON_JULIACALL_OPTLEVEL"] = "2"
try:
    import juliacall  # noqa: F401 — 在 torch 前加载, 避免 segfault
except ImportError:
    pass

import json
import logging
import os
import sys
import argparse
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from alpha.evaluation.backtester import AcademicBacktester
from alpha.config import Config, set_global_seed
from alpha.data.data_loader import CSI500Loader
from alpha.mining.preprocessor import DataPreprocessor
from alpha.mining.orchestrator import MiningOrchestrator
from alpha.mining.fm_regression import FamaMacBethRegressor
from alpha.mining.neutralize import fwl_neutralize

logger = logging.getLogger('WalkForward')

SAFE_MATH = {
    'abs': np.abs, 'square': np.square, 'sqrt': lambda x: np.sqrt(np.abs(x)),
    'sign': np.sign, 'min': np.minimum, 'max': np.maximum,
    'tanh': np.tanh, 'log1p': np.log1p,
    'inv': lambda x: np.where(np.abs(x) > 1e-10, 1.0 / x, 0.0),
    'sigmoid': lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500))),
    'div': lambda a, b: np.where(np.abs(b) > 1e-10, a / b, 0.0),
    '/': lambda a, b: np.where(np.abs(b) > 1e-10, a / b, 0.0),
}


def evaluate_formula_on_test(prep: DataPreprocessor, feats: List[str], formula: str,
                             ns_extra: Dict, fm_regressor: FamaMacBethRegressor):
    """在当前窗口的 test 集上评估公式, 返回 (raw_pred_te, FMRegressionResult).

    特征值取自当前窗口标准化后的 full_X, 与 orchestrator 行为一致.
    """
    ds = prep.get_subset(feats)
    X_all, y_all, d_all, s_all, amt_all = ds['all']
    ns = {f: X_all[:, i] for i, f in enumerate(feats)}
    ns.update(ns_extra)
    pred_all = eval(formula, {"__builtins__": {}}, ns)
    pred_all = np.where(np.isfinite(pred_all), pred_all, np.nan)

    te_mask = prep.te_mask
    pred_te = pred_all[te_mask]
    test_ret = y_all[te_mask]
    test_dates = d_all[te_mask]
    test_symbols = s_all[te_mask]
    test_amount = amt_all[te_mask]

    if Config.USE_RESIDUAL_NEUTRALIZATION:
        pure_pred, pure_ret = fwl_neutralize(pred_te, test_ret, test_dates, test_symbols, test_amount)
    else:
        pure_pred, pure_ret = pred_te, test_ret

    fm_result = fm_regressor.run(
        factor_values=pure_pred, returns=pure_ret,
        dates=test_dates, symbols=test_symbols,
    )
    return pred_te, fm_result


def parse_windows(windows_str: str) -> List[int]:
    """解析窗口选择字符串, 支持 '1,3' 和 '1-6' 格式 (1-based 窗口索引)."""
    result = []
    for part in windows_str.split(','):
        part = part.strip()
        if '-' in part:
            start, end = part.split('-', 1)
            result.extend(range(int(start), int(end) + 1))
        else:
            result.append(int(part))
    return sorted(set(result))


def load_combined_data(data_arg: str) -> pd.DataFrame:
    """加载一个或多个 parquet/csv 数据文件 (逗号分隔), 拼接并去重.

    两份 csi500 parquet 在 2021-06-30 交界日重叠, 这里按 (symbol, date)
    去重后排序, 形成连续面板.
    """
    paths = [p.strip() for p in data_arg.split(',') if p.strip()]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"数据文件不存在: {missing}")
    frames = []
    for p in paths:
        logger.info(f"加载数据: {p}")
        frames.append(CSI500Loader(path=p).load())
    df = pd.concat(frames, ignore_index=True)
    n0 = len(df)
    df = df.drop_duplicates(subset=['symbol', 'date']).sort_values(
        ['symbol', 'date']).reset_index(drop=True)
    df['date'] = pd.to_datetime(df['date'])
    if len(df) < n0:
        logger.info(f"拼接去重: {n0:,} -> {len(df):,} 行 (移除重叠边界日 {n0 - len(df):,} 行)")
    logger.info(f"合并数据覆盖: {df['date'].min().date()} ~ {df['date'].max().date()} "
                f"| {df['symbol'].nunique()} 只 | {df['date'].nunique()} 个交易日")
    return df


def main(data_path: str, max_trials: int, time_limit: int, market: str,
         train_months: int, val_months: int, test_months: int,
         max_windows: Optional[int], windows: Optional[List[int]] = None,
         list_windows: bool = False):
    set_global_seed()
    Config.MARKET = market
    Config.SPLIT_MODE = 'walkforward'
    Config.WF_TRAIN_MONTHS = train_months
    Config.WF_VAL_MONTHS = val_months
    Config.WF_TEST_MONTHS = test_months
  

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = os.path.join("factor_output_academic_v81", f"walkforward_{timestamp}")
    os.makedirs(root, exist_ok=True)
    Config.OUTPUT_DIR = root

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-7s | %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(root, 'mining.log'), mode='w', encoding='utf-8')
        ],
        force=True
    )

    logger.info(f"加载日频数据: {data_path}")
    df = load_combined_data(data_path)

    required_cols = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'amount']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"数据必须包含列: {required_cols}")

    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=['close', 'volume']).sort_values(['symbol', 'date']).reset_index(drop=True)

    prep = DataPreprocessor()
    prep.prepare_full_pool(df)

    n_windows = prep.n_windows
    if max_windows and max_windows > 0:
        n_windows = min(n_windows, max_windows)
    if n_windows <= 0:
        logger.error("无有效 walk-forward 窗口, 退出.")
        return

    if list_windows:
        logger.info(f"\n{'=' * 60}\n可用窗口一览 ({train_months}:{val_months}:{test_months} 月, "
                    f"共 {n_windows} 个, 1-based)\n{'=' * 60}")
        for w in range(1, n_windows + 1):
            spans = prep.wf_splitter.window_dates(w - 1)
            info = prep.wf_splitter.windows()[w - 1]
            logger.info(f"  w{w}: train {spans['train']} ({info['n_train']}日) | "
                        f"val {spans['val']} ({info['n_val']}日) | "
                        f"test {spans['test']} ({info['n_test']}日)")
        print(f"\n共 {n_windows} 个可用窗口. 使用 --windows 选择运行 (如 '--windows 1,3' 或 '--windows 1-6').")
        return

    if windows is not None:
        valid_windows = [w for w in windows if 1 <= w <= n_windows]
        if not valid_windows:
            logger.error(f"指定的窗口 {windows} 无有效窗口 (共 {n_windows} 个, 1-based), 退出.")
            return
        logger.info(f"选择性运行窗口: {valid_windows} (共 {n_windows} 个可用)")
    else:
        valid_windows = list(range(1, n_windows + 1))

    logger.info(f"\n{'=' * 60}\nWalk-Forward 挖掘 ({Config.WF_TRAIN_MONTHS}:{Config.WF_VAL_MONTHS}:"
                f"{Config.WF_TEST_MONTHS} 月) | 共 {len(valid_windows)} 个窗口 | {market}\n{'=' * 60}")

    fm_regressor = FamaMacBethRegressor(nw_lags=Config.NW_LAGS)
    backtester = AcademicBacktester()

    window_factors: Dict[int, List[Dict]] = {}

    # ================= 阶段 A: 逐窗口独立挖掘 =================
    for w in valid_windows:
        spans = prep.set_window(w - 1)
        wdir = os.path.join(root, f"window_{w}")
        os.makedirs(wdir, exist_ok=True)
        Config.OUTPUT_DIR = wdir

        window_info = {
            'window_idx': w,
            'train': spans['train'],
            'val': spans['val'],
            'test': spans['test'],
            'config': {
                'train_months': train_months,
                'val_months': val_months,
                'test_months': test_months,
            },
        }
        with open(os.path.join(wdir, 'window_info.json'), 'w', encoding='utf-8') as f:
            json.dump(window_info, f, indent=2, ensure_ascii=False)

        logger.info(f"\n{'=' * 60}\n窗口 w{w}\n"
                    f"  Train: {spans['train']}\n  Val:   {spans['val']}\n  Test:  {spans['test']}\n"
                    f"{'=' * 60}")

        orchestrator = MiningOrchestrator(
            prep, name=f"WF_w{w}",
            report_title=f"Walk-Forward 窗口 {w} 因子挖掘报告")
        prod = orchestrator.run(max_trials=max_trials, time_limit_min=time_limit)
        window_factors[w] = prod

        if prod:
            logger.info(f"窗口 w{w} 达标因子 {len(prod)} 个, 启动单因子回测 (Top-3)...")
            for factor in prod[:3]:
                fid, formula, feats = (factor['id'], factor['metrics']['formula'],
                                       factor['metrics']['features'])
                try:
                    pred_te, _ = evaluate_formula_on_test(
                        prep, feats, formula, orchestrator.pysr_ns, fm_regressor)
                    ds = prep.get_subset(feats)
                    backtester.run(pred_te, ds['test'][1], ds['test'][2], ds['test'][3], fid,
                            open_prices=prep.full_open[prep.te_mask] if prep.full_open is not None else None,
                            close_prices=prep.full_close[prep.te_mask] if prep.full_close is not None else None)
                except Exception as e:
                    logger.warning(f"窗口 w{w} {fid} 回测失败: {e}")

    # ================= 阶段 B: 跨窗口稳健性 (仅在被选窗口内) =================
    logger.info(f"\n{'=' * 60}\n跨窗口稳健性检验\n{'=' * 60}")
    cross_rows: List[Dict] = []
    for i in valid_windows:
        prod = window_factors.get(i, [])
        if not prod:
            continue
        for factor in prod:
            fid, formula, feats = (factor['id'], factor['metrics']['formula'],
                                   factor['metrics']['features'])
            src_t = factor['metrics'].get('FM_tstat', 0.0)
            for j in [w for w in valid_windows if w > i]:
                prep.set_window(j - 1)
                try:
                    _, fm_res = evaluate_formula_on_test(
                        prep, feats, formula, SAFE_MATH, fm_regressor)
                except Exception as e:
                    logger.warning(f"跨窗 {i}->{j} {fid} 评估失败: {e}")
                    continue
                if fm_res.n_periods < 30:
                    cross_rows.append({
                        'src_window': i, 'fid': fid, 'test_window': j,
                        'n_periods': fm_res.n_periods, 'passed': False, 'insufficient': True,
                    })
                    continue
                sign_ok = (src_t == 0.0) or (np.sign(fm_res.t_stat) == np.sign(src_t))
                passed = abs(fm_res.t_stat) >= Config.TSTAT_THRESHOLD and sign_ok
                cross_rows.append({
                    'src_window': i, 'fid': fid, 'test_window': j,
                    'n_periods': fm_res.n_periods,
                    'FM_tstat': round(float(fm_res.t_stat), 4),
                    'Rank_IC': round(float(fm_res.rank_ic_mean), 4),
                    'ICIR': round(float(fm_res.rank_icir), 4),
                    'passed': bool(passed),
                })

    # ================= 汇总输出 =================
    cross_df = pd.DataFrame(cross_rows)
    cross_csv = os.path.join(root, 'cross_window_robustness.csv')
    if not cross_df.empty:
        cross_df.to_csv(cross_csv, index=False)
    else:
        pd.DataFrame(columns=['src_window', 'fid', 'test_window', 'n_periods',
                              'FM_tstat', 'Rank_IC', 'ICIR', 'passed']).to_csv(cross_csv, index=False)

    summary = {'config': {
        'market': market, 'train_months': train_months, 'val_months': val_months,
        'test_months': test_months, 'max_trials': max_trials, 'time_limit_min': time_limit,
        'n_windows': n_windows,
    }}
    summary['windows'] = {}
    for i in valid_windows:
        factors = []
        for factor in window_factors.get(i, []):
            m = factor['metrics']
            factors.append({
                'id': factor['id'], 'formula': m.get('formula', ''),
                'features': factor.get('features', []),
                'test_FM_tstat': m.get('FM_tstat', 0.0), 'test_IC': m.get('Rank_IC', 0.0),
                'test_ICIR': m.get('ICIR', 0.0),
            })
        summary['windows'][str(i)] = {
            'n_factors': len(factors),
            'factors': factors,
        }

    if not cross_df.empty:
        if 'insufficient' in cross_df.columns:
            valid = cross_df[~cross_df['insufficient']]
        else:
            valid = cross_df
        summary['cross_window'] = {
            'n_evals': int(len(cross_df)),
            'n_passed': int(valid['passed'].sum()) if 'passed' in valid and len(valid) else 0,
            'hit_rate': float(valid['passed'].mean()) if len(valid) else 0.0,
            'mean_tstat': float(valid['FM_tstat'].mean()) if len(valid) else 0.0,
            'mean_ic': float(valid['Rank_IC'].mean()) if len(valid) else 0.0,
            'mean_icir': float(valid['ICIR'].mean()) if len(valid) else 0.0,
        }
    else:
        summary['cross_window'] = {'n_evals': 0}

    with open(os.path.join(root, 'walkforward_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    report_lines = [
        "=" * 90,
        f"Walk-Forward 滚动窗口实验汇总 ({train_months}:{val_months}:{test_months} 月)",
        "=" * 90,
    ]
    for i in valid_windows:
        report_lines.append(f"窗口 {i}: {len(window_factors.get(i, []))} 个达标因子")
    if summary['cross_window'].get('n_evals', 0) > 0:
        cw = summary['cross_window']
        report_lines.append("-" * 90)
        report_lines.append(f"跨窗口评估 {cw['n_evals']} 次 | 命中 {cw['n_passed']} | "
                            f"命中率 {cw['hit_rate']:.1%} | 平均 t={cw['mean_tstat']:.3f} | "
                            f"平均 IC={cw['mean_ic']:.4f} | 平均 ICIR={cw['mean_icir']:.3f}")
    report_text = "\n".join(report_lines)
    logger.info(f"\n{report_text}")
    with open(os.path.join(root, 'walkforward_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"\n完成！产出物: {os.path.abspath(root)}")


def console_main(argv=None):
    parser = argparse.ArgumentParser(description="Walk-Forward 滚动窗口实验 (36:12:12 月, 3年训练/1年验证/1年测试)")
    parser.add_argument('--data', type=str,
                        default=os.path.join('data', 'csi500_daily_2016-06-30_to_2021-06-30.parquet')
                                  + ',' + os.path.join('data', 'csi500_daily_2021-06-30_to_2026-06-30.parquet'),
                        help='数据文件, 支持逗号分隔多个 (自动拼接去重)')
    parser.add_argument('--trials', type=int, default=50)
    parser.add_argument('--time', type=int, default=80)
    parser.add_argument('--market', type=str, default='zs500', choices=['zs500', 'hs300'])
    parser.add_argument('--train-months', type=int, default=36)
    parser.add_argument('--val-months', type=int, default=12)
    parser.add_argument('--test-months', type=int, default=12)
    parser.add_argument('--max-windows', type=int, default=None,
                        help='最多运行的窗口数 (冒烟测试用), 默认全部')
    parser.add_argument('--windows', type=str, default=None,
                        help='指定要运行的窗口 (1-based), 逗号分隔 (如 "1,3") 或范围 (如 "1-6"), 默认全部')
    parser.add_argument('--list-windows', action='store_true',
                        help='仅打印各窗口时间段, 不挖掘')
    args = parser.parse_args(argv)
    windows = parse_windows(args.windows) if args.windows else None
    main(args.data, max_trials=args.trials, time_limit=args.time, market=args.market,
         train_months=args.train_months, val_months=args.val_months,
         test_months=args.test_months, max_windows=args.max_windows,
         windows=windows, list_windows=args.list_windows)


if __name__ == "__main__":
    console_main()
