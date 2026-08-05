#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
学术级日频单因子挖掘系统 — 完整版 (V9.1 ICIR先验版)
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["JULIA_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"  # Intel CPU

os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"  # 防止程序退出时崩溃
os.environ["PYTHON_JULIACALL_THREADS"] = "1"           # 强制单线程，与 JULIA_NUM_THREADS 保持一致
os.environ["PYTHON_JULIACALL_OPTLEVEL"] = "2"
try:
    import juliacall  # noqa: F401 — 在 torch 前加载，避免 segfault
except ImportError:
    pass

import numpy as np
import pandas as pd
import logging
import os
import sys
import argparse
from datetime import datetime
from alpha.evaluation.backtester import AcademicBacktester
from alpha.config import REPO_ROOT, Config, set_global_seed
from alpha.data.data_loader import CSI500Loader
from alpha.mining.preprocessor import DataPreprocessor
from alpha.mining.orchestrator import MiningOrchestrator
_REPO_ROOT = REPO_ROOT

logger = logging.getLogger('AcademicDualEngine')

def main(data_path: str, max_trials: int = 50, time_limit: int = 60, debug: bool = False):
    set_global_seed(42)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    Config.OUTPUT_DIR = os.path.join(Config.OUTPUT_ROOT, f"run_{timestamp}")
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-7s | %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(Config.OUTPUT_DIR, 'mining.log'), mode='w', encoding='utf-8')
        ],
        force=True
    )

    logger.info(f"加载日频数据: {data_path}")
    data_util = CSI500Loader(path=data_path)
    df = data_util.load()

    if debug:
        debug_dir = os.path.join(Config.OUTPUT_DIR, 'debug_subsample')
        os.makedirs(debug_dir, exist_ok=True)
        symbols = sorted(df['symbol'].unique())[:3]
        df = df[df['symbol'].isin(symbols)].copy()
        df = df.sort_values('date').groupby('symbol', sort=False).tail(60).reset_index(drop=True)
        max_trials = min(max_trials, 2)
        time_limit = min(time_limit, 2)
        logger.info(f"[DEBUG] 子样本: {len(symbols)} 只股票 × 60 交易日, "
                    f"max_trials={max_trials}, time_limit={time_limit}min")

    required_cols = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'amount']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"数据必须包含列: {required_cols}")

    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=['close', 'volume']).sort_values(['symbol', 'date']).reset_index(drop=True)

    prep = DataPreprocessor()
    prep.prepare_full_pool(df)

    mode = "大盘因子集" if Config.MARKET == 'hs300' else "标准因子集"
    logger.info(f"\n{'='*60}\n因子挖掘 ({mode})\n{'='*60}")
    orchestrator = MiningOrchestrator(prep)
    prod_factors = orchestrator.run(max_trials=max_trials, time_limit_min=time_limit)

    if prod_factors:
        logger.info("启动单因子回测...")
        backtester = AcademicBacktester()
        for factor in prod_factors[:3]:
            fid, formula, feats = factor['id'], factor['metrics']['formula'], factor['metrics']['features']
            ds = prep.get_subset(feats)
            X_all, d_all, s_all = ds['all'][0], ds['all'][2], ds['all'][3]
            ns = {f: X_all[:, i] for i, f in enumerate(feats)}
            ns.update(orchestrator.pysr_ns)
            try:
                pred_all = eval(formula, {"__builtins__": {}}, ns)
                pred_all = np.where(np.isfinite(pred_all), pred_all, np.nan)
                backtester.run(pred_all[prep.te_mask], ds['test'][1], ds['test'][2], ds['test'][3], fid)
            except Exception as e:
                logger.warning(f"{fid} 回测失败: {e}")

    print(f"\n完成！产出物: {os.path.abspath(Config.OUTPUT_DIR)}")


def console_main(argv=None):
    parser = argparse.ArgumentParser(description="学术级日频单因子挖掘系统 V9.1")
    parser.add_argument('--data', type=str,
                        default=os.path.join(_REPO_ROOT, 'data', 'csi500_daily_2020-07-20_to_2026-07-19.parquet'))
    parser.add_argument('--trials', type=int, default=60)
    parser.add_argument('--time', type=int, default=80)
    parser.add_argument('--debug', action='store_true',
                        help='调试模式: 小样本(3只股票×60日)快速冒烟')
    parser.add_argument('--market', type=str, default='zs500', choices=['zs500', 'hs300'],
                        help='市场类型: zs500 (中证500) 或 hs300 (沪深300)')
    args = parser.parse_args(argv)
    Config.MARKET = args.market
    main(args.data, max_trials=args.trials, time_limit=args.time, debug=args.debug)


if __name__ == "__main__":
    console_main()
