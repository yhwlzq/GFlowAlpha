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
from alpha.config import REPO_ROOT, Config, set_global_seed
from alpha.data.data_loader import CSI500Loader
from alpha.mining.preprocessor import DataPreprocessor
from alpha.mining.orchestrator import MiningOrchestrator

logger = logging.getLogger('AcademicDualEngine')

def main(data_path: str, max_trials: int = 50, time_limit: int = 60, debug: bool = False,
         run_mode: str = 'cross_sectional', seed: int = 42):
    set_global_seed(seed)
    Config.MODE = run_mode
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    Config.OUTPUT_DIR = os.path.join("factor_output_academic_v81", f"run_{timestamp}_s{seed}")
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
    logger.info(f"随机种子: {seed}")

    prep = DataPreprocessor()

    if run_mode == 'timing':
        # 时序择时模式: 加载成分股数据, 内部自动计算等权指数
        data_util = CSI500Loader(path=data_path)
        df = data_util.load()
        df['date'] = pd.to_datetime(df['date'])
        df = df.dropna(subset=['close', 'volume']).sort_values(['symbol', 'date']).reset_index(drop=True)
        logger.info(f"时序择时模式 | 成分股数据: {len(df)} 行, {df['symbol'].nunique()} 只股票")
        prep.prepare_timing_dataset(df)
    else:
        # 截面选股模式 (原有逻辑)
        data_util = CSI500Loader(path=data_path)
        df = data_util.load()

        required_cols = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'amount']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"数据必须包含列: {required_cols}")

        df['date'] = pd.to_datetime(df['date'])
        df = df.dropna(subset=['close', 'volume']).sort_values(['symbol', 'date']).reset_index(drop=True)
        prep.prepare_full_pool(df)

    mode_label = "时序择时" if run_mode == 'timing' else ("大盘因子集" if Config.MARKET == 'hs300' else "标准因子集")
    logger.info(f"\n{'='*60}\n因子挖掘 ({mode_label})\n{'='*60}")
    orchestrator = MiningOrchestrator(prep)
    prod_factors = orchestrator.run(max_trials=max_trials, time_limit_min=time_limit)

    if prod_factors:
        logger.info(f"达标因子 {len(prod_factors)} 个, 回测详情见 fm_summary_report.txt")

    print(f"\n完成！产出物: {os.path.abspath(Config.OUTPUT_DIR)}")


def console_main(argv=None):
    parser = argparse.ArgumentParser(description="学术级日频单因子挖掘系统 V9.1")
    parser.add_argument('--data', type=str,
                        default=os.path.join(REPO_ROOT, 'data', 'warmup','csi500_daily_2020-06-30_to_2026-06-30.parquet'))
    parser.add_argument('--trials', type=int, default=60)
    parser.add_argument('--time', type=int, default=150)
    parser.add_argument('--market', type=str, default='zs500', choices=['zs500', 'hs300'],
                        help='市场类型: zs500 (中证500) 或 hs300 (沪深300)')
    parser.add_argument('--pool', type=str, default='buildin', choices=['buildin', 'alpha158'],
                        help='特征池: buildin (自建语义因子池) 或 alpha158 (Qlib Alpha158 工程化因子池)')
    parser.add_argument('--mode', type=str, default='warmup', choices=['cold', 'warmup', 'ratio', 'month'],
                        help='切分模式: cold(冷启动, 数据首日含首日 36:12:12, 默认), '
                             'warmup(前12月回溯+36:12:12), ratio(6:2:2), month(兼容旧, 走 --anchor)')
    parser.add_argument('--run_mode', type=str, default='cross_sectional',
                        choices=['cross_sectional', 'timing'],
                        help='运行模式: cross_sectional (截面选股, 默认) 或 timing (时序择时)')
    parser.add_argument('--anchor', type=str, default='data_start', choices=['first_07_01', 'data_start'],
                        help='month 模式的锚点: first_07_01(07-01锚定, 剔锚前行) 或 data_start(数据首日, 含首日)')
    parser.add_argument('--seed', type=int, default=Config.SEED,
                        help='随机种子 (默认 42, 用于 multi-seed 鲁棒性实验)')
    args = parser.parse_args(argv)
    Config.MARKET = args.market
    Config.FEATURE_POOL = args.pool

    if args.mode == 'ratio':
        Config.SPLIT_MODE='ratio'
        Config.SPLIT_WARMUP = False
    elif args.mode == 'warmup':
        Config.SPLIT_MODE = 'month'
        Config.SPLIT_MONTH_ANCHOR='data_start'
        Config.SPLIT_WARMUP=True
    else:
        Config.SPLIT_MODE = 'month'
        Config.SPLIT_MONTH_ANCHOR='data_start'
        Config.SPLIT_WARMUP=False

    # 时序择时模式自动切换默认数据路径
    data_path = args.data
    if args.run_mode == 'timing' and 'csi500' in data_path:
        timing_default = os.path.join(REPO_ROOT, 'data', 'warmup', 'hs300_daily_2020-06-30_to_2026-06-30.parquet')
        if os.path.exists(timing_default):
            data_path = timing_default
            logger.info(f"时序择时模式: 自动切换数据至 {data_path}")

    # 统一更新全局种子 (含 PySR/GFlowNet/np/torch)
    Config.SEED = args.seed
    Config.PYSR_SEED = args.seed

    main(data_path, max_trials=args.trials, time_limit=args.time, run_mode=args.run_mode, seed=args.seed)


if __name__ == "__main__":
    console_main()
