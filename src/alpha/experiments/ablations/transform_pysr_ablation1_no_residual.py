#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
学术级日频单因子挖掘系统 — 消融实验 #1: 无双残差中性化
"""
import os
os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"
try:
    import juliacall  # noqa: F401 — 在 torch 前加载，避免 segfault
except ImportError:
    pass

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

Config.USE_RESIDUAL_NEUTRALIZATION = False

logger = logging.getLogger('AcademicDualEngine_Ablation1')


def main_ablation1_no_residual(data_path: str, max_trials: int = 50, time_limit: int = 60):
    set_global_seed(42)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    Config.OUTPUT_DIR = os.path.join(Config.OUTPUT_ROOT, f"ablation1_no_residual_{timestamp}")
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

    required_cols = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'amount']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"数据必须包含列: {required_cols}")

    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=['close', 'volume']).sort_values(['symbol', 'date']).reset_index(drop=True)

    prep = DataPreprocessor()
    prep.prepare_full_pool(df)

    logger.info("\n" + "="*60 + "\n消融#1: 因子挖掘 (无双残差中性化)\n" + "="*60)
    orchestrator = MiningOrchestrator(
        prep, name="Ablation#1", report_title="消融#1 因子挖掘汇总报告 (无双残差中性化)",
        registry_version="9.1_ICIR_Prior_Ablation1")
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

    print(f"\n消融#1 完成！产出物: {os.path.abspath(Config.OUTPUT_DIR)}")


if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    parser = argparse.ArgumentParser(description="消融实验 #1: 无双残差中性化")
    parser.add_argument('--data', type=str,
                        default=os.path.join(_REPO_ROOT, 'data', 'csi500_daily_2020-07-20_to_2026-07-19.parquet'))
    parser.add_argument('--trials', type=int, default=50)
    parser.add_argument('--time', type=int, default=60)
    args = parser.parse_args()
    main_ablation1_no_residual(args.data, max_trials=args.trials, time_limit=args.time)
