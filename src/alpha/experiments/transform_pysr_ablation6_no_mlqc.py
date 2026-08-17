#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
学术级日频单因子挖掘系统 — 消融实验 #6: 无奖励塑形 (No MLQC)
和 transform_pysr_primary.py 完全一致，仅去掉 _compute_reward 的奖励塑形，
GFlowNet 接收恒为 1.0 的 reward，对比查看奖励引导对因子质量的影响。
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
    import juliacall  # noqa: F401
except ImportError:
    pass

import logging
import os
import sys
import argparse
from datetime import datetime
from alpha.evaluation.backtester import AcademicBacktester
from alpha.config import Config, set_global_seed
from alpha.data.data_loader import CSI500Loader
from alpha.mining.preprocessor import DataPreprocessor
from alpha.mining.orchestrator import MiningOrchestrator

logger = logging.getLogger('AcademicDualEngine_Ablation6')


class NoMLQCOrchestrator(MiningOrchestrator):
    """和 MiningOrchestrator 完全一致，仅 _compute_reward 返回常数 1.0"""

    def _compute_reward(self, fm_result, used_features, formula=""):
        raw_ic = abs(fm_result.rank_ic_mean)
        debug = {
            't_ctrl_score': 0.0,
            'icir_score': 0.0,
            'ic_score': raw_ic,
            'final_reward': raw_ic,
            'reason': 'raw_rank_IC',
        }
        return raw_ic, debug


def main_ablation6_no_mlqc(data_path: str, max_trials: int = 50, time_limit: int = 60):
    set_global_seed()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    Config.OUTPUT_DIR = os.path.join("factor_output_academic_v81", f"ablation6_no_mlqc_{timestamp}")
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

    logger.info("\n" + "=" * 60 + "\n消融#6: 因子挖掘 (无奖励塑形 · No MLQC)\n" + "=" * 60)
    orchestrator = NoMLQCOrchestrator(prep)
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
                backtester.run(pred_all[prep.te_mask], ds['test'][1], ds['test'][2], ds['test'][3], fid,
                               open_prices=prep.full_open[prep.te_mask] if prep.full_open is not None else None,
                               close_prices=prep.full_close[prep.te_mask] if prep.full_close is not None else None)
            except Exception as e:
                logger.warning(f"{fid} 回测失败: {e}")

    print(f"\n消融#6 完成！产出物: {os.path.abspath(Config.OUTPUT_DIR)}")


if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    parser = argparse.ArgumentParser(description="消融实验 #6: 无奖励塑形 (No MLQC)")
    parser.add_argument('--data', type=str,
                        default='data/csi500_daily_2021-06-30_to_2026-06-30.parquet')
    parser.add_argument('--trials', type=int, default=60)
    parser.add_argument('--time', type=int, default=80)
    args = parser.parse_args()
    main_ablation6_no_mlqc(args.data, max_trials=args.trials, time_limit=args.time)
