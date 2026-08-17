#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
消融实验: 无多样性控制 (w/o Diversity Control)

和 transform_pysr_primary.py 完全一致, 仅关闭 Config.USE_DYNAMIC_DIVERSITY:
  1. diversity reward (diversity_mult) 关闭
  2. feature masking (soft/hard mask -> GFlowNet logits 惩罚) 关闭
  3. Jaccard penalty (sim_penalty) 关闭

保留: GFlowNet + MLQC (四层门禁 L1/L2/L3/L4 + 奖励塑形)。

用于对比: 有/无多样性控制对因子池 ICIR、两两相关性、单特征占比、
唯一公式占比等多样性指标的影响 (GFlowNet -> diverse sampling,
diversity control -> maintain heterogeneity)。
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
import numpy as np
import pandas as pd
from alpha.evaluation.backtester import AcademicBacktester
from alpha.config import REPO_ROOT, Config, set_global_seed
from alpha.data.data_loader import CSI500Loader
from alpha.mining.preprocessor import DataPreprocessor
from alpha.mining.orchestrator import MiningOrchestrator

logger = logging.getLogger('AcademicDualEngine_NoDiversity')


def main(data_path: str, max_trials: int = 50, time_limit: int = 60):
    set_global_seed()
    Config.USE_DYNAMIC_DIVERSITY = False
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    Config.OUTPUT_DIR = os.path.join("factor_output_academic_v81", f"ablation_no_diversity_{timestamp}")
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

    mode = "大盘因子集" if Config.MARKET == 'hs300' else "标准因子集"
    logger.info(f"\n{'='*60}\n因子挖掘 ({mode}) — w/o Diversity Control\n{'='*60}")
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
                backtester.run(pred_all[prep.te_mask], ds['test'][1], ds['test'][2], ds['test'][3], fid,
                               open_prices=prep.full_open[prep.te_mask] if prep.full_open is not None else None,
                               close_prices=prep.full_close[prep.te_mask] if prep.full_close is not None else None)
            except Exception as e:
                logger.warning(f"{fid} 回测失败: {e}")

    print(f"\n消融(no diversity)完成！产出物: {os.path.abspath(Config.OUTPUT_DIR)}")


def console_main(argv=None):
    parser = argparse.ArgumentParser(description="消融实验: 无多样性控制 (w/o Diversity Control)")
    parser.add_argument('--data', type=str,
                        default=os.path.join(REPO_ROOT, 'data', 'csi500_daily_2021-06-30_to_2026-06-30.parquet'))
    parser.add_argument('--trials', type=int, default=50)
    parser.add_argument('--time', type=int, default=80)
    parser.add_argument('--market', type=str, default='zs500', choices=['zs500', 'hs300'],
                        help='市场类型: zs500 (中证500) 或 hs300 (沪深300)')
    parser.add_argument('--pool', type=str, default='buildin', choices=['buildin', 'alpha158'],
                        help='特征池: buildin (自建语义因子池) 或 alpha158 (Qlib Alpha158 工程化因子池)')
    args = parser.parse_args(argv)
    Config.MARKET = args.market
    Config.FEATURE_POOL = args.pool
    main(args.data, max_trials=args.trials, time_limit=args.time)


if __name__ == "__main__":
    console_main()
