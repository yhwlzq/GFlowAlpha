#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
学术级日频单因子挖掘系统 — 消融实验 #5: 完全无约束
(无特征IC先验筛选 · 无GFlowNet · 无操作符限制 · 无计量检验 · 无奖励塑形 · 无去重)
"""
import os
os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"
try:
    import juliacall  # noqa: F401
except ImportError:
    pass

import logging
import os
import sys
import time
import types
from typing import List, Dict
import numpy as np
import pandas as pd
from datetime import datetime
from alpha.evaluation.backtester import AcademicBacktester

from alpha.config import Config, set_global_seed
from alpha.data.data_loader import CSI500Loader
from alpha.mining.safe_ops import SafeOps
from alpha.mining.preprocessor import DataPreprocessor
from alpha.mining.pysr_engine import PySRMiningEngine
from alpha.mining.registry import FactorRegistry

# ============================================================
# 消融开关：去除所有先导知识约束
# ============================================================
ALL_UN_OPS = ["square", "abs", "log1p", "sign", "inv", "sqrt", "cube", "tanh", "exp"]
ALL_BIN_OPS = ["+", "-", "*", "/"]

logger = logging.getLogger('AcademicDualEngine_Ablation5')


class MiningOrchestrator:
    def __init__(self, preprocessor: DataPreprocessor):
        self.prep = preprocessor

        self.registry = FactorRegistry(version="9.1_Unconstrained")
        _orig = self.registry.get_production_ready
        self.registry.get_production_ready = types.MethodType(
            lambda self: [{"id": fid, **info} for fid, info in self.data["factors"].items()],
            self.registry
        )

        self.pysr_ns = {
            "square": np.square, "sigmoid": SafeOps.safe_sigmoid,
            "log1p": SafeOps.safe_log1p, "inv": SafeOps.safe_inv,
            "sqrt": SafeOps.safe_sqrt, "cube": SafeOps.safe_cube,
            "tanh": SafeOps.safe_tanh, "exp": SafeOps.safe_exp,
            "/": SafeOps.protected_div,
            "sign": np.sign, "abs": np.abs,
        }
        logger.info(f"消融#5 (完全无约束) 就绪 | 特征池大小: {len(Config.FINAL_FEATURE_POOL)}")

    @staticmethod
    def _random_sample_features() -> tuple:
        pool_size = len(Config.FINAL_FEATURE_POOL)
        n_select = np.random.randint(Config.MIN_FEATURES, Config.MAX_FEATURES + 1)
        indices = sorted(np.random.choice(pool_size, n_select, replace=False).tolist())
        features = [Config.FINAL_FEATURE_POOL[i] for i in indices]
        return features, indices

    def run(self, max_trials: int = 50, time_limit_min: int = 60):
        start = time.time()
        trial = 0
        logger.info(f"消融#5 启动 | 上限:{max_trials}轮/{time_limit_min}分钟")

        while trial < max_trials:
            if (time.time() - start) / 60 > time_limit_min:
                break

            logger.info(f"\n{'=' * 50}\nTrial #{trial} (随机搜索 · 无约束)\n{'=' * 50}")

            feats, _ = self._random_sample_features()
            if len(feats) < Config.MIN_FEATURES:
                trial += 1
                continue

            ds = self.prep.get_subset(feats)
            X_tr, y_tr = ds['train'][0], ds['train'][1]

            engine = PySRMiningEngine()
            engine.un_ops = ALL_UN_OPS
            engine.bin_ops = ALL_BIN_OPS
            pysr_result = engine.run(pd.DataFrame(X_tr, columns=feats), y_tr)
            if not pysr_result:
                trial += 1
                continue
            formula, complexity = pysr_result

            try:
                X_all, y_all, d_all, s_all, amt_all = ds['all']
                ns = {f: X_all[:, i] for i, f in enumerate(feats)}
                ns.update(self.pysr_ns)
                pred_all = eval(formula, {"__builtins__": {}}, ns)
                pred_all = np.where(np.isfinite(pred_all), pred_all, np.nan)
            except Exception as e:
                logger.warning(f"Trial #{trial} 执行失败: {e}")
                trial += 1
                continue

            # === 无计量检验 · 直接入库 ===
            fid = f"ACAD_{trial:03d}"
            metrics = {
                'formula': formula,
                'features': feats,
                'complexity': complexity,
            }
            self.registry.register(fid, formula, metrics, feats)
            logger.info(f"{fid} 已入库 | formula: {formula[:80]} | cpx:{complexity}")

            trial += 1

        prod_factors = self.registry.get_production_ready()
        logger.info(f"共产生 {len(prod_factors)} 个因子")
        return prod_factors


def main_ablation5_no_constraints(data_path: str, max_trials: int = 50, time_limit: int = 60):
    set_global_seed()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    Config.OUTPUT_DIR = os.path.join("factor_output_academic_v81", f"ablation5_no_constraints_{timestamp}")
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

    logger.info("\n" + "=" * 60 + "\n消融#5: 因子挖掘 (完全无约束)\n" + "=" * 60)
    orchestrator = MiningOrchestrator(prep)
    prod_factors = orchestrator.run(max_trials=max_trials, time_limit_min=time_limit)

    if prod_factors:
        logger.info("启动单因子回测...")
        backtester = AcademicBacktester()
        for factor in prod_factors[:3]:
            fid = factor['id']
            formula = factor['metrics']['formula']
            feats = factor['metrics']['features']
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

    print(f"\n消融#5 完成！产出物: {os.path.abspath(Config.OUTPUT_DIR)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="消融实验 #5: 完全无约束")
    parser.add_argument('--data', type=str,
                        default='data/csi500_daily_2021-06-30_to_2026-06-30.parquet')
    parser.add_argument('--trials', type=int, default=50)
    parser.add_argument('--time', type=int, default=60)
    args = parser.parse_args()
    main_ablation5_no_constraints(args.data, max_trials=args.trials, time_limit=args.time)
