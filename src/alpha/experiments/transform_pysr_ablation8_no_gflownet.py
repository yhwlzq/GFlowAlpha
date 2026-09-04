#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
消融实验 #8: 无 GFlowNet（随机特征选择）

和 transform_pysr_primary.py 完全一致，仅去掉 GFlowNet：
  - 特征选择：随机采样取代 GFlowNet 策略采样
  - 无 GFlowNet 训练
  - 其余全部保留（PySR 配置、FM 回归、中性化、质量门禁、去重、样本外评估）

用于对比：有/无 GFlowNet 引导对生成公式质量和多样性的影响。
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
import time
import argparse
from datetime import datetime
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd

from alpha.evaluation.backtester import AcademicBacktester
from alpha.config import Config, set_global_seed, apply_split_mode
from alpha.data.data_loader import CSI500Loader
from alpha.mining.preprocessor import DataPreprocessor
from alpha.mining.orchestrator import MiningOrchestrator
from alpha.mining.pysr_engine import PySRMiningEngine

logger = logging.getLogger('AcademicDualEngine_Ablation8')


class NoGFlowNetOrchestrator(MiningOrchestrator):
    """继承 MiningOrchestrator，将 GFlowNet 采样替换为随机采样。"""

    @staticmethod
    def _random_sample_features() -> Tuple[List[str], List[int]]:
        pool_size = len(Config.FINAL_FEATURE_POOL)
        n_select = np.random.randint(Config.MIN_FEATURES, Config.MAX_FEATURES + 1)
        indices = sorted(np.random.choice(pool_size, n_select, replace=False).tolist())
        features = [Config.FINAL_FEATURE_POOL[i] for i in indices]
        return features, indices

    def run(self, max_trials: int = 50, time_limit_min: int = 60):
        start = time.time()
        best_tstat, trial = 0.0, 0
        logger.info(f"Orchestrator {self.name} (No GFlowNet) 启动 | 上限:{max_trials}轮/{time_limit_min}分钟")

        while trial < max_trials:
            if (time.time() - start) / 60 > time_limit_min:
                break

            logger.info(f"\n{'=' * 50}\nTrial #{trial} (随机搜索)\n{'=' * 50}")

            feats, _ = self._random_sample_features()
            if len(feats) < Config.MIN_FEATURES:
                trial += 1
                continue

            ds = self.prep.get_subset(feats)
            X_tr, y_tr_raw = ds['train'][0], ds['train'][1]
            y_tr = y_tr_raw

            engine = PySRMiningEngine()
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

                pred_val = pred_all[self.prep.val_mask]
                val_ret = ds['val'][1]
                val_dates = ds['val'][2]
                val_symbols = s_all[self.prep.val_mask]
                val_amount = amt_all[self.prep.val_mask]

            except Exception as e:
                logger.warning(f"Trial #{trial} 执行失败: {e}")
                trial += 1
                continue

            if Config.USE_RESIDUAL_NEUTRALIZATION:
                pure_pred, pure_ret = self._neutralize(pred_val, val_ret, val_dates, val_amount, self.prep.val_mask)
            else:
                pure_pred, pure_ret = pred_val, val_ret

            fm_result = self.fm_regressor.run(
                factor_values=pure_pred,
                returns=pure_ret,
                dates=val_dates,
                symbols=val_symbols
            )

            fid = f"ACAD_{trial:03d}"
            self.fm_regressor.print_report(fm_result, factor_name=fid)

            if abs(fm_result.t_stat) < Config.TSTAT_THRESHOLD:
                logger.warning(f"{fid} t-stat不达标 ({fm_result.t_stat:.3f} < {Config.TSTAT_THRESHOLD})")
                trial += 1
                continue

            if Config.USE_MONOTONICITY_GATE:
                if not self._check_monotonicity(fm_result.quantile_returns, fm_result.t_stat):
                    logger.warning(f"{fid} 单调性不达标")
                    trial += 1
                    continue

            used_feats = [f for f in feats if f in formula]
            structure_fp = self._normalize_structure(formula, feats)
            if structure_fp in self.structural_fingerprints:
                logger.warning(f"{fid} 结构重复 ({structure_fp[:60]}...)")
                trial += 1
                continue

            canon_sig = self._canonical_signature(formula, feats)

            logger.info(f"{fid} | Cpx:{complexity} | t:{fm_result.t_stat:.3f} | "
                         f"ICIR:{fm_result.rank_icir:.4f} | IC:{fm_result.rank_ic_mean:.4f}")

            # === Phase B: Test set evaluation for registry ===
            pred_te = pred_all[self.prep.te_mask]
            test_ret = ds['test'][1]
            test_dates = ds['test'][2]
            test_symbols = ds['test'][3]
            test_amount = ds['test'][4]

            if Config.USE_RESIDUAL_NEUTRALIZATION:
                pure_pred_te, pure_ret_te = self._neutralize(
                    pred_te, test_ret, test_dates, test_amount, self.prep.te_mask
                )
            else:
                pure_pred_te, pure_ret_te = pred_te, test_ret

            fm_result_test = self.fm_regressor.run(
                factor_values=pure_pred_te,
                returns=pure_ret_te,
                dates=test_dates,
                symbols=test_symbols
            )

            self.fm_regressor.print_report(fm_result_test, factor_name=f"{fid} (Test集样本外)")

            ok, reason, gate_status = self._registration_verdict(fm_result_test)
            if not ok:
                logger.warning(f"{fid} {reason} 不入库")
                trial += 1
                continue

            fm_dict = fm_result_test.to_dict()
            fm_dict['formula'] = formula
            fm_dict['features'] = feats
            fm_dict['complexity'] = complexity
            self.registry.register(fid, formula, fm_dict, feats, gate_status=gate_status)
            self.structural_fingerprints.add(structure_fp)
            self.structural_signatures.add(canon_sig)

            for feat in feats:
                self.feature_usage[feat] = self.feature_usage.get(feat, 0) + 1
            self.total_passed += 1
            self.registered_formulas.append((fid, formula, feats))

            if abs(fm_result.t_stat) > best_tstat:
                best_tstat = abs(fm_result.t_stat)
                logger.info(f"新纪录! (Val集) |t-stat| = {best_tstat:.4f}")

            trial += 1

        prod_factors = self.registry.get_production_ready()
        logger.info(f"注册表因子总数: {len(self.registry.data['factors'])} 个 | "
                     f"达生产标准: {len(prod_factors)} 个")
        self._generate_summary_report(prod_factors)
        return prod_factors


def main_ablation8_no_gflownet(data_path: str, max_trials: int = 50, time_limit: int = 60):
    set_global_seed()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    Config.OUTPUT_DIR = os.path.join("factor_output_academic_v81", f"ablation8_no_gflownet_{timestamp}")
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

    logger.info("\n" + "=" * 60 + "\n消融#8: 因子挖掘 (无 GFlowNet · 随机特征选择)\n" + "=" * 60)
    orchestrator = NoGFlowNetOrchestrator(prep)
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

    print(f"\n消融#8 完成！产出物: {os.path.abspath(Config.OUTPUT_DIR)}")


if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    parser = argparse.ArgumentParser(description="消融实验 #8: 无 GFlowNet (随机特征选择)")
    parser.add_argument('--data', type=str,
                        default='data/warmup/csi500_daily_2020-06-30_to_2026-06-30.parquet')
    parser.add_argument('--trials', type=int, default=50)
    parser.add_argument('--time', type=int, default=80)
    parser.add_argument('--mode', type=str, default='warmup', choices=['cold', 'warmup', 'ratio', 'month'],
                        help='切分模式(与主线一致): warmup(前12月回溯+36:12:12, 默认), cold/ratio/month')
    args = parser.parse_args()
    apply_split_mode(args.mode)
    logger.info(f"切分配置: mode={args.mode} (SPLIT_MODE={Config.SPLIT_MODE}, SPLIT_WARMUP={Config.SPLIT_WARMUP}, "
                f"SPLIT_MONTH_ANCHOR={Config.SPLIT_MONTH_ANCHOR})")
    main_ablation8_no_gflownet(args.data, max_trials=args.trials, time_limit=args.time)
