#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
消融实验 #7: 完全无 MLQC（无质量门禁 + 无奖励塑形）

和 transform_pysr_primary.py 完全一致，但去掉所有 MLQC 组件：
  1. 奖励塑形 → raw_IC（同 #6）
  2. 质量门禁（t-stat 阈值、单调性检查、单特征线性跳过、结构去重）全部移除
  3. 所有 PySR 发现的公式均入库

用于证明：MLQC 整体对因子数量和质量的系统性影响。

注意：FactorRegistry.register() 无条件写入所有公式，但 get_production_ready()
内部有 Config.TSTAT_THRESHOLD / ICIR_MIN / IC_THRESHOLD 后过滤；
分析时请直接读取 registry JSON 的全量因子列表。
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

logger = logging.getLogger('AcademicDualEngine_Ablation7')


class NoMLQCGatesOrchestrator(MiningOrchestrator):
    """继承 MiningOrchestrator，重写 run() 移除所有质量门禁，
    重写 _compute_reward 返回 raw_IC（无塑形）。"""

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

    def run(self, max_trials: int = 50, time_limit_min: int = 60):
        start = time.time()
        best_tstat, trial = 0.0, 0
        logger.info(f"Orchestrator {self.name} (No Gates) 启动 | 上限:{max_trials}轮/{time_limit_min}分钟")

        while trial < max_trials:
            if (time.time() - start) / 60 > time_limit_min:
                break

            if Config.USE_DYNAMIC_DIVERSITY:
                if self.soft_mask:
                    expired = [f for f, t in self.soft_mask.items() if t <= 0]
                    for f in expired:
                        del self.soft_mask[f]
                    for feat in list(self.soft_mask.keys()):
                        self.soft_mask[feat] -= 1
                if self.hard_mask:
                    expired = [f for f, t in self.hard_mask.items() if t <= 0]
                    for f in expired:
                        del self.hard_mask[f]
                    for feat in list(self.hard_mask.keys()):
                        self.hard_mask[feat] -= 1
                penalty_dict = {}
                for f in self.soft_mask:
                    if f in self.prep.feature_to_idx:
                        penalty_dict[self.prep.feature_to_idx[f]] = 5.0
                for f in self.hard_mask:
                    if f in self.prep.feature_to_idx:
                        penalty_dict[self.prep.feature_to_idx[f]] = 50.0
                self.gfn_sampler.set_feature_penalties(penalty_dict)

            logger.info(f"\n{'=' * 50}\nTrial #{trial}\n{'=' * 50}")

            traj = self.gfn_sampler.sample(batch_size=1)[0]
            feats = traj.features
            self.trial_count += 1
            for f in feats:
                self.trial_feature_counter[f] = self.trial_feature_counter.get(f, 0) + 1

            if Config.USE_DYNAMIC_DIVERSITY and self.trial_count > 15:
                for f, cnt in list(self.trial_feature_counter.items()):
                    ratio = cnt / self.trial_count
                    if ratio > 0.50 and f not in self.hard_mask and f not in self.soft_mask:
                        self.hard_mask[f] = 10
                        logger.info(f"硬屏蔽 {f} (出现率{ratio:.0%})")

            if len(feats) < Config.MIN_FEATURES:
                self.gfn_trainer.train_step(traj.log_pf, traj.log_pb, reward=0.01)
                trial += 1
                continue

            ds = self.prep.get_subset(feats)
            X_tr, y_tr_raw = ds['train'][0], ds['train'][1]

            tr_dates = ds['train'][2]
            y_tr = y_tr_raw

            engine = PySRMiningEngine()
            pysr_result = engine.run(pd.DataFrame(X_tr, columns=feats), y_tr)
            if not pysr_result:
                self.gfn_trainer.train_step(traj.log_pf, traj.log_pb, reward=0.01)
                trial += 1
                continue
            formula, complexity = pysr_result

            # === No MLQC: 跳过单特征线性门禁 ===

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
                self.gfn_trainer.train_step(traj.log_pf, traj.log_pb, reward=0.01)
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

            # === No MLQC: 跳过 t-stat 门禁 ===
            # === No MLQC: 跳过单调性门禁 ===
            # === No MLQC: 跳过结构重复门禁 ===

            used_feats = [f for f in feats if f in formula]
            structure_fp = self._normalize_structure(formula, feats)
            canon_sig = self._canonical_signature(formula, feats)

            reward, reward_debug = self._compute_reward(fm_result, feats, formula=formula)

            logger.info(
                f"{fid} | Reward:{reward:.3f} Cpx:{complexity} | "
                f"t:{reward_debug.get('t_ctrl_score', 0):.3f} "
                f"icir:{reward_debug.get('icir_score', 0):.3f} "
                f"ic:{reward_debug.get('ic_score', 0):.3f}"
            )

            self.gfn_trainer.train_step(traj.log_pf, traj.log_pb, reward=reward)

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

            fm_dict = fm_result_test.to_dict()
            fm_dict['formula'] = formula
            fm_dict['features'] = feats
            fm_dict['complexity'] = complexity
            fm_dict['single_feat'] = len(used_feats) <= 1
            fm_dict['val_t_stat'] = fm_result.t_stat
            self.registry.register(fid, formula, fm_dict, feats)
            self.structural_fingerprints.add(structure_fp)
            self.structural_signatures.add(canon_sig)

            for feat in feats:
                self.feature_usage[feat] = self.feature_usage.get(feat, 0) + 1
            self.total_passed += 1
            self.pass_counter += 1
            self.registered_formulas.append((fid, formula, feats))

            if Config.USE_DYNAMIC_DIVERSITY:
                if self.pass_counter >= 1:
                    self.pass_counter = 0
                    formula_feats = [f for f in feats if f in formula]
                    if formula_feats:
                        for feat in formula_feats:
                            if feat not in self.soft_mask:
                                self.soft_mask[feat] = 10
                                logger.info(f"软屏蔽 {feat} (公式核心特征)")
                    else:
                        sorted_feats = sorted(self.feature_usage.items(), key=lambda x: -x[1])
                        for feat, _ in sorted_feats[:2]:
                            if feat not in self.soft_mask:
                                self.soft_mask[feat] = 10
                                logger.info(f"软屏蔽 {feat} (出现{self.feature_usage[feat]}次)")

            if abs(fm_result.t_stat) > best_tstat:
                best_tstat = abs(fm_result.t_stat)
                logger.info(f"新纪录! (Val集) |t-stat| = {best_tstat:.4f}")

            trial += 1

        prod_factors = self.registry.get_production_ready()
        logger.info(f"注册表因子总数: {len(self.registry.data['factors'])} 个 | "
                     f"达生产标准: {len(prod_factors)} 个")
        self._generate_summary_report(prod_factors)
        return prod_factors


def main_ablation7_no_mlqc_gates(data_path: str, max_trials: int = 50, time_limit: int = 60):
    set_global_seed()
    Config.USE_DYNAMIC_DIVERSITY = False
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    Config.OUTPUT_DIR = os.path.join("factor_output_academic_v81", f"ablation7_no_mlqc_gates_{timestamp}")
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

    logger.info("\n" + "=" * 60 + "\n消融#7: 因子挖掘 (完全无 MLQC · 无门禁 + 无塑形)\n" + "=" * 60)
    orchestrator = NoMLQCGatesOrchestrator(prep)
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

    print(f"\n消融#7 完成！产出物: {os.path.abspath(Config.OUTPUT_DIR)}")


if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    parser = argparse.ArgumentParser(description="消融实验 #7: 完全无 MLQC (无门禁 + 无塑形)")
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
    main_ablation7_no_mlqc_gates(args.data, max_trials=args.trials, time_limit=args.time)
