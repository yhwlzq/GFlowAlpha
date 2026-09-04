#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
学术级日频单因子挖掘系统 — 基准实验 #10: RL + MQLCV（MLQC 四层结果编码为 RL 终止奖励）

和 transform_pysr_primary.py 完全一致，仅替换 GFlowNet 终止奖励来源：

  主流程: R(τ) = FLOOR + (TOP-FLOOR) * sigmoid(t)ᵂᵗ * sigmoid(ICIR)ᵂᵢr * sigmoid(IC)ᵂᵢc   (连续塑形)
  本实验: R(τ) = FLOOR + (TOP-FLOOR) * Σ(wᵢ·lᵢ) / Σwᵢ                                    (加权分段)

  四层 MLQC 门禁结果 lᵢ ∈ {0,1}:
    L1 单特征线性 | L2 t-stat 阈值 | L3 单调性 | L4 结构去重
  其余组件全部保留: 门禁、FWL 中性化、多样性软屏蔽、sign 错位惩罚、reward floor。
  未通过全部四层的因子仍不入库（同主流程），但其轨迹获得逐层加权的分级奖励用于 RL 训练，
  用以对比「连续奖励塑形 vs 仅由质量控制结果驱动的离散式奖励」对因子质量与多样性的影响。
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
from typing import List, Tuple
import numpy as np
import pandas as pd

from alpha.evaluation.backtester import AcademicBacktester
from alpha.config import Config, REPO_ROOT, set_global_seed, apply_split_mode
from alpha.data.data_loader import CSI500Loader
from alpha.mining.preprocessor import DataPreprocessor
from alpha.mining.orchestrator import MiningOrchestrator
from alpha.mining.pysr_engine import PySRMiningEngine

logger = logging.getLogger('AcademicDualEngine_Ablation10')

# === R(τ) 四层权重 (可调; 加权越多 → 通过该层对奖励的边际贡献越大) ===
MLQCV_L1_SINGLE_FEAT_W = 1.0   # L1: 非单特征线性公式
MLQCV_L2_TSTAT_W = 2.0         # L2: |t-stat| >= TSTAT_THRESHOLD
MLQCV_L3_MONO_W = 1.5          # L3: 分位收益单调性
MLQCV_L4_DEDUP_W = 1.5         # L4: 结构指纹不重复


class RLMLQCVOrchestrator(MiningOrchestrator):
    """继承 MiningOrchestrator, 仅将终止奖励改为 MLQC 四层结果的加权分段编码。

    run() 每轮顺序判定四层布尔, 计算 R(τ) 训练 RL, 再决定是否入库。
    """

    def _rl_mlqcv_reward(self, fm_result, used_features, formula="",
                         layers: Tuple[bool, bool, bool, bool] = (False, False, False, False),
                         canon_dup_penalty: float = 1.0) -> Tuple[float, dict]:
        debug = {}
        weights = np.array([MLQCV_L1_SINGLE_FEAT_W, MLQCV_L2_TSTAT_W,
                            MLQCV_L3_MONO_W, MLQCV_L4_DEDUP_W], dtype=float)
        passed = np.array([bool(x) for x in layers], dtype=float)
        frac = passed.dot(weights) / weights.sum()
        reward = (Config.REWARD_SOFT_FLOOR +
                  (Config.REWARD_TOP - Config.REWARD_SOFT_FLOOR) * frac)
        debug['layer_mask'] = [bool(x) for x in layers]
        debug['layer_weights'] = weights.tolist()
        debug['layer_frac'] = float(frac)

        raw_ic = fm_result.rank_ic_mean
        if np.sign(fm_result.t_stat) * np.sign(raw_ic) < 0 or \
           np.sign(fm_result.t_stat) * np.sign(fm_result.rank_icir) < 0:
            reward *= Config.SIGN_MISMATCH_PENALTY
            debug['sign_penalty'] = Config.SIGN_MISMATCH_PENALTY
            debug['reason'] = 'sign_mismatch'

        if formula:
            eff_feats = self._compute_formula_effective_features(formula, used_features)
            if eff_feats <= 1:
                reward *= 0.2
                debug['single_feat_penalty'] = 0.2

        if Config.USE_DYNAMIC_DIVERSITY and used_features:
            novel_count = 0
            for feat in used_features:
                usage = self.feature_usage.get(feat, 0)
                if usage <= max(1, int(self.total_passed * 0.15)):
                    novel_count += 1
            diversity_mult = min(1.5, 1.0 + 0.15 * novel_count)
            reward *= diversity_mult
            debug['div_mult'] = diversity_mult

            sim_penalty = 1.0
            for _, _, reg_feats in self.registered_formulas:
                sim = self._compute_formula_similarity(used_features, reg_feats)
                if sim > 0.75:
                    sim_penalty = 0.1
                    break
            if sim_penalty < 1.0:
                reward *= sim_penalty
                debug['sim_penalty'] = sim_penalty

        if canon_dup_penalty < 1.0:
            reward *= canon_dup_penalty
            debug['canon_dup_penalty'] = canon_dup_penalty

        reward = max(reward, Config.REWARD_SOFT_FLOOR)
        debug['final_reward'] = reward
        return reward, debug

    def run(self, max_trials: int = 50, time_limit_min: int = 60):
        start = time.time()
        best_tstat, trial = 0.0, 0
        logger.info(f"Orchestrator {self.name} (RL+MQLCV) 启动 | 上限:{max_trials}轮/{time_limit_min}分钟")

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
                self.gfn_trainer.train_step(traj.log_pf, traj.log_pb, reward=Config.REWARD_HARD)
                trial += 1
                continue

            ds = self.prep.get_subset(feats)
            X_tr, y_tr = ds['train'][0], ds['train'][1]

            engine = PySRMiningEngine()
            pysr_result = engine.run(pd.DataFrame(X_tr, columns=feats), y_tr)
            if not pysr_result:
                self.gfn_trainer.train_step(traj.log_pf, traj.log_pb, reward=Config.REWARD_HARD)
                trial += 1
                continue
            formula, complexity = pysr_result

            used_feats = [f for f in feats if f in formula]

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
                self.gfn_trainer.train_step(traj.log_pf, traj.log_pb, reward=Config.REWARD_HARD)
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

            # === 四层 MLQC 结果判定 (不再提前 continue, 每层都参与 R(τ)) ===
            l1 = len(used_feats) > 1 or '(' in formula
            l2 = abs(fm_result.t_stat) >= Config.TSTAT_THRESHOLD
            l3 = True
            if Config.USE_MONOTONICITY_GATE:
                mono_ok = self._check_monotonicity(fm_result.quantile_returns, fm_result.t_stat)
                l3 = bool(mono_ok)
            structure_fp = self._normalize_structure(formula, feats)
            l4 = structure_fp not in self.structural_fingerprints

            canon_sig = self._canonical_signature(formula, feats)
            canon_dup_penalty = 1.0
            if canon_sig in self.structural_signatures:
                logger.info(f"{fid} 特征无关结构重复 ({canon_sig[:50]}...)，施加软惩罚")
                canon_dup_penalty = 0.3

            reward, reward_debug = self._rl_mlqcv_reward(
                fm_result, used_features, formula=formula,
                layers=(l1, l2, l3, l4), canon_dup_penalty=canon_dup_penalty)

            layer_str = "".join("\u2713" if x else "\u2717" for x in (l1, l2, l3, l4))
            diversity_part = f" div:{reward_debug.get('div_mult', 1.0):.2f}x" if 'div_mult' in reward_debug else ""
            sim_part = " SIM-PENALTY" if 'sim_penalty' in reward_debug else ""
            canon_part = f" canon:{reward_debug.get('canon_dup_penalty', 1.0):.2f}x" if canon_dup_penalty < 1.0 else ""
            if not Config.USE_DYNAMIC_DIVERSITY:
                diversity_part = ""
                sim_part = ""
            logger.info(
                f"{fid} | Reward(Rτ):{reward:.3f} Cpx:{complexity} | 层[{layer_str}] "
                f"(L1:{'✓' if l1 else '✗'} L2:{'✓' if l2 else '✗'} "
                f"L3:{'✓' if l3 else '✗'} L4:{'✓' if l4 else '✗'}) "
                f"frac:{reward_debug['layer_frac']:.2f}{diversity_part}{sim_part}{canon_part}"
            )

            # Train GFlowNet on R(τ) — 无论是否全部通过都训练 (分级奖励)
            self.gfn_trainer.train_step(traj.log_pf, traj.log_pb, reward=reward)

            # === MQLCV: 门禁与主流程一致, 仅四层全过才入库 ===
            if not (l1 and l2 and l3 and l4):
                failed = [name for name, ok in
                          (('L1单特征', l1), ('L2 t-stat', l2), ('L3单调性', l3), ('L4去重', l4)) if not ok]
                logger.warning(f"{fid} 未通过全部 MLQC 四层 (失败: {', '.join(failed)})，不入库")
                trial += 1
                continue

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
            rc = Config.get_reward_config()
            strong_icir = abs(fm_result_test.rank_icir) >= rc['icir']['threshold']
            strong_ic = abs(fm_result_test.rank_ic_mean) >= rc['ic']['threshold']
            gate_status = 'STRONG' if (strong_icir and strong_ic) else 'WEAK'
            self.registry.register(fid, formula, fm_dict, feats, gate_status=gate_status)
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
        logger.info(f"达标因子: {len(prod_factors)} 个 | "
                     f"注册表因子总数: {len(self.registry.data['factors'])} 个")
        self._generate_summary_report(prod_factors)
        return prod_factors


def main_ablation10_rl_mlqcv(data_path: str, max_trials: int = 50, time_limit: int = 60):
    set_global_seed()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    Config.OUTPUT_DIR = os.path.join("factor_output_academic_v81", f"ablation10_rl_mlqcv_{timestamp}")
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

    logger.info("\n" + "=" * 60 + "\n基准#10: 因子挖掘 (RL+MQLCV · 四层结果编码为终止奖励)\n" + "=" * 60)
    orchestrator = RLMLQCVOrchestrator(prep)
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

    print(f"\n基准#10 完成！产出物: {os.path.abspath(Config.OUTPUT_DIR)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="基准实验 #10: RL+MQLCV (MLQC 四层结果编码为终止奖励)")
    parser.add_argument('--data', type=str,
                        default=os.path.join(REPO_ROOT, 'data', 'warmup', 'csi500_daily_2020-06-30_to_2026-06-30.parquet'))
    parser.add_argument('--trials', type=int, default=50)
    parser.add_argument('--time', type=int, default=80)
    parser.add_argument('--market', type=str, default='zs500', choices=['zs500', 'hs300'],
                        help='市场类型: zs500 (中证500) 或 hs300 (沪深300)')
    parser.add_argument('--pool', type=str, default='buildin', choices=['buildin', 'alpha158'],
                        help='特征池: buildin (自建语义因子池) 或 alpha158 (Qlib Alpha158 工程化因子池)')
    parser.add_argument('--mode', type=str, default='warmup', choices=['cold', 'warmup', 'ratio', 'month'],
                        help='切分模式(与主线一致): warmup(前12月回溯+36:12:12, 默认), cold/ratio/month')
    args = parser.parse_args()
    Config.MARKET = args.market
    Config.FEATURE_POOL = args.pool
    apply_split_mode(args.mode)
    logger.info(f"切分配置: mode={args.mode} (SPLIT_MODE={Config.SPLIT_MODE}, SPLIT_WARMUP={Config.SPLIT_WARMUP}, "
                f"SPLIT_MONTH_ANCHOR={Config.SPLIT_MONTH_ANCHOR})")
    main_ablation10_rl_mlqcv(args.data, max_trials=args.trials, time_limit=args.time)