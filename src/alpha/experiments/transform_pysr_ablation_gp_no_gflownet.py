#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
消融实验: GP 替代 GFlowNet（遗传规划公式直接挖掘）

  - 因子生成：gplearn SymbolicRegressor 在全量特征池上演进，直接产出公式
  - 无 GFlowNet 引导式搜索
  - 无 PySR 精化
  - MLQC 纯事后筛选（FWL 中性化 → FM 回归 → t-stat/ICIR/单调性/去重）

用于对比：GFlowNet 引导式搜索 vs 传统 GP 盲目搜索 + 事后质控。
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
import sys
import time
import re
import argparse
from datetime import datetime
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd

from alpha.evaluation.backtester import AcademicBacktester
from alpha.config import Config, set_global_seed
from alpha.data.data_loader import CSI500Loader
from alpha.mining.preprocessor import DataPreprocessor
from alpha.mining.orchestrator import MiningOrchestrator
from alpha.mining.safe_ops import SafeOps

from gplearn.genetic import SymbolicRegressor

logger = logging.getLogger('AcademicDualEngine_GP')

GP_FUNCTION_SET = ('add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv', 'sin', 'cos')

GP_NS = {
    'add': np.add,
    'sub': np.subtract,
    'mul': np.multiply,
    'div': SafeOps.protected_div,
    'sqrt': lambda x: np.sqrt(np.maximum(x, 0)),
    'log': lambda x: np.log(np.abs(x) + 1e-10),
    'abs': np.abs,
    'neg': np.negative,
    'inv': SafeOps.safe_inv,
    'sin': np.sin,
    'cos': np.cos,
}

FEATURE_RE = re.compile(r'X(\d+)')


class GPFactorMiner:
    """每轮训练一个 SymbolicRegressor 并返回最优公式及用到的特征。"""

    def __init__(self, population_size: int = 300, generations: int = 15,
                 tournament_size: int = 15, parsimony_coefficient: float = 0.001,
                 n_jobs: int = 1, random_state: Optional[int] = None):
        self.population_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.parsimony_coefficient = parsimony_coefficient
        self.n_jobs = n_jobs
        self.random_state = random_state

    def mine(self, X_tr: np.ndarray, y_tr: np.ndarray,
             feature_names: List[str]) -> Tuple[Optional[str], Optional[List[str]], Optional[int]]:
        est = SymbolicRegressor(
            population_size=self.population_size,
            generations=self.generations,
            tournament_size=self.tournament_size,
            function_set=GP_FUNCTION_SET,
            const_range=None,
            parsimony_coefficient=self.parsimony_coefficient,
            stopping_criteria=0.0,
            random_state=self.random_state,
            verbose=0,
            n_jobs=self.n_jobs,
        )
        try:
            est.fit(X_tr, y_tr)
        except Exception as e:
            logger.warning(f"GP 训练失败: {e}")
            return None, None, None

        program = est._program
        if program is None:
            return None, None, None

        gplearn_formula = str(program)

        used_indices = sorted(set(int(m) for m in FEATURE_RE.findall(gplearn_formula)))
        if not used_indices:
            return gplearn_formula, [], None

        used_feats = [feature_names[i] for i in used_indices if i < len(feature_names)]
        complexity = len(program.program) if program.program else None

        return gplearn_formula, used_feats, complexity


class GPOrchestrator(MiningOrchestrator):
    """继承 MiningOrchestrator，用 GP 直接挖掘替代 GFlowNet + PySR。"""

    def __init__(self, preprocessor: DataPreprocessor, **kwargs):
        super().__init__(preprocessor, **kwargs)
        self.gp_miner = GPFactorMiner(
            population_size=300,
            generations=15,
            tournament_size=15,
            parsimony_coefficient=0.001,
            n_jobs=1,
        )
        self.gp_ns = dict(GP_NS)

    def run(self, max_trials: int = 50, time_limit_min: int = 60):
        start = time.time()
        best_tstat, trial = 0.0, 0
        logger.info(
            f"Orchestrator {self.name} (GP) 启动 | "
            f"上限:{max_trials}轮/{time_limit_min}分钟"
        )

        all_feat_names = list(Config.FINAL_FEATURE_POOL)

        while trial < max_trials:
            if (time.time() - start) / 60 > time_limit_min:
                break

            logger.info(f"\n{'=' * 50}\nTrial #{trial}\n{'=' * 50}")

            ds_all = self.prep.get_subset(all_feat_names)
            X_tr_all, y_tr_raw = ds_all['train'][0], ds_all['train'][1]

            gplearn_formula, used_feats, complexity = self.gp_miner.mine(
                X_tr_all, y_tr_raw, all_feat_names
            )

            if not gplearn_formula or not used_feats:
                logger.warning(f"Trial #{trial} GP 未产出有效公式")
                trial += 1
                continue

            if len(used_feats) <= 1 and '(' not in gplearn_formula:
                logger.info(f"Trial #{trial} 单特征线性公式，跳过: {gplearn_formula}")
                trial += 1
                continue

            used_indices = sorted(set(int(m) for m in FEATURE_RE.findall(gplearn_formula)))
            used_feat_names = [all_feat_names[i] for i in used_indices]

            ds = self.prep.get_subset(used_feat_names)

            try:
                X_all, y_all, d_all, s_all, amt_all = ds['all']

                ns = {f'X{orig_idx}': X_all[:, i]
                      for i, orig_idx in enumerate(used_indices)}
                ns.update(self.gp_ns)
                pred_all = eval(gplearn_formula, {"__builtins__": {}}, ns)
                pred_all = np.where(np.isfinite(pred_all), pred_all, np.nan)

                if np.all(np.isnan(pred_all)):
                    logger.warning(f"Trial #{trial} 全部为 NaN，跳过")
                    trial += 1
                    continue

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
                pure_pred, pure_ret = self._neutralize(
                    pred_val, val_ret, val_dates, val_amount
                )
            else:
                pure_pred, pure_ret = pred_val, val_ret

            fm_result = self.fm_regressor.run(
                factor_values=pure_pred,
                returns=pure_ret,
                dates=val_dates,
                symbols=val_symbols,
            )

            fid = f"GP_{trial:03d}"
            self.fm_regressor.print_report(fm_result, factor_name=fid)

            if abs(fm_result.t_stat) < Config.TSTAT_THRESHOLD:
                logger.warning(
                    f"{fid} t-stat不达标 ({fm_result.t_stat:.3f} < {Config.TSTAT_THRESHOLD})"
                )
                trial += 1
                continue

            if Config.USE_MONOTONICITY_GATE:
                if not self._check_monotonicity(fm_result.quantile_returns, fm_result.t_stat):
                    logger.warning(f"{fid} 单调性不达标")
                    trial += 1
                    continue

            structure_fp = self._normalize_structure(gplearn_formula, used_feat_names)
            if structure_fp in self.structural_fingerprints:
                logger.warning(f"{fid} 结构重复 ({structure_fp[:60]}...)")
                trial += 1
                continue

            canon_sig = self._canonical_signature(gplearn_formula, used_feat_names)
            canon_dup_penalty = 1.0
            if canon_sig in self.structural_signatures:
                logger.info(
                    f"{fid} 特征无关结构重复 ({canon_sig[:50]}...)，施加软惩罚"
                )
                canon_dup_penalty = 0.3

            reward, reward_debug = self._compute_reward(fm_result, used_feat_names, formula=gplearn_formula)
            if canon_dup_penalty < 1.0:
                reward *= canon_dup_penalty
                reward_debug['canon_dup_penalty'] = canon_dup_penalty

            diversity_part = (
                f" div:{reward_debug.get('div_mult', 1.0):.2f}x"
                if 'div_mult' in reward_debug else ""
            )
            sim_part = " SIM-PENALTY" if 'sim_penalty' in reward_debug else ""
            canon_part = (
                f" canon:{reward_debug.get('canon_dup_penalty', 1.0):.2f}x"
                if reward_debug.get('canon_dup_penalty', 1.0) < 1.0 else ""
            )
            logger.info(
                f"{fid} | Reward:{reward:.3f} Cpx:{complexity} | "
                f"t:{reward_debug.get('t_ctrl_score', 0):.3f} "
                f"icir:{reward_debug.get('icir_score', 0):.3f} "
                f"ic:{reward_debug.get('ic_score', 0):.3f}"
                f"{diversity_part}{sim_part}{canon_part}"
            )

            pred_te = pred_all[self.prep.te_mask]
            test_ret = ds['test'][1]
            test_dates = ds['test'][2]
            test_symbols = ds['test'][3]
            test_amount = ds['test'][4]

            if Config.USE_RESIDUAL_NEUTRALIZATION:
                pure_pred_te, pure_ret_te = self._neutralize(
                    pred_te, test_ret, test_dates, test_amount
                )
            else:
                pure_pred_te, pure_ret_te = pred_te, test_ret

            fm_result_test = self.fm_regressor.run(
                factor_values=pure_pred_te,
                returns=pure_ret_te,
                dates=test_dates,
                symbols=test_symbols,
            )

            self.fm_regressor.print_report(
                fm_result_test, factor_name=f"{fid} (Test集样本外)"
            )

            fm_dict = fm_result_test.to_dict()
            fm_dict['formula'] = gplearn_formula
            fm_dict['features'] = used_feat_names
            fm_dict['complexity'] = complexity if complexity is not None else 0
            self.registry.register(fid, gplearn_formula, fm_dict, used_feat_names)
            self.structural_fingerprints.add(structure_fp)
            self.structural_signatures.add(canon_sig)

            for feat in used_feat_names:
                self.feature_usage[feat] = self.feature_usage.get(feat, 0) + 1
            self.total_passed += 1
            self.registered_formulas.append((fid, gplearn_formula, used_feat_names))

            if abs(fm_result.t_stat) > best_tstat:
                best_tstat = abs(fm_result.t_stat)
                logger.info(f"新纪录! (Val集) |t-stat| = {best_tstat:.4f}")

            trial += 1

        prod_factors = self.registry.get_production_ready()
        logger.info(
            f"注册表因子总数: {len(self.registry.data['factors'])} 个 | "
            f"达生产标准: {len(prod_factors)} 个"
        )
        self._generate_summary_report(prod_factors)
        return prod_factors


def main_gp_no_gflownet(data_path: str, max_trials: int = 50, time_limit: int = 60):
    set_global_seed()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    Config.OUTPUT_DIR = os.path.join(
        "factor_output_academic_v81", f"ablation_gp_no_gflownet_{timestamp}"
    )
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-7s | %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                os.path.join(Config.OUTPUT_DIR, 'mining.log'),
                mode='w', encoding='utf-8',
            ),
        ],
        force=True,
    )

    logger.info(f"加载日频数据: {data_path}")
    data_util = CSI500Loader(path=data_path)
    df = data_util.load()

    required_cols = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'amount']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"数据必须包含列: {required_cols}")

    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=['close', 'volume']).sort_values(
        ['symbol', 'date']
    ).reset_index(drop=True)

    prep = DataPreprocessor()
    prep.prepare_full_pool(df)

    logger.info(
        "\n" + "=" * 60 + "\n"
        "消融: GP 公式直接挖掘 (替代 GFlowNet + PySR)\n"
        "配置: SymbolicRegressor(pop=300, gen=15)\n"
        f"函数集: {GP_FUNCTION_SET}\n"
        + "=" * 60
    )
    orchestrator = GPOrchestrator(prep)
    prod_factors = orchestrator.run(
        max_trials=max_trials, time_limit_min=time_limit
    )

    if prod_factors:
        logger.info("启动单因子回测...")
        backtester = AcademicBacktester()
        for factor in prod_factors[:3]:
            fid = factor['id']
            formula = factor['metrics']['formula']
            feats = factor['metrics']['features']
            ds = prep.get_subset(feats)
            X_all, d_all, s_all = ds['all'][0], ds['all'][2], ds['all'][3]

            all_feat_names = list(Config.FINAL_FEATURE_POOL)
            bt_indices = [all_feat_names.index(f) for f in feats]
            ns = {f'X{orig_idx}': X_all[:, i]
                  for i, orig_idx in enumerate(bt_indices)}
            ns.update(GP_NS)

            try:
                pred_all = eval(formula, {"__builtins__": {}}, ns)
                pred_all = np.where(np.isfinite(pred_all), pred_all, np.nan)
                backtester.run(
                    pred_all[prep.te_mask],
                    ds['test'][1],
                    ds['test'][2],
                    ds['test'][3],
                    fid,
                    open_prices=prep.full_open[prep.te_mask] if prep.full_open is not None else None,
                    close_prices=prep.full_close[prep.te_mask] if prep.full_close is not None else None,
                )
            except Exception as e:
                logger.warning(f"{fid} 回测失败: {e}")

    print(f"\n消融(GP)完成！产出物: {os.path.abspath(Config.OUTPUT_DIR)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="消融实验: GP 替代 GFlowNet (遗传规划公式直接挖掘)"
    )
    parser.add_argument(
        '--data', type=str,
        default='data/csi500_daily_2021-06-30_to_2026-06-30.parquet',
    )
    parser.add_argument('--trials', type=int, default=60)
    parser.add_argument('--time', type=int, default=80)
    args = parser.parse_args()
    main_gp_no_gflownet(args.data, max_trials=args.trials, time_limit=args.time)
