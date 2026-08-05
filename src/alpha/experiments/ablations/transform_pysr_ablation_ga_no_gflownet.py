#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
消融实验: GA 替代 GFlowNet（遗传算法特征子集选择）

和 transform_pysr_primary.py 完全一致，仅将 GFlowNet 替换为 GA：
  - 特征选择：GA 种群进化取代 GFlowNet 策略采样
  - GA 使用 MLQC reward 作为适应度驱动进化
  - 无 GFlowNet 训练
  - 其余全部保留（PySR 配置、FM 回归、中性化、质量门禁、去重、样本外评估）

用于对比：GFlowNet 引导 vs GA 搜索 vs 随机搜索 对公式质量和多样性的影响。
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
import argparse
from datetime import datetime
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd

from alpha.evaluation.backtester import AcademicBacktester
from alpha.config import REPO_ROOT, Config, set_global_seed
from alpha.data.data_loader import CSI500Loader
from alpha.mining.preprocessor import DataPreprocessor
from alpha.mining.orchestrator import MiningOrchestrator
from alpha.mining.pysr_engine import PySRMiningEngine

logger = logging.getLogger('AcademicDualEngine_GA')


class GAFeatureSelector:
    """遗传算法特征子集选择器。

    维护一个种群，每个个体是一个特征索引子集（3-5个特征）。
    适应度 = MLQC reward 的指数移动平均。
    每 POP_SIZE 轮进化一次（锦标赛选择 + 并集交叉 + 替换变异）。
    """

    def __init__(self, pool_size: int, min_select: int, max_select: int,
                 pop_size: int = 10, mutation_rate: float = 0.3):
        self.pool_size = pool_size
        self.min_select = min_select
        self.max_select = max_select
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate

        # population: list of (indices: List[int], fitness: float, age: int)
        self.population: List[Tuple[List[int], float, int]] = []
        self._init_population()

        self.generation = 0
        self.total_trials = 0

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _random_subset(self) -> List[int]:
        n = int(np.random.randint(self.min_select, self.max_select + 1))
        return sorted(np.random.choice(self.pool_size, n, replace=False).tolist())

    def _init_population(self) -> None:
        self.population = [(self._random_subset(), 0.0, 0) for _ in range(self.pop_size)]

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------
    def select(self) -> Tuple[List[int], int]:
        """返回 (特征索引列表, 个体ID)，个体 ID 用于后续 update_fitness。"""
        if self.total_trials < self.pop_size:
            idx = self.total_trials
        else:
            candidates = np.random.choice(len(self.population), 3, replace=False)
            best_idx = max(candidates, key=lambda i: self.population[i][1])
            idx = int(best_idx)

        indices, fitness, age = self.population[idx]
        self.population[idx] = (indices, fitness, age + 1)
        self.total_trials += 1
        return list(indices), idx

    def update_fitness(self, idx: int, reward: float, alpha: float = 0.3) -> None:
        """用新 reward 更新个体的适应度（指数移动平均）。"""
        indices, old_fit, age = self.population[idx]
        if old_fit == 0.0:
            new_fit = reward
        else:
            new_fit = alpha * reward + (1 - alpha) * old_fit
        self.population[idx] = (indices, new_fit, age)

    def evolve(self) -> None:
        """进化到下一代。"""
        self.generation += 1
        sorted_idx = sorted(
            range(len(self.population)),
            key=lambda i: self.population[i][1],
            reverse=True,
        )

        # 精英保留 top 2
        new_pop: List[Tuple[List[int], float, int]] = [
            (list(self.population[i][0]), self.population[i][1], 0)
            for i in sorted_idx[:2]
        ]

        while len(new_pop) < self.pop_size:
            if np.random.random() < 0.15:
                new_pop.append((self._random_subset(), 0.0, 0))
                continue

            p1_idx = max(
                np.random.choice(len(self.population), 3, replace=False),
                key=lambda i: self.population[i][1],
            )
            p2_idx = max(
                np.random.choice(len(self.population), 3, replace=False),
                key=lambda i: self.population[i][1],
            )
            child = self._crossover(self.population[p1_idx][0], self.population[p2_idx][0])
            child = self._mutate(child)
            new_pop.append((child, 0.0, 0))

        self.population = new_pop[:self.pop_size]
        best_fit = max(p[1] for p in self.population)
        logger.info(
            f"GA 进化 → 第 {self.generation} 代 | "
            f"种群最佳适应度: {best_fit:.4f}"
        )

    # ------------------------------------------------------------------
    # 遗传算子
    # ------------------------------------------------------------------
    def _crossover(self, p1: List[int], p2: List[int]) -> List[int]:
        """并集交叉：取两亲本的并集，若超出 MAX_FEATURES 则随机裁剪。"""
        combined = sorted(set(p1 + p2))
        if len(combined) < self.min_select:
            available = [i for i in range(self.pool_size) if i not in combined]
            n_needed = self.min_select - len(combined)
            if available and n_needed > 0:
                extra = np.random.choice(available, min(n_needed, len(available)), replace=False).tolist()
                combined = sorted(combined + extra)
        elif len(combined) > self.max_select:
            combined = sorted(np.random.choice(combined, self.max_select, replace=False).tolist())
        return combined

    def _mutate(self, subset: List[int]) -> List[int]:
        """替换变异：以 mutation_rate 概率替换一个特征。"""
        if np.random.random() >= self.mutation_rate:
            return subset
        available = [i for i in range(self.pool_size) if i not in subset]
        if not available:
            return subset
        remove_idx = int(np.random.randint(len(subset)))
        new_feat = int(np.random.choice(available))
        subset = [f for i, f in enumerate(subset) if i != remove_idx] + [new_feat]
        return sorted(subset)

    # ------------------------------------------------------------------
    # 报告
    # ------------------------------------------------------------------
    def report(self) -> str:
        lines = [f"GA 报告 (第{self.generation}代, {self.total_trials}试次):"]
        for i, (indices, fitness, age) in enumerate(self.population):
            feat_str = ",".join(str(x) for x in indices)
            lines.append(f"  [{i:2d}] fit={fitness:.4f} age={age:2d}  [{feat_str}]")
        return "\n".join(lines)


class GAOrchestrator(MiningOrchestrator):
    """继承 MiningOrchestrator，用 GA 特征选择替代 GFlowNet。"""

    def __init__(self, preprocessor: DataPreprocessor, **kwargs):
        super().__init__(preprocessor, **kwargs)
        self.ga_selector = GAFeatureSelector(
            pool_size=len(Config.FINAL_FEATURE_POOL),
            min_select=Config.MIN_FEATURES,
            max_select=Config.MAX_FEATURES,
            pop_size=10,
            mutation_rate=0.3,
        )

    def run(self, max_trials: int = 50, time_limit_min: int = 60):
        start = time.time()
        best_tstat, trial = 0.0, 0
        logger.info(
            f"Orchestrator {self.name} (GA) 启动 | "
            f"上限:{max_trials}轮/{time_limit_min}分钟"
        )

        while trial < max_trials:
            if (time.time() - start) / 60 > time_limit_min:
                break

            # === Step 1: GA selects feature subset ===
            feats_idx, indiv_idx = self.ga_selector.select()
            feats = [Config.FINAL_FEATURE_POOL[i] for i in feats_idx]
            logger.info(
                f"\n{'=' * 50}\n"
                f"Trial #{trial} (GA 个体 #{indiv_idx}, 代#{self.ga_selector.generation})\n"
                f"特征: {feats}\n"
                f"{'=' * 50}"
            )

            if len(feats) < Config.MIN_FEATURES:
                self.ga_selector.update_fitness(indiv_idx, 0.01)
                trial += 1
                continue

            # === Step 2: PySR symbolic regression on selected features ===
            ds = self.prep.get_subset(feats)
            X_tr, y_tr_raw = ds['train'][0], ds['train'][1]
            y_tr = y_tr_raw

            engine = PySRMiningEngine()
            pysr_result = engine.run(pd.DataFrame(X_tr, columns=feats), y_tr)
            if not pysr_result:
                self.ga_selector.update_fitness(indiv_idx, 0.01)
                trial += 1
                continue
            formula, complexity = pysr_result

            # === Step 3: Evaluate formula on full data ===
            used_feats = [f for f in feats if f in formula]
            if len(used_feats) <= 1 and '(' not in formula:
                logger.info(f"Trial #{trial} 单特征线性公式，跳过")
                self.ga_selector.update_fitness(indiv_idx, 0.01)
                trial += 1
                continue

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
                self.ga_selector.update_fitness(indiv_idx, 0.01)
                trial += 1
                continue

            # === Step 4: FWL neutralization + Fama-MacBeth regression ===
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

            fid = f"GA_{trial:03d}"
            self.fm_regressor.print_report(fm_result, factor_name=fid)

            # === Step 5: MLQC quality gates ===
            # 5a: t-stat threshold
            if abs(fm_result.t_stat) < Config.TSTAT_THRESHOLD:
                logger.warning(
                    f"{fid} t-stat不达标 ({fm_result.t_stat:.3f} < {Config.TSTAT_THRESHOLD})"
                )
                self.ga_selector.update_fitness(indiv_idx, 0.01)
                trial += 1
                continue

            # 5b: monotonicity
            if Config.USE_MONOTONICITY_GATE:
                if not self._check_monotonicity(fm_result.quantile_returns, fm_result.t_stat):
                    logger.warning(f"{fid} 单调性不达标")
                    self.ga_selector.update_fitness(indiv_idx, 0.01)
                    trial += 1
                    continue

            # 5c: structural dedup
            structure_fp = self._normalize_structure(formula, feats)
            if structure_fp in self.structural_fingerprints:
                logger.warning(f"{fid} 结构重复 ({structure_fp[:60]}...)")
                self.ga_selector.update_fitness(indiv_idx, 0.005)
                trial += 1
                continue

            # 5d: canonical signature dedup
            canon_sig = self._canonical_signature(formula, feats)
            canon_dup_penalty = 1.0
            if canon_sig in self.structural_signatures:
                logger.info(
                    f"{fid} 特征无关结构重复 ({canon_sig[:50]}...)，施加软惩罚"
                )
                canon_dup_penalty = 0.3

            # === Step 6: MLQC reward ===
            reward, reward_debug = self._compute_reward(fm_result, feats, formula=formula)
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

            # === Step 7: GA fitness update ===
            self.ga_selector.update_fitness(indiv_idx, reward)

            # === Step 8: Test set evaluation & registry ===
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
            fm_dict['formula'] = formula
            fm_dict['features'] = feats
            fm_dict['complexity'] = complexity
            self.registry.register(fid, formula, fm_dict, feats)
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

            # === Step 9: GA evolution (every pop_size trials) ===
            if trial % self.ga_selector.pop_size == 0 and trial > 0:
                logger.info("\n" + self.ga_selector.report())
                self.ga_selector.evolve()

        # === End of trials ===
        logger.info("\n" + self.ga_selector.report())

        prod_factors = self.registry.get_production_ready()
        logger.info(
            f"注册表因子总数: {len(self.registry.data['factors'])} 个 | "
            f"达生产标准: {len(prod_factors)} 个"
        )
        self._generate_summary_report(prod_factors)
        return prod_factors


def main_ga_no_gflownet(data_path: str, max_trials: int = 50, time_limit: int = 60):
    set_global_seed(42)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    Config.OUTPUT_DIR = os.path.join(
        Config.OUTPUT_ROOT, f"ablation_ga_no_gflownet_{timestamp}"
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
        "消融: GA 特征选择 (替代 GFlowNet)\n"
        + "=" * 60
    )
    orchestrator = GAOrchestrator(prep)
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
            ns = {f: X_all[:, i] for i, f in enumerate(feats)}
            ns.update(orchestrator.pysr_ns)
            try:
                pred_all = eval(formula, {"__builtins__": {}}, ns)
                pred_all = np.where(np.isfinite(pred_all), pred_all, np.nan)
                backtester.run(
                    pred_all[prep.te_mask],
                    ds['test'][1],
                    ds['test'][2],
                    ds['test'][3],
                    fid,
                )
            except Exception as e:
                logger.warning(f"{fid} 回测失败: {e}")

    print(f"\n消融(GA)完成！产出物: {os.path.abspath(Config.OUTPUT_DIR)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="消融实验: GA 替代 GFlowNet (遗传算法特征子集选择)"
    )
    parser.add_argument(
        '--data', type=str,
        default=os.path.join(_REPO_ROOT, 'data', 'csi500_daily_2020-07-20_to_2026-07-19.parquet'),
    )
    parser.add_argument('--trials', type=int, default=60)
    parser.add_argument('--time', type=int, default=80)
    args = parser.parse_args()
    main_ga_no_gflownet(args.data, max_trials=args.trials, time_limit=args.time)
