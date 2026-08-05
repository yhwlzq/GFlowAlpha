#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GP Baseline: gplearn 遗传规划因子挖掘
对比对象: transform_pysr_primary.py (GFlowNet + PySR)

数据加载、特征预处理、FM评估与transform_pysr_primary.py完全一致，
仅替换因子挖掘引擎为gplearn的SymbolicRegressor，确保对比公平。
"""

import os
import sys
import json
import logging
import time
import numpy as np
np.seterr(all='ignore')
import pandas as pd
from scipy.stats import spearmanr
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import warnings
from scipy.stats import ConstantInputWarning
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=ConstantInputWarning)

# ── 确保能找到 transform_pysr_primary.py ──
    
    

from alpha.config import Config, set_global_seed
from alpha.data.data_loader import CSI500Loader
from alpha.mining.preprocessor import DataPreprocessor
from alpha.mining.fm_regression import FamaMacBethRegressor, FMRegressionResult
from alpha.evaluation.backtester import AcademicBacktester
from alpha.mining.neutralize import fwl_neutralize

# re-use logger name from primary pipeline
logger = logging.getLogger('AcademicDualEngine')

try:
    from gplearn.fitness import make_fitness
    from gplearn.functions import make_function
    from gplearn.genetic import SymbolicRegressor
    HAS_GPLEARN = True
except ImportError:
    HAS_GPLEARN = False
    logger.error("gplearn 未安装: pip install gplearn")


def compute_icir_from_preds(y_true: np.ndarray, y_pred: np.ndarray,
                         dates: np.ndarray, min_stocks: int = 30,
                         min_periods: int = 10) -> float:
    """按日期分组计算 Rank IC 序列，返回 ICIR (mean/std)。"""
    N = min(len(y_true), len(y_pred), len(dates))
    if N < 10:
        return 0.0
    df = pd.DataFrame({'date': pd.to_datetime(dates[:N]).normalize(),
                       'y': y_true[:N], 'pred': y_pred[:N]}).dropna()
    ic_list = []
    for _, grp in df.groupby('date'):
        if len(grp) > min_stocks:
            y_vals = grp['y'].values
            p_vals = grp['pred'].values
            if y_vals.std() < 1e-10 or p_vals.std() < 1e-10:
                continue
            with np.errstate(invalid='ignore'), warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                warnings.simplefilter('ignore', UserWarning)
                ic = spearmanr(y_vals, p_vals)[0]
            if np.isfinite(ic):
                ic_list.append(ic)
    if len(ic_list) < min_periods:
        return 0.0
    return float(np.mean(ic_list) / (np.std(ic_list, ddof=1) + 1e-8))


# ─────────────────────────────────────────────────────────────
#  自定义 gplearn 适应度函数 (ICIR)
#  gplearn 签名为 f(y, y_pred, sample_weight)，不传外部数据。
#  我们通过闭包注入 dates 数组。
# ─────────────────────────────────────────────────────────────
def _make_icir_fitness(val_dates: np.ndarray):
    """工厂函数：将 val_dates 注入适应度闭包（验证集 ICIR）。"""
    def _icir(y, y_pred, sample_weight=None):
        return compute_icir_from_preds(y, y_pred, val_dates,
                                       min_stocks=30, min_periods=10)
    return _icir


# ─────────────────────────────────────────────────────────────
#  安全算子 (与 transform_pysr_primary.py SafeOps 一致)
# ─────────────────────────────────────────────────────────────
def _safe_log1p(x):
    x = np.clip(x, -1e4, 1e4).astype(np.float32)
    return np.log1p(np.abs(x)) * np.sign(x)

def _safe_inv(x):
    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, -1e4, 1e4)
    mask = np.abs(x) > 1e-8
    safe_x = np.where(mask, x, 1.0)
    return np.where(mask, 1.0 / safe_x, 0.0).astype(np.float32)

def _safe_div(x1, x2):
    x1 = np.asarray(x1, dtype=np.float32)
    x2 = np.asarray(x2, dtype=np.float32)
    x2 = np.clip(x2, -1e4, 1e4)
    mask = np.abs(x2) > 1e-8
    safe_x2 = np.where(mask, x2, 1.0)
    return np.where(mask, x1 / safe_x2, 0.0).astype(np.float32)

def _safe_square(x):
    x = np.clip(x, -1e4, 1e4).astype(np.float32)
    res = np.square(x)
    return np.clip(res, -1e8, 1e8).astype(np.float32)

def _safe_abs(x):
    return np.abs(x).astype(np.float32)

def _safe_sign(x):
    return np.sign(x).astype(np.float32)


# ─────────────────────────────────────────────────────────────
#  GP 基线主流程
# ─────────────────────────────────────────────────────────────
def run_gp_baseline(data_path: str,
                    max_generations: int = 50,
                    population_size: int = 1000,
                     tournament_size: int = 5,
                    hall_of_fame_size: int = 20,
                     parsimony_coefficient: float = 0.05,
                    random_state: int = 42,
                    n_jobs: int = 4,
                    ) -> Tuple[List[dict], List[dict], DataPreprocessor]:
    """运行 GP 基线实验，返回 (达标因子列表, preprocessor)。"""

    if not HAS_GPLEARN:
        raise RuntimeError("gplearn 未安装，无法运行 GP 基线")

    logger.info("=" * 60)
    logger.info(" GP 基线因子挖掘 (gplearn SymbolicRegressor)")
    logger.info(f"  generations={max_generations}, pop={population_size}, "
                f"tournament={tournament_size}")
    logger.info("=" * 60)

    # ── 1. 加载数据 ──────────────────────────────────────────
    set_global_seed(random_state)
    loader = CSI500Loader(path=data_path)
    df_raw = loader.load()

    prep = DataPreprocessor()
    prep.prepare_full_pool(df_raw)
    logger.info(f"特征映射: {dict(enumerate(prep.valid_features))}")

    # ── 2. 提取训练数据 ─────────────────────────────────────
    X_train = prep.full_X[prep.tr_mask]
    y_train = prep.full_y[prep.tr_mask]
    train_dates = prep.full_dates[prep.tr_mask]

    X_val = prep.full_X[prep.val_mask]
    y_val = prep.full_y[prep.val_mask]
    val_dates = prep.full_dates[prep.val_mask]

    X_test = prep.full_X[prep.te_mask]
    y_test = prep.full_y[prep.te_mask]
    test_dates = prep.full_dates[prep.te_mask]
    test_symbols = prep.full_symbols[prep.te_mask]
    test_amount = prep.full_amount[prep.te_mask]

    logger.info(f"训练集: {X_train.shape[0]:,} 样本 × {X_train.shape[1]} 特征")
    logger.info(f"验证集: {X_val.shape[0]:,} 样本")
    logger.info(f"测试集: {X_test.shape[0]:,} 样本")

    # ── 3. 定义 gplearn 函数集与适应度 ──────────────────────
    log1p_func = make_function(function=_safe_log1p, name='log1p', arity=1)
    inv_func = make_function(function=_safe_inv, name='inv', arity=1)
    div_func = make_function(function=_safe_div, name='div', arity=2)
    tanh_func = make_function(function=np.tanh, name='tanh', arity=1)
    square_func = make_function(function=_safe_square, name='square', arity=1)
    abs_func = make_function(function=_safe_abs, name='abs', arity=1)
    sign_func = make_function(function=_safe_sign, name='sign', arity=1)

    function_set = [
        'add', 'sub', 'mul', div_func,
        tanh_func, square_func, abs_func, log1p_func, sign_func, inv_func,
    ]

    icir_fitness = make_fitness(
        function=_make_icir_fitness(val_dates),
        greater_is_better=True,
    )

    # ── 4. 运行 GP 进化 ────────────────────────────────────
    gp = SymbolicRegressor(
        population_size=population_size,
        generations=max_generations,
        tournament_size=5,
        stopping_criteria=5.0,
        function_set=function_set,
        metric=icir_fitness,
        parsimony_coefficient=0.05,
        init_depth=(2, 4),
        p_crossover=0.4,
        p_subtree_mutation=0.3,
        p_hoist_mutation=0.1,
        p_point_mutation=0.2,
        p_point_replace=0.05,
        max_samples=0.8,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=1,
    )

    logger.info("\n" + "=" * 60)
    logger.info("🔍 开始 GP 进化搜索...")
    logger.info("=" * 60)
    t0 = time.time()

    gp.fit(X_train, y_train)

    elapsed = time.time() - t0
    logger.info(f"GP 进化完成，耗时 {elapsed:.1f}s")

    # ── 5. 提取 Hall of Fame: 遍历所有代的种群, 按公式字符串去重后取 TopK ─
    #     对应 PySR 的 model.equations_ (全历史 Pareto 前沿)
    seen = {}
    for gen_idx, gen_progs in enumerate(gp._programs):
        for p in gen_progs:
            if p is None:
                continue
            key = str(p)
            if key not in seen or p.fitness_ > seen[key].fitness_:
                seen[key] = p
    unique_progs = sorted(seen.values(), key=lambda p: p.fitness_, reverse=True)
    programs = unique_progs[:hall_of_fame_size]
    logger.info(f"共 {len(gp._programs)} 代, 全局去重后 {len(unique_progs)} 个唯一公式, "
                f"取 top {len(programs)}")

    # ── 6. 在测试集上评估每个程序 ──────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("📊 GP 挖掘结果评估 (测试集)")
    logger.info("=" * 60)

    fm = FamaMacBethRegressor(nw_lags=Config.NW_LAGS)
    bt = AcademicBacktester(top_quantile=0.2)
    gp_factors = []

    for i, prog in enumerate(programs[:hall_of_fame_size]):
        if prog is None:
            continue

        formula_str = str(prog)
        y_pred_test = prog.execute(X_test)

        valid = np.isfinite(y_pred_test) & np.isfinite(y_test)
        if valid.sum() < 500:
            continue

        pred_te = y_pred_test[valid]
        ret_te = y_test[valid]
        d_te = test_dates[valid]
        s_te = test_symbols[valid]
        a_te = test_amount[valid]

        # Backtester: raw data (matches pipeline)
        bt_result = bt.run(pred_te, ret_te, d_te, s_te, f'GP_{i:03d}')

        # FM regression: FWL-neutralized (matches pipeline registry)
        fv, rv = fwl_neutralize(pred_te, ret_te, d_te, s_te, a_te)
        ok = np.isfinite(fv) & np.isfinite(rv)
        fm_result = fm.run(fv[ok], rv[ok], d_te[ok], s_te[ok])

        entry = {
            'id': f'GP_{i:03d}',
            'formula': formula_str,
            'fitness': float(prog.fitness_),
            'length': int(prog.length_),
            't_stat': fm_result.t_stat,
            'icir': fm_result.rank_icir,
            'rank_ic': fm_result.rank_ic_mean,
            'ls_spread': fm_result.long_short_spread,
            'n_periods': fm_result.n_periods,
        }
        if bt_result:
            entry['annualized_return'] = bt_result['annualized_return']
            entry['sharpe_ratio'] = bt_result['sharpe_ratio']
            entry['quantile_monthly_rets'] = bt_result['quantile_monthly_rets']
        gp_factors.append(entry)

        sig = '*' * min(3, max(0, int(abs(fm_result.t_stat) // 1.5)))
        bt_str = f"年化={bt_result['annualized_return']*100:.1f}% Sharpe={bt_result['sharpe_ratio']:.2f}" if bt_result else "回测无结果"
        logger.info(
            f"GP_{i:03d} | fitness={prog.fitness_:.4f} "
            f"| t={fm_result.t_stat:6.3f}{sig} "
            f"| ICIR={fm_result.rank_icir:.4f} "
            f"| {bt_str} "
            f"| len={prog.length_:2d} "
            f"| {formula_str[:60]}"
        )

    # ── 7. 筛选达标因子 (|t-stat| > 2.0) ──────────────────
    valid_factors = [f for f in gp_factors if abs(f['t_stat']) > Config.TSTAT_THRESHOLD]
    valid_factors.sort(key=lambda x: abs(x['t_stat']), reverse=True)

    logger.info("\n" + "=" * 60)
    logger.info(f"✅ GP 共挖掘出 {len(valid_factors)}/{len(gp_factors)} 个达标因子 "
              f"(t > {Config.TSTAT_THRESHOLD})")
    logger.info("=" * 60)

    if valid_factors:
        logger.info(f"\nTop 5 达标因子:")
        for f in valid_factors[:5]:
            logger.info(
                f"  {f['id']:>8s} | t={f['t_stat']:6.3f} | "
                f"ICIR={f['icir']:.4f} | IC={f['rank_ic']:.4f} | "
                f"{f['formula'][:80]}"
            )

    return valid_factors, gp_factors, prep


# ─────────────────────────────────────────────────────────────
#  入口
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    DATA_PATH = os.path.join(
        Config.OUTPUT_ROOT,
        'legacy',
        'CSI500_4Years_2022-06-30_to_2026-06-29.csv',
    )

    valid_factors, gp_factors, prep = run_gp_baseline(
        data_path=DATA_PATH,
        max_generations=50,
        population_size=1000,
        tournament_size=5,
        hall_of_fame_size=20,
        parsimony_coefficient=0.05,
        random_state=42,
        n_jobs=4,
    )

    # ── 保存结果 ──
    output = {
        'experiment': 'GP_baseline_gplearn',
        'config': {
            'max_generations': 50,
            'population_size': 1000,
            'tournament_size': 5,
            'function_set': ['add', 'sub', 'mul', 'div',
                             'tanh', 'square', 'abs', 'log1p', 'sign', 'inv'],
            'parsimony_coefficient': 0.05,
            'init_depth': '(2, 4)',
            'max_samples': 0.8,
            'fitness': 'validation_ICIR',
        },
        'n_total_factors': len(valid_factors),
        'valid_factors': valid_factors,
    }

    # 也保存一份完整列表（含未达标因子）
    all_output = {
        'experiment': 'GP_baseline_gplearn_all',
        'config': output['config'],
        'n_total_factors': len(gp_factors),
        'factors': gp_factors,
    }

    out_path = os.path.join(Config.OUTPUT_ROOT, 'gp_baseline_results.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    all_path = os.path.join(Config.OUTPUT_ROOT, 'gp_baseline_all_factors.json')
    with open(all_path, 'w', encoding='utf-8') as f:
        json.dump(all_output, f, indent=2, ensure_ascii=False)

    logger.info(f"\n✅ GP 基线结果已保存至 {out_path}")
    logger.info(f"   (完整列表: {all_path})")