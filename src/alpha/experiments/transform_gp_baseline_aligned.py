#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GP Baseline (方法学对齐版): gplearn 遗传规划因子挖掘
对比对象: transform_pysr_primary.py (GFlowNet + PySR)

与 transform_gp_baseline.py 的方法学差异 (对齐主流水线):
  1. 验证集验证: 达标筛选 (t-stat ≥ TSTAT_THRESHOLD) 在【验证集】上做,
     GP 进化时先做全历史公式去重再取 Hall of Fame.
  2. 测试集只做样本外报告: 仅在通过验证集门禁的因子上, 于测试集跑
     FM 回归并写入 registry, 不再参与任何筛选.
  3. FWL 中性化: 验证集与测试集的 FM 前均剔除 ln(amount) 暴露,
     与主流水线指标口径一致 (避免 size 污染虚高显著性).
  4. 结构去重: 报告因子做特征无关的结构指纹去重, 避免重复公式虚增数量.

说明: gplearn 的适应度函数只在训练数据上评估, 因此搜索适应度使用
训练集 ICIR (同 transform_gp_baseline.py), 而真正用于模型选择的
"验证" 在进化后通过验证集 FM t-stat 门禁完成, 测试集保持干净.
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
import time
import warnings
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from datetime import datetime
import sympy as sp

from alpha.config import Config, set_global_seed, apply_split_mode
from alpha.data.data_loader import CSI500Loader
from alpha.mining.safe_ops import SafeOps
from alpha.mining.preprocessor import DataPreprocessor
from alpha.mining.fm_regression import FamaMacBethRegressor
from alpha.mining.registry import FactorRegistry, is_production_ready
from alpha.mining.neutralize import fwl_neutralize

logger = logging.getLogger('GPBaselineAligned')

try:
    from gplearn.fitness import make_fitness
    from gplearn.functions import make_function
    from gplearn.genetic import SymbolicRegressor
    HAS_GPLEARN = True
except ImportError:
    HAS_GPLEARN = False
    logger.error("gplearn 未安装: pip install gplearn")


def _safe_std(x, ddof=0):
    with np.errstate(over='ignore', invalid='ignore'):
        return np.std(x, ddof=ddof)


def compute_icir_from_preds(y_true: np.ndarray, y_pred: np.ndarray,
                            dates: np.ndarray, min_stocks: int = 30,
                            min_periods: int = 10) -> float:
    N = min(len(y_true), len(y_pred), len(dates))
    if N < 10:
        return -1.0
    if _safe_std(y_pred) < 1e-8:
        return -1.0
    df = pd.DataFrame({'date': pd.to_datetime(dates[:N]).normalize(),
                       'y': y_true[:N], 'pred': y_pred[:N]})
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    ic_list = []
    for _, grp in df.groupby('date'):
        if len(grp) > min_stocks:
            pred_vals = grp['pred'].values
            y_vals = grp['y'].values
            if _safe_std(pred_vals) < 1e-8 or _safe_std(y_vals) < 1e-8:
                continue
            with np.errstate(invalid='ignore'), warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                warnings.simplefilter('ignore', UserWarning)
                ic = spearmanr(y_vals, pred_vals)[0]
            if np.isfinite(ic):
                ic_list.append(ic)
    if len(ic_list) < min_periods:
        return -1.0
    ic_mean = np.mean(ic_list)
    ic_std = _safe_std(ic_list, ddof=1)
    if ic_std < 1e-8:
        return -1.0
    return float(ic_mean / ic_std)


def _make_icir_fitness(train_dates: np.ndarray):
    def _icir(y, y_pred, sample_weight=None):
        return compute_icir_from_preds(y, y_pred, train_dates, min_stocks=30, min_periods=10)
    return _icir


def _safe_square(x):
    x = np.clip(np.asarray(x, dtype=np.float64), -1e4, 1e4)
    res = np.square(x)
    return np.clip(res, -1e8, 1e8).astype(np.float32)


def _safe_abs(x):
    return np.abs(np.asarray(x, dtype=np.float64)).astype(np.float32)


def _safe_sign(x):
    return np.sign(np.asarray(x, dtype=np.float64)).astype(np.float32)


def _safe_log1p(x):
    return SafeOps.safe_log1p(x)


def _safe_inv(x):
    return SafeOps.safe_inv(x)


def _safe_div(x1, x2):
    return SafeOps.protected_div(x1, x2)


def _structure_fingerprint(formula: str, feature_names: List[str]) -> str:
    """特征无关的结构指纹: 特征名替换为占位符后 sympy 归一化."""
    used = [f for f in feature_names if f in formula]
    if not used:
        return formula
    s = formula
    for i, f in enumerate(used):
        s = s.replace(f, f'_F{i}_')
    try:
        expr = sp.expand(sp.sympify(s))
        return str(expr)
    except Exception:
        return s


def write_summary_report(gp_factors: List[dict]):
    if not gp_factors:
        return

    def _fmt_table(entries: List[dict]) -> str:
        header = (f"{'ID':<12} {'Val-t':>8} {'t-stat':>8} {'p-value':>10} {'Coef':>10} "
                  f"{'Rank_IC':>8} {'ICIR':>8} {'L-S':>10} {'R²':>8} {'公式'}")
        lines = [header, "-" * 100]
        for f in sorted(entries, key=lambda x: abs(x.get('FM_tstat', 0)), reverse=True):
            sig = "***" if abs(f.get('FM_tstat', 0)) >= 3.0 else (
                "**" if abs(f.get('FM_tstat', 0)) >= 2.0 else "")
            lines.append(
                f"{f['id']:<12} {f.get('Val_tstat', 0):>7.3f} "
                f"{f.get('FM_tstat', 0):>7.3f}{sig:<3} "
                f"{f.get('FM_pvalue', 1):>10.6f} {f.get('FM_coef', 0):>10.6f} "
                f"{f.get('Rank_IC', 0):>8.4f} {f.get('ICIR', 0):>8.4f} "
                f"{f.get('LS_spread', 0):>10.6f} {f.get('FM_R2_avg', 0):>8.4f} "
                f"{f.get('formula', '')[:50]}"
            )
        return "\n".join(lines)

    passed = [f for f in gp_factors if is_production_ready(f)]
    passed_ids = {f['id'] for f in passed}
    failed = [f for f in gp_factors if f['id'] not in passed_ids]
    report_lines = [
        "=" * 100,
        "GP Baseline (对齐版) 因子挖掘汇总报告",
        "=" * 100,
        f"统一达标标准: |t|≥{Config.PRODUCT_FMT_THRESHOLD}, "
        f"|ICIR|≥{Config.PRODUCT_ICIR_THRESHOLD}, |Rank_IC|≥{Config.PRODUCT_IC_THRESHOLD}, "
        f"截面期数≥30",
        "-" * 100,
        f"[1/2] 达标因子 ({len(passed)} 个)",
        _fmt_table(passed),
        "-" * 100,
        f"[2/2] 未达标因子 ({len(failed)} 个)",
        _fmt_table(failed),
    ]
    report_text = "\n".join(report_lines)
    logger.info(f"\n{report_text}")
    report_path = os.path.join(Config.OUTPUT_DIR, "fm_summary_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    logger.info(f"汇总报告已保存至: {report_path}")


def run_gp_baseline_aligned(data_path: str,
                            max_generations: int = 40,
                            population_size: int = 500,
                            tournament_size: int = 20,
                            hall_of_fame_size: int = 15,
                            parsimony_coefficient: float = 0.002,
                            max_samples: float = 0.7,
                            random_state: int = Config.GP_SEED,
                            n_jobs: int = 1,
                            ) -> Tuple[List[dict], DataPreprocessor]:

    if not HAS_GPLEARN:
        raise RuntimeError("gplearn 未安装")

    logger.info("=" * 60)
    logger.info(f"GP Baseline (对齐版) | generations={max_generations}, pop={population_size}")
    logger.info("=" * 60)

    set_global_seed()
    logger.info("=" * 60)
    logger.info(f"GP Baseline (对齐版) | generations={max_generations}, pop={population_size}")
    loader = CSI500Loader(path=data_path)
    df_raw = loader.load()

    prep = DataPreprocessor()
    prep.prepare_full_pool(df_raw)

    X_train = prep.full_X[prep.tr_mask]
    y_train = prep.full_y[prep.tr_mask]
    train_dates = prep.full_dates[prep.tr_mask]

    X_val = prep.full_X[prep.val_mask]
    y_val = prep.full_y[prep.val_mask]
    val_dates = prep.full_dates[prep.val_mask]
    val_symbols = prep.full_symbols[prep.val_mask]
    val_amount = prep.full_amount[prep.val_mask]

    X_test = prep.full_X[prep.te_mask]
    y_test = prep.full_y[prep.te_mask]
    test_dates = prep.full_dates[prep.te_mask]
    test_symbols = prep.full_symbols[prep.te_mask]
    test_amount = prep.full_amount[prep.te_mask]

    logger.info(f"训练集: {X_train.shape[0]:,} x {X_train.shape[1]} | "
                f"验证集: {X_val.shape[0]:,} | 测试集: {X_test.shape[0]:,}")

    square_func = make_function(function=_safe_square, name='square', arity=1)
    abs_func = make_function(function=_safe_abs, name='abs', arity=1)
    sign_func = make_function(function=_safe_sign, name='sign', arity=1)
    log1p_func = make_function(function=_safe_log1p, name='log1p', arity=1)
    inv_func = make_function(function=_safe_inv, name='inv', arity=1)
    div_func = make_function(function=_safe_div, name='div', arity=2)

    function_set = ['add', 'sub', 'mul', div_func,
                    square_func, abs_func, log1p_func, sign_func, inv_func]

    icir_fitness = make_fitness(function=_make_icir_fitness(train_dates), greater_is_better=True)

    gp = SymbolicRegressor(
        population_size=population_size,
        generations=max_generations,
        tournament_size=tournament_size,
        stopping_criteria=4.0,
        function_set=function_set,
        metric=icir_fitness,
        parsimony_coefficient=parsimony_coefficient,
        p_crossover=0.7,
        p_subtree_mutation=0.1,
        p_hoist_mutation=0.05,
        p_point_mutation=0.1,
        p_point_replace=0.05,
        max_samples=max_samples,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=1,
    )

    logger.info("\nGP 进化搜索开始...")
    t0 = time.time()
    gp.fit(X_train, y_train)
    elapsed = time.time() - t0
    logger.info(f"GP 完成，耗时 {elapsed:.1f}s")

    # 全历史代际去重后取 Hall of Fame (同 gp_baseline.py, 避免重复公式)
    seen = {}
    for gen_progs in gp._programs:
        for p in gen_progs:
            if p is None:
                continue
            key = str(p)
            if key not in seen or p.fitness_ > seen[key].fitness_:
                seen[key] = p
    unique_progs = sorted(seen.values(), key=lambda p: p.fitness_, reverse=True)
    programs = unique_progs[:hall_of_fame_size]
    logger.info(f"全历史去重后 {len(unique_progs)} 个唯一公式, 取 top {len(programs)}")

    fm = FamaMacBethRegressor(nw_lags=Config.NW_LAGS)
    feature_names = list(Config.FINAL_FEATURE_POOL)

    # ── Phase A: 验证集门禁 (t-stat + 结构去重) ──────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("Phase A: 验证集门禁筛选")
    logger.info("=" * 60)
    selected = []
    fingerprints = set()
    for i, prog in enumerate(programs):
        if prog is None:
            continue
        formula_str = str(prog)
        y_pred_val = prog.execute(X_val)
        valid = np.isfinite(y_pred_val) & np.isfinite(y_val)
        if valid.sum() < 500:
            logger.info(f"GP_{i:03d} 验证集有效样本过少，跳过")
            continue

        fv, rv = fwl_neutralize(y_pred_val[valid], y_val[valid],
                                val_dates[valid], val_symbols[valid], val_amount[valid])
        ok = np.isfinite(fv) & np.isfinite(rv)
        if ok.sum() < 500:
            logger.info(f"GP_{i:03d} 中性化后有效样本过少，跳过")
            continue
        fm_val = fm.run(fv[ok], rv[ok], val_dates[valid][ok], val_symbols[valid][ok])

        if abs(fm_val.t_stat) < Config.TSTAT_THRESHOLD:
            logger.info(f"GP_{i:03d} 验证集 t-stat 不达标 ({fm_val.t_stat:.3f} < "
                        f"{Config.TSTAT_THRESHOLD}) | fitness={prog.fitness_:.4f}")
            continue

        fp = _structure_fingerprint(formula_str, feature_names)
        if fp in fingerprints:
            logger.info(f"GP_{i:03d} 结构重复，淘汰 | t={fm_val.t_stat:.3f} | {formula_str[:60]}")
            continue
        fingerprints.add(fp)
        selected.append((i, prog, formula_str, fm_val))
        sig = '*' * min(3, max(0, int(abs(fm_val.t_stat) // 1.5)))
        logger.info(f"GP_{i:03d} 通过验证集门禁 | t={fm_val.t_stat:6.3f}{sig} | "
                    f"ICIR={fm_val.rank_icir:.4f} | len={prog.length_:2d} | {formula_str[:50]}")

    logger.info(f"\n通过验证集门禁: {len(selected)}/{len(programs)} 个")

    # ── Phase B: 测试集样本外报告 (仅报告, 不筛选) ────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("Phase B: 测试集样本外报告 (FWL 中性化)")
    logger.info("=" * 60)
    registry = FactorRegistry(version="GP_baseline_gplearn_aligned")
    gp_factors = []
    for i, prog, formula_str, fm_val in selected:
        y_pred_test = prog.execute(X_test)
        valid = np.isfinite(y_pred_test) & np.isfinite(y_test)
        if valid.sum() < 500:
            logger.warning(f"GP_{i:03d} 测试集有效样本过少，跳过")
            continue

        fv_te, rv_te = fwl_neutralize(y_pred_test[valid], y_test[valid],
                                      test_dates[valid], test_symbols[valid], test_amount[valid])
        ok = np.isfinite(fv_te) & np.isfinite(rv_te)
        if ok.sum() < 500:
            logger.warning(f"GP_{i:03d} 测试集中性化后有效样本过少，跳过")
            continue
        fm_test = fm.run(fv_te[ok], rv_te[ok], test_dates[valid][ok], test_symbols[valid][ok])

        fid = f'GP_{i:03d}'
        metrics = fm_test.to_dict()
        metrics['formula'] = formula_str
        metrics['Val_tstat'] = fm_val.t_stat
        metrics['fitness'] = float(prog.fitness_)
        metrics['length'] = int(prog.length_)

        gp_factors.append({
            'id': fid, 'formula': formula_str,
            'fitness': float(prog.fitness_), 'length': int(prog.length_),
            'Val_tstat': fm_val.t_stat,
            'FM_tstat': fm_test.t_stat, 'FM_pvalue': fm_test.p_value,
            'FM_coef': fm_test.coefficient, 'FM_se': fm_test.std_error,
            'Rank_IC': fm_test.rank_ic_mean, 'ICIR': fm_test.rank_icir,
            'LS_spread': fm_test.long_short_spread, 'FM_R2_avg': fm_test.avg_r_squared,
            'FM_n_periods': fm_test.n_periods,
        })

        registry.register(fid, formula_str, metrics)

        sig = '*' * min(3, max(0, int(abs(fm_test.t_stat) // 1.5)))
        logger.info(f"{fid} | Val-t={fm_val.t_stat:6.3f} | Test-t={fm_test.t_stat:6.3f}{sig} | "
                    f"ICIR={fm_test.rank_icir:.4f} | len={prog.length_:2d} | {formula_str[:50]}")

    n_prod = sum(1 for f in gp_factors if is_production_ready(f))
    logger.info(f"\nGP 对齐版达标因子: {n_prod}/{len(selected)} "
                f"(验证门禁 + 统一生产标准: |t|≥{Config.PRODUCT_FMT_THRESHOLD}, "
                f"ICIR≥{Config.PRODUCT_ICIR_THRESHOLD}, IC≥{Config.PRODUCT_IC_THRESHOLD})")
    write_summary_report(gp_factors)
    return gp_factors, prep


def main(data_path: str, max_generations: int = 40, population_size: int = 500,
         time_limit_min: int = 60):
    set_global_seed()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    Config.OUTPUT_DIR = os.path.join("factor_output_academic_v81", f"gp_baseline_aligned_{timestamp}")
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
    valid_factors, prep = run_gp_baseline_aligned(
        data_path=data_path,
        max_generations=max_generations,
        population_size=population_size,
        tournament_size=20,
        hall_of_fame_size=15,
        parsimony_coefficient=0.002,
        max_samples=0.7,
        random_state=Config.GP_SEED,
        n_jobs=1,
    )

    print(f"\n完成！产出物: {os.path.abspath(Config.OUTPUT_DIR)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GP Baseline (对齐版): gplearn 遗传规划因子挖掘")
    parser.add_argument('--data', type=str,
                        default='data/warmup/csi500_daily_2020-06-30_to_2026-06-30.parquet')
    parser.add_argument('--generations', type=int, default=40)
    parser.add_argument('--population', type=int, default=500)
    parser.add_argument('--time-limit', type=int, default=60)
    parser.add_argument('--mode', type=str, default='warmup', choices=['cold', 'warmup', 'ratio', 'month'],
                        help='切分模式(与主线一致): warmup(前12月回溯+36:12:12, 默认), cold/ratio/month')
    args = parser.parse_args()
    apply_split_mode(args.mode)
    logger.info(f"切分配置: mode={args.mode} (SPLIT_MODE={Config.SPLIT_MODE}, SPLIT_WARMUP={Config.SPLIT_WARMUP}, "
                f"SPLIT_MONTH_ANCHOR={Config.SPLIT_MONTH_ANCHOR})")
    main(args.data, max_generations=args.generations,
         population_size=args.population, time_limit_min=args.time_limit)
