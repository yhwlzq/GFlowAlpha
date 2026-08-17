#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GP Baseline: gplearn 遗传规划因子挖掘 (内联版)
对比对象: transform_pysr_primary.py (GFlowNet + PySR)
"""
import os
import sys
import logging
import json
import time
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from datetime import datetime
import statsmodels.api as sm

from alpha.config import Config, set_global_seed
from alpha.data.data_loader import CSI500Loader
from alpha.mining.safe_ops import SafeOps, TimeSeriesOps
from alpha.features.feature_registry import UltimateDailyFeatureRegistry
from alpha.mining.preprocessor import DataPreprocessor
from alpha.mining.fm_regression import FMRegressionResult, FamaMacBethRegressor
from alpha.mining.registry import FactorRegistry, is_production_ready, format_fm_table
from alpha.mining.neutralize import fwl_neutralize

logger = logging.getLogger('GPBaseline')

try:
    from gplearn.fitness import make_fitness
    from gplearn.functions import make_function
    from gplearn.genetic import SymbolicRegressor
    HAS_GPLEARN = True
except ImportError:
    HAS_GPLEARN = False
    logger.error("gplearn 未安装: pip install gplearn")


class _ICIRMetric:
    """向量化逐日 Rank-IC / ICIR 适应度。

    预计算日期分组索引 (仅一次), 适应度评估时:
      1) 尊重 gplearn 的 sample_weight, 只用被采样的行 (0.7 训练子样本),
         修复 OOB 行混入训练适应度的泄漏, 单次调用节省 30% 计算;
      2) 纯 numpy 逐截面 rank-Spearman, 跳过每轮重建 DataFrame/date 转换/
         pandas groupby (原实现单次 ~0.4s 是运行永不结束的根因).
    """

    def __init__(self, dates: np.ndarray, min_stocks: int = 30, min_periods: int = 10):
        self.min_stocks = min_stocks
        self.min_periods = min_periods
        day = pd.to_datetime(dates).normalize().values.astype('datetime64[D]')
        self.uniq_days, self.group_ids = np.unique(day, return_inverse=True)
        self.n_groups = len(self.uniq_days)

    def __call__(self, y: np.ndarray, y_pred: np.ndarray,
                 sample_weight: np.ndarray = None) -> float:
        y = np.asarray(y, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        if sample_weight is None:
            keep = np.isfinite(y) & np.isfinite(y_pred)
        else:
            keep = ((np.asarray(sample_weight, dtype=np.float64) > 0)
                    & np.isfinite(y) & np.isfinite(y_pred))
        n = int(keep.sum())
        if n < self.min_stocks:
            return -1.0

        g = self.group_ids[keep]
        order = np.argsort(g, kind='stable')
        gs = g[order]
        ys = y[keep][order]
        ps = y_pred[keep][order]
        edges = np.searchsorted(gs, np.arange(self.n_groups + 1))

        ic_list = []
        for i in range(self.n_groups):
            a, b = int(edges[i]), int(edges[i + 1])
            m = b - a
            if m < self.min_stocks:
                continue
            yy = ys[a:b]
            pp = ps[a:b]
            ry = yy.argsort().argsort().astype(np.float64)
            rp = pp.argsort().argsort().astype(np.float64)
            ry -= ry.mean()
            rp -= rp.mean()
            denom = np.sqrt((ry * ry).sum() * (rp * rp).sum())
            if denom < 1e-8:
                continue
            ic = float((ry * rp).sum() / denom)
            if np.isfinite(ic):
                ic_list.append(ic)

        if len(ic_list) < self.min_periods:
            return -1.0
        ic_mean = float(np.mean(ic_list))
        ic_std = float(np.std(ic_list, ddof=1))
        if ic_std < 1e-8:
            return -1.0
        return ic_mean / ic_std


def _make_icir_fitness(train_dates: np.ndarray):
    metric = _ICIRMetric(train_dates, min_stocks=30, min_periods=10)

    def _icir(y, y_pred, sample_weight=None):
        return metric(y, y_pred, sample_weight)

    return _icir


def _safe_square(x):
    return np.square(np.asarray(x, dtype=np.float64)).astype(np.float32)


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


def write_summary_report(gp_factors: List[dict]):
    if not gp_factors:
        return
    entries = [{'id': f['id'], 'metrics': f} for f in gp_factors]
    passed = [{'id': f['id'], 'metrics': f} for f in gp_factors
              if is_production_ready(f)]
    passed_ids = {e['id'] for e in passed}
    failed = [e for e in entries if e['id'] not in passed_ids]
    report_lines = [
        "=" * 90,
        "GP Baseline 因子挖掘汇总报告",
        "=" * 90,
        f"统一达标标准: |t|≥{Config.PRODUCT_FMT_THRESHOLD}, "
        f"|ICIR|≥{Config.PRODUCT_ICIR_THRESHOLD}, |Rank_IC|≥{Config.PRODUCT_IC_THRESHOLD}, "
        f"截面期数≥30",
        "-" * 90,
        f"[1/2] 达标因子 ({len(passed)} 个)",
        format_fm_table(passed),
        "-" * 90,
        f"[2/2] 未达标因子 ({len(failed)} 个)",
        format_fm_table(failed),
    ]
    report_text = "\n".join(report_lines)
    logger.info(f"\n{report_text}")
    report_path = os.path.join(Config.OUTPUT_DIR, "fm_summary_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    logger.info(f"汇总报告已保存至: {report_path}")


def run_gp_baseline(data_path: str,
                    max_generations: int = 40,
                    population_size: int = 500,
                    tournament_size: int = 20,
                    hall_of_fame_size: int = 15,
                    parsimony_coefficient: float = 0.002,
                    max_samples: float = 0.7,
                    random_state: int = Config.GP_SEED,
                    n_jobs: int = 2,
                    stopping_criteria: float = 4.0,
                    use_fwl: bool = True,
                    ) -> Tuple[List[dict], DataPreprocessor]:

    if not HAS_GPLEARN:
        raise RuntimeError("gplearn 未安装")

    logger.info("=" * 60)
    logger.info(f"GP 基线 | generations={max_generations}, pop={population_size}, "
                f"n_jobs={n_jobs}, stopping={stopping_criteria}")
    logger.info("=" * 60)

    set_global_seed()
    loader = CSI500Loader(path=data_path)
    df_raw = loader.load()

    Config.CLUSTER_FEATURE_POOL = True
    prep = DataPreprocessor()
    prep.prepare_full_pool(df_raw)

    X_train = prep.full_X[prep.tr_mask]
    y_train = prep.full_y[prep.tr_mask]
    train_dates = prep.full_dates[prep.tr_mask]
    X_test = prep.full_X[prep.te_mask]
    y_test = prep.full_y[prep.te_mask]
    test_dates = prep.full_dates[prep.te_mask]
    test_symbols = prep.full_symbols[prep.te_mask]
    test_amount = prep.full_amount[prep.te_mask]

    logger.info(f"训练集: {X_train.shape[0]:,} x {X_train.shape[1]} | 测试集: {X_test.shape[0]:,}")

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
        stopping_criteria=stopping_criteria,
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
        low_memory=True,
    )

    logger.info("\nGP 进化搜索开始...")
    t0 = time.time()
    gp.fit(X_train, y_train)
    elapsed = time.time() - t0
    logger.info(f"GP 完成，耗时 {elapsed:.1f}s")

    final_gen = gp._programs[-1]
    unique_progs = list(dict.fromkeys(final_gen))
    unique_progs = [p for p in unique_progs if p is not None]
    unique_progs.sort(key=lambda p: p.fitness_, reverse=True)
    programs = unique_progs[:hall_of_fame_size]
    logger.info(f"末代 {len(final_gen)} 个, 去重后 {len(unique_progs)} 个, top {len(programs)}")

    registry = FactorRegistry(version="GP_baseline_gplearn")
    gp_factors = []
    for i, prog in enumerate(programs[:hall_of_fame_size]):
        if prog is None:
            continue
        formula_str = str(prog)
        y_pred_test = prog.execute(X_test)

        # ── 统一评估尺子: 与 GFlowAlpha 完全相同的测试集评估流程 (FWL 中性化) ──
        #    搜索适应度保持 GP 最佳实践 (train ICIR, raw); 仅测试集检验走同一把尺.
        if use_fwl:
            fv, rv = fwl_neutralize(y_pred_test, y_test,
                                    test_dates, test_symbols, test_amount)
            ok = np.isfinite(fv) & np.isfinite(rv)
            fm = FamaMacBethRegressor(nw_lags=Config.NW_LAGS)
            fm_result = fm.run(factor_values=fv[ok], returns=rv[ok],
                               dates=test_dates[ok], symbols=test_symbols[ok])
        else:
            fm = FamaMacBethRegressor(nw_lags=Config.NW_LAGS)
            fm_result = fm.run(factor_values=y_pred_test, returns=y_test,
                               dates=test_dates, symbols=test_symbols)

        metrics = fm_result.to_dict()
        metrics['formula'] = formula_str
        metrics['fitness'] = float(prog.fitness_)
        metrics['length'] = int(prog.length_)

        fid = f'GP_{i:03d}'
        gp_factors.append({
            'id': fid, 'formula': formula_str,
            'fitness': float(prog.fitness_), 'length': int(prog.length_),
            'FM_tstat': fm_result.t_stat, 'FM_pvalue': fm_result.p_value,
            'FM_coef': fm_result.coefficient, 'FM_se': fm_result.std_error,
            'Rank_IC': fm_result.rank_ic_mean, 'ICIR': fm_result.rank_icir,
            'LS_spread': fm_result.long_short_spread, 'FM_R2_avg': fm_result.avg_r_squared,
            'FM_n_periods': fm_result.n_periods,
        })

        registry.register(fid, formula_str, metrics)

        sig = '*' * min(3, max(0, int(abs(fm_result.t_stat) // 1.5)))
        logger.info(f"GP_{i:03d} | fitness={prog.fitness_:.4f} | t={fm_result.t_stat:6.3f}{sig} | "
                    f"ICIR={fm_result.rank_icir:.4f} | len={prog.length_:2d}")

    n_prod = sum(1 for f in gp_factors if is_production_ready(f))
    logger.info(f"\nGP 达标因子: {n_prod}/{len(gp_factors)} "
                f"(统一生产标准: |t|≥{Config.PRODUCT_FMT_THRESHOLD}, "
                f"ICIR≥{Config.PRODUCT_ICIR_THRESHOLD}, IC≥{Config.PRODUCT_IC_THRESHOLD})")
    write_summary_report(gp_factors)
    return gp_factors, prep


def main(data_path: str, max_generations: int = 40, population_size: int = 500,
         n_jobs: int = 2, use_fwl: bool = True):
    set_global_seed()
    sys.stdout.reconfigure(line_buffering=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    Config.OUTPUT_DIR = os.path.join("factor_output_academic_v81", f"gp_baseline_{timestamp}")
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
    gp_factors, prep = run_gp_baseline(
        data_path=data_path,
        max_generations=max_generations,
        population_size=population_size,
        tournament_size=20,
        hall_of_fame_size=15,
        parsimony_coefficient=0.002,
        max_samples=0.7,
        random_state=Config.GP_SEED,
        n_jobs=n_jobs,
        stopping_criteria=4.0,
        use_fwl=use_fwl,
    )

    print(f"\n完成！产出物: {os.path.abspath(Config.OUTPUT_DIR)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GP Baseline: gplearn 遗传规划因子挖掘")
    # hs300_daily_2022-06-30_to_2026-06-30.parquet
    parser.add_argument('--data', type=str,
                        default='data/csi500_daily_2021-06-30_to_2026-06-30.parquet')
    parser.add_argument('--generations', type=int, default=40)
    parser.add_argument('--population', type=int, default=500)
    parser.add_argument('--jobs', type=int, default=2,
                        help='并行度 (默认 2; 大值时注意内存)')
    parser.add_argument('--no-fwl', action='store_true',
                        help='跳过 FWL 中性化 (测试集 FM 用 raw 数据, 默认 FWL)')
    args = parser.parse_args()
    main(args.data, max_generations=args.generations,
         population_size=args.population, n_jobs=args.jobs,
         use_fwl=not args.no_fwl)
