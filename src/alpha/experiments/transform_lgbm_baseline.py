#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
学术级日频单因子挖掘系统 — LightGBM 黑盒基线 (基准 #12)

对比对象: transform_pysr_primary.py (GFlowNet + PySR + MLQC), GP 基线

方法学 (与符号搜索完全同口径):
  1. 特征: 与符号搜索完全相同, 使用聚类正交化后的 FINAL_FEATURE_POOL (85 个原子因子),
     同一 6:2:2 划分与标准化 (DataPreprocessor.prepare_full_pool)。
  2. 训练: LightGBM 以 L2 回归预测前向收益, train 集拟合, val 集 early stopping
     (防过拟合, 非质量门禁), random_state 取 Config.LGBM_SEED。
  3. 预测: 同一 test 集 (te_mask) 上模型输出即因子得分。
  4. 评估: test 集 FWL 中性化 (剔除 ln(amount)) 后
     Fama-MacBeth, 报告 Rank_IC / ICIR / FM t-stat 等; 另附原始 IC/ICIR 参照。
  5. 产出: lgbm_config.json / fm_summary_report.txt / feature_importance.csv /
     与 primary 及其它 registry 的对比表 (comparison_*.csv / .md)。
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import logging
import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd

import lightgbm as lgb

from alpha.config import Config, REPO_ROOT, set_global_seed
from alpha.data.data_loader import CSI500Loader
from alpha.mining.preprocessor import DataPreprocessor
from alpha.mining.fm_regression import FamaMacBethRegressor
from alpha.mining.neutralize import fwl_neutralize

logger = logging.getLogger('LGBMBaseline')


def _production_stats(fm_result, n_features: int) -> Dict:
    """Compare-table row stats for a single black-box factor (同 Config 阈值)."""
    t = abs(fm_result.t_stat)
    icir = abs(fm_result.rank_icir)
    ic = abs(fm_result.rank_ic_mean)
    n_prod = int(t >= Config.TSTAT_THRESHOLD and
                 icir >= Config.ICIR_MIN and
                 ic >= Config.IC_THRESHOLD and
                 fm_result.n_periods >= 30)
    return {
        "n_registered": 1,
        "n_production": n_prod,
        "mean_abs_t": round(t, 4),
        "max_abs_t": round(t, 4),
        "mean_abs_icir": round(icir, 4),
        "mean_abs_ic": round(ic, 4),
        "feature_union": n_features,
        "structure_unique": 1,
    }


def _load_registry_stats(path: str, tag: str) -> Dict:
    """Best-effort stats from a standalone registry JSON."""
    if not os.path.exists(path):
        logger.warning(f"对比 registry 不存在: {path}")
        return {"tag": tag, "path": path, "n_registered": 0, "n_production": 0}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    factors = data.get("factors", {})
    t_vals, icir_vals, ic_vals = [], [], []
    feats = set()
    formulas = set()
    n_prod = 0
    for fid, info in factors.items():
        m = info.get("metrics", info)
        t = abs(m.get("FM_tstat", 0))
        icir = abs(m.get("ICIR", 0))
        ic = abs(m.get("Rank_IC", 0))
        n_periods = m.get("FM_n_periods", 0)
        t_vals.append(t)
        icir_vals.append(icir)
        ic_vals.append(ic)
        formulas.add(str(m.get("formula", "")))
        for f in info.get("features", []) or []:
            feats.add(f)
        if (t >= Config.TSTAT_THRESHOLD and icir >= Config.ICIR_MIN
                and ic >= Config.IC_THRESHOLD and n_periods >= 30):
            n_prod += 1
    return {
        "tag": tag,
        "path": path,
        "n_registered": len(factors),
        "n_production": n_prod,
        "mean_abs_t": float(np.mean(t_vals)) if t_vals else np.nan,
        "max_abs_t": float(np.max(t_vals)) if t_vals else np.nan,
        "mean_abs_icir": float(np.mean(icir_vals)) if icir_vals else np.nan,
        "mean_abs_ic": float(np.mean(ic_vals)) if ic_vals else np.nan,
        "feature_union": len(feats),
        "structure_unique": len(formulas),
    }


def write_comparison(rows: List[Dict], output_dir: str):
    cols = ["tag", "path", "n_registered", "n_production",
            "mean_abs_t", "max_abs_t", "mean_abs_icir", "mean_abs_ic",
            "feature_union", "structure_unique"]
    df = pd.DataFrame(rows, columns=cols)
    csv_path = os.path.join(output_dir, "comparison_baseline.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8")
    md_path = os.path.join(output_dir, "comparison_baseline.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "---|" * len(cols) + "\n")
        for _, row in df.iterrows():
            cells = [str(row[c]) for c in cols]
            f.write("| " + " | ".join(cells) + " |\n")
    logger.info(f"\n对比表已保存: {csv_path} / {md_path}")
    print(df.to_string(index=False))


def _build_model(params: Dict, seed: int):
    return lgb.LGBMRegressor(
        objective="regression",
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        num_leaves=params["num_leaves"],
        max_depth=params["max_depth"],
        min_child_samples=params["min_child_samples"],
        subsample=params["subsample"],
        subsample_freq=1,
        colsample_bytree=params["colsample_bytree"],
        n_jobs=params["n_jobs"],
        random_state=seed,
        verbose=-1,
    )


def run_lgbm_baseline(data_path: str,
                      n_estimators: int = 1000,
                      learning_rate: float = 0.05,
                      num_leaves: int = 31,
                      max_depth: int = -1,
                      min_child_samples: int = 20,
                      subsample: float = 0.8,
                      colsample_bytree: float = 1.0,
                      es_rounds: int = 200,
                      n_jobs: int = 1,
                      seeds: int = 1,
                      compare: Optional[List[str]] = None):
    set_global_seed()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    Config.OUTPUT_DIR = os.path.join("factor_output_academic_v81", f"lgbm_baseline_{timestamp}")
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

    feature_names = list(Config.FINAL_FEATURE_POOL)
    logger.info(f"特征池 (与符号搜索相同): {len(feature_names)} 个 | "
                f"总样本: {len(prep.full_X):,}")

    X_tr = prep.full_X[prep.tr_mask]
    y_tr = prep.full_y[prep.tr_mask]
    X_val = prep.full_X[prep.val_mask]
    y_val = prep.full_y[prep.val_mask]
    X_te = prep.full_X[prep.te_mask]

    test_ret = prep.full_y[prep.te_mask]
    test_dates = prep.full_dates[prep.te_mask]
    test_symbols = prep.full_symbols[prep.te_mask]
    test_amount = prep.full_amount[prep.te_mask]

    logger.info(f"LightGBM 训练集: {X_tr.shape} | 验证集(early stopping): {X_val.shape} | "
                f"测试集: {X_te.shape} | seeds={seeds}")

    params = {
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "num_leaves": num_leaves,
        "max_depth": max_depth,
        "min_child_samples": min_child_samples,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "n_jobs": n_jobs,
    }

    fm = FamaMacBethRegressor(nw_lags=Config.NW_LAGS)
    seed_metrics = []
    best_model = None
    best_iters = 0

    for s in range(seeds):
        seed = Config.LGBM_SEED + s
        logger.info(f"\n=== LGBM seed={seed} 训练开始 ===")
        model = _build_model(params, seed)
        t0 = time.time()
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                  eval_metric="l2", callbacks=[lgb.early_stopping(es_rounds, verbose=False)])
        elapsed = time.time() - t0
        iters = model.best_iteration_ or n_estimators
        best_iters = iters
        logger.info(f"seed={seed} 完成 | {elapsed:.1f}s | best_iteration={iters} | "
                    f"val_l2={model.best_score_.get('valid_0', {}).get('l2', float('nan')):.6f}")

        # ---- 预测 test 集 (同一 te_mask) ----
        pred_te = model.predict(X_te)
        pred_te = np.where(np.isfinite(pred_te), pred_te, np.nan)

        # ---- 原始 FM (未中性化, 透明参照) ----
        fm_raw = fm.run(pred_te, test_ret, test_dates, test_symbols)

        # ---- FWL 中性化 + FM (统一评估通道) ----
        pure_pred, pure_ret = fwl_neutralize(pred_te, test_ret, test_dates,
                                             test_symbols, test_amount)
        ok = np.isfinite(pure_pred) & np.isfinite(pure_ret)
        fm_test = fm.run(pure_pred[ok], pure_ret[ok],
                         test_dates[ok], test_symbols[ok])

        fid = f"LGBM_{s:03d}"
        fm.print_report(fm_test, factor_name=f"{fid} (Test集样本外, FWL+FM)")
        logger.info(f"{fid} | raw_IC={fm_raw.rank_ic_mean:.4f} raw_ICIR={fm_raw.rank_icir:.4f} | "
                    f"FWL IC={fm_test.rank_ic_mean:.4f} ICIR={fm_test.rank_icir:.4f} "
                    f"t={fm_test.t_stat:.3f}")

        seed_metrics.append({
            "seed": seed, "fid": fid, "iterations": iters,
            "t_stat": fm_test.t_stat, "p_value": fm_test.p_value,
            "coef": fm_test.coefficient, "se": fm_test.std_error,
            "Rank_IC": fm_test.rank_ic_mean, "ICIR": fm_test.rank_icir,
            "LS_spread": fm_test.long_short_spread, "R2_avg": fm_test.avg_r_squared,
            "n_periods": fm_test.n_periods,
            "raw_Rank_IC": fm_raw.rank_ic_mean, "raw_ICIR": fm_raw.rank_icir,
            "quantile_returns": fm_test.quantile_returns,
        })
        if best_model is None:
            best_model = model
            fm_main = fm_test
            raw_main = fm_raw

    metrics_df = pd.DataFrame([{k: v for k, v in m.items() if k != "quantile_returns"}
                               for m in seed_metrics])

    # === 汇总报告 ===
    report_lines = [
        "=" * 90,
        "LightGBM 黑盒基线因子挖掘汇总报告 (Test集样本外, FWL+FM)",
        "=" * 90,
        f"特征池: {len(feature_names)} 个原子因子 (FINAL_FEATURE_POOL, 与符号搜索相同)",
        f"模型: LGBMRegressor | params={json.dumps(params, ensure_ascii=False)} | "
        f"best_iteration={best_iters} | seeds={seeds}",
        "-" * 90,
        f"{'ID':<12} {'t-stat':>8} {'p-value':>10} {'Coef':>10} {'NW-SE':>10} "
        f"{'Rank_IC':>8} {'ICIR':>8} {'L-S':>10} {'R²':>8} {'原始IC':>8}",
        "-" * 90,
    ]
    for m in seed_metrics:
        sig = "***" if abs(m['t_stat']) >= 3.0 else ("**" if abs(m['t_stat']) >= 2.0 else "")
        report_lines.append(
            f"{m['fid']:<12} {m['t_stat']:>7.3f}{sig:<3} {m['p_value']:>10.6f} "
            f"{m['coef']:>10.6f} {m['se']:>10.6f} {m['Rank_IC']:>8.4f} "
            f"{m['ICIR']:>8.4f} {m['LS_spread']:>10.6f} {m['R2_avg']:>8.4f} "
            f"{m['raw_Rank_IC']:>8.4f}"
        )
    if seeds > 1:
        report_lines.append("-" * 90)
        report_lines.append("多种子稳健性 (mean ± std):")
        for col, name in [("t_stat", "FM t-stat"), ("Rank_IC", "Rank_IC"), ("ICIR", "ICIR"),
                          ("coef", "FM coef"), ("LS_spread", "L-S")]:
            vals = metrics_df[col].astype(float)
            report_lines.append(f"  {name:<12}: {vals.mean():>10.4f} ± {vals.std(ddof=1):>10.4f}")
        qr = np.array([np.asarray(m["quantile_returns"], dtype=float) for m in seed_metrics if m["quantile_returns"]])
        if qr.size:
            report_lines.append(f"  分位收益(mean): " + " → ".join(f"{v:.4f}" for v in qr.mean(axis=0)))
    report_text = "\n".join(report_lines)
    logger.info(f"\n{report_text}")
    report_path = os.path.join(Config.OUTPUT_DIR, "fm_summary_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    logger.info(f"报告已保存至: {report_path}")

    # === 特征重要性 ===
    imp = np.asarray(best_model.feature_importances_, dtype=float)
    imp_df = pd.DataFrame({"feature": feature_names, "gain_importance": imp}).sort_values(
        "gain_importance", ascending=False).reset_index(drop=True)
    imp_path = os.path.join(Config.OUTPUT_DIR, "feature_importance.csv")
    imp_df.to_csv(imp_path, index=False, encoding="utf-8")
    logger.info(f"特征重要性已保存: {imp_path} | top5: "
                + ", ".join(imp_df['feature'].head(5).tolist()))

    # === 对比表 ===
    main_stats = _production_stats(fm_main, len(feature_names))
    main_stats["path"] = Config.OUTPUT_DIR
    rows = [{"tag": "LGBM+全部原子因子 (黑盒)", **main_stats}]
    for tag, path in (compare or []):
        rows.append(_load_registry_stats(path, tag))
    write_comparison(rows, Config.OUTPUT_DIR)

    # === 保存配置与指标 ===
    cfg_path = os.path.join(Config.OUTPUT_DIR, "lgbm_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump({
            "data": data_path,
            "feature_pool_size": len(feature_names),
            "feature_pool": feature_names,
            "params": params,
            "early_stopping_rounds": es_rounds,
            "best_iteration": best_iters,
            "seeds": seeds,
            "seed_base": Config.LGBM_SEED,
            "train_rows": int(X_tr.shape[0]),
            "val_rows": int(X_val.shape[0]),
            "test_rows": int(X_te.shape[0]),
            "metrics_seed0": {k: v for k, v in seed_metrics[0].items() if k != "quantile_returns"},
            "aggregate": {
                "t_stat_mean": float(metrics_df['t_stat'].astype(float).mean()),
                "t_stat_std": float(metrics_df['t_stat'].astype(float).std(ddof=1)),
                "Rank_IC_mean": float(metrics_df['Rank_IC'].astype(float).mean()),
                "ICIR_mean": float(metrics_df['ICIR'].astype(float).mean()),
            } if seeds > 1 else None,
        }, f, indent=2, ensure_ascii=False)

    print(f"\nLightGBM 基线完成！产出物: {os.path.abspath(Config.OUTPUT_DIR)}")
    return metrics_df, seed_metrics, prep


def console_main(argv=None):
    parser = argparse.ArgumentParser(description="LightGBM 黑盒基线 (85 原子因子, 与符号搜索同特征/同测试集, FWL+FM 评估)")
    parser.add_argument('--data', type=str,
                        default=os.path.join(REPO_ROOT, 'data', 'csi500_daily_2021-06-30_to_2026-06-30.parquet'))
    parser.add_argument('--estimators', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--leaves', type=int, default=31)
    parser.add_argument('--depth', type=int, default=-1)
    parser.add_argument('--min-child-samples', type=int, default=20)
    parser.add_argument('--subsample', type=float, default=0.8)
    parser.add_argument('--colsample', type=float, default=1.0)
    parser.add_argument('--es-rounds', type=int, default=200)
    parser.add_argument('--n-jobs', type=int, default=1)
    parser.add_argument('--seeds', type=int, default=1, help='独立随机种子数 (第 i 个 = LGBM_SEED+i)')
    parser.add_argument('--market', type=str, default='zs500', choices=['zs500', 'hs300'])
    parser.add_argument('--pool', type=str, default='buildin', choices=['buildin', 'alpha158'])
    parser.add_argument('--compare', type=str, default=None,
                        help='逗号分隔 tag:path 列表, 与其它方法 registry 对比')
    args = parser.parse_args(argv)

    Config.MARKET = args.market
    Config.FEATURE_POOL = args.pool
    compare = None
    if args.compare:
        compare = []
        for item in args.compare.split(','):
            if ':' in item:
                tag, path = item.split(':', 1)
            else:
                tag, path = os.path.basename(item), item
            compare.append((tag, path))

    run_lgbm_baseline(
        data_path=args.data,
        n_estimators=args.estimators,
        learning_rate=args.lr,
        num_leaves=args.leaves,
        max_depth=args.depth,
        min_child_samples=args.min_child_samples,
        subsample=args.subsample,
        colsample_bytree=args.colsample,
        es_rounds=args.es_rounds,
        n_jobs=args.n_jobs,
        seeds=args.seeds,
        compare=compare,
    )


if __name__ == "__main__":
    console_main()
