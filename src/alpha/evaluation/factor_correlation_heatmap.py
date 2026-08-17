#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子相关性热力图绘制工具。

读取 registry_academic.json，解析完整因子及其依赖的原子特征，
使用 Spearman 秩相关系数计算相关矩阵，输出热力图和关键指标。

用法:
  python factor_correlation_heatmap.py --registry path/to/registry_academic.json
  python factor_correlation_heatmap.py --registry ... --data path/to/data.parquet
"""
import argparse
import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

from alpha.config import REPO_ROOT, Config, set_global_seed
from alpha.data.data_loader import CSI500Loader
from alpha.mining.preprocessor import DataPreprocessor
from alpha.mining.safe_ops import SafeOps

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger('FactorCorrelation')

SAFE_MATH = {
    'abs': np.abs, 'square': np.square, 'sqrt': SafeOps.safe_sqrt,
    'sign': np.sign, 'min': np.minimum, 'max': np.maximum,
    'tanh': SafeOps.safe_tanh, 'log1p': SafeOps.safe_log1p,
    'inv': SafeOps.safe_inv, 'sigmoid': SafeOps.safe_sigmoid,
    'div': SafeOps.protected_div, '/': SafeOps.protected_div,
    'cube': lambda x: np.power(x, 3), 'exp': SafeOps.safe_exp,
}


def load_registry(path: str) -> Tuple[Dict[str, str], List[str]]:
    """解析 registry JSON，返回 {fid: formula} 和所有原子特征去重列表."""
    with open(path) as f:
        registry = json.load(f)
    factors = registry.get('factors', {})
    if not factors:
        logger.error("注册表中没有因子")
        sys.exit(1)
    formulas: Dict[str, str] = {}
    all_feats: set = set()
    for fid, info in factors.items():
        m = info.get('metrics', info)
        formula = m.get('formula', '')
        feats = m.get('features', info.get('features', []))
        if formula:
            formulas[fid] = formula
            all_feats.update(feats)
    logger.info(f"解析到 {len(formulas)} 个因子, {len(all_feats)} 个原子特征")
    return formulas, sorted(all_feats)


def compute_factor_values(prep: DataPreprocessor,
                          formulas: Dict[str, str],
                          all_feats: List[str],
                          ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """计算所有因子 pred 和原子特征值（仅在测试集上）."""
    te_mask = prep.te_mask
    n_test = te_mask.sum()
    logger.info(f"测试集样本数: {n_test}")

    te_dates = prep.full_dates[te_mask]
    te_symbols = prep.full_symbols[te_mask]

    # 提取原子特征
    feat_to_idx = prep.feature_to_idx
    atomic_data = {}
    missing = []
    for feat in all_feats:
        if feat in feat_to_idx:
            col = prep.full_X[te_mask, feat_to_idx[feat]]
            atomic_data[feat] = col
        else:
            missing.append(feat)
    if missing:
        logger.warning(f"原子特征缺失（跳过）: {missing}")
    atomic_df = pd.DataFrame(atomic_data)
    atomic_df.index = pd.MultiIndex.from_arrays([te_dates, te_symbols],
                                                names=['date', 'symbol'])
    logger.info(f"原子特征矩阵: {atomic_df.shape}")

    # 计算因子 pred
    pred_dict: Dict[str, np.ndarray] = {}
    for fid, formula in formulas.items():
        used_feats = [f for f in all_feats if f in formula]
        ns = {}
        for feat in used_feats:
            if feat in feat_to_idx:
                ns[feat] = prep.full_X[te_mask, feat_to_idx[feat]]
            else:
                ns[feat] = np.full(n_test, np.nan)
        ns.update(SAFE_MATH)
        try:
            pred = eval(formula, {"__builtins__": {}}, ns)
            pred = np.where(np.isfinite(pred), pred, np.nan)
        except Exception as e:
            logger.warning(f"{fid} eval 失败: {e}")
            pred = np.full(n_test, np.nan)
        pred_dict[fid] = pred

    pred_df = pd.DataFrame(pred_dict)
    pred_df.index = pd.MultiIndex.from_arrays([te_dates, te_symbols],
                                              names=['date', 'symbol'])
    logger.info(f"因子预测矩阵: {pred_df.shape}")
    return pred_df, atomic_df


def compute_correlation_matrices(pred_df: pd.DataFrame,
                                 atomic_df: pd.DataFrame,
                                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算 Spearman 秩相关矩阵.

    Returns:
        full_corr: 全矩阵 (N+M) × (N+M)
        factor_corr: 因子×因子子矩阵
        factor_atomic_corr: 因子×原子子矩阵
    """
    combined = pd.concat([pred_df, atomic_df], axis=1)
    combined = combined.dropna(how='all')
    n_factors = pred_df.shape[1]

    full_corr, _ = spearmanr(combined, axis=0, nan_policy='omit')
    full_corr = np.nan_to_num(full_corr, nan=0.0)

    factor_corr = full_corr[:n_factors, :n_factors]
    factor_atomic_corr = full_corr[:n_factors, n_factors:]

    return full_corr, factor_corr, factor_atomic_corr


def plot_heatmap(corr_matrix: np.ndarray,
                 row_labels: List[str],
                 col_labels: List[str],
                 title: str,
                 file_prefix: str,
                 figsize: Tuple[int, int] = (12, 10),
                 annot: bool = True):
    """绘制相关性热力图并保存."""
    fig, ax = plt.subplots(figsize=figsize)
    square = (len(corr_matrix) == len(row_labels) == len(col_labels))
    mask = np.eye(len(corr_matrix), dtype=bool) if square else None

    cmap = sns.diverging_palette(240, 10, as_cmap=True)

    sns.heatmap(corr_matrix, xticklabels=col_labels, yticklabels=row_labels,
                cmap=cmap, center=0, vmin=-1, vmax=1,
                annot=annot, fmt='.2f', linewidths=0.3,
                square=square, mask=mask,
                cbar_kws={'shrink': 0.8, 'label': 'Spearman ρ'})

    ax.set_title(title, fontsize=14, fontweight='bold', pad=16)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    for ext in ['pdf', 'png']:
        fp = f'{file_prefix}.{ext}'
        fig.savefig(fp, dpi=300, bbox_inches='tight')
        logger.info(f"保存: {os.path.abspath(fp)}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='因子相关性热力图绘制')
    parser.add_argument('--registry', type=str,
                        default=os.path.join(REPO_ROOT, 'factor_output_academic_v81',
                                             'ablation9_random_no_mlqc_20260807_082338', 'registry_academic.json'),
                        help='注册表 JSON 路径')
    parser.add_argument('--data', type=str,
                        default=os.path.join(REPO_ROOT, 'data',
                                             'csi500_daily_2021-06-30_to_2026-06-30.parquet'))
    parser.add_argument('--output', type=str, default=None,
                        help='输出目录；默认自动创建带时间戳的运行目录')
    parser.add_argument('--seed', type=int, default=Config.SEED)
    args = parser.parse_args()

    set_global_seed(args.seed)

    if args.output is None:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output = os.path.join(REPO_ROOT, 'outputs',
                                   f'factor_correlation_{ts}')
    os.makedirs(args.output, exist_ok=True)

    # ① 解析 registry
    formulas, all_feats = load_registry(args.registry)
    factor_ids = list(formulas.keys())

    if len(factor_ids) < 2:
        logger.error("因子数量不足 2 个，无法计算相关性")
        sys.exit(1)

    # ② 加载数据
    logger.info("加载数据...")
    loader = CSI500Loader(path=args.data)
    df = loader.load()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)

    prev = Config.TARGET_FACTOR_POOL_SIZE
    Config.TARGET_FACTOR_POOL_SIZE = 999
    prep = DataPreprocessor()
    prep.prepare_full_pool(df)
    Config.TARGET_FACTOR_POOL_SIZE = prev

    # ③ 计算因子值
    pred_df, atomic_df = compute_factor_values(prep, formulas, all_feats)

    # ④ Spearman 相关矩阵
    full_corr, factor_corr, factor_atomic_corr = compute_correlation_matrices(
        pred_df, atomic_df)

    # ⑤ 打印关键指标
    n_f = len(factor_ids)
    np.fill_diagonal(factor_corr, np.nan)
    max_off_diag = np.nanmax(np.abs(factor_corr))
    mean_off_diag = np.nanmean(np.abs(factor_corr))

    print("\n" + "=" * 60)
    print("  因子相关性分析报告 (Spearman 秩相关)")
    print("=" * 60)
    print(f"\n  因子数量: {n_f}")
    print(f"  原子特征数量: {len(all_feats)}")
    print(f"  测试集样本量: {len(pred_df)}")

    print(f"\n  ┌─ 因子间相关性 (Factor×Factor) ──────────────────────")
    print(f"  │ 最大非对角 |ρ|: {max_off_diag:.4f}")
    print(f"  │ 平均非对角 |ρ|: {mean_off_diag:.4f}")
    print(f"  └────────────────────────────────────────────────")

    # 找最大相关对
    if n_f >= 2:
        triu_idx = np.triu_indices(n_f, k=1)
        flat_corr = factor_corr[triu_idx]
        flat_labels = [f"{factor_ids[i]}×{factor_ids[j]}"
                       for i, j in zip(triu_idx[0], triu_idx[1])]
        if len(flat_corr) > 0:
            max_idx = np.nanargmax(np.abs(flat_corr))
            print(f"  最相关对: {flat_labels[max_idx]} = {flat_corr[max_idx]:.4f}")

    factor_atomic_abs = np.abs(factor_atomic_corr)
    print(f"\n  ┌─ 因子×原子特征相关性 (Factor×Atomic) ───────────────")
    print(f"  │ 平均 |ρ|: {np.nanmean(factor_atomic_abs):.4f}")
    print(f"  │ 最大 |ρ|: {np.nanmax(factor_atomic_abs):.4f}")
    print(f"  └────────────────────────────────────────────────")

    # ⑥ 绘制热力图
    out_prefix = os.path.join(args.output, 'factor_correlation_heatmap')

    # 因子×因子
    plot_heatmap(
        corr_matrix=full_corr[:n_f, :n_f],
        row_labels=factor_ids,
        col_labels=factor_ids,
        title='Factor × Factor — Spearman Rank Correlation',
        file_prefix=out_prefix + '_factors',
    )

    # 因子×原子
    plot_heatmap(
        corr_matrix=full_corr[:n_f, n_f:],
        row_labels=factor_ids,
        col_labels=list(atomic_df.columns),
        title='Factor × Atomic Feature — Spearman Rank Correlation',
        file_prefix=out_prefix + '_factor_atomic',
        figsize=(14, 8),
    )

    # 全矩阵大图
    all_labels = factor_ids + [f'A:{c}' for c in atomic_df.columns]
    plot_heatmap(
        corr_matrix=full_corr,
        row_labels=all_labels,
        col_labels=all_labels,
        title='Full Matrix — Spearman Rank Correlation',
        file_prefix=out_prefix + '_full',
        figsize=(max(14, len(all_labels) * 0.4),
                 max(10, len(all_labels) * 0.4)),
        annot=len(all_labels) <= 30,
    )

    print(f"\n  输出目录: {os.path.abspath(args.output)}")
    print("=" * 60)


if __name__ == '__main__':
    main()
