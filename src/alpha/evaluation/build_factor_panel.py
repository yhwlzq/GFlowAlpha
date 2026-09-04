#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
registry_academic.json 中的因子公式 → 全样本(每股×每日)因子值面板

用于打通: registry(公式) → factor_panel.parquet → factor_to_ls_returns.py → orthogonality_check.py
与主实验同源评估: 复用 DataPreprocessor + 标准化特征 + safe_math eval,
保证公式里的常量系数与挖矿时口径一致.

输出列: date, symbol, ret, <fid1>, <fid2>, ...
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import json
import logging
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.tools import add_constant

from alpha.config import Config, apply_split_mode
from alpha.data.data_loader import CSI500Loader
from alpha.mining.preprocessor import DataPreprocessor

logger = logging.getLogger("BuildFactorPanel")

SAFE_MATH = {
    'abs': np.abs, 'square': np.square, 'sqrt': np.sqrt,
    'sign': np.sign, 'min': np.minimum, 'max': np.maximum,
    'tanh': np.tanh,
    'log1p': lambda x: np.log1p(np.abs(x)) * np.sign(x),
    'inv': lambda x: np.where(np.abs(x) > 1e-8, 1.0 / x, 0.0),
    'sigmoid': lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500))),
}


def eval_registry_to_panel(prep: DataPreprocessor, registry: dict) -> pd.DataFrame:
    factors = registry.get('factors', {})
    if not factors:
        raise ValueError("registry 中无 factors")

    meta = pd.DataFrame({
        'date': pd.to_datetime(prep.full_dates),
        'symbol': prep.full_symbols,
    })
    if prep.full_y is not None:
        ret_base = prep.full_y
    elif prep.full_y_raw is not None:
        ret_base = prep.full_y_raw
    else:
        ret_base = np.zeros(len(prep.full_dates), dtype=np.float64)
    panel = meta
    panel['ret'] = pd.to_numeric(np.asarray(ret_base, dtype=np.float64), errors='coerce')

    for fid, info in factors.items():
        m = info.get('metrics', info)
        formula = m.get('formula', '')
        feats = m.get('features', info.get('features', []))
        if not formula or not feats:
            logger.warning(f"{fid} 无公式/特征，跳过")
            continue

        feats = [f for f in feats if f in prep.feature_to_idx]
        if len(feats) < len(m.get('features', info.get('features', []))):
            logger.warning(f"{fid} 部分特征缺失，使用 {feats}")

        X_sub = np.asarray(prep.full_X[:, [prep.feature_to_idx[f] for f in feats]], dtype=np.float64)
        ns = {f: X_sub[:, i] for i, f in enumerate(feats)}
        ns.update(SAFE_MATH)

        try:
            pred = eval(formula, {"__builtins__": {}}, ns)
            pred = np.where(np.isfinite(pred), pred, np.nan)
        except Exception as e:
            logger.error(f"{fid} eval 失败: {e}")
            pred = np.full(len(prep.full_dates), np.nan)

        panel[fid] = pred

    return panel


def neutralize_factor_by_amount(panel: pd.DataFrame, amount: np.ndarray,
                                turn: np.ndarray | None = None,
                                min_cross: int = 30) -> pd.DataFrame:
    """逐截面 FWL 中性化: 每个交易日将各因子列对 1 + ln(amount) (+turn) 回归取残差.

    与挖矿门禁同口径: Config.NEUTRALIZE_CONTROLS 决定控制变量集,
    用于正交性检验前剥离因子中的市值暴露, 使 L-S 收益测试的是"中性化后的纯信号".
    若 amount 全为 0 (数据无成交额) 则原样返回.
    """
    if amount is None or np.all(np.asarray(amount, dtype=np.float64) == 0):
        logger.warning("无成交额数据 (amount 全 0), 跳过中性化")
        return panel

    factor_cols = [c for c in panel.columns if c not in ('date', 'symbol', 'ret')]
    if not factor_cols:
        return panel

    ln_amt = np.log(np.maximum(np.asarray(amount, dtype=np.float64), 1.0))
    use_turn = turn is not None and not np.all(np.asarray(turn, dtype=np.float64) == 0)
    if use_turn:
        turn_val = np.asarray(turn, dtype=np.float64)
    resid = {c: np.full(len(panel), np.nan, dtype=np.float64) for c in factor_cols}
    date_idx = pd.Series(pd.to_datetime(panel['date']).values, index=panel.index)

    for dt, idx in date_idx.groupby(date_idx).groups.items():
        idx = pd.Index(idx)
        if len(idx) < min_cross:
            for c in factor_cols:
                resid[c][idx] = panel.loc[idx, c].to_numpy(dtype=np.float64)
            continue
        cols = [ln_amt[idx]]
        if use_turn:
            cols.append(turn_val[idx])
        X = add_constant(np.column_stack(cols))
        for c in factor_cols:
            y = panel.loc[idx, c].to_numpy(dtype=np.float64)
            m = sm.OLS(y, X).fit()
            resid[c][idx] = m.resid

    for c in factor_cols:
        panel[c] = resid[c]
    ctrl = '1+ln(amount)+turn' if use_turn else '1+ln(amount)'
    logger.info(f"FWL 中性化完成: {len(factor_cols)} 个因子列 (逐截面 {ctrl})")
    return panel


def main():
    parser = argparse.ArgumentParser(description="registry 公式 → 全样本因子值面板")
    parser.add_argument('--data', type=str, required=True, help='日频 parquet/csv 面板')
    parser.add_argument('--registry', type=str, required=True, help='registry_academic.json 路径')
    parser.add_argument('--fids', nargs='*', default=None,
                        help='仅提取指定因子 id (默认全部)')
    parser.add_argument('--market', type=str, default='zs500', choices=['zs500', 'hs300'])
    parser.add_argument('--output', default='factor_panel.parquet', help='输出 parquet 路径')
    parser.add_argument('--neutralize', action='store_true',
                        help='逐截面 FWL ln(amount) 市值中性化因子值后再输出 (旧口径)')
    parser.add_argument('--neutralize-multi', action='store_true',
                        help='按 Config.NEUTRALIZE_CONTROLS 做横截面中性化 (默认仅 amount, 可改为 amount+turn)')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-7s | %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    with open(args.registry, 'r', encoding='utf-8') as f:
        registry = json.load(f)
    if args.fids:
        registry['factors'] = {k: v for k, v in registry['factors'].items() if k in set(args.fids)}

    Config.MARKET = args.market
    Config.TARGET_FACTOR_POOL_SIZE = 999
    apply_split_mode('warmup')

    loader = CSI500Loader(path=args.data)
    df = loader.load()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)

    prep = DataPreprocessor()
    prep.prepare_full_pool(df)

    panel = eval_registry_to_panel(prep, registry)
    if args.neutralize_multi:
        use_turn = 'turn' in Config.NEUTRALIZE_CONTROLS
        panel = neutralize_factor_by_amount(
            panel, prep.full_amount,
            turn=prep.full_turn if use_turn else None)
    elif args.neutralize:
        panel = neutralize_factor_by_amount(panel, prep.full_amount)
    out_cols = ['date', 'symbol', 'ret'] + [k for k in registry['factors'].keys() if k in panel.columns]
    panel = panel[out_cols]
    panel = panel.dropna(subset=['ret']).sort_values(['symbol', 'date']).reset_index(drop=True)

    panel.to_parquet(args.output, index=False)
    logger.info(f"已保存: {args.output} ({len(panel):,} 行, {len(registry['factors'])} 个因子)")
    logger.info(f"列: {list(panel.columns)}")
    logger.info(f"日期范围: {panel['date'].min()} ~ {panel['date'].max()}")


if __name__ == "__main__":
    main()