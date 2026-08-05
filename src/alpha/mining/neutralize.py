import numpy as np
import pandas as pd
from statsmodels.api import add_constant, OLS
from typing import Tuple


def fwl_neutralize(
    factor_vals: np.ndarray, ret: np.ndarray,
    dates: np.ndarray, symbols: np.ndarray, amount: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """FWL 双残差中性化: 截面回归剔除 ln(amount) 暴露.

    对每个交易日独立执行:
      1. factor ~ 1 + ln(amount)  → resid = pure factor
      2. ret    ~ 1 + ln(amount)  → resid = pure return

    Args:
        factor_vals: 原始因子值 (N,)
        ret:  forward 收益率 (N,)
        dates:  日期标签 (N,)
        symbols: 股票代码 (N,) — 当前未使用, 保留供未来扩展 (可传空数组)
        amount:  成交额, size proxy (N,)

    Returns:
        pure_factor:  残差因子值 (N,)
        pure_ret:     残差收益率 (N,)
    """
    if amount is None or np.all(amount == 0):
        return factor_vals.copy(), ret.copy()

    ln_amt = np.log(np.maximum(amount, 1.0))
    df_n = pd.DataFrame({
        'date': pd.to_datetime(dates).normalize(),
        'factor': factor_vals, 'ret': ret, 'ln_amt': ln_amt,
    }).dropna()

    pure_f = np.full(len(df_n), np.nan, dtype=np.float64)
    pure_r = np.full(len(df_n), np.nan, dtype=np.float64)

    for dt, grp in df_n.groupby('date'):
        idx = grp.index
        if len(grp) < 30:
            pure_f[idx] = grp['factor'].values
            pure_r[idx] = grp['ret'].values
            continue
        X = add_constant(grp['ln_amt'].values)
        try:
            pure_f[idx] = OLS(grp['factor'].values, X).fit().resid
            pure_r[idx] = OLS(grp['ret'].values, X).fit().resid
        except Exception:
            pure_f[idx] = grp['factor'].values
            pure_r[idx] = grp['ret'].values

    return pure_f, pure_r
