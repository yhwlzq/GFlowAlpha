import numpy as np
import pandas as pd
from typing import Callable

class SafeOps:
    @staticmethod
    def protected_div(left, right, default=0.0):
        left, right = np.asarray(left, dtype=np.float32), np.asarray(right, dtype=np.float32)
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            return np.where(np.abs(right) >= 1e-8, left / right, np.float32(default))

    @staticmethod
    def safe_sigmoid(x):
        x = np.clip(np.asarray(x, dtype=np.float32), -500, 500)
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def safe_log1p(x):
        x = np.asarray(x, dtype=np.float32)
        return np.log1p(np.clip(x, -0.999, None))

    @staticmethod
    def safe_inv(x):
        x = np.asarray(x, dtype=np.float32)
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            return np.where(np.abs(x) > 1e-8, 1.0 / x, 0.0)

    @staticmethod
    def safe_cube(x):
        x = np.clip(np.asarray(x, dtype=np.float32), -10, 10)
        return np.power(x, 3)

    @staticmethod
    def safe_exp(x):
        return np.exp(np.clip(np.asarray(x, dtype=np.float32), -20, 20))

    @staticmethod
    def safe_sqrt(x):
        x = np.asarray(x, dtype=np.float32)
        return np.sqrt(np.maximum(x, 0.0))

    @staticmethod
    def safe_tanh(x):
        return np.tanh(np.clip(np.asarray(x, dtype=np.float32), -10, 10))

    @staticmethod
    def rolling_ols_intercept(y, x, window):
        """向量化滚动 OLS 截距项 — cumsum + 正规方程，无 apply。"""
        y = np.asarray(y, dtype=np.float64)
        x = np.asarray(x, dtype=np.float64)
        n = len(y)
        if n < window:
            return np.full(n, np.nan, dtype=np.float32)
        cs_y = np.cumsum(y)
        cs_x = np.cumsum(x)
        cs_xy = np.cumsum(x * y)
        cs_xx = np.cumsum(x * x)
        sum_y = cs_y[window - 1:] - np.concatenate(([0.0], cs_y[:n - window]))
        sum_x = cs_x[window - 1:] - np.concatenate(([0.0], cs_x[:n - window]))
        sum_xy = cs_xy[window - 1:] - np.concatenate(([0.0], cs_xy[:n - window]))
        sum_xx = cs_xx[window - 1:] - np.concatenate(([0.0], cs_xx[:n - window]))
        w = np.float64(window)
        denom = w * sum_xx - sum_x * sum_x
        slope = np.where(np.abs(denom) > 1e-16, (w * sum_xy - sum_x * sum_y) / denom, 0.0)
        intercept = (sum_y - slope * sum_x) / w
        result = np.full(n, np.nan, dtype=np.float64)
        result[window - 1:] = intercept
        return result.astype(np.float32)

    @staticmethod
    def rolling_ols_slope(y, x, window):
        """向量化滚动 OLS 斜率项 (Beta) — cumsum + 正规方程，无 apply。"""
        y = np.asarray(y, dtype=np.float64)
        x = np.asarray(x, dtype=np.float64)
        n = len(y)
        if n < window:
            return np.full(n, np.nan, dtype=np.float32)
        cs_y = np.cumsum(y)
        cs_x = np.cumsum(x)
        cs_xy = np.cumsum(x * y)
        cs_xx = np.cumsum(x * x)
        sum_y = cs_y[window - 1:] - np.concatenate(([0.0], cs_y[:n - window]))
        sum_x = cs_x[window - 1:] - np.concatenate(([0.0], cs_x[:n - window]))
        sum_xy = cs_xy[window - 1:] - np.concatenate(([0.0], cs_xy[:n - window]))
        sum_xx = cs_xx[window - 1:] - np.concatenate(([0.0], cs_xx[:n - window]))
        w = np.float64(window)
        denom = w * sum_xx - sum_x * sum_x
        slope = np.where(np.abs(denom) > 1e-16, (w * sum_xy - sum_x * sum_y) / denom, 0.0)
        result = np.full(n, np.nan, dtype=np.float64)
        result[window - 1:] = slope
        return result.astype(np.float32)

    @staticmethod
    def rolling_ols_residual(y, x, window):
        """向量化滚动 OLS 残差: y - beta*x - alpha。"""
        slope = SafeOps.rolling_ols_slope(y, x, window)
        intercept = SafeOps.rolling_ols_intercept(y, x, window)
        y = np.asarray(y, dtype=np.float64)
        x = np.asarray(x, dtype=np.float64)
        predicted = slope * x + intercept
        residual = y - predicted
        return residual.astype(np.float32)

    @staticmethod
    def safe_obv(close, volume):
        """OBV (On Balance Volume) — 向量化实现。"""
        close = np.asarray(close, dtype=np.float64)
        volume = np.asarray(volume, dtype=np.float64)
        direction = np.sign(np.diff(close, prepend=close[0]))
        return np.cumsum(direction * volume).astype(np.float32)

    @staticmethod
    def safe_rsrs(high, low, window=20):
        """RSRS 阻力支撑相对强度 — OLS(H~L).slope × R²，向量化。"""
        slope = SafeOps.rolling_ols_slope(high, low, window)
        high_s = np.asarray(high, dtype=np.float64)
        low_s = np.asarray(low, dtype=np.float64)
        corr = pd.Series(high_s).rolling(window).corr(pd.Series(low_s))
        r2 = (corr ** 2).fillna(0.0).values
        return (slope * r2).astype(np.float32)

    @staticmethod
    def safe_mfi(high, low, close, volume, window=14):
        """MFI 资金流量指标 — 向量化。"""
        high = pd.Series(np.asarray(high, dtype=np.float64))
        low = pd.Series(np.asarray(low, dtype=np.float64))
        close = pd.Series(np.asarray(close, dtype=np.float64))
        volume = pd.Series(np.asarray(volume, dtype=np.float64))
        tp = (high + low + close) / 3.0
        mf = tp * volume
        tp_diff = tp.diff()
        pos_mf = mf.where(tp_diff > 0, 0.0)
        neg_mf = mf.where(tp_diff < 0, 0.0)
        pos_sum = pos_mf.rolling(window, min_periods=1).sum()
        neg_sum = neg_mf.rolling(window, min_periods=1).sum().clip(lower=1e-8)
        mfr = pos_sum / neg_sum
        return (100.0 - 100.0 / (1.0 + mfr)).astype(np.float32).values


class TimeSeriesOps:
    def __init__(self, date_col: np.ndarray, symbol_col: np.ndarray):
        self.date_col, self.symbol_col = np.asarray(date_col), np.asarray(symbol_col)
        self.unique_symbols = np.unique(self.symbol_col)
        self.symbol_slices = {sym: np.where(self.symbol_col == sym)[0] for sym in self.unique_symbols}

    def _apply_per_symbol(self, x: np.ndarray, func: Callable, **kwargs) -> np.ndarray:
        result = np.zeros(len(x), dtype=np.float32)
        for indices in self.symbol_slices.values():
            if len(indices) < 2:
                continue
            sym_res = func(pd.Series(x[indices]), **kwargs)
            result[indices] = sym_res.values if hasattr(sym_res, 'values') else np.asarray(sym_res, dtype=np.float32)
        return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def ts_rank(self, x, window=10):
        return self._apply_per_symbol(x, lambda s: s.rolling(window, min_periods=2).apply(
            lambda arr: pd.Series(arr).rank(pct=True).iloc[-1], raw=True).fillna(0.5))

    def ts_delta(self, x, period=5):
        return self._apply_per_symbol(x, lambda s: s.diff(period).fillna(0))
