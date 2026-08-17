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
