import logging
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from alpha.config import Config

logger = logging.getLogger(__name__)


class WalkForwardSplitter:
    """18:6:3 月度滚动 walk-forward 分割器.

    每个窗口: train = 18 月, val = 6 月, test = 3 月 (季度, 生产调仓周期).
    test 窗以 3 月为步长不重叠滚动. 边界施加 purge(1日) + embargo(2交易日)
    防止 HORIZON=1 的标签窗口泄漏到相邻集.
    """

    def __init__(self, dates: np.ndarray,
                 train_months: int = None, val_months: int = None, test_months: int = None):
        self.dates = pd.to_datetime(dates).normalize().values
        self.train_months = train_months or Config.WF_TRAIN_MONTHS
        self.val_months = val_months or Config.WF_VAL_MONTHS
        self.test_months = test_months or Config.WF_TEST_MONTHS
        self.purge = Config.WF_PURGE_DAYS
        self.embargo = Config.WF_EMBARGO_DAYS
        self.unique_days = np.sort(np.unique(self.dates))
        self._month_off = self._month_offset(self.unique_days)
        self.min_test_days = 30   # FM 回归要求至少 30 个截面期, 残缺 test 窗直接丢弃

    @staticmethod
    def _month_offset(days: np.ndarray) -> np.ndarray:
        base_m = days.min().astype('datetime64[M]').astype(int)
        return days.astype('datetime64[M]').astype(int) - base_m

    def windows(self) -> Optional[Dict[str, np.ndarray]]:
        """返回 {window_idx: {'train': day_mask, 'val': day_mask, 'test': day_mask}}. 按天数掩码."""
        n_days = len(self.unique_days)
        last_month = int(self._month_off[-1])
        wm = self.train_months + self.val_months + self.test_months
        windows = {}
        i = 0
        while True:
            train_l, train_r = self.test_months * i, self.test_months * i + self.train_months
            val_l, val_r = train_r, train_r + self.val_months
            test_l, test_r = val_r, val_r + self.test_months
            if test_l > last_month:
                break

            train_day = np.zeros(n_days, dtype=bool)
            val_day = np.zeros(n_days, dtype=bool)
            test_day = np.zeros(n_days, dtype=bool)
            train_day[(self._month_off >= train_l) & (self._month_off < train_r)] = True
            val_day[(self._month_off >= val_l) & (self._month_off < val_r)] = True
            test_day[(self._month_off >= test_l) & (self._month_off < test_r)] = True

            train_day, val_day, test_day = self._purge_embargo(train_day, val_day, test_day)

            if int(test_day.sum()) < self.min_test_days:
                logger.info(f"窗口 {i}: test 窗仅 {int(test_day.sum())} 日 (<{self.min_test_days}), 丢弃残缺窗口")
                break

            windows[i] = {
                'train': train_day, 'val': val_day, 'test': test_day,
                'n_train': int(train_day.sum()), 'n_val': int(val_day.sum()),
                'n_test': int(test_day.sum()),
            }
            i += 1
        return windows or None

    def _purge_embargo(self, tr_d: np.ndarray, va_d: np.ndarray, te_d: np.ndarray):
        """在 train|val 与 val|test 边界施加 purge + embargo 掩码 (按交易日)."""
        tr_idx = np.where(tr_d)[0]
        va_idx = np.where(va_d)[0]
        te_idx = np.where(te_d)[0]

        if len(va_idx) >= self.embargo:
            va_d[va_idx[:self.embargo]] = False      # embargo: 去掉 val 开头 embargo 日
        if len(tr_idx) >= self.purge:
            tr_d[tr_idx[-self.purge:]] = False       # purge: 去掉 train 末尾 purge 日

        if len(te_idx) >= self.embargo:
            te_d[te_idx[:self.embargo]] = False
        if len(va_idx) >= self.purge + self.embargo:
            va_d[va_idx[-self.purge:]] = False
        return tr_d, va_d, te_d

    def to_row_masks(self, window_idx: int, row_dates: np.ndarray) -> Dict[str, np.ndarray]:
        """将指定窗口的天级掩码映射到行级掩码."""
        w = self.windows()[window_idx]
        rd = pd.to_datetime(row_dates).values
        day_index = dict(zip(self.unique_days, range(len(self.unique_days))))
        idx = np.array([day_index[d] for d in rd])
        return {k: w[k][idx] for k in ('train', 'val', 'test')}

    @property
    def unique(self):
        return self.unique_days

    def window_dates(self, window_idx: int) -> Dict[str, str]:
        w = self.windows()[window_idx]
        def _span(mask):
            days = self.unique_days[mask]
            return f"{pd.Timestamp(days.min()).date()}~{pd.Timestamp(days.max()).date()}"
        return {'train': _span(w['train']), 'val': _span(w['val']), 'test': _span(w['test'])}