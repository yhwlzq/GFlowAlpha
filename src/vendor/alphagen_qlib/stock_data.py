# -*- coding: utf-8 -*-
"""
alphagen_qlib.stock_data — qlib-free drop-in replacement.

替换上游 AlphaSAGE 对 `alphagen_qlib.stock_data` 的唯一依赖：
上游只为获得 `StockData` (数据张量 + 回溯/未来天数 元信息) 与 `FeatureType`
枚举, 从不触碰 qlib 语义本身。这里用 numpy/pandas 面板直接构造相同的
`[T, n_features, n_stocks]` 布局 (行 = 交易日, 由旧到新; 前
`max_backtrack_days` 行为回溯暖身区, 后 `max_future_days` 行为未来区),
零 qlib, 与 vendored `alphagen` 其余代码的接口逐字段对齐。
"""
from typing import List, Optional, Tuple, Union
from enum import IntEnum

import pandas as pd
import torch

__all__ = ["FeatureType", "StockData", "ParquetStockData"]


class FeatureType(IntEnum):
    OPEN = 0
    CLOSE = 1
    HIGH = 2
    LOW = 3
    VOLUME = 4
    VWAP = 5


class ParquetStockData:
    """数据持有人, 对齐上游 `StockData` 接口 (见 `alphagen/data/expression.py`
    与 `alphagen/models/alpha_pool.py` 的使用面)。

    张量布局:
        `self.data`:      torch.Tensor [T, n_features, n_stocks], 行=交易日(旧->新)
        `max_backtrack_days`: 行 0..mb-1 为回溯暖身区 (供 Ref/Ts* 向前看历史)
        `n_days`         : 有效窗口天数 = T - mb - mf
        `max_future_days`: 行末尾的未来区 (本工程 target 只用历史, 恒为 0)
    """

    def __init__(
        self,
        data: torch.Tensor,
        dates: pd.Index,
        stock_ids: pd.Index,
        max_backtrack_days: int = 100,
        max_future_days: int = 0,
        device: Optional[torch.device] = None,
    ) -> None:
        self.data = data
        self._dates = dates
        self._stock_ids = stock_ids
        self.max_backtrack_days = max_backtrack_days
        self.max_future_days = max_future_days
        self.device = device if device is not None else data.device
        self._features = list(FeatureType)
        self.df_bak = None

    @classmethod
    def from_panel(
        cls,
        feature_values: Union[list, "torch.Tensor"],
        dates: pd.Index,
        stock_ids: pd.Index,
        max_backtrack_days: int = 100,
        max_future_days: int = 0,
        device: Optional[torch.device] = None,
    ) -> "ParquetStockData":
        """从 (n_features 个) [T, n_stocks] 面板 numpy 数组堆叠构造。"""
        import numpy as np

        if isinstance(feature_values, list):
            feature_values = np.stack(feature_values, axis=1)  # [T, n_feat, n_stock]
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tensor = torch.tensor(feature_values.astype(np.float32), dtype=torch.float, device=device)
        return cls(tensor, dates, stock_ids,
                   max_backtrack_days=max_backtrack_days,
                   max_future_days=max_future_days,
                   device=device)

    @property
    def n_features(self) -> int:
        return len(self._features)

    @property
    def n_stocks(self) -> int:
        return self.data.shape[-1]

    @property
    def n_days(self) -> int:
        return self.data.shape[0] - self.max_backtrack_days - self.max_future_days

    def make_dataframe(
        self,
        data: Union[torch.Tensor, List[torch.Tensor]],
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """导出 (n_days * n_stocks) 长表, 兼容上游 `StockData.make_dataframe`。"""
        if isinstance(data, list):
            data = torch.stack(data, dim=2)
        if len(data.shape) == 2:
            data = data.unsqueeze(2)
        if columns is None:
            columns = [str(i) for i in range(data.shape[2])]
        n_days, n_stocks, n_columns = data.shape
        if self.n_days != n_days:
            raise ValueError(
                f"number of days in the provided tensor ({n_days}) doesn't "
                f"match that of the current StockData ({self.n_days})")
        if self.n_stocks != n_stocks:
            raise ValueError(
                f"number of stocks in the provided tensor ({n_stocks}) doesn't "
                f"match that of the current StockData ({self.n_stocks})")
        if len(columns) != n_columns:
            raise ValueError(
                f"size of columns ({len(columns)}) doesn't match with "
                f"tensor feature count ({data.shape[2]})")
        if self.max_future_days == 0:
            date_index = self._dates[self.max_backtrack_days:]
        else:
            date_index = self._dates[self.max_backtrack_days:-self.max_future_days]
        index = pd.MultiIndex.from_product([date_index, self._stock_ids])
        data = data.reshape(-1, n_columns)
        return pd.DataFrame(data.detach().cpu().numpy(), index=index, columns=columns)


StockData = ParquetStockData