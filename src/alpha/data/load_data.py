#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中证500 日频数据加载与清洗
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


class CSI500Loader:
    """
    中证500 日频数据加载与清洗

    Usage:
        loader = CSI500Loader("path/to/data.csv")
        df = loader.load()
    """

    def __init__(self, path: str = Path("data/csi500_daily_2021-06-30_to_2026-06-30.parquet"), min_bars: int = 252, winsor_pct: tuple = (0.01, 0.99)):
        self.path = path
        self.min_bars = min_bars
        self.winsor_pct = winsor_pct

    def load(self) -> pd.DataFrame:
        df = self._read()
        df = self._clean(df)
        print(df.columns)
        print("***********")
        return df

    # ── 读取 ──────────────────────────────────────────────────────

    def _read(self) -> pd.DataFrame:
        ext = os.path.splitext(self.path)[1].lower()
        if ext == ".csv":
            df = pd.read_csv(self.path, parse_dates=["date"])
        elif ext in (".parquet", ".pq"):
            df = pd.read_parquet(self.path)
        else:
            raise ValueError(f"不支持的文件格式: {ext}，仅支持 .csv / .parquet")
        
        df = df.rename(columns={"date": "datetime", "code": "symbol"})
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["symbol"] = df["symbol"].astype(str)
        for col in ["open", "high", "low", "close", "volume", "amount", "turn", "pctChg"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df.sort_values(["symbol", "datetime"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    # ── 清洗 ──────────────────────────────────────────────────────

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        n0 = len(df)
        logger.info(f"原始样本: {n0:,} 行, {df['symbol'].nunique()} 只股票")

        # 1. 剔除 ST / *ST / PT
        st_mask = df["symbol"].str.contains(r"ST|\*ST|PT", case=True, na=False)
        df = df[~st_mask].copy()
        n_st = st_mask.sum()
        if n_st:
            logger.info(f"剔除 ST 股: {n_st:,} 行")

        # 2. 剔除停牌 (volume=0 且 OHLC 四价平齐)
        zero_vol = (df["volume"] == 0) | df["volume"].isna()
        flat_price = (
            (df["open"] == df["high"]) & (df["high"] == df["low"]) & (df["low"] == df["close"])
        )
        suspended = zero_vol & flat_price
        df = df[~suspended].copy()
        if suspended.sum():
            logger.info(f"剔除停牌: {suspended.sum():,} 行")

        # 3. 剔除价格异常 (high < max(open,close) 或 low > min(open,close))
        anomaly = (
            (df["high"] < df[["open", "close"]].max(axis=1))
            | (df["low"] > df[["open", "close"]].min(axis=1))
        )
        df = df[~anomaly].copy()
        if anomaly.sum():
            logger.info(f"剔除价格异常: {anomaly.sum():,} 行")

        # 4. 剔除上市不足 min_bars 个交易日的股票
        counts = df.groupby("symbol").size()
        valid = counts[counts >= self.min_bars].index
        dropped = len(counts) - len(valid)
        df = df[df["symbol"].isin(valid)].copy()
        if dropped:
            logger.info(f"剔除上市不足 {self.min_bars} 天: {dropped} 只")

        # 5. 收益率缩尾 (每日截面)
        for dt, grp in df.groupby("datetime", sort=False):
            lo = grp["pctChg"].quantile(self.winsor_pct[0])
            hi = grp["pctChg"].quantile(self.winsor_pct[1])
            df.loc[grp.index, "pctChg"] = grp["pctChg"].clip(lower=lo, upper=hi)

        # 最终整理
        order = ["datetime", "symbol", "open", "high", "low", "close",
                 "volume", "amount", "turn", "pctChg"]
        df = df[[c for c in order if c in df.columns]]
        for col in ["open", "high", "low", "close", "volume", "amount", "turn", "pctChg"]:
            if col in df.columns:
                df[col] = df[col].astype(np.float64)

        logger.info(
            f"清洗完成: {len(df):,} 行, {df['symbol'].nunique()} 只股票, "
            f"{df['datetime'].nunique()} 个交易日"
        )
        return df


# ============================================================================
# 便捷函数
# ============================================================================

def load_csi500(path: str, **kwargs) -> pd.DataFrame:
    return CSI500Loader(path, **kwargs).load()


if __name__ == "__main__":
    instance = CSI500Loader().load()
