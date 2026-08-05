import logging
import os
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

class CSI500Loader:
    def __init__(self, path: str = Path(__file__).parent.parent / "CSI500_4Years_2022-06-30_to_2026-06-29.csv",
                 min_bars: int = 252, winsor_pct: tuple = (0.01, 0.99)):
        self.path = str(path)
        self.min_bars = min_bars
        self.winsor_pct = winsor_pct

    def load(self) -> pd.DataFrame:
        df = self._read()
        df = self._clean(df)
        return df

    def _read(self) -> pd.DataFrame:
        ext = os.path.splitext(self.path)[1].lower()
        if ext == ".csv":
            df = pd.read_csv(self.path, parse_dates=["date"])
        elif ext in (".parquet", ".pq"):
            df = pd.read_parquet(self.path)
        else:
            raise ValueError(f"unsupported format: {ext}")
        df = df.rename(columns={"code": "symbol"})
        df["date"] = pd.to_datetime(df["date"])
        df["symbol"] = df["symbol"].astype(str)
        for col in ["open", "high", "low", "close", "volume", "amount", "turn", "pctChg"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df.sort_values(["symbol", "date"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        n0 = len(df)
        logger.info(f"原始样本: {n0:,} 行, {df['symbol'].nunique()} 只股票")
        st_mask = df["symbol"].str.contains(r"ST|\*ST|PT", case=True, na=False)
        df = df[~st_mask].copy()
        n_st = st_mask.sum()
        if n_st:
            logger.info(f"剔除 ST 股: {n_st:,} 行")
        zero_vol = (df["volume"] == 0) | df["volume"].isna()
        flat_price = (
            (df["open"] == df["high"]) & (df["high"] == df["low"]) & (df["low"] == df["close"])
        )
        suspended = zero_vol & flat_price
        df = df[~suspended].copy()
        if suspended.sum():
            logger.info(f"剔除停牌: {suspended.sum():,} 行")
        anomaly = (
            (df["high"] < df[["open", "close"]].max(axis=1))
            | (df["low"] > df[["open", "close"]].min(axis=1))
        )
        df = df[~anomaly].copy()
        if anomaly.sum():
            logger.info(f"剔除价格异常: {anomaly.sum():,} 行")
        counts = df.groupby("symbol").size()
        valid = counts[counts >= self.min_bars].index
        dropped = len(counts) - len(valid)
        df = df[df["symbol"].isin(valid)].copy()
        if dropped:
            logger.info(f"剔除上市不足 {self.min_bars} 天: {dropped} 只")
        for dt, grp in df.groupby("date", sort=False):
            lo = grp["pctChg"].quantile(self.winsor_pct[0])
            hi = grp["pctChg"].quantile(self.winsor_pct[1])
            df.loc[grp.index, "pctChg"] = grp["pctChg"].clip(lower=lo, upper=hi)
        order = ["date", "symbol", "open", "high", "low", "close",
                 "volume", "amount", "turn", "pctChg"]
        df = df[[c for c in order if c in df.columns]]
        for col in ["open", "high", "low", "close", "volume", "amount", "turn", "pctChg"]:
            if col in df.columns:
                df[col] = df[col].astype(np.float64)
        logger.info(
            f"清洗完成: {len(df):,} 行, {df['symbol'].nunique()} 只股票, "
            f"{df['date'].nunique()} 个交易日"
        )
        return df
