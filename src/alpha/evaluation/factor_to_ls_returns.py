#!/usr/bin/env python3
"""面板数据 → 等权多空组合日收益 (固定 20/80 分位)"""
import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def panel_to_ls_returns(
    panel: pd.DataFrame,
    factor_cols: list = None,
    top_quantile: float = 0.2,
    min_stocks: int = 30,
) -> pd.DataFrame:
    df = panel.copy()
    df["date"] = pd.to_datetime(df["date"])

    skip = {"date", "symbol", "ret", "open", "high", "low", "close", "volume", "amount",
            "industry_id", "turn", "pct_chg"}
    if factor_cols is None:
        factor_cols = [c for c in df.columns if c not in skip and df[c].dtype in (np.float64, np.int64)]

    if not factor_cols:
        raise ValueError("未找到因子列")

    logger.info(f"因子列: {factor_cols}")
    logger.info(f"日期范围: {df['date'].min().date()} ~ {df['date'].max().date()}, "
                f"股票数: {df['symbol'].nunique()}")

    results = {"date": sorted(df["date"].unique())}
    date_idx = {d: i for i, d in enumerate(results["date"])}

    for fc in factor_cols:
        ls_arr = np.full(len(results["date"]), np.nan)

        for dt, grp in df.groupby("date"):
            if dt not in date_idx:
                continue
            sub = grp[["symbol", "ret", fc]].dropna()
            if len(sub) < min_stocks:
                continue

            n_long = max(int(len(sub) * top_quantile), 1)
            n_short = max(int(len(sub) * top_quantile), 1)

            sorted_sub = sub.sort_values(fc, ascending=False)
            long_ret = sorted_sub.head(n_long)["ret"].mean()
            short_ret = sorted_sub.tail(n_short)["ret"].mean()
            ls_arr[date_idx[dt]] = long_ret - short_ret

        out_col = f"{fc}_LS"
        results[out_col] = ls_arr
        valid = np.isfinite(ls_arr)
        logger.info(f"  {out_col}: {valid.sum()}/{len(ls_arr)} 天有效")

    out_df = pd.DataFrame(results)
    out_df["date"] = pd.to_datetime(out_df["date"])
    return out_df


def main():
    parser = argparse.ArgumentParser(description="面板数据 → 等权 L-S 组合日收益")
    parser.add_argument("--panel", required=True, help="面板数据路径 (CSV/Parquet)")
    parser.add_argument("--factors", nargs="*", help="因子列名 (默认自动识别)")
    parser.add_argument("--top-quantile", type=float, default=0.2, help="多空分位阈值 (默认 0.2)")
    parser.add_argument("--min-stocks", type=int, default=30, help="最小截面股票数 (默认 30)")
    parser.add_argument("--start-date", type=str, default=None, help="过滤起始日期 YYYY-MM-DD (默认不限)")
    parser.add_argument("--end-date", type=str, default=None, help="过滤结束日期 YYYY-MM-DD (默认不限)")
    parser.add_argument("--output", default="ls_returns.csv", help="输出路径")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    ext = os.path.splitext(args.panel)[1].lower()
    panel = pd.read_parquet(args.panel) if ext == ".parquet" else pd.read_csv(args.panel)
    panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)

    if args.start_date or args.end_date:
        panel["date"] = pd.to_datetime(panel["date"])
        if args.start_date:
            panel = panel[panel["date"] >= pd.Timestamp(args.start_date)]
        if args.end_date:
            panel = panel[panel["date"] <= pd.Timestamp(args.end_date)]
        logger.info(f"日期过滤: {args.start_date or '不限'} ~ {args.end_date or '不限'}, "
                    f"剩余 {len(panel):,} 行")

    out = panel_to_ls_returns(panel, args.factors, args.top_quantile, args.min_stocks)
    out.to_csv(args.output, index=False, encoding="utf-8-sig")
    logger.info(f"已保存: {args.output} ({len(out)} 天)")
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
