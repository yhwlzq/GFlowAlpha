#!/usr/bin/env python3
"""下载 FF5 日频因子 + 计算 Amihud 流动性，输出 ff5_liq.csv"""
import argparse
import logging
import os
import sys
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FF5_PATH = (
    "https://www.factorwar.com/wp-content/uploads/2026/05/"
    "Fama-French-五因子模型（经典算法）日收益率（截至到20260331）.csv"
)
FF5_URL = urllib.parse.quote(FF5_PATH, safe=":/?=&%")


def download_ff5(output_path: str) -> pd.DataFrame:
    logger.info(f"从 factorwar.com 下载 FF5 日频数据...")
    urllib.request.urlretrieve(FF5_URL, output_path)
    df = pd.read_csv(output_path, encoding="gbk")
    df.columns = [c.strip() for c in df.columns]

    date_col = df.columns[0]
    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"], format="mixed")

    rename_map = {}
    for c in df.columns:
        cl = c.lower()
        if "市场" in c and "超额" not in c:
            rename_map[c] = "MKT_raw"
        elif "市场" in c and "超额" in c:
            rename_map[c] = "MKT"
        elif "规模" in c or "smb" in cl:
            rename_map[c] = "SMB"
        elif "价值" in c or "hml" in cl:
            rename_map[c] = "HML"
        elif "盈利" in c or "rmw" in cl:
            rename_map[c] = "RMW"
        elif "投资" in c or "cma" in cl:
            rename_map[c] = "CMA"
        elif "无风险" in c or "rf" in cl:
            rename_map[c] = "RF"

    df = df.rename(columns=rename_map)

    if "MKT" not in df.columns and "MKT_raw" in df.columns and "RF" in df.columns:
        df["MKT"] = df["MKT_raw"] - df["RF"]
    elif "MKT" not in df.columns:
        for c in df.columns:
            if c not in ("date", "RF") and df[c].dtype in (np.float64, np.int64):
                df = df.rename(columns={c: "MKT"})
                break

    keep = [c for c in ["date", "MKT", "SMB", "HML", "RMW", "CMA"] if c in df.columns]
    df = df[keep].dropna(subset=["date"])

    for c in ["MKT", "SMB", "HML", "RMW", "CMA"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values("date").reset_index(drop=True)
    logger.info(f"FF5 数据: {len(df)} 天, {df['date'].min().date()} ~ {df['date'].max().date()}")
    return df


def compute_amihud_liq(panel: pd.DataFrame) -> pd.DataFrame:
    logger.info("计算 Amihud 流动性因子...")
    df = panel.copy()
    df["date"] = pd.to_datetime(df["date"])

    if "ret" not in df.columns:
        if "pctChg" in df.columns:
            df["ret"] = pd.to_numeric(df["pctChg"], errors="coerce") / 100.0
        else:
            logger.warning("面板数据中无 ret/pctChg 列，跳过 LIQ 计算")
            return None

    if "volume" in df.columns:
        vol_col = "volume"
    elif "amount" in df.columns:
        vol_col = "amount"
    else:
        logger.warning("面板数据中无 volume/amount 列，跳过 LIQ 计算")
        return None

    df["ret_abs"] = df["ret"].abs()
    df["vol_log"] = np.log(np.maximum(df[vol_col].astype(float), 1.0))
    df["amihud"] = df["ret_abs"] / df["vol_log"]
    df = df[np.isfinite(df["amihud"])]

    liq = df.groupby("date")["amihud"].agg(["mean", "count"]).reset_index()
    liq = liq[liq["count"] >= 30].rename(columns={"mean": "LIQ"})
    liq = liq[["date", "LIQ"]].sort_values("date").reset_index(drop=True)
    logger.info(f"Amihud LIQ: {len(liq)} 天")
    return liq


def main():
    parser = argparse.ArgumentParser(description="下载 FF5 + 计算 Amihud LIQ")
    parser.add_argument("--panel", type=str, help="面板数据路径 (CSV/Parquet)，用于计算 LIQ")
    parser.add_argument("--output", type=str, default="ff5_liq.csv", help="输出路径")
    parser.add_argument("--existing", type=str, help="已有 ff5_liq.csv 路径（仅检查格式）")
    parser.add_argument("--check", action="store_true", help="仅检查格式")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if args.existing and args.check:
        df = pd.read_csv(args.existing, parse_dates=["date"])
        required = {"date", "MKT", "SMB", "HML", "RMW", "CMA"}
        missing = required - set(df.columns)
        if missing:
            logger.error(f"缺少列: {missing}")
            sys.exit(1)
        logger.info(f"格式检查通过: {len(df)} 行, {df['date'].min()} ~ {df['date'].max()}")
        return

    ff5 = download_ff5(args.output)

    liq = None
    if args.panel:
        ext = os.path.splitext(args.panel)[1].lower()
        if ext == ".parquet":
            panel = pd.read_parquet(args.panel)
        else:
            panel = pd.read_csv(args.panel)
        liq = compute_amihud_liq(panel)

    if liq is not None and not liq.empty:
        result = pd.merge(ff5, liq, on="date", how="inner")
        logger.info(f"合并后: {len(result)} 天")
    else:
        result = ff5
        logger.warning("无 LIQ 数据，仅输出 FF5")

    result.to_csv(args.output, index=False, encoding="utf-8-sig")
    logger.info(f"已保存: {args.output}")
    print(result.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
