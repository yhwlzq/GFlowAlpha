import baostock as bs
import pandas as pd
from datetime import datetime, timedelta
import os
import argparse


def fetch_csi500_daily(
    start_date: str | None = None,
    end_date: str | None = None,
    lookback_years: int = 5,
    adjustflag: str = "2",
    output_format: str = "parquet",
    output_dir: str | None = None,
) -> str:
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=lookback_years * 365)).strftime("%Y-%m-%d")

    print(f"时间窗口: {start_date} 至 {end_date}")
    print(f"输出格式: {output_format}")

    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"Baostock 登录失败: {lg.error_msg}")

    try:
        print("正在获取中证500成分股列表...")
        rs = bs.query_zz500_stocks(date=end_date)
        zz500_stocks = []
        while (rs.error_code == "0") & rs.next():
            zz500_stocks.append(rs.get_row_data())

        df_stocks = pd.DataFrame(zz500_stocks, columns=rs.fields)
        stock_codes = df_stocks["code"].tolist()
        print(f"共获取到 {len(stock_codes)} 只成分股。")

        fields = "date,code,open,high,low,close,volume,amount,turn,pctChg"

        all_data = []
        total = len(stock_codes)

        for i, code in enumerate(stock_codes):
            if (i + 1) % 50 == 0 or i == 0:
                print(f"下载进度: {i + 1}/{total} ({code})...")

            rs_k = bs.query_history_k_data_plus(
                code,
                fields,
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag=adjustflag,
            )

            data_list = []
            while (rs_k.error_code == "0") & rs_k.next():
                data_list.append(rs_k.get_row_data())

            if data_list:
                df = pd.DataFrame(data_list, columns=rs_k.fields)
                all_data.append(df)

        if not all_data:
            raise RuntimeError("未获取到有效数据，请检查网络连接或 Baostock 服务器状态。")

        print("正在合并数据并进行类型转换...")
        final_df = pd.concat(all_data, ignore_index=True)

        numeric_cols = ["open", "high", "low", "close", "volume", "amount", "turn", "pctChg"]
        final_df[numeric_cols] = final_df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        final_df.dropna(subset=["close", "volume"], inplace=True)
        final_df.sort_values(["code", "date"], inplace=True)
        final_df.reset_index(drop=True, inplace=True)

        if output_dir is None:
            output_dir = os.getcwd()

        os.makedirs(output_dir, exist_ok=True)

        if output_format == "csv":
            filename = f"csi500_daily_{start_date}_to_{end_date}.csv"
            filepath = os.path.join(output_dir, filename)
            final_df.to_csv(filepath, index=False, encoding="utf-8-sig")
        else:
            filename = f"csi500_daily_{start_date}_to_{end_date}.parquet"
            filepath = os.path.join(output_dir, filename)
            final_df.to_parquet(filepath, index=False)

        print(f"文件已保存至: {os.path.abspath(filepath)}")
        print(f"数据总行数: {len(final_df)} 条, 成分股数: {final_df['code'].nunique()}")

        return os.path.abspath(filepath)

    finally:
        bs.logout()


def fetch_hs300_daily(
    start_date: str = "2022-06-30",
    end_date: str = "2026-06-30",
    adjustflag: str = "2",
    output_format: str = "parquet",
    output_dir: str | None = None,
) -> str:
    print(f"时间窗口: {start_date} 至 {end_date}")
    print(f"输出格式: {output_format}")

    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"Baostock 登录失败: {lg.error_msg}")

    try:
        print("正在获取沪深300成分股列表...")
        rs = bs.query_hs300_stocks(date=end_date)
        hs300_stocks = []
        while (rs.error_code == "0") & rs.next():
            hs300_stocks.append(rs.get_row_data())

        df_stocks = pd.DataFrame(hs300_stocks, columns=rs.fields)
        stock_codes = df_stocks["code"].tolist()
        print(f"共获取到 {len(stock_codes)} 只成分股。")

        fields = "date,code,open,high,low,close,volume,amount,turn,pctChg"

        all_data = []
        total = len(stock_codes)

        for i, code in enumerate(stock_codes):
            if (i + 1) % 50 == 0 or i == 0:
                print(f"下载进度: {i + 1}/{total} ({code})...")

            rs_k = bs.query_history_k_data_plus(
                code,
                fields,
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag=adjustflag,
            )

            data_list = []
            while (rs_k.error_code == "0") & rs_k.next():
                data_list.append(rs_k.get_row_data())

            if data_list:
                df = pd.DataFrame(data_list, columns=rs_k.fields)
                all_data.append(df)

        if not all_data:
            raise RuntimeError("未获取到有效数据，请检查网络连接或 Baostock 服务器状态。")

        print("正在合并数据并进行类型转换...")
        final_df = pd.concat(all_data, ignore_index=True)

        numeric_cols = ["open", "high", "low", "close", "volume", "amount", "turn", "pctChg"]
        final_df[numeric_cols] = final_df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        final_df.dropna(subset=["close", "volume"], inplace=True)
        final_df.sort_values(["code", "date"], inplace=True)
        final_df.reset_index(drop=True, inplace=True)

        if output_dir is None:
            output_dir = os.getcwd()

        os.makedirs(output_dir, exist_ok=True)

        if output_format == "csv":
            filename = f"hs300_daily_{start_date}_to_{end_date}.csv"
            filepath = os.path.join(output_dir, filename)
            final_df.to_csv(filepath, index=False, encoding="utf-8-sig")
        else:
            filename = f"hs300_daily_{start_date}_to_{end_date}.parquet"
            filepath = os.path.join(output_dir, filename)
            final_df.to_parquet(filepath, index=False)

        print(f"文件已保存至: {os.path.abspath(filepath)}")
        print(f"数据总行数: {len(final_df)} 条, 成分股数: {final_df['code'].nunique()}")

        return os.path.abspath(filepath)

    finally:
        bs.logout()


def main():
    parser = argparse.ArgumentParser(description="从 Baostock 获取指数日频历史数据")
    parser.add_argument("--index", type=str, default="csi500", choices=["csi500", "hs300"],
                        help="指数: csi500(中证500) 或 hs300(沪深300) (默认 csi500)")
    parser.add_argument("--start-date", type=str, default=None, help="2020-06-30")
    parser.add_argument("--end-date", type=str, default=None, help="2026-06-30")
    parser.add_argument("--lookback-years", type=int, default=5, help="回溯年数 (默认 5)，仅 csi500 使用")
    parser.add_argument("--adjustflag", type=str, default="2", choices=["1", "2", "3"],
                        help="复权方式: 1=后复权, 2=前复权(默认), 3=不复权")
    parser.add_argument("--format", type=str, default="parquet", choices=["parquet", "csv"],
                        help="输出格式 (默认 parquet)")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录 (默认当前目录)")

    args = parser.parse_args()

    if args.index == "hs300":
        filepath = fetch_hs300_daily(
            start_date=args.start_date or "2021-06-30",
            end_date=args.end_date or "2026-06-30",
            adjustflag=args.adjustflag,
            output_format=args.format,
            output_dir=args.output_dir,
        )
    else:
        filepath = fetch_csi500_daily(
            start_date=args.start_date or "2020-06-30",
            end_date=args.end_date or "2026-06-30",
            lookback_years=args.lookback_years,
            adjustflag=args.adjustflag,
            output_format=args.format,
            output_dir=args.output_dir,
        )
    print(f"完成: {filepath}")


if __name__ == "__main__":
    main()
