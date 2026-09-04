#!/usr/bin/env python3
"""因子 L-S 收益 vs FF5+LIQ 时序回归 (Newey-West HAC 校正)"""
import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_hac
from scipy.stats import t as t_dist

logger = logging.getLogger(__name__)


def run_orthogonality(
    factors_df: pd.DataFrame,
    ff5_df: pd.DataFrame,
    nw_lags: int = 5,
    alpha_threshold: float = 2.0,
    drop_liq: bool = False,
    fids: list[str] | None = None,
) -> pd.DataFrame:
    ff5_df = ff5_df.copy()
    all_ff5_cols = [c for c in ff5_df.columns if c != "date"]
    if drop_liq:
        ff5_df = ff5_df.drop(columns=["LIQ"], errors="ignore")
    factor_cols = [c for c in factors_df.columns if c != "date"]
    if fids:
        requested = [f if f in factors_df.columns else f + "_LS" for f in fids]
        factor_cols = [c for c in requested if c in factors_df.columns]
    ff5_cols = [c for c in ff5_df.columns if c != "date"]

    # 健壮性 (合并前): 剔除覆盖率过低的因子列 (如公式特征缺失仍全 NaN),
    # 否则任何一列全 NaN 会把 inner 连接的 dropna 帧清空, 导致整个回归崩溃.
    min_valid = max(30, int(len(factors_df) * 0.5))
    skipped = [fc for fc in factor_cols
               if fc not in factors_df.columns
               or np.isfinite(factors_df[fc]).sum() < min_valid]
    if skipped:
        logger.warning(f"跳过覆盖率不足的因子列: {skipped}")
    factor_cols = [fc for fc in factor_cols if fc not in skipped]
    if not factor_cols:
        logger.error("无有效因子列可回归 (均缺失/覆盖率不足), 终止")
        sys.exit(2)

    merged = pd.merge(factors_df, ff5_df, on="date", how="inner").dropna()
    if len(merged) < min_valid:
        logger.error(f"合并后有效样本过少: {len(merged)} 天 (< {min_valid}), 终止")
        sys.exit(2)
    logger.info(f"合并数据: {len(merged)} 天, {merged['date'].min().date()} ~ {merged['date'].max().date()}")

    dropped_liq = "LIQ" in all_ff5_cols and drop_liq
    rows = []
    for fc in factor_cols:
        y = merged[fc].values
        X = merged[ff5_cols].values
        X = sm.add_constant(X)

        model = sm.OLS(y, X).fit()

        nw_cov = cov_hac(model, nlags=nw_lags, use_correction=True)
        nw_se = np.sqrt(np.diag(nw_cov))
        n_params = len(model.params)
        nw_t = np.where(nw_se > 1e-10, model.params / nw_se, 0.0)
        df_freedom = len(y) - n_params
        nw_p = np.array([2 * (1 - t_dist.cdf(abs(t), df_freedom)) for t in nw_t])

        alpha = model.params[0]
        t_alpha = nw_t[0]
        r2 = model.rsquared

        if abs(t_alpha) > alpha_threshold:
            judgment = "Pure Alpha"
        elif abs(t_alpha) > 1.5:
            judgment = "Marginal"
        else:
            judgment = "Risk-proxy"

        row = {
            "Factor": fc,
            "Alpha": round(alpha, 6),
            "t_Alpha": round(t_alpha, 3),
            "p_Alpha": round(nw_p[0], 4),
        }
        for i, col in enumerate(ff5_cols, start=1):
            row[f"Beta_{col}"] = round(model.params[i], 6)
            row[f"t_{col}"] = round(nw_t[i], 3)

        if dropped_liq:
            row["Beta_LIQ"] = np.nan
            row["t_LIQ"] = np.nan

        row["R2"] = round(r2, 4)
        row["N"] = len(y)
        row["N_start"] = merged["date"].min().date()
        row["N_end"] = merged["date"].max().date()
        row["Judgment"] = judgment
        rows.append(row)

    return pd.DataFrame(rows)


def print_table(results: pd.DataFrame):
    ff5_cols = [c.replace("Beta_", "") for c in results.columns if c.startswith("Beta_")]

    print(f"\n{'=' * 110}")
    print(f"  正交性检验: 因子 L-S 收益 vs FF5 + LIQ  (Newey-West HAC 校正)")
    print(f"  回归区间: {results['N_start'].iloc[0]} ~ {results['N_end'].iloc[0]}  "
          f"| 天数: {results['N'].iloc[0]}")
    print(f"{'=' * 110}")

    header = f"{'Factor':<16} {'Alpha':>9} {'t(α)':>7} {'p(α)':>7}"
    for col in ff5_cols:
        header += f" {'β_'+col:>9} {'t_'+col:>7}"
    header += f" {'R²':>6} {'Judgment':<14}"
    print(header)
    print(f"{'-' * 110}")

    for _, row in results.iterrows():
        line = f"{row['Factor']:<16} {row['Alpha']:>9.5f} {row['t_Alpha']:>7.2f} {row['p_Alpha']:>7.3f}"
        for col in ff5_cols:
            beta = row.get(f"Beta_{col}", 0)
            t_val = row.get(f"t_{col}", 0)
            sig = ""
            if pd.isna(t_val):
                line += f" {'N/A':>9} {'—':>7}"
                continue
            if abs(t_val) > 2.5:
                sig = "***"
            elif abs(t_val) > 1.96:
                sig = "**"
            elif abs(t_val) > 1.64:
                sig = "*"
            line += f" {beta:>9.5f} {t_val:>6.2f}{sig}"
        line += f" {row['R2']:>6.3f} {row['Judgment']:<14}"
        print(line)

    print(f"{'-' * 110}")
    n_alpha = (results["Judgment"] == "Pure Alpha").sum()
    n_total = len(results)
    print(f"  Pure Alpha: {n_alpha}/{n_total}  |  "
          f"*** p<0.001  ** p<0.05  * p<0.10")
    print(f"{'=' * 110}\n")


def print_jfds_table(results: pd.DataFrame, fids: list[str] | None = None) -> str:
    """生成 JFE/JFDS 风格两行式 Markdown 表格 (系数行 + Newey-West t 值行)。

    每因子两行:
      - 行1: Alpha, t(Alpha), 各 β 系数, R²
      - 行2: 各 β 的 t 值 (括号内, |t|>1.96 加粗)
    返回 Markdown 字符串。
    """
    ff5_cols = [c.replace("Beta_", "") for c in results.columns if c.startswith("Beta_")]
    ff5_cols = [c for c in ff5_cols if c != "LIQ"]

    rows = results.copy()
    if fids:
        requested = [f if f in rows["Factor"].tolist() else f + "_LS" for f in fids]
        rows = rows[rows["Factor"].isin(requested)]
        order = {f: i for i, f in enumerate(requested)}
        rows = rows.sort_values(by="Factor", key=lambda s: s.map(order)).reset_index(drop=True)
    else:
        rows = rows.reset_index(drop=True)

    def fmt_beta(v):
        if pd.isna(v):
            return "N/A"
        return f"{v:.3f}".replace("-", "−")

    def fmt_t(v):
        if pd.isna(v):
            return "—"
        s = f"{v:.2f}".replace("-", "−")
        return f"**{s}**" if abs(v) > 1.96 else s

    header_cols = ["Factor", "Alpha", "t(Alpha)"] + [f"β_{c}" for c in ff5_cols] + ["R²"]
    header = "| " + " | ".join(header_cols) + " |"
    sep = "|" + "|".join(["---"] * len(header_cols)) + "|"

    lines = [header, sep]
    for _, r in rows.iterrows():
        factor = r["Factor"].replace("_LS", "")
        coef_row = [factor, f"{r['Alpha']:.4f}", f"({r['t_Alpha']:.2f})"]
        t_row = ["", "", ""]
        for c in ff5_cols:
            coef_row.append(fmt_beta(r.get(f"Beta_{c}")))
            t_row.append(f"({fmt_t(r.get(f't_{c}'))})")
        coef_row.append(f"{r['R2']:.3f}")
        t_row.append("")
        lines.append("| " + " | ".join(coef_row) + " |")
        lines.append("| " + " | ".join(t_row) + " |")

    md = "\n".join(lines) + "\n"
    print(md)
    return md


def main():
    parser = argparse.ArgumentParser(description="正交性检验: 因子 vs FF5+LIQ")
    parser.add_argument("--factors", required=True, help="因子 L-S 收益 CSV (date + 因子列)")
    parser.add_argument("--ff5", required=True, help="FF5+LIQ 日收益 CSV (date + MKT/SMB/...)")
    parser.add_argument("--nw-lag", type=int, default=5, help="Newey-West 滞后阶数 (默认 5)")
    parser.add_argument("--alpha-threshold", type=float, default=2.0, help="α 显著性阈值 (默认 2.0)")
    parser.add_argument("--drop-liq", action="store_true",
                        help="从回归中剔除 LIQ 因子 (LIQ 量纲小而均值非零, 可能毒化截距)")
    parser.add_argument("--fids", nargs="*", default=None,
                        help="要报告的因子列表 (registry 顺序); 缺省报告全部")
    parser.add_argument("--output", default="orthogonality_result.csv", help="输出 CSV 路径")
    parser.add_argument("--output-md", default=None,
                        help="输出 JFE/JFDS 风格 Markdown 表格路径 (缺省不生成)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    factors_df = pd.read_csv(args.factors, parse_dates=["date"])
    ff5_df = pd.read_csv(args.ff5, parse_dates=["date"])

    results = run_orthogonality(factors_df, ff5_df, args.nw_lag, args.alpha_threshold, args.drop_liq, args.fids)
    print_table(results)

    if args.output_md:
        md = print_jfds_table(results, args.fids)
        with open(args.output_md, "w", encoding="utf-8") as f:
            f.write(md)
        logger.info(f"已保存: {args.output_md}")

    results.to_csv(args.output, index=False, encoding="utf-8-sig")
    logger.info(f"已保存: {args.output}")


if __name__ == "__main__":
    main()
