#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared AcademicBacktester — single source of truth for all ablation/primary scripts.
Fixed: correctly compounds long/short separately within each month, then subtracts.
Also reports Q1–Q5 quantile monthly returns for monotonicity plotting.
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AcademicBacktester:
    def __init__(self, top_quantile: float = 0.2, n_groups: int = 0, cost_bps: float = 30.0):
        self.top_quantile = top_quantile
        self.n_groups = n_groups if n_groups > 0 else int(1.0 / top_quantile)
        self.cost_bps = cost_bps

    def run(self, pred: np.ndarray, ret: np.ndarray, datetimes: np.ndarray,
            symbols: np.ndarray, fid: str = ""):
        df = pd.DataFrame({
            'date': pd.to_datetime(datetimes).normalize(),
            'symbol': symbols, 'pred': pred, 'ret': ret
        }).dropna()
        if df.empty:
            return

        df['month'] = df['date'].dt.to_period('M')

        monthly_ls_rets = []
        monthly_quantile_rets = {f'Q{i+1}': [] for i in range(self.n_groups)}

        for month, grp in df.groupby('month'):
            first_day = grp['date'].min()
            cross_section = grp[grp['date'] == first_day]
            if len(cross_section) < 50:
                continue

            # Assign quantile groups based on first-day cross-section
            cross_section = cross_section.sort_values('pred')
            labels = [f'Q{i+1}' for i in range(self.n_groups)]
            try:
                cross_section['quantile'] = pd.qcut(
                    cross_section['pred'],
                    q=self.n_groups,
                    labels=labels,
                    duplicates='drop'
                )
            except ValueError:
                logger.warning(f"  {fid}: qcut 失败({month}), 跳过该月")
                continue
            quantile_map = dict(zip(cross_section['symbol'], cross_section['quantile']))
            grp = grp.copy()
            grp['quantile'] = grp['symbol'].map(quantile_map)

            long_syms = set(cross_section[cross_section['quantile'] == f'Q{self.n_groups}']['symbol'])
            short_syms = set(cross_section[cross_section['quantile'] == 'Q1']['symbol'])

            daily_quantile_rets = {f'Q{i+1}': [] for i in range(self.n_groups)}
            daily_long_rets = []
            daily_short_rets = []

            for day, day_grp in grp.groupby('date'):
                for q_name, q_grp in day_grp.groupby('quantile'):
                    daily_quantile_rets[q_name].append(q_grp['ret'].mean())

                long_ret = day_grp[day_grp['symbol'].isin(long_syms)]['ret'].mean()
                short_ret = day_grp[day_grp['symbol'].isin(short_syms)]['ret'].mean()
                daily_long_rets.append(long_ret)
                daily_short_rets.append(short_ret)

            # Apply transaction cost on rebalance day (first day of month)
            if self.cost_bps > 0 and len(daily_long_rets) > 0:
                cost_per_leg = self.cost_bps / 2 / 10000
                daily_long_rets[0] = (1 + daily_long_rets[0]) * (1 - cost_per_leg) - 1
                daily_short_rets[0] = (1 + daily_short_rets[0]) * (1 - cost_per_leg) - 1

            # Correct: compound long and short separately, then subtract
            month_long_ret = float(np.prod(1 + np.array(daily_long_rets, dtype=np.float64))) - 1
            month_short_ret = float(np.prod(1 + np.array(daily_short_rets, dtype=np.float64))) - 1
            month_ls_ret = month_long_ret - month_short_ret
            monthly_ls_rets.append(month_ls_ret)

            for q_name in daily_quantile_rets:
                if daily_quantile_rets[q_name]:
                    month_q_ret = float(np.prod(1 + np.array(daily_quantile_rets[q_name], dtype=np.float64))) - 1
                    monthly_quantile_rets[q_name].append(month_q_ret)

        if not monthly_ls_rets:
            return

        pnl_df = pd.DataFrame({
            'month': [str(m) for m in df['month'].unique()[:len(monthly_ls_rets)]],
            'net_ret': monthly_ls_rets
        }).set_index('month')
        pnl_df['cumulative'] = (1 + pnl_df['net_ret']).cumprod()

        ann_ret = float(pnl_df['cumulative'].iloc[-1] ** (12 / len(pnl_df)) - 1)
        sharpe = float(pnl_df['net_ret'].mean() / (pnl_df['net_ret'].std() + 1e-8) * np.sqrt(12))

        quantile_summary = {}
        for q_name, rets in monthly_quantile_rets.items():
            quantile_summary[q_name] = float(np.mean(rets)) if rets else 0.0

        logger.info(f"🏆 {fid} 学术回测: 年化={ann_ret:.2%}, 月度Sharpe={sharpe:.2f}, months={len(pnl_df)}")
        logger.info(f"📊 {fid} 五分位平均月收益: "
                    + ", ".join(f"{q}={quantile_summary.get(q, 0):.4%}"
                               for q in [f'Q{i+1}' for i in range(self.n_groups)]))
        return {
            'annualized_return': ann_ret,
            'sharpe_ratio': sharpe,
            'quantile_monthly_rets': quantile_summary,
            'cumulative_ls': pnl_df['cumulative'].values,
        }
