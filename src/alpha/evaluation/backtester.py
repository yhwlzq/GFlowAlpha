#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared AcademicBacktester — single source of truth for all ablation/primary scripts.
[Fixed] 1. 交易成本严格与换手率(Turnover)挂钩, 每腿按完整往返费率收取.
[Fixed] 2. 停牌/缺失数据使用 reindex 填0，防止隐形权重再分配(幸存者偏差).
[Fixed] 3. 使用 rank(method='first') 确定性分箱，因子值重复不再导致分箱失败.
[Fixed] 4. 统一 Long-only 的年化与 Sharpe 计算频次(日频, 252).
[Fixed] 5. 按 (month, ret) 配对保存，避免月度被跳过时标签错位.
[Fixed] 6. T+1 开盘执行口径 (可选): 信号 T 日收盘生成 → T+1 日开盘建仓,
        执行日收益 = close/open (剔除隔夜), 持有日收益 = close/prev_close;
        开盘一字涨停(多头)/一字跌停(空头)无法成交的标的在构建时置零剔除.
        未传 open/close 时自动回退旧"收盘执行"口径, 向后兼容.
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AcademicBacktester:
    def __init__(self, top_quantile: float = 0.2, n_groups: int = 0,
                 cost_bps: float = 30.0, long_only: bool = False):
        """
        Args:
            top_quantile: 多头分位阈值.
            n_groups: 分位数, 默认由 top_quantile 推导.
            cost_bps: 单组合完整往返交易成本 (bps, 2×单边). 默认 30.0 (即单边 15bps).
            long_only: True 时主指标取 long-only 日频口径 (252), 同时仍报告 Ann L-S.
        """
        self.top_quantile = top_quantile
        self.n_groups = n_groups if n_groups > 0 else int(1.0 / top_quantile)
        self.cost_bps = cost_bps
        self.long_only = long_only
        self.cost_per_roundtrip = self.cost_bps / 10000.0

    def run(self, pred: np.ndarray, ret: np.ndarray, datetimes: np.ndarray,
            symbols: np.ndarray, fid: str = "",
            open_prices: np.ndarray | None = None,
            close_prices: np.ndarray | None = None,
            limit_pct: float = 0.10):
        """
        提供 open_prices / close_prices (与 pred/ret 逐行同序) 时启用 T+1 开盘执行口径:
          信号 T 日收盘生成 → 下一交易日开盘建仓; 执行日收益 = close/open (剔除隔夜),
          持有日收益 = close/prev_close; 开盘即一字涨停(多头)/一字跌停(空头)无法成交的
          标的在构建时置零剔除 (limit_pct 为涨跌停幅度, <=0 时不做成交门槛).
        缺省时回退旧的"收盘执行, 信号日当天开始计收益"口径.
        """
        cols = {
            'date': pd.to_datetime(datetimes).normalize(),
            'symbol': symbols, 'pred': pred, 'ret': ret,
        }
        if open_prices is not None and close_prices is not None:
            cols['open'] = np.asarray(open_prices, dtype=np.float64)
            cols['close'] = np.asarray(close_prices, dtype=np.float64)
        df = pd.DataFrame(cols).dropna(subset=['date', 'symbol', 'pred', 'ret'])
        if df.empty:
            return None

        use_open_exec = 'open' in df.columns and 'close' in df.columns
        if use_open_exec:
            df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
            df['prev_close'] = df.groupby('symbol')['close'].shift(1)
            df['entry_ratio'] = df['close'] / df['open']
            df['held_ratio'] = df['close'] / df['prev_close']
            cal = pd.DatetimeIndex(sorted(pd.unique(df['date'])))
            next_day = {d: cal[i + 1] for i, d in enumerate(cal[:-1])}
        else:
            df = df.dropna()

        df['month'] = df['date'].dt.to_period('M')

        ls_series = []                     # (month, ls_ret)
        quantile_series = {f'Q{i+1}': [] for i in range(self.n_groups)}
        long_daily_all = []
        ls_daily_all = []

        prev_long = None
        prev_short = None
        long_turnovers = []
        short_turnovers = []

        exec_label = "T+1开盘执行" if use_open_exec else "收盘执行(旧口径)"
        for month, grp in df.groupby('month'):
            first_day = grp['date'].min()
            cross_section = grp[grp['date'] == first_day]
            if len(cross_section) < 50:
                continue

            cross_section = cross_section.sort_values('pred').copy()
            labels = [f'Q{i+1}' for i in range(self.n_groups)]
            # [Fix 3] 确定性分箱: rank 后无重复值, qcut 永不因重复值失败
            try:
                cross_section['quantile'] = pd.qcut(
                    cross_section['pred'].rank(method='first'),
                    q=self.n_groups, labels=labels)
            except ValueError:
                logger.warning(f"  {fid}: qcut 失败({month}), 跳过该月")
                continue

            quantile_map = dict(zip(cross_section['symbol'], cross_section['quantile']))
            grp = grp.copy()
            grp['quantile'] = grp['symbol'].map(quantile_map)

            quantile_syms = {q: list(cross_section.loc[cross_section['quantile'] == q, 'symbol'])
                             for q in labels}
            long_syms = quantile_syms[f'Q{self.n_groups}']
            short_syms = quantile_syms['Q1']

            # ── T+1 开盘执行: 定位执行日, 并对开盘涨跌停标的做成交门槛 ──
            if use_open_exec:
                entry_day = next_day.get(first_day)
                if entry_day is None:
                    logger.warning(f"  {fid}: {month} 无信号日次交易日, 跳过")
                    continue
                held_days = [pd.Timestamp(d) for d in
                             sorted(pd.unique(grp['date'][grp['date'] >= entry_day]))]
                if limit_pct and limit_pct > 0:
                    e_rows = grp[grp['date'] == entry_day].set_index('symbol')
                    up = e_rows['prev_close'] * (1 + limit_pct)   # 涨停价
                    dn = e_rows['prev_close'] * (1 - limit_pct)   # 跌停价
                    # NaN(无前收盘/停牌) 的标的低于/高于判断均为 False, 保守剔除
                    long_fill = set(e_rows.index[e_rows['open'] < up])
                    short_fill = set(e_rows.index[e_rows['open'] > dn])
                    long_syms = [s for s in long_syms if s in long_fill]
                    short_syms = [s for s in short_syms if s in short_fill]
            else:
                entry_day = first_day
                held_days = [pd.Timestamp(d) for d in sorted(pd.unique(grp['date']))]

            if not held_days or not long_syms or not short_syms:
                logger.warning(f"  {fid}: {month} 无可成交标的或无持有日, 跳过")
                continue

            # [Fix 1] 每腿换手率单独计算 (首月建仓视为 1.0)
            if prev_long is None:
                long_turnover = 1.0
            else:
                long_turnover = 1.0 - len(set(long_syms) & prev_long) / float(max(len(long_syms), 1))
                long_turnovers.append(long_turnover)
            if prev_short is None:
                short_turnover = 1.0
            else:
                short_turnover = 1.0 - len(set(short_syms) & prev_short) / float(max(len(short_syms), 1))
                short_turnovers.append(short_turnover)
            prev_long = set(long_syms)
            prev_short = set(short_syms)

            daily_quantile_rets = {q: [] for q in labels}
            daily_long_rets = []
            daily_short_rets = []

            # 单位统一: 新口径用比值(close/open 等), 缺失标的按现金(1.0=0收益)计,
            # 再减 1.0 转成收益率; 旧口径 ret 本身即收益率, 缺失按 0.
            fill_val = 1.0 if use_open_exec else 0.0
            for day in held_days:
                # [Fix 2] reindex 填现金/0, 保持分母(权重)不变
                day_grp = grp[grp['date'] == day]
                day_indexed = day_grp.set_index('symbol')
                if use_open_exec and day == entry_day:
                    ratio_col = 'entry_ratio'      # 执行日: open→close, 不含隔夜
                elif use_open_exec:
                    ratio_col = 'held_ratio'       # 持有日: 前收→收盘
                else:
                    ratio_col = 'ret'              # 旧口径: 收盘→收盘 (前向)
                for q_name in labels:
                    q_ret = (day_indexed.reindex(quantile_syms[q_name])[ratio_col]
                             .fillna(fill_val).mean() - fill_val)
                    daily_quantile_rets[q_name].append(q_ret)
                long_ret = (day_indexed.reindex(long_syms)[ratio_col]
                            .fillna(fill_val).mean() - fill_val)
                short_ret = (day_indexed.reindex(short_syms)[ratio_col]
                             .fillna(fill_val).mean() - fill_val)
                daily_long_rets.append(long_ret)
                daily_short_rets.append(short_ret)

            # [Fix 1] 调仓日扣成本: 每腿收完整往返 = 换手率 × cost_bps.
            # 注意空头腿: r_S 是"按做多口径"记录的篮子收益, 空头 P&L = -r_S,
            # 故成本应加到 r_S 上 (使 L-S = r_L - r_S 净扣 c_L + c_S), 而非减.
            if self.cost_bps > 0 and len(daily_long_rets) > 0:
                daily_long_rets[0] -= long_turnover * self.cost_per_roundtrip
                daily_short_rets[0] += short_turnover * self.cost_per_roundtrip

            month_long_ret = float(np.prod(1 + np.array(daily_long_rets, dtype=np.float64))) - 1
            month_short_ret = float(np.prod(1 + np.array(daily_short_rets, dtype=np.float64))) - 1
            month_ls_ret = month_long_ret - month_short_ret

            ls_series.append((month, month_ls_ret))
            long_daily_all.extend(daily_long_rets)
            ls_daily_all.extend(
                dl - ds for dl, ds in zip(daily_long_rets, daily_short_rets))

            for q_name in labels:
                if daily_quantile_rets[q_name]:
                    month_q_ret = float(np.prod(1 + np.array(daily_quantile_rets[q_name], dtype=np.float64))) - 1
                    quantile_series[q_name].append(month_q_ret)

        if not ls_series:
            return None

        # ── L-S 月度指标 (月频, √12) ──
        months = [str(m) for m, _ in ls_series]
        ls_rets = np.array([r for _, r in ls_series], dtype=np.float64)
        ls_cum = np.cumprod(1 + ls_rets)
        ls_ann = float(ls_cum[-1] ** (12 / len(ls_cum)) - 1)
        ls_sharpe = float(ls_rets.mean() / (ls_rets.std() + 1e-8) * np.sqrt(12))
        ls_max_dd = _max_drawdown(ls_cum)

        # ── L-S 日频指标 (日频, √252) ──
        ls_arr = np.array(ls_daily_all, dtype=np.float64)
        ls_daily_nav = np.cumprod(1 + ls_arr)
        n_days = max(len(ls_arr), 1)
        ls_daily_ann = float(ls_daily_nav[-1] ** (252 / n_days) - 1)
        ls_daily_sharpe = float(ls_arr.mean() / (ls_arr.std() + 1e-8) * np.sqrt(252))
        ls_daily_max_dd = _max_drawdown(ls_daily_nav)

        # ── Long-only 日频指标 (日频, √252) ──
        long_arr = np.array(long_daily_all, dtype=np.float64)
        long_nav = np.cumprod(1 + long_arr)
        long_ann = float(long_nav[-1] ** (252 / n_days) - 1)
        long_sharpe = float(long_arr.mean() / (long_arr.std() + 1e-8) * np.sqrt(252))
        long_max_dd = _max_drawdown(long_nav)

        # [Fix 4] 主指标口径随 long_only 切换:
        #   - long_only: 日频 long-only
        #   - 默认:      L-S 日频 (月频 L-S 字段保留为 ls_* 向后兼容)
        if self.long_only:
            ann_ret, sharpe, max_dd = long_ann, long_sharpe, long_max_dd
        else:
            ann_ret, sharpe, max_dd = ls_daily_ann, ls_daily_sharpe, ls_daily_max_dd

        quantile_summary = {q: (float(np.mean(rets)) if rets else 0.0)
                            for q, rets in quantile_series.items()}
        avg_turnover = float(np.mean(long_turnovers)) if long_turnovers else 0.0
        avg_turnover_short = float(np.mean(short_turnovers)) if short_turnovers else 0.0

        strategy_label = "long-only" if self.long_only else "多空"
        logger.info(
            f"🏆 {fid} 学术回测({strategy_label}/{exec_label}): 年化={ann_ret:.2%}, "
            f"Sharpe={sharpe:.2f}, Ann L-S={ls_ann:.2%}, MaxDD={max_dd:.2%}, "
            f"月均换手={avg_turnover:.2%}, months={len(ls_series)}, days={n_days}")
        logger.info(f"📊 {fid} " + ", ".join(
            f"{q}={quantile_summary.get(q, 0):.4%}" for q in list(quantile_series.keys())))

        return {
            # 向后兼容键 (所有旧消费者零改动)
            'annualized_return': ann_ret,
            'sharpe_ratio': sharpe,
            'quantile_monthly_rets': quantile_summary,
            'cumulative_ls': ls_cum,
            # 新增指标
            'max_drawdown': max_dd,
            'ls_annualized_return': ls_ann,
            'ls_sharpe_ratio': ls_sharpe,
            'ls_max_drawdown': ls_max_dd,
            'ls_daily_annualized': ls_daily_ann,
            'ls_daily_sharpe': ls_daily_sharpe,
            'ls_daily_max_drawdown': ls_daily_max_dd,
            'ls_daily_returns': ls_arr,
            'cumulative_ls_daily': ls_daily_nav,
            'long_annualized_return': long_ann,
            'long_sharpe_ratio': long_sharpe,
            'long_max_drawdown': long_max_dd,
            'avg_monthly_turnover': avg_turnover,
            'avg_monthly_turnover_short': avg_turnover_short,
            'long_only': self.long_only,
            'n_months': len(ls_series),
            'n_days': n_days,
            'months': months,
            'cumulative_long': long_nav,
        }


def _max_drawdown(nav: np.ndarray) -> float:
    """从净值序列计算最大回撤 (负值表示回撤幅度)."""
    nav = np.asarray(nav, dtype=np.float64)
    if len(nav) < 2:
        return 0.0
    running_max = np.maximum.accumulate(nav)
    dd = nav / running_max - 1.0
    return float(np.min(dd))
