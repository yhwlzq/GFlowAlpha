import logging
import numpy as np
import pandas as pd
from alpha.config import Config

logger = logging.getLogger(__name__)

class UltimateDailyFeatureRegistry:
    @classmethod
    def build_ultimate_factors(cls, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("构建日频基础因子库...")
        df = df.sort_values(['symbol', 'date']).copy()
        factor_dfs = []

        for sym, grp in df.groupby('symbol'):
            c, o, h, l = grp['close'], grp['open'], grp['high'], grp['low']
            v, amt = grp['volume'], grp['amount']
            tr = grp.get('turn', pd.Series(np.nan, index=grp.index))
            feats = {}
            feats['date'] = grp['date'].values
            feats['symbol'] = sym

            vwap = amt / v.clip(lower=1)
            log_ret = np.log(c.clip(lower=1e-8) / c.shift(1).clip(lower=1e-8))
            hl_range = (h - l).clip(lower=1e-8)
            ret = c.pct_change(1)

            for d in [1, 2, 3, 5, 10, 20, 60]:
                feats[f'ret_{d}d'] = c.pct_change(d)
                feats[f'rev_{d}d'] = -feats[f'ret_{d}d']

            roll_max_252 = c.rolling(252).max()
            roll_min_252 = c.rolling(252).min()
            valid_252 = (~roll_max_252.isna()) & (roll_max_252 > 0) & (~roll_min_252.isna()) & (roll_min_252 > 0)
            feats['dist_52w_high'] = np.where(valid_252, c / roll_max_252 - 1, np.nan)
            feats['dist_52w_low'] = np.where(valid_252, c / roll_min_252 - 1, np.nan)

            feats['mom_smooth_20d'] = (c / c.shift(20) - 1) / (log_ret.rolling(20).std() * np.sqrt(20) + 1e-8)
            feats['info_discrete_20d'] = log_ret.rolling(20).mean() / (log_ret.rolling(20).std() + 1e-8)

            for d in [5, 10, 20, 60]:
                feats[f'vol_{d}d'] = log_ret.rolling(d).std() * np.sqrt(252)

            feats['skew_20d'] = log_ret.rolling(20).skew()
            feats['kurt_20d'] = log_ret.rolling(20).kurt()

            down_vol = log_ret.clip(upper=0).rolling(20).std() * np.sqrt(252)
            up_vol = log_ret.clip(lower=0).rolling(20).std() * np.sqrt(252)
            feats['vol_skew_ratio'] = down_vol / (up_vol + 1e-8)

            feats['max_ret_20d'] = log_ret.rolling(20).max()
            feats['min_ret_20d'] = log_ret.rolling(20).min()
            feats['extreme_vol_20d'] = feats['max_ret_20d'] - feats['min_ret_20d']

            feats['parkinson_vol_20d'] = np.sqrt(
                (np.log(h.clip(lower=1e-8) / l.clip(lower=1e-8)) ** 2).rolling(20).mean() / (4 * np.log(2))) * np.sqrt(252)

            feats['ivol_20d'] = (log_ret - log_ret.rolling(20).mean()).rolling(20).std() * np.sqrt(252)

            feats['amihud_5d'] = (log_ret.abs() / amt.clip(lower=1e8)).rolling(5).mean()
            feats['amihud_20d'] = (log_ret.abs() / amt.clip(lower=1e8)).rolling(20).mean()

            turnover_base_60 = v.rolling(60).mean().clip(lower=1)
            for d in [5, 20, 60]:
                feats[f'turnover_{d}d'] = (v / turnover_base_60).rolling(d).mean()
            feats['turnover_std_20d'] = (v / turnover_base_60).rolling(20).std()

            feats['illiq_spread'] = (h - l) / c
            feats['volume_shock_5d'] = v / v.rolling(20).mean().clip(lower=1)
            feats['amount_shock_5d'] = amt / amt.rolling(20).mean().clip(lower=1)
            feats['liquidity_var'] = (amt / c.clip(lower=1)).rolling(20).std()

            feats['clv'] = (2 * c - h - l) / hl_range
            feats['upper_shadow'] = (h - np.maximum(c, o)) / hl_range
            feats['lower_shadow'] = (np.minimum(c, o) - l) / hl_range
            feats['body_ratio'] = (c - o).abs() / hl_range
            feats['gap'] = (o - c.shift(1)) / c.shift(1).clip(lower=1e-8)

            feats['smart_money_10d'] = ((c - vwap) / c.clip(lower=1e-8)).rolling(10).sum()
            feats['smart_money_20d'] = ((c - vwap) / c.clip(lower=1e-8)).rolling(20).sum()
            feats['vwap_dev_5d'] = ((c - vwap) / c.clip(lower=1e-8)).rolling(5).mean()
            feats['vwap_dev_20d'] = ((c - vwap) / c.clip(lower=1e-8)).rolling(20).mean()

            feats['intraday_mom'] = (c - (h + l + o + c) / 4) / ((h + l + o + c) / 4).clip(lower=1e-8)
            feats['price_accel'] = ret - ret.shift(1)

            for d in [5, 10, 20]:
                feats[f'pv_corr_{d}d'] = c.rolling(d).corr(v)
                feats[f'logpv_corr_{d}d'] = log_ret.rolling(d).corr(np.log(v.clip(lower=1)))

            obv = (np.sign(c.diff()) * v).cumsum()
            feats['obv_slope_10d'] = obv.rolling(10).apply(
                lambda x: np.polyfit(np.arange(10), x, 1)[0], raw=True)

            feats['money_flow_10d'] = (feats['clv'] * v).rolling(10).sum()
            feats['money_flow_20d'] = (feats['clv'] * v).rolling(20).sum()
            feats['pv_divergence'] = c.pct_change(10) - v.pct_change(10)

            for d in [5, 10, 20, 60]:
                ma = c.rolling(d).mean()
                feats[f'bias_{d}d'] = (c - ma) / ma.clip(lower=1e-8)
                feats[f'ma_cross_{d}d'] = c / ma.clip(lower=1e-8) - 1

            for d in [6, 14, 24]:
                delta = c.diff()
                gain = delta.clip(lower=0).rolling(d).mean()
                loss = -delta.clip(upper=0).rolling(d).mean()
                feats[f'rsi_{d}'] = 100 - (100 / (1 + gain / loss.clip(lower=1e-8)))

            ema12 = c.ewm(span=12).mean()
            ema26 = c.ewm(span=26).mean()
            feats['macd_hist'] = ema12 - ema26
            feats['macd_signal'] = feats['macd_hist'].ewm(span=9).mean()
            feats['boll_width_20d'] = (c.rolling(20).std() * 2) / c.rolling(20).mean().clip(lower=1e-8)

            # === HS300 大盘增强因子 (华泰因子体系) ===
            # A. 换手率类 (6个)
            feats['turnover_std_5d'] = tr.rolling(5).std()
            feats['turnover_decay_20d'] = tr.rolling(20).apply(
                lambda x: x.autocorr() if len(x) > 1 else np.nan)
            feats['turnover_skew_20d'] = tr.rolling(20).skew()
            feats['turnover_ma_ratio_5_60'] = tr.rolling(5).mean() / tr.rolling(60).mean().clip(lower=1e-8)
            feats['turnover_illuminated_20d'] = tr.rolling(20).mean() * log_ret.rolling(20).std()
            feats['amt_turn_corr_20d'] = amt.rolling(20).corr(tr)

            # B. 技术因子 - 量价背离类 (5个)
            feats['volume_std_5d'] = (v / v.rolling(20).mean().clip(lower=1)).rolling(5).std()
            feats['close_volume_corr_change_10d'] = (
                c.rolling(10).corr(v) - c.shift(5).rolling(10).corr(v.shift(5))
            )
            feats['volume_skew_20d'] = v.rolling(20).skew()
            feats['ret_volume_corr_10d'] = ret.rolling(10).corr(v)
            feats['high_low_volume_corr_10d'] = hl_range.rolling(10).corr(v)

            # C. 波动率增强类 (4个)
            feats['vol_stability_20d'] = feats['vol_5d'] / feats['vol_20d'].clip(lower=1e-8)
            feats['downside_vol_20d'] = (ret.clip(upper=0) ** 2).rolling(20).mean()
            feats['vol_expansion_20d'] = feats['vol_20d'] / feats['vol_60d'].clip(lower=1e-8) - 1
            feats['resid_vol_20d'] = (log_ret - log_ret.rolling(20).mean()).rolling(20).std() * np.sqrt(252)

            factor_dfs.append(pd.DataFrame(feats))

        full_factors = pd.concat(factor_dfs, ignore_index=True)

        if Config.USE_RANK_FEATURES:
            # ============================================================
            #  横截面排名特征 (Cross-Sectional Rank)
            # ============================================================
            logger.info("构建横截面排名增强因子...")
            full_factors = full_factors.sort_values(['symbol', 'date']).reset_index(drop=True)

            rank_targets = [
                'ret_1d', 'ret_5d', 'ret_20d',
                'vol_20d', 'turnover_20d',
                'body_ratio', 'pv_divergence',
                'amihud_20d', 'dist_52w_high'
            ]

            for col in rank_targets:
                if col in full_factors.columns:
                    full_factors[f'rank_{col}'] = full_factors.groupby('date')[col].transform(
                        lambda x: (x.rank(pct=True, method='average') - 0.5) / 0.5)

            # ============================================================
            #  高阶交互特征 (Higher-Order Interactions)
            # ============================================================
            logger.info("构建高阶交互特征...")

            full_factors['vol_cone_5_20'] = full_factors['vol_5d'] / (full_factors['vol_20d'] + 1e-8)
            full_factors['pv_div_accel'] = full_factors.groupby('symbol')['pv_divergence'].transform(
                lambda x: x - x.shift(5))
            full_factors['body_ratio_mom'] = full_factors.groupby('symbol')['body_ratio'].transform(
                lambda x: x - x.shift(5))
            full_factors['extreme_up_days_5d'] = full_factors.groupby('symbol')['ret_1d'].transform(
                lambda x: (x > 0.05).rolling(5, min_periods=1).sum() / 5.0)
            full_factors['extreme_down_days_5d'] = full_factors.groupby('symbol')['ret_1d'].transform(
                lambda x: (x < -0.05).rolling(5, min_periods=1).sum() / 5.0)

        full_factors.fillna(0.0, inplace=True)
        for c in full_factors.columns:
            if full_factors[c].dtype == 'float64':
                full_factors[c] = full_factors[c].astype(np.float32)
        factor_cols = [c for c in full_factors.columns if c not in ['date', 'symbol']]
        assert Config.INITIAL_FEATURE_COLS == [], "build_ultimate_factors 不应被重复调用"
        Config.INITIAL_FEATURE_COLS = factor_cols
        logger.info(f"基础因子库构建完成，共 {len(factor_cols)} 个原子因子。")
        return full_factors


class LargeCapFeatureRegistry:
    """大盘股特征注册表 — 专为沪深 300 设计。

    与 UltimateDailyFeatureRegistry 的区别:
      - 删除 K 线形态 / 微观结构 / 超短期反转（大盘上无效）
      - 新增长周期动量 (120d/252d)
      - 新增市场相对特征（等权市场收益，超额 / Beta / 相对强度）
      - 新增质量/稳定性代理（夏普 / 卡玛 / 回撤）
      - 删除冗余因子，总量 ~60-65 个
    """

    @classmethod
    def build_large_cap_factors(cls, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("构建大盘因子库 (LargeCapFeatureRegistry)...")
        df = df.sort_values(['symbol', 'date']).copy()
        df['ret'] = df.groupby('symbol')['close'].pct_change(1)

        market_ret = cls._compute_market_return(df)
        df = df.merge(market_ret, on='date', how='left')

        factor_dfs = []
        for sym, grp in df.groupby('symbol'):
            c, o, h, l = grp['close'], grp['open'], grp['high'], grp['low']
            v, amt, tr = grp['volume'], grp['amount'], grp.get('turn', pd.Series(np.nan, index=grp.index))
            mkt_ret_series = grp['market_ret']

            feats = {}
            feats['date'] = grp['date'].values
            feats['symbol'] = sym

            vwap = amt / v.clip(lower=1)
            log_ret = np.log(c.clip(lower=1e-8) / c.shift(1).clip(lower=1e-8))
            hl_range = (h - l).clip(lower=1e-8)
            ret_1d = c.pct_change(1)

            # === 1. 动量 / 反转 (保留 5d+, 删除 1d/2d/3d) ===
            for d in [5, 10, 20, 60]:
                feats[f'ret_{d}d'] = c.pct_change(d)
                feats[f'rev_{d}d'] = -feats[f'ret_{d}d']
            feats['ret_120d'] = c.pct_change(120)
            feats['ret_252d'] = c.pct_change(252)

            # === 2. 波动率 ===
            for d in [5, 10, 20, 60]:
                feats[f'vol_{d}d'] = log_ret.rolling(d).std() * np.sqrt(252)

            feats['parkinson_vol_20d'] = np.sqrt(
                (np.log(h.clip(lower=1e-8) / l.clip(lower=1e-8)) ** 2).rolling(20).mean() / (4 * np.log(2))) * np.sqrt(252)

            feats['vol_skew_ratio'] = (
                log_ret.clip(upper=0).rolling(20).std() * np.sqrt(252)) / (
                log_ret.clip(lower=0).rolling(20).std() * np.sqrt(252) + 1e-8)

            feats['vol_stability_20d'] = feats['vol_5d'] / feats['vol_20d'].clip(lower=1e-8)
            feats['vol_expansion_20d'] = feats['vol_20d'] / feats['vol_60d'].clip(lower=1e-8) - 1
            feats['downside_vol_20d'] = (ret_1d.clip(upper=0) ** 2).rolling(20).mean()

            max_vol_252 = log_ret.rolling(20).std().rolling(252).max() * np.sqrt(252)
            feats['vol_percentile_252d'] = feats['vol_20d'] / max_vol_252.clip(lower=1e-8)

            # === 3. 分布特征 ===
            feats['skew_20d'] = log_ret.rolling(20).skew()
            feats['kurt_20d'] = log_ret.rolling(20).kurt()
            feats['max_ret_20d'] = log_ret.rolling(20).max()
            feats['min_ret_20d'] = log_ret.rolling(20).min()

            # === 4. 流动性 / 换手率 ===
            feats['amihud_5d'] = (log_ret.abs() / amt.clip(lower=1e8)).rolling(5).mean()
            feats['amihud_20d'] = (log_ret.abs() / amt.clip(lower=1e8)).rolling(20).mean()

            turnover_base_60 = v.rolling(60).mean().clip(lower=1)
            for d in [5, 20, 60]:
                feats[f'turnover_{d}d'] = (v / turnover_base_60).rolling(d).mean()
            feats['turnover_std_20d'] = (v / turnover_base_60).rolling(20).std()
            feats['turnover_ma_ratio_5_60'] = tr.rolling(5).mean() / tr.rolling(60).mean().clip(lower=1e-8)
            feats['turnover_illuminated_20d'] = tr.rolling(20).mean() * log_ret.rolling(20).std()
            feats['amt_turn_corr_20d'] = amt.rolling(20).corr(tr)

            feats['illiq_spread'] = (h - l) / c
            feats['volume_shock_5d'] = v / v.rolling(20).mean().clip(lower=1)
            feats['amount_shock_5d'] = amt / amt.rolling(20).mean().clip(lower=1)
            feats['liquidity_var'] = (amt / c.clip(lower=1)).rolling(20).std()
            feats['volume_std_5d'] = (v / v.rolling(20).mean().clip(lower=1)).rolling(5).std()

            # === 5. 相关性 ===
            for d in [5, 10, 20]:
                feats[f'pv_corr_{d}d'] = c.rolling(d).corr(v)
            feats['ret_volume_corr_10d'] = ret_1d.rolling(10).corr(v)
            feats['high_low_volume_corr_10d'] = hl_range.rolling(10).corr(v)
            feats['pv_divergence'] = c.pct_change(10) - v.pct_change(10)

            # === 6. 技术指标 ===
            for d in [5, 10, 20, 60]:
                ma = c.rolling(d).mean()
                feats[f'bias_{d}d'] = (c - ma) / ma.clip(lower=1e-8)
                feats[f'ma_cross_{d}d'] = c / ma.clip(lower=1e-8) - 1

            for d in [6, 14, 24]:
                delta = c.diff()
                gain = delta.clip(lower=0).rolling(d).mean()
                loss = -delta.clip(upper=0).rolling(d).mean()
                feats[f'rsi_{d}'] = 100 - (100 / (1 + gain / loss.clip(lower=1e-8)))

            ema12 = c.ewm(span=12).mean()
            ema26 = c.ewm(span=26).mean()
            feats['macd_hist'] = ema12 - ema26
            feats['macd_signal'] = feats['macd_hist'].ewm(span=9).mean()
            feats['boll_width_20d'] = (c.rolling(20).std() * 2) / c.rolling(20).mean().clip(lower=1e-8)

            # === 7. 52 周位置 (大盘上最强的量价信号) ===
            roll_max_252 = c.rolling(252).max()
            roll_min_252 = c.rolling(252).min()
            valid_252 = (~roll_max_252.isna()) & (roll_max_252 > 0) & (~roll_min_252.isna()) & (roll_min_252 > 0)
            feats['dist_52w_high'] = np.where(valid_252, c / roll_max_252 - 1, np.nan)
            feats['dist_52w_low'] = np.where(valid_252, c / roll_min_252 - 1, np.nan)

            # === 8. 市场相对特征 (关键新增) ===
            for d in [5, 20, 60]:
                stock_ret = c.pct_change(d)
                mkt_ret_d = mkt_ret_series.rolling(d).apply(
                    lambda x: np.prod(1 + x.values) - 1 if len(x) == d else np.nan, raw=False)
                feats[f'excess_ret_{d}d'] = stock_ret - mkt_ret_d

            feats['relative_strength_20d'] = c.pct_change(20) / (
                mkt_ret_series.rolling(20).apply(
                    lambda x: np.prod(1 + x.values) - 1, raw=False).clip(lower=1e-8))

            stock_daily_ret = ret_1d
            mkt_daily_ret = mkt_ret_series
            cov_60 = stock_daily_ret.rolling(60).cov(mkt_daily_ret)
            var_60 = mkt_daily_ret.rolling(60).var().clip(lower=1e-8)
            feats['market_beta_60d'] = cov_60 / var_60

            # === 9. 质量 / 稳定性代理 ===
            feats['info_discrete_20d'] = log_ret.rolling(20).mean() / (log_ret.rolling(20).std() + 1e-8)
            feats['mom_smooth_20d'] = (c / c.shift(20) - 1) / (log_ret.rolling(20).std() * np.sqrt(20) + 1e-8)

            feats['sharpe_60d'] = ret_1d.rolling(60).mean() / ret_1d.rolling(60).std().clip(lower=1e-8)
            roll_max_60 = c.rolling(60).max()
            drawdown_60 = c / roll_max_60 - 1
            feats['max_drawdown_60d'] = drawdown_60.rolling(60).min()
            feats['calmar_252d'] = c.pct_change(252) / (drawdown_60.rolling(252).min().clip(lower=1e-8)).abs()

            pos_sum = ret_1d.clip(lower=0).rolling(60).sum()
            neg_sum = (-ret_1d.clip(upper=0)).rolling(60).sum().clip(lower=1e-8)
            feats['gain_loss_60d'] = pos_sum / neg_sum

            factor_dfs.append(pd.DataFrame(feats))

        full_factors = pd.concat(factor_dfs, ignore_index=True)

        full_factors.fillna(0.0, inplace=True)
        for c in full_factors.columns:
            if full_factors[c].dtype == 'float64':
                full_factors[c] = full_factors[c].astype(np.float32)
        factor_cols = [c for c in full_factors.columns if c not in ['date', 'symbol', 'market_ret', 'ret']]
        assert Config.INITIAL_FEATURE_COLS == [], "build_large_cap_factors 不应被重复调用"
        Config.INITIAL_FEATURE_COLS = factor_cols
        logger.info(f"大盘因子库构建完成，共 {len(factor_cols)} 个原子因子。")
        return full_factors

    @staticmethod
    def _compute_market_return(df: pd.DataFrame) -> pd.DataFrame:
        df_tmp = df[['date', 'symbol', 'ret']].copy()
        daily_mkt = df_tmp.groupby('date')['ret'].mean().reset_index()
        daily_mkt.columns = ['date', 'market_ret']
        daily_mkt['date'] = pd.to_datetime(daily_mkt['date'])
        return daily_mkt
