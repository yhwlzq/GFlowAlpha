import logging
import numpy as np
import pandas as pd
from alpha.config import Config

logger = logging.getLogger(__name__)

try:
    from alpha.mining.safe_ops import SafeOps
except ImportError:
    SafeOps = None
    logger.warning("SafeOps 未找到，部分高级回归与技术指标将跳过。")

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

            feats['amihud_5d'] = (log_ret.abs() / amt.clip(lower=1e-8)).rolling(5).mean()
            feats['amihud_20d'] = (log_ret.abs() / amt.clip(lower=1e-8)).rolling(20).mean()

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
                'amihud_20d'
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

        full_factors.replace([np.inf, -np.inf], np.nan, inplace=True)
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
      - 删除 K 线形态 / 微观结构（大盘上无效）
      - 保留动量/反转全窗口 (1d/2d/3d/5d/...+60d), T+1 短周期反转信号
      - 新增市场相对特征（等权市场收益，超额 / Beta / 相对强度）
      - 新增质量/稳定性代理（夏普 / 卡玛 / 回撤）
      - 删除冗余因子，总量 ~60-65 个
    """

    @classmethod
    def build_large_cap_factors(cls, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("构建大盘因子库 (LargeCapFeatureRegistry) - GFlowNet 优化版...")
        df = df.sort_values(['symbol', 'date']).copy()
        df['ret'] = df.groupby('symbol')['close'].pct_change(1)
        df['log_ret'] = np.log(df['close'].clip(lower=1e-8) / df.groupby('symbol')['close'].shift(1).clip(lower=1e-8))

        market_ret = cls._compute_market_return(df)
        df = df.merge(market_ret, on='date', how='left')

        factor_dfs = []
        for sym, grp in df.groupby('symbol'):
            c, o, h, l = grp['close'], grp['open'], grp['high'], grp['low']
            v, amt, tr = grp['volume'], grp['amount'], grp.get('turn', pd.Series(np.nan, index=grp.index))
            log_ret, ret_1d, mkt_ret = grp['log_ret'], grp['ret'], grp['market_ret']
            vwap = amt / v.clip(lower=1)
            hl_range = (h - l).clip(lower=1e-8)

            feats = {'symbol': sym, 'date': grp['date'].values}

            # === 1. 动量 + 反转 ===
            for d in [1, 2, 3, 5, 10, 20, 60]:
                feats[f'ret_{d}d'] = c.pct_change(d)
                feats[f'rev_{d}d'] = -feats[f'ret_{d}d']

            # === 1b. K线形态 (精简) ===
            feats['body_ratio'] = (c - o).abs() / hl_range

            # === 2. 波动率 ===
            for d in [5, 10, 20, 60]:
                feats[f'vol_{d}d'] = log_ret.rolling(d).std() * np.sqrt(252)

            feats['parkinson_vol_20d'] = np.sqrt(
                (np.log(h.clip(lower=1e-8) / l.clip(lower=1e-8)) ** 2).rolling(20).mean() / (4 * np.log(2))
            ) * np.sqrt(252)

            feats['vol_skew_ratio'] = log_ret.clip(upper=0).rolling(20).std() / (log_ret.clip(lower=0).rolling(20).std() + 1e-8)
            feats['vol_stability_20d'] = feats['vol_5d'] / feats['vol_20d'].clip(lower=1e-8)
            feats['vol_expansion_20d'] = feats['vol_20d'] / feats['vol_60d'].clip(lower=1e-8) - 1
            feats['downside_vol_20d'] = (ret_1d.clip(upper=0) ** 2).rolling(20).mean()

            # === 3. 分布特征 ===
            feats['skew_20d'] = log_ret.rolling(20).skew()
            feats['kurt_20d'] = log_ret.rolling(20).kurt()
            feats['max_ret_20d'] = log_ret.rolling(20).max()
            feats['min_ret_20d'] = log_ret.rolling(20).min()

            # === 4. 流动性 / 换手率 ===
            feats['amihud_5d'] = (log_ret.abs() / amt.clip(lower=1e-8)).rolling(5).mean()
            feats['amihud_20d'] = (log_ret.abs() / amt.clip(lower=1e-8)).rolling(20).mean()

            turnover_base_60 = v.rolling(60).mean().clip(lower=1)
            for d in [5, 20, 60]:
                feats[f'turnover_{d}d'] = (v / turnover_base_60).rolling(d).mean()
            feats['turnover_std_20d'] = (v / turnover_base_60).rolling(20).std()
            feats['turnover_ma_ratio_5_60'] = tr.rolling(5).mean() / tr.rolling(60).mean().clip(lower=1e-8)
            feats['turnover_illuminated_20d'] = tr.rolling(20).mean() * log_ret.rolling(20).std()
            feats['amt_turn_corr_20d'] = amt.rolling(20).corr(tr)

            feats['illiq_spread'] = hl_range / c.clip(lower=1e-8)
            feats['volume_shock_5d'] = v / v.rolling(20).mean().clip(lower=1)
            feats['amount_shock_5d'] = amt / amt.rolling(20).mean().clip(lower=1)
            feats['liquidity_var'] = (amt / c.clip(lower=1)).rolling(20).std()
            feats['volume_std_5d'] = (v / v.rolling(20).mean().clip(lower=1)).rolling(5).std()
            feats['amt_concentration_20d'] = amt.rolling(20).apply(
                lambda x: np.nansum(np.sort(x)[-5:]) / np.nansum(x) if np.nansum(x) > 0 else np.nan, raw=True)
            feats['amihud_ratio_5_20'] = feats['amihud_5d'] / feats['amihud_20d'].clip(lower=1e-8)

            # === 5. 相关性 ===
            for d in [5, 10, 20]:
                feats[f'pv_corr_{d}d'] = c.rolling(d).corr(v)
            feats['ret_volume_corr_10d'] = ret_1d.rolling(10).corr(v)
            feats['high_low_volume_corr_10d'] = hl_range.rolling(10).corr(v)
            feats['pv_divergence'] = c.pct_change(10) - v.pct_change(10)
            feats['vol_lead_corr_10d'] = v.shift(1).rolling(10).corr(ret_1d)

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

            bb_mid = c.rolling(20).mean()
            bb_std = c.rolling(20).std()
            bb_upper = bb_mid + 2 * bb_std
            bb_lower = bb_mid - 2 * bb_std
            feats['boll_width_20d'] = (bb_std * 2) / bb_mid.clip(lower=1e-8)
            feats['boll_pctb_20d'] = (c - bb_lower) / (bb_upper - bb_lower).clip(lower=1e-8)

            tp = (h + l + c) / 3.0
            feats['cci_20d'] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std().clip(lower=1e-8))

            h14 = h.rolling(14).max()
            l14 = l.rolling(14).min()
            feats['williams_r_14d'] = (h14 - c) / (h14 - l14).clip(lower=1e-8) * -100

            # === 7. 市场相对特征 (向量化优化) ===
            for d in [5, 20, 60]:
                stock_ret_d = np.exp(log_ret.rolling(d).sum()) - 1
                mkt_ret_d = np.exp(mkt_ret.rolling(d).sum()) - 1
                feats[f'excess_ret_{d}d'] = stock_ret_d - mkt_ret_d

            feats['relative_strength_20d'] = (np.exp(log_ret.rolling(20).sum()) - 1) / (
                np.exp(mkt_ret.rolling(20).sum()).clip(lower=1e-8) - 1 + 1e-8)

            feats['market_beta_60d'] = ret_1d.rolling(60).cov(mkt_ret) / mkt_ret.rolling(60).var().clip(lower=1e-8)

            # === 8. 质量 / 稳定性代理 ===
            feats['info_discrete_20d'] = log_ret.rolling(20).mean() / (log_ret.rolling(20).std() + 1e-8)
            feats['mom_smooth_20d'] = (c / c.shift(20) - 1) / (log_ret.rolling(20).std() * np.sqrt(20) + 1e-8)
            feats['sharpe_60d'] = ret_1d.rolling(60).mean() / ret_1d.rolling(60).std().clip(lower=1e-8)

            roll_max_60 = c.rolling(60).max()
            drawdown_60 = c / roll_max_60 - 1
            feats['max_drawdown_60d'] = drawdown_60.rolling(60).min()

            pos_sum = ret_1d.clip(lower=0).rolling(60).sum()
            neg_sum = (-ret_1d.clip(upper=0)).rolling(60).sum().clip(lower=1e-8)
            feats['gain_loss_60d'] = pos_sum / neg_sum

            # === 9. 市场回归 Alpha / Beta 增强 (SafeOps 保护) ===
            if SafeOps is not None:
                ret_vals = ret_1d.values.astype(np.float64)
                mkt_vals = mkt_ret.values.astype(np.float64)
                for d in [60, 120, 250]:
                    feats[f'rolling_alpha_{d}d'] = SafeOps.rolling_ols_intercept(ret_vals, mkt_vals, d)
                feats['market_beta_120d'] = SafeOps.rolling_ols_slope(ret_vals, mkt_vals, 120)
                feats['beta_stability_60d'] = feats['market_beta_60d'] / np.clip(feats['market_beta_120d'], 1e-8, None)

                predicted_60 = feats['market_beta_60d'] * mkt_ret + pd.Series(
                    np.asarray(SafeOps.rolling_ols_intercept(ret_vals, mkt_vals, 60)), index=ret_1d.index)
                feats['ivol_60d'] = (ret_1d - predicted_60).rolling(60).std() * np.sqrt(252)

                feats['rsrs_20d'] = SafeOps.safe_rsrs(h, l, 20)

                obv = pd.Series(SafeOps.safe_obv(c, v), index=c.index)
                feats['obv_trend_20d'] = obv.rolling(20).apply(
                    lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] / (np.std(x) + 1e-8) if len(x) == 20 else np.nan, raw=True)

                feats['mfi_14d'] = SafeOps.safe_mfi(h, l, c, v, 14)

            # === 10. 隔夜 / 日内模式 ===
            feats['overnight_ret_20d'] = (o / c.shift(1).clip(lower=1e-8) - 1).rolling(20).mean()
            feats['intraday_ret_20d'] = (c / o.clip(lower=1e-8) - 1).rolling(20).mean()
            feats['close_vwap_dev_20d'] = ((c - vwap) / vwap.clip(lower=1e-8)).rolling(20).mean()
            feats['open_efficiency_20d'] = ((c - o) / o.clip(lower=1e-8)).rolling(20).std()

            # === 11. 分布 / 尾部风险 (优化版：剔除大盘股无效的极端计数，替换为特异质收益) ===
            # feats['extreme_up_ratio_20d'] 和 feats['extreme_down_ratio_20d'] 已删除
            # 原因: 截面变异度≈0，导致 FM 回归设计矩阵奇异 (n_periods=0)

            # 特异质收益 (Idiosyncratic Return) - 大盘股 Alpha 的核心来源
            if SafeOps is not None:
                feats['idio_ret_20d'] = SafeOps.rolling_ols_residual(ret_vals, mkt_vals, 20)
            else:
                feats['idio_ret_20d'] = ret_1d - mkt_ret.rolling(20).mean()

            # 成交额动量 (Amount Momentum) - 大盘股资金流向的更好代理
            feats['amt_mom_20d'] = amt.rolling(20).mean() / amt.rolling(60).mean().clip(lower=1e-8)

            # 量价协方差 (Price-Volume Covariance) - 捕捉大盘股的主力建仓/派发
            feats['pv_cov_20d'] = ret_1d.rolling(20).cov(np.log(v.clip(lower=1)))

            feats['var_95_20d'] = log_ret.rolling(20).quantile(0.05)

            # === 12. 换手率 / 流动性增强 ===
            feats['turnover_mom_20d'] = tr / tr.shift(20).clip(lower=1e-8) - 1
            feats['turnover_vol_20d'] = tr.rolling(20).std() / tr.rolling(20).mean().clip(lower=1e-8)

            factor_dfs.append(pd.DataFrame(feats))

        full_factors = pd.concat(factor_dfs, ignore_index=True)
        full_factors.replace([np.inf, -np.inf], np.nan, inplace=True)

        # ==========================================================
        # 截面标准化 (MAD Winsorize + Z-score)
        # ==========================================================
        exclude_cols = {'date', 'symbol', 'market_ret', 'ret', 'log_ret'}
        factor_cols = [
            c for c in full_factors.columns
            if c not in exclude_cols and pd.api.types.is_numeric_dtype(full_factors[c])
        ]

        logger.info(f"执行截面标准化 (MAD Winsorize + Z-score) 于 {len(factor_cols)} 个因子...")
        for col in factor_cols:
            full_factors[col] = full_factors.groupby('date')[col].transform(lambda x: x.fillna(x.median()))
            median = full_factors.groupby('date')[col].transform('median')
            mad = full_factors.groupby('date')[col].transform(lambda x: np.median(np.abs(x - np.median(x))))
            upper_bound = median + 3 * 1.4826 * mad
            lower_bound = median - 3 * 1.4826 * mad
            full_factors[col] = np.clip(full_factors[col], lower_bound, upper_bound)
            mean = full_factors.groupby('date')[col].transform('mean')
            std = full_factors.groupby('date')[col].transform('std').clip(lower=1e-8)
            full_factors[col] = (full_factors[col] - mean) / std

        for c in full_factors.columns:
            if full_factors[c].dtype == 'float64':
                full_factors[c] = full_factors[c].astype(np.float32)

        if hasattr(Config, 'INITIAL_FEATURE_COLS') and not Config.INITIAL_FEATURE_COLS:
            Config.INITIAL_FEATURE_COLS = factor_cols
        logger.info(f"大盘因子库构建完成，共 {len(factor_cols)} 个已标准化的原子因子。")
        return full_factors

    @staticmethod
    def _compute_market_return(df: pd.DataFrame) -> pd.DataFrame:
        df_tmp = df[['date', 'symbol', 'ret']].copy()
        daily_mkt = df_tmp.groupby('date')['ret'].median().reset_index()
        daily_mkt.columns = ['date', 'market_ret']
        daily_mkt['date'] = pd.to_datetime(daily_mkt['date'])
        return daily_mkt


class TimingFeatureRegistry:
    """时序择时因子库 — 从成分股 OHLCV 计算市场级因子 (含截面统计量)."""

    @staticmethod
    def build_timing_factors(df_stocks: pd.DataFrame) -> pd.DataFrame:
        """从成分股日线数据构建时序择时因子.

        Args:
            df_stocks: 成分股日线数据, 需含 date/symbol/open/high/low/close/volume/amount.

        Returns:
            DataFrame, index=date, columns=因子名.
        """
        logger.info("构建时序择时因子库 (TimingFeatureRegistry)...")
        df_raw = df_stocks.copy()
        df_raw['date'] = pd.to_datetime(df_raw['date'])

        # ── 从成分股计算等权指数 (仅用于 target 和部分 TS 因子) ──
        df = df_raw.groupby('date').agg(
            open=('open', 'mean'),
            high=('high', 'mean'),
            low=('low', 'mean'),
            close=('close', 'mean'),
            volume=('volume', 'mean'),
            amount=('amount', 'mean'),
        ).reset_index().sort_values('date').reset_index(drop=True)

        c = df['close'].values.astype(np.float64)
        o = df['open'].values.astype(np.float64)
        h = df['high'].values.astype(np.float64)
        l = df['low'].values.astype(np.float64)
        v = df['volume'].values.astype(np.float64)
        amt = df['amount'].values.astype(np.float64)
        ret = pd.Series(c).pct_change().values
        log_ret = np.log(c / np.roll(c, 1))
        log_ret[0] = 0.0

        feats = {'date': df['date'].values}

        def _rolling_std(arr, w):
            s = pd.Series(arr)
            return s.rolling(w, min_periods=max(1, w // 2)).std().values

        def _rolling_mean(arr, w):
            s = pd.Series(arr)
            return s.rolling(w, min_periods=max(1, w // 2)).mean().values

        def _rolling_sum(arr, w):
            s = pd.Series(arr)
            return s.rolling(w, min_periods=max(1, w // 2)).sum().values

        def _rolling_max(arr, w):
            s = pd.Series(arr)
            return s.rolling(w, min_periods=max(1, w // 2)).max().values

        def _rolling_min(arr, w):
            s = pd.Series(arr)
            return s.rolling(w, min_periods=max(1, w // 2)).min().values

        def _rolling_skew(arr, w):
            s = pd.Series(arr)
            return s.rolling(w, min_periods=max(1, w // 2)).skew().values

        def _rolling_kurt(arr, w):
            s = pd.Series(arr)
            return s.rolling(w, min_periods=max(1, w // 2)).kurt().values

        def _ema(arr, span):
            s = pd.Series(arr)
            return s.ewm(span=span, adjust=False).mean().values

        # ── 动量因子 ──
        for w in [1, 3, 5, 10, 20, 60]:
            feats[f'ret_{w}d'] = pd.Series(c).pct_change(w).values

        # ── 反转因子 ──
        for w in [5, 10, 20, 60]:
            feats[f'rev_{w}d'] = -pd.Series(c).pct_change(w).values

        # ── 波动率因子 ──
        for w in [5, 10, 20, 60]:
            feats[f'vol_{w}d'] = _rolling_std(ret, w)

        # Parkinson volatility
        hl_ratio = np.log(h / l)
        feats['parkinson_vol_20d'] = np.sqrt(_rolling_sum(hl_ratio ** 2, 20) / (20 * np.log(2)))
        feats['parkinson_vol_60d'] = np.sqrt(_rolling_sum(hl_ratio ** 2, 60) / (60 * np.log(2)))

        # Garman-Klass volatility
        gk = 0.5 * np.log(h / l) ** 2 - (2 * np.log(2) - 1) * np.log(c / o) ** 2
        feats['garman_klass_vol_20d'] = np.sqrt(_rolling_sum(gk, 20) / 20)
        feats['garman_klass_vol_60d'] = np.sqrt(_rolling_sum(gk, 60) / 60)

        # 波动率比率
        vol_5 = pd.Series(ret).rolling(5).std().values
        vol_20 = pd.Series(ret).rolling(20).std().values
        vol_60 = pd.Series(ret).rolling(60).std().values
        feats['vol_ratio_5_20'] = vol_5 / np.clip(vol_20, 1e-8, None)
        feats['vol_ratio_20_60'] = vol_20 / np.clip(vol_60, 1e-8, None)
        feats['vol_expansion_20d'] = vol_20 / np.clip(_rolling_std(ret, 60), 1e-8, None)

        # ── 均线偏离因子 ──
        for w in [5, 10, 20, 60, 120]:
            ma = _rolling_mean(c, w)
            feats[f'bias_{w}d'] = (c - ma) / np.clip(ma, 1e-8, None)

        # ── RSI ──
        for w in [6, 14, 24]:
            delta = pd.Series(c).diff()
            gain = delta.clip(lower=0).rolling(w).mean()
            loss = (-delta.clip(upper=0)).rolling(w).mean()
            rs = gain / loss.clip(lower=1e-8)
            feats[f'rsi_{w}d'] = (100 - 100 / (1 + rs)).values

        # ── 布林带 ──
        ma20 = _rolling_mean(c, 20)
        std20 = _rolling_std(c, 20)
        upper = ma20 + 2 * std20
        lower = ma20 - 2 * std20
        feats['boll_width_20d'] = (upper - lower) / np.clip(ma20, 1e-8, None)
        feats['boll_pctb_20d'] = (c - lower) / np.clip(upper - lower, 1e-8, None)

        # ── MACD ──
        ema12 = _ema(c, 12)
        ema26 = _ema(c, 26)
        macd_line = ema12 - ema26
        signal_line = _ema(macd_line, 9)
        feats['macd_hist'] = macd_line - signal_line
        feats['macd_signal'] = signal_line / np.clip(c, 1e-8, None)

        # ── CCI ──
        tp = (h + l + c) / 3
        tp_ma20 = _rolling_mean(tp, 20)
        tp_md20 = pd.Series(tp).rolling(20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
        feats['cci_20d'] = (tp - tp_ma20) / np.clip(0.015 * tp_md20, 1e-8, None)

        # ── 成交量因子 ──
        vol_ma5 = _rolling_mean(v, 5)
        vol_ma20 = _rolling_mean(v, 20)
        vol_ma60 = _rolling_mean(v, 60)
        feats['volume_ratio_5_20'] = vol_ma5 / np.clip(vol_ma20, 1e-8, None)
        feats['volume_ratio_10_60'] = _rolling_mean(v, 10) / np.clip(vol_ma60, 1e-8, None)
        feats['volume_shock_5d'] = v / np.clip(vol_ma20, 1e-8, None)

        # OBV slope
        obv = np.cumsum(np.sign(ret) * v)
        obv_slope = pd.Series(obv).rolling(20).apply(
            lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] / (np.std(x) + 1e-8) if len(x) > 1 else 0,
            raw=True
        ).values
        feats['obv_slope_20d'] = obv_slope

        # Amount momentum
        feats['amt_mom_20d'] = pd.Series(amt).pct_change(20).values

        # ── 动量质量因子 ──
        for w in [20, 60]:
            r = pd.Series(ret)
            feats[f'sharpe_{w}d'] = (r.rolling(w).mean() / r.rolling(w).std().clip(lower=1e-8)).values

        # Sortino ratio
        for w in [20, 60]:
            r = pd.Series(ret)
            down_vol = r.clip(upper=0).rolling(w).std().clip(lower=1e-8)
            feats[f'sortino_{w}d'] = (r.rolling(w).mean() / down_vol).values

        # Gain/Loss ratio
        for w in [20, 60]:
            r = pd.Series(ret)
            gain = r.clip(lower=0).rolling(w).mean()
            loss = (-r.clip(upper=0)).rolling(w).mean().clip(lower=1e-8)
            feats[f'gain_loss_{w}d'] = (gain / loss).values

        # Max/Min return
        feats['max_ret_20d'] = _rolling_max(ret, 20)
        feats['min_ret_20d'] = _rolling_min(ret, 20)
        feats['max_ret_60d'] = _rolling_max(ret, 60)
        feats['min_ret_60d'] = _rolling_min(ret, 60)

        # ── 趋势强度 ──
        # Aroon Oscillator (25-day)
        aroon_up = pd.Series(h).rolling(25).apply(lambda x: np.argmax(x) / 24, raw=True).values
        aroon_down = pd.Series(l).rolling(25).apply(lambda x: np.argmin(x) / 24, raw=True).values
        feats['aroon_osc_25d'] = aroon_up - aroon_down

        # Linear regression slope / std
        for w in [20, 60]:
            slope = pd.Series(c).rolling(w).apply(
                lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] / (np.std(x) + 1e-8) if len(x) > 1 else 0,
                raw=True
            ).values
            feats[f'trend_slope_{w}d'] = slope

        # ── 偏度/峰度 ──
        feats['skew_20d'] = _rolling_skew(ret, 20)
        feats['kurt_20d'] = _rolling_kurt(ret, 20)

        # ── 组合因子 ──
        r20 = pd.Series(ret)
        feats['momentum_vol_ratio'] = (r20.rolling(20).mean() * 20) / np.clip(r20.rolling(20).std() * np.sqrt(20), 1e-8, None)
        bias20 = feats['bias_20d']
        vol20 = feats['vol_20d']
        feats['reversal_vol_ratio'] = bias20 / np.clip(vol20, 1e-8, None)

        # Williams %R
        for w in [14]:
            hh = pd.Series(h).rolling(w).max()
            ll = pd.Series(l).rolling(w).min()
            feats[f'williams_r_{w}d'] = ((hh - c) / np.clip(hh - ll, 1e-8, None) * -100)

        # MFI (Money Flow Index)
        tp_mfi = (h + l + c) / 3
        mf = tp_mfi * v
        pos_mf = pd.Series(np.where(tp_mfi > np.roll(tp_mfi, 1), mf, 0)).rolling(14).sum()
        neg_mf = pd.Series(np.where(tp_mfi <= np.roll(tp_mfi, 1), mf, 0)).rolling(14).sum()
        mfr = pos_mf / neg_mf.clip(lower=1e-8)
        feats['mfi_14d'] = (100 - 100 / (1 + mfr)).values

        # ══════════════════════════════════════════════════════════════
        # 新增特征: 宏观/市场微观结构 + Regime 特征
        # ══════════════════════════════════════════════════════════════

        # ── 波动率期限结构 (Volatility Term Structure) ──
        vol_5d = pd.Series(ret).rolling(5).std()
        vol_20d = pd.Series(ret).rolling(20).std()
        vol_60d_s = pd.Series(ret).rolling(60).std()
        feats['vol_term_5_60'] = (vol_5d / vol_60d_s.clip(lower=1e-8)).values
        feats['vol_term_20_60'] = (vol_20d / vol_60d_s.clip(lower=1e-8)).values
        feats['vol_term_slope'] = ((vol_5d - vol_20d) / vol_60d_s.clip(lower=1e-8)).values

        # ── 波动率的波动率 (Vol of Vol) ──
        feats['vol_of_vol_20d'] = _rolling_std(vol_20d.values, 20)
        feats['vol_of_vol_60d'] = _rolling_std(vol_20d.values, 60)

        # ── 成交量-价格背离 (Volume-Price Divergence) ──
        # 价涨量缩 或 价跌量增 → 反转信号
        ret_5d = pd.Series(ret).rolling(5).mean()
        vol_chg_5d = pd.Series(v).pct_change(5)
        feats['vol_price_div_5d'] = (ret_5d.values - vol_chg_5d.values)

        # 成交量动量加速度
        vol_mom_10d = pd.Series(v).pct_change(10)
        vol_mom_20d = pd.Series(v).pct_change(20)
        feats['vol_mom_accel'] = (vol_mom_10d - vol_mom_20d).values

        # ── 累积/派发模式 (Accumulation/Distribution Proxy) ──
        # CLV = [(close - low) - (high - close)] / (high - low)
        clv = ((c - l) - (h - c)) / np.clip(h - l, 1e-8, None)
        ad_line = np.cumsum(clv * v)
        feats['ad_line_slope_20d'] = pd.Series(ad_line).rolling(20).apply(
            lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] / (np.std(x) + 1e-8) if len(x) > 1 else 0,
            raw=True
        ).values
        feats['ad_line_slope_60d'] = pd.Series(ad_line).rolling(60).apply(
            lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] / (np.std(x) + 1e-8) if len(x) > 1 else 0,
            raw=True
        ).values

        # ── Regime 特征: 波动率分位数 ──
        # 当前波动率在历史分布中的位置
        vol_20d_arr = vol_20d.values
        feats['vol_regime_60d'] = pd.Series(vol_20d_arr).rolling(60).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) >= 20 else 0.5,
            raw=False
        ).values
        feats['vol_regime_120d'] = pd.Series(vol_20d_arr).rolling(120).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) >= 30 else 0.5,
            raw=False
        ).values

        # ── Regime 特征: 趋势状态 ──
        # 多均线排列: MA5 > MA10 > MA20 > MA60 → 强多头
        ma5 = _rolling_mean(c, 5)
        ma10 = _rolling_mean(c, 10)
        ma20 = _rolling_mean(c, 20)
        ma60 = _rolling_mean(c, 60)
        # 趋势得分: 每满足一个相邻MA多头排列 +1, 空头 -1
        trend_score = np.zeros(len(c))
        for i in range(len(c)):
            score = 0
            if ma5[i] > ma10[i]: score += 1
            else: score -= 1
            if ma10[i] > ma20[i]: score += 1
            else: score -= 1
            if ma20[i] > ma60[i]: score += 1
            else: score -= 1
            trend_score[i] = score / 3.0
        feats['trend_regime'] = trend_score

        # ── Regime 特征: 均值回复强度 ──
        # 价格偏离 60日均线的幅度 × 波动率调整后的信号
        bias_60 = (c - ma60) / np.clip(ma60, 1e-8, None)
        feats['mean_rev_signal'] = (-bias_60 / np.clip(vol_60d_s.values, 1e-8, None))

        # ── Regime 特征: 市场情绪 (Fear/Greed Proxy) ──
        # 上涨日比例 (20日) × 波动率下降 → Greed
        up_day = (pd.Series(ret) > 0).astype(float)
        up_ratio_20d = up_day.rolling(20).mean()
        vol_chg = vol_20d.pct_change(10)
        feats['sentiment_greed'] = (up_ratio_20d.values - 0.5) * 2 * (-np.sign(vol_chg.values))

        # ── Regime 特征: 动量崩溃风险 ──
        # 短期动量 vs 长期动量的分歧度
        ret_20d = pd.Series(ret).rolling(20).mean()
        ret_60d = pd.Series(ret).rolling(60).mean()
        feats['momentum_divergence'] = (ret_20d - ret_60d).values / np.clip(vol_60d_s.values, 1e-8, None)

        # ── 尾部风险特征 ──
        # 条件VaR (CVaR) 近似: 最差5%日均回报
        feats['cvar_20d'] = pd.Series(ret).rolling(20).apply(
            lambda x: np.mean(x[x <= np.percentile(x, 5)]) if len(x[x <= np.percentile(x, 5)]) > 0 else np.mean(x),
            raw=True
        ).values
        feats['cvar_60d'] = pd.Series(ret).rolling(60).apply(
            lambda x: np.mean(x[x <= np.percentile(x, 5)]) if len(x[x <= np.percentile(x, 5)]) > 0 else np.mean(x),
            raw=True
        ).values

        # ── 流动性冲击特征 ──
        # Amihud 非流动性 (|ret| / volume)
        illiq = np.abs(ret) / np.clip(v, 1e-8, None)
        feats['amihud_20d'] = _rolling_mean(illiq, 20)
        feats['amihud_60d'] = _rolling_mean(illiq, 60)
        feats['amihud_ratio'] = feats['amihud_20d'] / np.clip(feats['amihud_60d'], 1e-8, None)

        # ── 日内模式特征 ──
        # 上影线比例 (高点压力)
        upper_shadow = (h - np.maximum(o, c)) / np.clip(h - l, 1e-8, None)
        feats['upper_shadow_20d'] = _rolling_mean(upper_shadow, 20)

        # 下影线比例 (低点支撑)
        lower_shadow = (np.minimum(o, c) - l) / np.clip(h - l, 1e-8, None)
        feats['lower_shadow_20d'] = _rolling_mean(lower_shadow, 20)

        # 实体大小 (趋势强度)
        body = np.abs(c - o) / np.clip(h - l, 1e-8, None)
        feats['candle_body_20d'] = _rolling_mean(body, 20)

        # ══════════════════════════════════════════════════════════════
        # 截面因子 — 从原始成分股数据逐日计算
        # ══════════════════════════════════════════════════════════════
        dates_list = sorted(df_raw['date'].unique())
        n_stocks = df_raw['symbol'].nunique()
        logger.info(f"计算截面因子 ({n_stocks} 只股票, {len(dates_list)} 个交易日)...")

        cross_feats = {k: [] for k in [
            'cross_ret_disp_5d', 'cross_ret_disp_20d',
            'cross_vol_disp_20d',
            'up_ratio_5d', 'up_ratio_20d',
            'down_ratio_5d', 'down_ratio_20d',
            'cross_skew_20d', 'cross_kurt_20d',
            'large_ret_disp_20d', 'small_ret_disp_20d',
            'big_small_ret_diff_20d',
            'cross_turnover_disp_20d',
            'high_low_corr_20d',
        ]}

        for dt in dates_list:
            day_df = df_raw[df_raw['date'] == dt].copy()

            # 逐股计算 ret_5d, ret_20d, vol_20d, turnover
            stock_rets = {}
            stock_vols = {}
            stock_turns = {}
            for sym, grp in day_df.groupby('symbol'):
                if len(grp) < 21:
                    continue
                c_s = grp['close'].values
                v_s = grp['volume'].values
                log_ret_s = np.log(c_s / np.roll(c_s, 1))
                log_ret_s[0] = 0.0

                stock_rets[sym] = {
                    'ret_5d': (c_s[-1] / c_s[-6] - 1) if len(c_s) >= 6 else np.nan,
                    'ret_20d': (c_s[-1] / c_s[-21] - 1) if len(c_s) >= 21 else np.nan,
                }
                stock_vols[sym] = pd.Series(log_ret_s).rolling(20).std().iloc[-1] if len(log_ret_s) >= 20 else np.nan
                if 'turn' in grp.columns:
                    stock_turns[sym] = grp['turn'].iloc[-1] if not pd.isna(grp['turn'].iloc[-1]) else np.nan

            ret_5d_vals = np.array([stock_rets[s]['ret_5d'] for s in stock_rets if np.isfinite(stock_rets[s]['ret_5d'])])
            ret_20d_vals = np.array([stock_rets[s]['ret_20d'] for s in stock_rets if np.isfinite(stock_rets[s]['ret_20d'])])
            vol_20d_vals = np.array([stock_vols[s] for s in stock_vols if np.isfinite(stock_vols[s])])

            # 市场广度
            up_5d = np.mean(ret_5d_vals > 0) if len(ret_5d_vals) > 0 else 0.5
            up_20d = np.mean(ret_20d_vals > 0) if len(ret_20d_vals) > 0 else 0.5
            down_5d = np.mean(ret_5d_vals < 0) if len(ret_5d_vals) > 0 else 0.5
            down_20d = np.mean(ret_20d_vals < 0) if len(ret_20d_vals) > 0 else 0.5

            # 截面离散度
            ret_disp_5d = np.std(ret_5d_vals) if len(ret_5d_vals) > 2 else 0.0
            ret_disp_20d = np.std(ret_20d_vals) if len(ret_20d_vals) > 2 else 0.0
            vol_disp_20d = np.std(vol_20d_vals) if len(vol_20d_vals) > 2 else 0.0

            # 截面偏度/峰度
            if len(ret_20d_vals) > 5:
                s20 = pd.Series(ret_20d_vals)
                cross_skew = s20.skew()
                cross_kurt = s20.kurtosis()
            else:
                cross_skew = 0.0
                cross_kurt = 0.0

            # 大盘股 vs 小盘股 (按 ret_20d 排序, 前50大 vs 后50小)
            if len(ret_20d_vals) >= 100:
                sorted_ret = np.sort(ret_20d_vals)
                large_ret = sorted_ret[-50:]
                small_ret = sorted_ret[:50]
                large_disp = np.std(large_ret)
                small_disp = np.std(small_ret)
                big_small_diff = np.mean(large_ret) - np.mean(small_ret)
            else:
                large_disp = 0.0
                small_disp = 0.0
                big_small_diff = 0.0

            # 换手率离散度
            turn_vals = np.array([stock_turns[s] for s in stock_turns if np.isfinite(stock_turns[s])])
            turn_disp = np.std(turn_vals) if len(turn_vals) > 2 else 0.0

            # 个股间收益相关性均值 (随机采样50对, 避免计算量过大)
            corr_mean = 0.0
            if len(day_df) > 0 and n_stocks >= 10:
                pivot = df_raw[df_raw['date'] <= dt].pivot_table(
                    index='date', columns='symbol', values='close'
                ).tail(60)
                if pivot.shape[1] >= 10 and pivot.shape[0] >= 20:
                    ret_mat = pivot.pct_change().dropna()
                    if ret_mat.shape[0] >= 20:
                        corr_mat = ret_mat.corr()
                        mask = np.triu(np.ones_like(corr_mat, dtype=bool), k=1)
                        upper_vals = corr_mat.values[mask]
                        upper_vals = upper_vals[np.isfinite(upper_vals)]
                        if len(upper_vals) > 0:
                            sample_n = min(500, len(upper_vals))
                            corr_mean = np.mean(np.random.choice(upper_vals, sample_n, replace=False))

            cross_feats['cross_ret_disp_5d'].append(ret_disp_5d)
            cross_feats['cross_ret_disp_20d'].append(ret_disp_20d)
            cross_feats['cross_vol_disp_20d'].append(vol_disp_20d)
            cross_feats['up_ratio_5d'].append(up_5d)
            cross_feats['up_ratio_20d'].append(up_20d)
            cross_feats['down_ratio_5d'].append(down_5d)
            cross_feats['down_ratio_20d'].append(down_20d)
            cross_feats['cross_skew_20d'].append(cross_skew)
            cross_feats['cross_kurt_20d'].append(cross_kurt)
            cross_feats['large_ret_disp_20d'].append(large_disp)
            cross_feats['small_ret_disp_20d'].append(small_disp)
            cross_feats['big_small_ret_diff_20d'].append(big_small_diff)
            cross_feats['cross_turnover_disp_20d'].append(turn_disp)
            cross_feats['high_low_corr_20d'].append(corr_mean)

        cross_df = pd.DataFrame(cross_feats)
        cross_df['date'] = np.array(dates_list)
        logger.info(f"截面因子计算完成, 共 {len(cross_feats)} 个因子。")

        # ── 组装 DataFrame ──
        full_factors = pd.DataFrame(feats)
        full_factors = full_factors.merge(cross_df, on='date', how='left')

        factor_cols = [c for c in full_factors.columns if c != 'date']

        # 滚动时间序列标准化 (expanding window, 避免前瞻偏差)
        logger.info(f"执行时间序列标准化 (expanding Z-score) 于 {len(factor_cols)} 个因子...")
        for col in factor_cols:
            s = full_factors[col]
            expanding_mean = s.expanding(min_periods=20).mean()
            expanding_std = s.expanding(min_periods=20).std().clip(lower=1e-8)
            full_factors[col] = (s - expanding_mean) / expanding_std

        for c in full_factors.columns:
            if full_factors[c].dtype == 'float64':
                full_factors[c] = full_factors[c].astype(np.float32)

        if hasattr(Config, 'INITIAL_FEATURE_COLS') and not Config.INITIAL_FEATURE_COLS:
            Config.INITIAL_FEATURE_COLS = factor_cols
        logger.info(f"时序择时因子库构建完成，共 {len(factor_cols)} 个因子。")
        return full_factors
