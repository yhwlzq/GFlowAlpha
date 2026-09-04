import logging
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t as t_dist
from dataclasses import dataclass, field
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_hac
from alpha.config import Config

logger = logging.getLogger(__name__)


@dataclass
class FMRegressionResult:
    coefficient: float = 0.0
    std_error: float = 0.0
    t_stat: float = 0.0
    p_value: float = 1.0
    n_periods: int = 0
    avg_r_squared: float = 0.0
    avg_adj_r_squared: float = 0.0
    coef_positive_ratio: float = 0.0
    coef_significant_ratio: float = 0.0
    quantile_returns: List[float] = field(default_factory=list)
    long_short_spread: float = 0.0
    rank_ic_mean: float = 0.0
    rank_icir: float = 0.0

    def is_significant(self, threshold: float = 2.0) -> bool:
        return abs(self.t_stat) >= threshold

    def to_dict(self) -> Dict:
        return {
            'FM_coef': round(self.coefficient, 6),
            'FM_se': round(self.std_error, 6),
            'FM_tstat': round(self.t_stat, 4),
            'FM_pvalue': round(self.p_value, 6),
            'FM_n_periods': self.n_periods,
            'FM_R2_avg': round(self.avg_r_squared, 4),
            'FM_coef_pos_ratio': round(self.coef_positive_ratio, 4),
            'FM_coef_sig_ratio': round(self.coef_significant_ratio, 4),
            'LS_spread': round(self.long_short_spread, 6),
            'Rank_IC': round(self.rank_ic_mean, 4),
            'ICIR': round(self.rank_icir, 4),
        }


class FamaMacBethRegressor:
    def __init__(self, nw_lags: int = None, n_groups: int = 5):
        self.nw_lags = nw_lags if nw_lags is not None else Config.NW_LAGS
        self.n_groups = n_groups

    def run(self, factor_values: np.ndarray, returns: np.ndarray,
            dates: np.ndarray, symbols: np.ndarray,
            controls: Optional[Dict[str, np.ndarray]] = None) -> FMRegressionResult:
        result = FMRegressionResult()
        df = pd.DataFrame({
            'date': pd.to_datetime(dates).normalize(),
            'symbol': symbols,
            'factor': factor_values,
            'ret': returns
        }).dropna()

        if controls:
            for k, v in controls.items():
                if len(v) == len(factor_values):
                    df[k] = v
            df = df.dropna()

        min_dates = Config.get_market_config().get('fm_min_stocks', 30)
        if df.empty or df['date'].nunique() < min_dates:
            logger.warning(f"FM 回归: 有效数据不足 (需至少 {min_dates} 个截面期)")
            return result

        beta_series = []
        r2_series = []
        adj_r2_series = []
        single_t_series = []

        for dt, grp in df.groupby('date'):
            n_stocks = len(grp)
            if n_stocks < Config.get_market_config().get('fm_min_stocks', 30):
                continue

            y = grp['ret'].values
            X_cols = [grp['factor'].values]
            if controls:
                for ctrl_name in controls.keys():
                    if ctrl_name in grp.columns:
                        ctrl_vals = grp[ctrl_name].values
                        if grp[ctrl_name].dtype == 'object' or grp[ctrl_name].nunique() > 20:
                            dummies = pd.get_dummies(ctrl_vals, drop_first=True).values
                            X_cols.append(dummies)
                        else:
                            X_cols.append(ctrl_vals.reshape(-1, 1))

            X = np.column_stack(X_cols) if len(X_cols) > 1 else X_cols[0].reshape(-1, 1)
            X = sm.add_constant(X)

            try:
                cond = np.linalg.cond(X)
                if cond > 1e10:
                    continue
            except:
                continue

            try:
                model = sm.OLS(y, X, missing='drop').fit()
                beta_hat = model.params[1]
                beta_series.append(beta_hat)

                r2_series.append(model.rsquared)
                adj_r2_series.append(model.rsquared_adj)

                if len(model.tvalues) > 1:
                    single_t_series.append(model.tvalues[1])
                else:
                    single_t_series.append(0.0)

            except Exception:
                continue

        beta_arr = np.array(beta_series)
        n_periods = len(beta_arr)
        result.n_periods = n_periods

        if n_periods < 10:
            logger.warning(f"FM 回归: 有效截面期数过少 ({n_periods})")
            return result

        mean_beta = np.mean(beta_arr)
        result.coefficient = float(mean_beta)

        try:
            const_X = np.ones((n_periods, 1))
            ols_model = sm.OLS(beta_arr, const_X).fit()

            nw_cov = cov_hac(ols_model, nlags=self.nw_lags, use_correction=True)
            nw_se = float(np.sqrt(nw_cov[0, 0]))
            result.std_error = nw_se

            if nw_se > 1e-10:
                result.t_stat = float(mean_beta / nw_se)
            else:
                result.t_stat = 0.0

            df_freedom = n_periods - 1
            result.p_value = float(2 * (1 - t_dist.cdf(abs(result.t_stat), df_freedom)))

        except Exception as e:
            logger.warning(f"Newey-West 调整失败 ({e})，使用朴素标准误")
            naive_se = float(np.std(beta_arr, ddof=1) / np.sqrt(n_periods))
            result.std_error = naive_se
            result.t_stat = float(mean_beta / naive_se) if naive_se > 1e-10 else 0.0
            df_freedom = n_periods - 1
            result.p_value = float(2 * (1 - t_dist.cdf(abs(result.t_stat), df_freedom)))

        result.avg_r_squared = float(np.mean(r2_series)) if r2_series else 0.0
        result.avg_adj_r_squared = float(np.mean(adj_r2_series)) if adj_r2_series else 0.0

        result.coef_positive_ratio = float(np.mean(beta_arr > 0))
        single_t_arr = np.array(single_t_series)
        result.coef_significant_ratio = float(np.mean(np.abs(single_t_arr) > 1.96))

        result.quantile_returns = self._compute_quantile_returns(df)
        if len(result.quantile_returns) >= 2:
            result.long_short_spread = float(result.quantile_returns[-1] - result.quantile_returns[0])

        ic_list = []
        for _, grp in df.groupby('date'):
            if len(grp) > 30:
                ic = spearmanr(grp['factor'], grp['ret'])[0]
                if np.isfinite(ic):
                    ic_list.append(ic)
        if ic_list:
            result.rank_ic_mean = float(np.mean(ic_list))
            ic_std = float(np.std(ic_list, ddof=1)) if len(ic_list) > 1 else 1e-8
            result.rank_icir = float(result.rank_ic_mean / ic_std)

        return result

    def _compute_quantile_returns(self, df: pd.DataFrame) -> List[float]:
        try:
            df = df.copy()
            df['quantile'] = df.groupby('date')['factor'].transform(
                lambda x: pd.qcut(x, self.n_groups, labels=False, duplicates='drop')
            )
            group_rets = df.groupby('quantile')['ret'].mean().sort_index()
            return group_rets.values.tolist()
        except:
            return []

    def print_report(self, result: FMRegressionResult, factor_name: str = "Factor"):
        sig_marker = "***" if abs(result.t_stat) >= 3.0 else ("**" if abs(result.t_stat) >= 2.0 else "")
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Fama-MacBeth 回归报告 | {factor_name}")
        logger.info(f"{'=' * 60}")
        logger.info(f"  有效截面期数:       {result.n_periods}")
        logger.info(f"  平均系数:           {result.coefficient:.6f} {sig_marker}")
        logger.info(f"  NW标准误 (SE):      {result.std_error:.6f}")
        logger.info(f"  FM t-stat:          {result.t_stat:.4f} {sig_marker}")
        logger.info(f"  p-value:            {result.p_value:.6f}")
        logger.info(f"  平均 R²:            {result.avg_r_squared:.4f}")
        logger.info(f"  系数正占比:          {result.coef_positive_ratio:.2%}")
        logger.info(f"  单期显著占比:        {result.coef_significant_ratio:.2%}")
        logger.info(f"  Rank IC:            {result.rank_ic_mean:.4f}")
        logger.info(f"  ICIR:               {result.rank_icir:.4f}")
        logger.info(f"  多空收益 (L-S):     {result.long_short_spread:.6f}")
        if result.quantile_returns:
            ret_str = " → ".join([f"{r:.4f}" for r in result.quantile_returns])
            logger.info(f"  分组收益:            {ret_str}")
        logger.info(f"  NW滞后阶数:          {self.nw_lags}")
        logger.info(f"{'=' * 60}\n")


@dataclass
class TimingResult:
    """时序择时评估结果."""
    t_stat: float = 0.0
    p_value: float = 1.0
    coefficient: float = 0.0
    std_error: float = 0.0
    direction_accuracy: float = 0.5
    hit_rate: float = 0.5
    timing_sharpe: float = 0.0
    annual_return: float = 0.0
    max_drawdown: float = 0.0
    profit_factor: float = 0.0
    n_periods: int = 0
    r_squared: float = 0.0

    def is_significant(self, threshold: float = None) -> bool:
        if threshold is None:
            threshold = Config.TIMING_SHARPE_THRESHOLD
        return abs(self.timing_sharpe) >= threshold

    def to_dict(self) -> Dict:
        return {
            'Timing_tstat': round(self.t_stat, 4),
            'Timing_pvalue': round(self.p_value, 6),
            'Timing_coef': round(self.coefficient, 6),
            'Timing_dir_acc': round(self.direction_accuracy, 4),
            'Timing_hit_rate': round(self.hit_rate, 4),
            'Timing_sharpe': round(self.timing_sharpe, 4),
            'Timing_ann_ret': round(self.annual_return, 4),
            'Timing_max_dd': round(self.max_drawdown, 4),
            'Timing_profit_factor': round(self.profit_factor, 4),
            'Timing_n': self.n_periods,
            'Timing_R2': round(self.r_squared, 4),
        }


class TimingEvaluator:
    """时序择时评估器 — 替代 FM 回归, 用时序 OLS + 方向准确率."""

    def __init__(self, nw_lags: int = None):
        self.nw_lags = nw_lags if nw_lags is not None else Config.NW_LAGS

    def run(self, factor_values: np.ndarray, returns: np.ndarray,
            dates: np.ndarray) -> TimingResult:
        """时序择时评估.

        Args:
            factor_values: 因子预测值 (T,).
            returns: 指数前瞻收益 (T,).
            dates: 日期数组 (T,).

        Returns:
            TimingResult.
        """
        result = TimingResult()
        mask = np.isfinite(factor_values) & np.isfinite(returns)
        f = factor_values[mask]
        r = returns[mask]
        n = len(f)

        if n < 50:
            logger.warning(f"TimingEvaluator: 有效样本不足 ({n} < 50)")
            return result

        result.n_periods = n

        # ── 时序 OLS: r[t] = alpha + beta * f[t] + eps ──
        X = sm.add_constant(f)
        try:
            model = sm.OLS(r, X).fit()
            result.coefficient = float(model.params[1])
            result.r_squared = float(model.rsquared)

            # Newey-West 调整
            try:
                nw_cov = cov_hac(model, nlags=self.nw_lags, use_correction=True)
                nw_se = float(np.sqrt(nw_cov[1, 1]))
                result.std_error = nw_se
                if nw_se > 1e-10:
                    result.t_stat = float(model.params[1] / nw_se)
                else:
                    result.t_stat = 0.0
            except Exception:
                result.std_error = float(model.bse[1])
                result.t_stat = float(model.tvalues[1])

            df_freedom = n - 2
            result.p_value = float(2 * (1 - t_dist.cdf(abs(result.t_stat), df_freedom)))

        except Exception as e:
            logger.warning(f"TimingEvaluator OLS 失败: {e}")
            return result

        # ── 方向准确率 ──
        pred_sign = np.sign(f)
        actual_sign = np.sign(r)
        result.direction_accuracy = float(np.mean(pred_sign == actual_sign))
        result.hit_rate = result.direction_accuracy

        # ── 择时 Sharpe (假设信号>0时持仓, 信号<=0时空仓) ──
        position = np.where(f > 0, 1.0, 0.0)
        strategy_ret = position * r
        if np.std(strategy_ret) > 1e-10:
            result.timing_sharpe = float(np.mean(strategy_ret) / np.std(strategy_ret) * np.sqrt(252))
        else:
            result.timing_sharpe = 0.0

        # ── 年化收益 ──
        cumulative = np.cumprod(1 + strategy_ret)
        if len(cumulative) > 0 and cumulative[-1] > 0:
            years = n / 252
            result.annual_return = float(cumulative[-1] ** (1 / years) - 1) if years > 0 else 0.0

        # ── 最大回撤 ──
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / np.clip(peak, 1e-8, None)
        result.max_drawdown = float(np.min(drawdown))

        # ── 盈亏比 ──
        gains = strategy_ret[strategy_ret > 0]
        losses = strategy_ret[strategy_ret < 0]
        if len(gains) > 0 and len(losses) > 0:
            result.profit_factor = float(np.sum(gains) / abs(np.sum(losses)))
        else:
            result.profit_factor = 0.0

        return result

    def print_report(self, result: TimingResult, factor_name: str = "Factor"):
        sig_marker = "***" if abs(result.t_stat) >= 3.0 else ("**" if abs(result.t_stat) >= 2.0 else "")
        logger.info(f"\n{'=' * 60}")
        logger.info(f"时序择时评估报告 | {factor_name}")
        logger.info(f"{'=' * 60}")
        logger.info(f"  有效样本数:         {result.n_periods}")
        logger.info(f"  回归系数:           {result.coefficient:.6f} {sig_marker}")
        logger.info(f"  NW标准误 (SE):      {result.std_error:.6f}")
        logger.info(f"  t-stat:             {result.t_stat:.4f} {sig_marker}")
        logger.info(f"  p-value:            {result.p_value:.6f}")
        logger.info(f"  R²:                 {result.r_squared:.4f}")
        logger.info(f"  方向准确率:          {result.direction_accuracy:.2%}")
        logger.info(f"  择时 Sharpe:         {result.timing_sharpe:.4f}")
        logger.info(f"  年化收益:            {result.annual_return:.2%}")
        logger.info(f"  最大回撤:            {result.max_drawdown:.2%}")
        logger.info(f"  盈亏比:              {result.profit_factor:.4f}")
        logger.info(f"  NW滞后阶数:          {self.nw_lags}")
        logger.info(f"{'=' * 60}\n")
