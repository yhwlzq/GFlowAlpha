import logging
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import statsmodels.api as sm

logger = logging.getLogger(__name__)


class FactorScreener:
    @staticmethod
    def winsorize_mad(series, n=5.0):
        median = series.median()
        mad = np.abs(series - median).median()
        return series.clip(median - n * 1.4826 * mad, median + n * 1.4826 * mad)

    def screen(self, factor_values, returns, dates, symbols, amount):
        df = pd.DataFrame({
            'date': pd.to_datetime(dates).normalize(), 'symbol': symbols,
            'factor': factor_values, 'ret': returns, 'log_amount': np.log(amount.clip(lower=1.0))
        }).dropna()

        raw_icir = self._calc_icir(df, 'factor', 'ret')

        neutral_ic_list = []
        for dt, grp in df.groupby('date'):
            if len(grp) < 30:
                continue
            y = grp['factor'].values
            X = sm.add_constant(grp['log_amount'].values.reshape(-1, 1))
            try:
                res = sm.OLS(y, X).fit().resid
                ic = spearmanr(res, grp['ret'].values)[0]
                if np.isfinite(ic):
                    neutral_ic_list.append(ic)
            except:
                continue
        neutral_icir = np.mean(neutral_ic_list) / (np.std(neutral_ic_list) + 1e-8) if neutral_ic_list else 0.0

        passed = neutral_icir > 0.15
        return passed, raw_icir, neutral_icir

    def _calc_icir(self, df, f_col, r_col):
        ic_list = []
        for _, grp in df.groupby('date'):
            if len(grp) > 20:
                ic = spearmanr(grp[f_col], grp[r_col])[0]
                if np.isfinite(ic):
                    ic_list.append(ic)
        return np.mean(ic_list) / (np.std(ic_list) + 1e-8) if ic_list else 0.0
