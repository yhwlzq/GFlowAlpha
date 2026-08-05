import numpy as np
import pandas as pd
from alpha.evaluation.backtester import AcademicBacktester


def _synthetic_panel(n_symbols=60, n_months=6, days_per_month=21, seed=7):
    rng = np.random.default_rng(seed)
    dates = []
    symbols = []
    preds = []
    rets = []
    month_start = pd.Timestamp('2025-01-01')
    for m in range(n_months):
        month = month_start + pd.DateOffset(months=m)
        trading_days = pd.bdate_range(month, periods=days_per_month)
        alpha = rng.normal(0, 1, n_symbols)
        for d in trading_days:
            for s in range(n_symbols):
                dates.append(d)
                symbols.append(f'S{s:04d}')
                preds.append(alpha[s] + rng.normal(0, 0.01))
                # forward return positively correlated with pred
                rets.append(0.05 * alpha[s] + rng.normal(0, 0.05))
    return (np.array(preds), np.array(rets),
            np.array(dates, dtype='datetime64[ns]'), np.array(symbols))


def test_backtester_runs_and_produces_metrics():
    pred, ret, dates, symbols = _synthetic_panel()
    bt = AcademicBacktester(top_quantile=0.2)
    metrics = bt.run(pred, ret, dates, symbols, fid='TEST_001')
    assert metrics is not None
    assert isinstance(metrics, dict)


def test_backtester_positive_alpha_gives_positive_spread():
    pred, ret, dates, symbols = _synthetic_panel(seed=11)
    bt = AcademicBacktester(top_quantile=0.2)
    metrics = bt.run(pred, ret, dates, symbols, fid='TEST_002')
    assert metrics is not None
    assert metrics['annualized_return'] > 0
    assert 'quantile_monthly_rets' in metrics
    assert metrics['sharpe_ratio'] > 0
