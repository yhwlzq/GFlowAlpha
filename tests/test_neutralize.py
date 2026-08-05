import numpy as np
import pandas as pd
from alpha.mining.neutralize import fwl_neutralize


def test_fwl_neutralize_all_zero_amount_passthrough():
    factor = np.array([1.0, 2.0, 3.0, 4.0])
    ret = np.array([0.1, -0.2, 0.05, 0.03])
    dates = np.array(['2025-01-01'] * 4, dtype='datetime64[ns]')
    symbols = np.array(['A', 'B', 'C', 'D'])
    pf, pr = fwl_neutralize(factor, ret, dates, symbols, np.zeros(4))
    np.testing.assert_array_equal(pf, factor)
    np.testing.assert_array_equal(pr, ret)


def test_fwl_neutralize_produces_residuals():
    rng = np.random.default_rng(3)
    n = 120
    dates = np.repeat(pd.to_datetime(['2025-01-02', '2025-01-03', '2025-01-06']), 40)
    symbols = np.tile([f'S{i:03d}' for i in range(40)], 3)
    amount = rng.uniform(1e7, 1e9, n)
    factor = 2.0 * np.log(amount) + rng.normal(0, 1, n)
    ret = rng.normal(0, 0.02, n)
    pf, pr = fwl_neutralize(factor, ret, dates, symbols, amount)
    assert len(pf) == n
    assert len(pr) == n
    # residual factor should be uncorrelated with size proxy within dates
    df = pd.DataFrame({'date': dates, 'pf': pf, 'ln_amt': np.log(amount)}).dropna()
    corr_by_day = df.groupby('date').apply(
        lambda g: np.corrcoef(g['pf'], g['ln_amt'])[0, 1] if g['ln_amt'].std() > 0 else 0.0,
        include_groups=False,
    )
    assert np.abs(corr_by_day).max() < 0.2
