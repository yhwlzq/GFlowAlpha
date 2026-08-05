import numpy as np
from alpha.mining.safe_ops import SafeOps, TimeSeriesOps


def test_protected_div_by_zero():
    assert SafeOps.protected_div(1.0, 0.0) == 0.0
    assert SafeOps.protected_div(1.0, 2.0) == 0.5


def test_safe_log1p_negative_is_finite():
    assert np.isfinite(SafeOps.safe_log1p(-1.0))


def test_safe_log1p_extreme():
    assert np.isfinite(SafeOps.safe_log1p(-100.0))


def test_safe_sqrt_negative_is_finite():
    assert np.isfinite(SafeOps.safe_sqrt(-4.0))
    assert SafeOps.safe_sqrt(-4.0) == 0.0


def test_safe_sigmoid_bounds():
    assert 0.0 < SafeOps.safe_sigmoid(0.0) < 1.0
    assert np.isfinite(SafeOps.safe_sigmoid(1e6))


def test_time_series_rank():
    dates = np.tile(np.arange(10), 2)
    symbols = np.repeat(['A', 'B'], 10)
    x = np.tile(np.arange(10, dtype=float), 2)
    ts = TimeSeriesOps(dates, symbols)
    ranked = ts.ts_rank(x, window=5)
    assert ranked.shape == x.shape
    assert not np.isnan(ranked).all()
