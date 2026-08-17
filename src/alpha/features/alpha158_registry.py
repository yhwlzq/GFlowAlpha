import logging
import numpy as np
import pandas as pd
from alpha.config import Config

logger = logging.getLogger(__name__)

_EPS = 1e-12


def _rolling_lm(series: pd.Series, w: int):
    """滑动窗口线性回归三件套 (BETA/RSQR/RESI), 对齐 Qlib 语义.

    对窗口内 y 与 x=0..L-1 回归 (L=窗口内有效点数):
      slope = (L*Sxy - Sx*Sy) / (L*Sx2 - Sx^2)
      rsq   = r^2 (y 滚动 std≈0 时置 NaN)
      resi  = y[-1] - (slope*(L-1)+intercept)   # 最新点残差
    头部不足 w 个有效点时返回 NaN (整体只损失前 ~60 行).
    """
    y = series.to_numpy(dtype=np.float64)
    n = len(y)
    out_s, out_r, out_e = (np.full(n, np.nan) for _ in range(3))
    if n < 2:
        return out_s, out_r, out_e

    from numpy.lib.stride_tricks import sliding_window_view
    if n >= w:
        W = sliding_window_view(y, w)           # (n-w+1, w), 行=旧->新
        x = np.arange(w, dtype=np.float64)
        Sx, Sx2 = x.sum(), (x * x).sum()
        Sy = W.sum(axis=1)
        Sxy = W.dot(x)
        denom = w * Sx2 - Sx * Sx
        slope = (w * Sxy - Sx * Sy) / denom

        Syy = (W * W).sum(axis=1)
        Syy_den = w * Syy - Sy * Sy
        r2 = np.full_like(slope, np.nan)
        with np.errstate(invalid='ignore', divide='ignore'):
            corr2 = ((w * Sxy - Sx * Sy) ** 2) / (denom * Syy_den)
            ok = (denom > 0) & (Syy_den > 0) & np.isfinite(corr2)
            r2[ok] = corr2[ok]

        ybar = Sy / w
        xbar = Sx / w
        interp = ybar - slope * xbar
        pred_new = slope * (w - 1) + interp
        resi = W[:, -1] - pred_new

        out_s[w - 1:] = slope
        out_r[w - 1:] = r2
        out_e[w - 1:] = resi
    return out_s, out_r, out_e


def _rolling_idx(series: pd.Series, w: int, which: str) -> pd.Series:
    """滑动窗口 Argmax/Argmin, 语义: 距今位置 (旧=1, 新=w). 头部不足 w 时 NaN."""
    y = series.to_numpy(dtype=np.float64)
    n = len(y)
    out = np.full(n, np.nan)
    if n >= w:
        from numpy.lib.stride_tricks import sliding_window_view
        W = sliding_window_view(y, w)
        idx = W.argmax(axis=1) + 1 if which == 'max' else W.argmin(axis=1) + 1
        out[w - 1:] = idx.astype(np.float64)
    return pd.Series(out, index=series.index)


class Alpha158FeatureRegistry:
    """Qlib Alpha158 工程化因子池 (自实现, 与 Microsooft/qlib Alpha158DL 默认配置一致).

    默认配置:
      kbar    : 9 个 K 线形态因子 (KMID/KLEN/KMID2/KUP/KUP2/KLOW/KLOW2/KSFT/KSFT2)
      price   : 4 个当日价格比 (OPEN0/HIGH0/LOW0/VWAP0)
      rolling : 29 族 × windows [5,10,20,30,60] = 145 个滚动因子

    合计 = 9 + 4 + 145 = 158 个特征.

    滚动语义对齐 Qlib:
      - Mean/Std/Max/Min/Quantile/Rank/Corr 用 pandas rolling(min_periods=1)
      - Slope/BETA = polyfit(x=0..L-1, y) 斜率或滑动窗口 OLS
      - Rsquare     = 拟合直线与 y 的相关系数平方 (y 滚动 std≈0 时置 NaN, atol≈2e-05)
      - Resi        = 最新点相对回归直线的残差 (y[-1] - pred(x=L-1))
      - IdxMax/IdxMin = argmax/argmin+1 表示距今位置
    """

    _WINDOWS = [5, 10, 20, 30, 60]

    @classmethod
    def build_alpha158_factors(cls, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("构建 Alpha158 工程化因子库 (158 特征)...")
        df = df.sort_values(['symbol', 'date']).copy()
        factor_dfs = []

        for sym, grp in df.groupby('symbol'):
            c, o, h, l = grp['close'], grp['open'], grp['high'], grp['low']
            v = grp['volume']
            amt = grp.get('amount', pd.Series(np.nan, index=grp.index))
            vwap = amt / v.clip(lower=_EPS)

            prev_c = c.shift(1)
            d_c = c - prev_c
            abs_d_c = d_c.abs()
            prev_v = v.shift(1)
            d_v = v - prev_v
            abs_d_v = d_v.abs()

            feats = {
                'date': grp['date'].values,
                'symbol': sym,
            }

            # === 1. K 线形态 (9) ===
            feats['KMID'] = (2 * c - h - l) / o.clip(lower=_EPS)
            feats['KLEN'] = (h - l) / o.clip(lower=_EPS)
            feats['KMID2'] = (c - o) / (h - l + _EPS)
            feats['KUP'] = (h - np.maximum(o, c)) / o.clip(lower=_EPS)
            feats['KUP2'] = (h - np.maximum(o, c)) / (h - l + _EPS)
            feats['KLOW'] = (np.minimum(o, c) - l) / o.clip(lower=_EPS)
            feats['KLOW2'] = (np.minimum(o, c) - l) / (h - l + _EPS)
            feats['KSFT'] = (2 * c - h - l) / o.clip(lower=_EPS)
            feats['KSFT2'] = (2 * c - h - l) / (h - l + _EPS)

            # === 2. 当日价格比 (4) ===
            feats['OPEN0'] = o / c.clip(lower=_EPS)
            feats['HIGH0'] = h / c.clip(lower=_EPS)
            feats['LOW0'] = l / c.clip(lower=_EPS)
            feats['VWAP0'] = vwap / c.clip(lower=_EPS)

            # === 3. 滚动因子 (29 族 × 5 窗口) ===
            log_v = np.log(v.clip(lower=_EPS) + 1)
            close_change_ratio = (c / prev_c.clip(lower=_EPS) - 1)
            log_v_change = np.log(v / prev_v.clip(lower=_EPS) + 1)

            for d in cls._WINDOWS:
                rol = c.rolling(d, min_periods=1)

                feats[f'ROC{d}'] = c.shift(d) / c.clip(lower=_EPS)
                feats[f'MA{d}'] = rol.mean() / c.clip(lower=_EPS)
                feats[f'STD{d}'] = rol.std() / c.clip(lower=_EPS)

                bslope, brsq, bresi = _rolling_lm(c, d)
                feats[f'BETA{d}'] = bslope / c.to_numpy()
                feats[f'RSQR{d}'] = brsq
                feats[f'RESI{d}'] = bresi / c.to_numpy()

                feats[f'MAX{d}'] = h.rolling(d, min_periods=1).max() / c.clip(lower=_EPS)
                feats[f'MIN{d}'] = l.rolling(d, min_periods=1).min() / c.clip(lower=_EPS)
                feats[f'QTLU{d}'] = c.rolling(d, min_periods=1).quantile(0.8) / c.clip(lower=_EPS)
                feats[f'QTLD{d}'] = c.rolling(d, min_periods=1).quantile(0.2) / c.clip(lower=_EPS)
                feats[f'RANK{d}'] = c.rolling(d, min_periods=1).rank(pct=True)

                rsv_num = c - l.rolling(d, min_periods=1).min()
                rsv_den = h.rolling(d, min_periods=1).max() - l.rolling(d, min_periods=1).min() + _EPS
                feats[f'RSV{d}'] = rsv_num / rsv_den

                feats[f'IMAX{d}'] = _rolling_idx(h, d, 'max') / d
                feats[f'IMIN{d}'] = _rolling_idx(l, d, 'min') / d
                feats[f'IMXD{d}'] = (feats[f'IMAX{d}'] - feats[f'IMIN{d}']) / d

                feats[f'CORR{d}'] = np.nan_to_num(c.rolling(d, min_periods=1).corr(log_v))
                feats[f'CORD{d}'] = np.nan_to_num(
                    c.rolling(d, min_periods=1).corr(log_v_change))

                feats[f'CNTP{d}'] = (d_c > 0).rolling(d, min_periods=1).mean()
                feats[f'CNTN{d}'] = (d_c < 0).rolling(d, min_periods=1).mean()
                feats[f'CNTD{d}'] = feats[f'CNTP{d}'] - feats[f'CNTN{d}']

                sum_up = d_c.clip(lower=0).rolling(d, min_periods=1).sum()
                sum_abs = abs_d_c.rolling(d, min_periods=1).sum() + _EPS
                feats[f'SUMP{d}'] = sum_up / sum_abs
                feats[f'SUMN{d}'] = (-d_c).clip(lower=0).rolling(d, min_periods=1).sum() / sum_abs
                feats[f'SUMD{d}'] = feats[f'SUMP{d}'] - feats[f'SUMN{d}']

                feats[f'VMA{d}'] = v.rolling(d, min_periods=1).mean() / (v + _EPS)
                feats[f'VSTD{d}'] = v.rolling(d, min_periods=1).std() / (v + _EPS)

                wv = close_change_ratio.abs() * v
                feats[f'WVMA{d}'] = wv.rolling(d, min_periods=1).std() / (
                    wv.rolling(d, min_periods=1).mean() + _EPS)

                vsum_abs = abs_d_v.rolling(d, min_periods=1).sum() + _EPS
                feats[f'VSUMP{d}'] = d_v.clip(lower=0).rolling(d, min_periods=1).sum() / vsum_abs
                feats[f'VSUMN{d}'] = (-d_v).clip(lower=0).rolling(d, min_periods=1).sum() / vsum_abs
                feats[f'VSUMD{d}'] = feats[f'VSUMP{d}'] - feats[f'VSUMN{d}']

            factor_dfs.append(pd.DataFrame(feats))

        full_factors = pd.concat(factor_dfs, ignore_index=True)

        full_factors.replace([np.inf, -np.inf], np.nan, inplace=True)
        full_factors.fillna(0.0, inplace=True)
        for col in full_factors.columns:
            if full_factors[col].dtype == 'float64':
                full_factors[col] = full_factors[col].clip(-1e3, 1e3).astype(np.float32)
        factor_cols = [c for c in full_factors.columns if c not in ['date', 'symbol']]
        assert Config.INITIAL_FEATURE_COLS == [], "build_alpha158_factors 不应被重复调用"
        Config.INITIAL_FEATURE_COLS = factor_cols
        logger.info(f"Alpha158 因子库构建完成，共 {len(factor_cols)} 个特征。")
        return full_factors