import logging
from typing import List, Dict, Optional, Tuple
import numpy as np
from scipy.stats import spearmanr, norm

logger = logging.getLogger(__name__)


def whites_reality_check(ic_series: List[np.ndarray], n_boot: int = 10000,
                         seed: int = 42) -> Dict:
    """White's (2000) Reality Check 于逐日 Rank-IC 序列.

    Args:
        ic_series: 每个通过 val 筛选的候选在各自 test 窗的逐日截面 IC 序列.
        n_boot:    stationary bootstrap 次数.

    Returns:
        {'V': 观测的 max t-stat, 'p_value': RC p 值, 'n_factors': 候选数,
         'n_days': 最大序列长度, 'boot_max': bootstrap 分布 (均值/95分位)}
    """
    if not ic_series:
        return {'V': 0.0, 'p_value': 1.0, 'n_factors': 0, 'n_days': 0,
                'boot_max_mean': 0.0, 'boot_max_q95': 0.0}

    rng = np.random.default_rng(seed)
    n_factors = len(ic_series)
    obs_t = []
    centered = []
    for s in ic_series:
        s = np.asarray(s, dtype=np.float64)
        s = s[np.isfinite(s)]
        if len(s) < 5:
            obs_t.append(0.0)
            centered.append(np.array([0.0]))
            continue
        m, sd = np.mean(s), np.std(s, ddof=1)
        obs_t.append(m / (sd / np.sqrt(len(s))) if sd > 0 else 0.0)
        # White's RC null: 真实均值=0 → 用中心化序列 bootstrap
        centered.append(s - np.mean(s))
    obs_t = np.array(obs_t)
    V = float(np.max(obs_t))

    # Stationary bootstrap (Politis & Romano): 中心化序列, 保留截面相关性 (同时间索引)
    lens = [len(c) for c in centered]
    max_len = max(lens)
    boot_maxes = []
    for b in range(n_boot):
        t = np.arange(max_len)
        idx = _stationary_bootstrap(t, rng)
        boot_t = []
        for s in centered:
            k = len(s)
            sub = idx[idx < k]
            if len(sub) == 0:
                boot_t.append(0.0)
                continue
            bs = s[sub]
            m, sd = np.mean(bs), np.std(bs, ddof=1)
            boot_t.append(m / (sd / np.sqrt(len(sub))) if sd > 0 else 0.0)
        boot_maxes.append(float(np.max(boot_t)))

    boot_maxes = np.array(boot_maxes)
    p_value = float(np.mean(boot_maxes >= V))
    return {
        'V': V, 'p_value': p_value, 'n_factors': n_factors, 'n_days': max_len,
        'boot_max_mean': float(np.mean(boot_maxes)),
        'boot_max_q95': float(np.quantile(boot_maxes, 0.95)),
        'boot_max_std': float(np.std(boot_maxes)),
    }


def _stationary_bootstrap(t: np.ndarray, rng: np.random.Generator,
                          mean_block: int = 10) -> np.ndarray:
    n = len(t)
    out = np.empty(n, dtype=t.dtype)
    i, pos = 0, int(rng.integers(0, n))
    while i < n:
        L = int(rng.geometric(1.0 / mean_block))
        for j in range(L):
            if i >= n:
                break
            out[i] = t[pos]
            i += 1
            pos = (pos + 1) % n
        pos = int(rng.integers(0, n))
    return out


def bootstrap_selection_ci(best_ic: np.ndarray, all_ic: List[np.ndarray],
                           n_boot: int = 10000, seed: int = 42,
                           alpha: float = 0.05) -> Dict:
    """选择调整置信区间: 考虑'从 k 个候选中挑 max'的 selection effect.

    Returns:
        {'mean': 最佳候选均值 IC, 'ci_naive': 未调整 CI,
         'ci_sel': 选择调整 CI (采用 bootstrap 分布, 因多次试验膨胀而更宽)}
    """
    if best_ic is None or len(best_ic) < 5 or not all_ic:
        return {'mean': 0.0, 'ci_naive': (0.0, 0.0), 'ci_sel': (0.0, 0.0)}

    rng = np.random.default_rng(seed + 1)
    best = np.asarray(best_ic, dtype=np.float64)
    best = best[np.isfinite(best)]
    m = np.mean(best)
    se = np.std(best, ddof=1) / np.sqrt(len(best))
    z = norm.ppf(1 - alpha / 2)
    ci_naive = (m - z * se, m + z * se)

    # 选择调整: 对每个候选做 bootstrap, 取 bootstrap max 分布的调整量
    boot_max_mean = []
    for _ in range(n_boot):
        cand_means = []
        for s in all_ic:
            s = np.asarray(s, dtype=np.float64)
            s = s[np.isfinite(s)]
            if len(s) < 5:
                continue
            idx = rng.integers(0, len(s), size=len(s))
            cand_means.append(np.mean(s[idx]))
        if cand_means:
            boot_max_mean.append(float(np.max(cand_means)))
    if not boot_max_mean:
        return {'mean': float(m), 'ci_naive': ci_naive, 'ci_sel': ci_naive}

    boot_max_mean = np.array(boot_max_mean)
    inflate = float(np.quantile(boot_max_mean, 1 - alpha) - np.mean(boot_max_mean))
    ci_sel = (m - inflate, m + inflate)
    return {'mean': float(m), 'ci_naive': tuple(map(float, ci_naive)),
            'ci_sel': tuple(map(float, ci_sel)), 'inflation': inflate}


def aggregate_daily_ic(factor_values: np.ndarray, returns: np.ndarray,
                       dates: np.ndarray) -> np.ndarray:
    """按交易日计算逐日截面 Spearman Rank-IC 序列."""
    df = {}
    for dt in np.unique(dates):
        m = dates == dt
        x, y = factor_values[m], returns[m]
        if len(x) < 30 or np.ptp(x) < 1e-10 or np.ptp(y) < 1e-10:
            continue
        ic = spearmanr(x, y)[0]
        if np.isfinite(ic):
            df[dt] = ic
    if not df:
        return np.array([])
    return np.array(list(df.values()))