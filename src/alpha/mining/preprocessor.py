import logging
import warnings
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from alpha.config import Config
from alpha.features.feature_registry import (
    UltimateDailyFeatureRegistry, LargeCapFeatureRegistry, TimingFeatureRegistry,
)
from alpha.features.alpha158_registry import Alpha158FeatureRegistry
from .walkforward import WalkForwardSplitter

logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        self.full_X = self.full_y = None
        self.full_X_raw = self.full_y_raw = None
        self.full_dates = self.full_symbols = None
        self.tr_mask = self.val_mask = self.te_mask = None
        self.valid_features = []
        self.feature_to_idx = {}
        self.full_amount = None
        self.full_turn = None
        self.full_open = None
        self.full_close = None
        self.wf_splitter = None
        self.n_windows = 0

    def _dedup_near_duplicates(self, X: np.ndarray, valid_cols: List[str]) -> Tuple[np.ndarray, List[str]]:
        """近重复去重: 仅剔除 Spearman 正相关 > 阈值的镜像对 (按 |IC| 保最高).

        负相关 (如 rev_* vs ret_* 反转/动量镜像对) 不作为重复, 全部保留.
        使用 Union-Find 将高正相关特征聚成连通分量, 每组只保留 |IC| 最高的.
        """
        threshold = getattr(Config, 'NEAR_DUPLICATE_CORR', 0.99)
        n = len(valid_cols)
        if n < 2:
            return X, valid_cols

        corr_matrix, _ = spearmanr(X)

        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for i in range(n):
            for j in range(i + 1, n):
                c = corr_matrix[i, j]
                if c > threshold:
                    union(i, j)

        groups: Dict[int, List[int]] = {}
        for i in range(n):
            root = find(i)
            groups.setdefault(root, []).append(i)

        ic_scores = {}
        for i, col in enumerate(valid_cols):
            ic = spearmanr(X[:, i], self.full_y)[0]
            ic_scores[col] = abs(ic) if np.isfinite(ic) else 0.0

        keep_indices = []
        removed = []
        for members in groups.values():
            if len(members) == 1:
                keep_indices.append(members[0])
            else:
                members.sort(key=lambda x: ic_scores.get(valid_cols[x], 0), reverse=True)
                keep_indices.append(members[0])
                for idx in members[1:]:
                    removed.append(valid_cols[idx])

        if removed:
            logger.info(f"近重复去重: 剔除 {len(removed)} 个正相关 >{threshold:.2f} 的冗余因子")
            for f in removed[:10]:
                logger.info(f"  - {f}")
            if len(removed) > 10:
                logger.info(f"  ... 及另外 {len(removed) - 10} 个")

        keep_indices.sort()
        new_valid_cols = [valid_cols[i] for i in keep_indices]
        new_X = X[:, keep_indices]
        return new_X, new_valid_cols

    def prepare_full_pool(self, df_raw: pd.DataFrame):
        logger.info("构建日频数据集并执行特征质量门禁...")
        if Config.FEATURE_POOL == 'alpha158':
            df_feat = df_raw[['date', 'symbol', 'open', 'high', 'low', 'close',
                              'volume', 'amount']].dropna(subset=['close', 'volume']).copy()
            df_feat = df_feat.sort_values(['symbol', 'date']).reset_index(drop=True)
            df_raw = df_feat
            full_factors = Alpha158FeatureRegistry.build_alpha158_factors(df_raw)
        elif Config.MARKET == 'hs300':
            full_factors = LargeCapFeatureRegistry.build_large_cap_factors(df_raw)
        elif Config.MARKET == 'zs500':
            full_factors = UltimateDailyFeatureRegistry.build_ultimate_factors(df_raw)
        else:
            full_factors = UltimateDailyFeatureRegistry.build_ultimate_factors(df_raw)

        df_raw = df_raw.sort_values(['symbol', 'date'])
        labels = df_raw.groupby('symbol')['close'].pct_change(Config.HORIZON).shift(-Config.HORIZON)

        full_factors['forward_ret'] = labels.values
        if 'amount' in df_raw.columns:
            amt_map = df_raw[['symbol', 'date', 'amount']].drop_duplicates(subset=['symbol', 'date'])
            full_factors = full_factors.merge(amt_map, on=['symbol', 'date'], how='left')
        if 'turn' in df_raw.columns:
            turn_map = df_raw[['symbol', 'date', 'turn']].drop_duplicates(subset=['symbol', 'date'])
            full_factors = full_factors.merge(turn_map, on=['symbol', 'date'], how='left')
        if {'open', 'close'}.issubset(df_raw.columns):
            oc_map = df_raw[['symbol', 'date', 'open', 'close']].drop_duplicates(subset=['symbol', 'date'])
            full_factors = full_factors.merge(oc_map, on=['symbol', 'date'], how='left')
        full_factors = full_factors.dropna(subset=['forward_ret', 'date', 'symbol'])

        self.full_dates = full_factors['date'].values
        self.full_symbols = full_factors['symbol'].values
        self.full_y = full_factors['forward_ret'].values.astype(np.float32)
        self.full_amount = full_factors['amount'].values.astype(np.float64) if 'amount' in full_factors.columns else np.zeros(len(full_factors), dtype=np.float64)
        self.full_turn = full_factors['turn'].values.astype(np.float64) if 'turn' in full_factors.columns else np.zeros(len(full_factors), dtype=np.float64)
        self.full_open = full_factors['open'].values.astype(np.float64) if 'open' in full_factors.columns else None
        self.full_close = full_factors['close'].values.astype(np.float64) if 'close' in full_factors.columns else None

        X_raw = full_factors[Config.INITIAL_FEATURE_COLS].values.astype(np.float32)

        logger.info("执行特征质量门禁...")
        warnings.filterwarnings('ignore', message='invalid value encountered in reduce')
        valid_indices = []
        for i, feat in enumerate(Config.INITIAL_FEATURE_COLS):
            col_data = X_raw[:, i]
            nan_ratio = np.isnan(col_data).mean()
            if nan_ratio > Config.NaN_RATIO_CAP:
                logger.info(f"  剔除因子 {feat}: 缺失率 {nan_ratio:.1%} > {Config.NaN_RATIO_CAP:.0%}")
                continue
            if np.nanvar(col_data) < 1e-6:
                continue
            valid_indices.append(i)

        X_clean = X_raw[:, valid_indices]
        valid_cols = [Config.INITIAL_FEATURE_COLS[i] for i in valid_indices]
        logger.info(f"质量门禁后保留 {len(valid_cols)} 个因子。")

        if getattr(Config, 'DEDUP_NEAR_DUPLICATES', False):
            X_clean, valid_cols = self._dedup_near_duplicates(X_clean, valid_cols)

        if getattr(Config, 'DROP_HIGH_NAN_ROWS', False) and X_clean.shape[1] > 0:
            row_nan_ratio = np.isnan(X_clean).mean(axis=1)
            keep = row_nan_ratio <= Config.HIGH_NAN_ROW_FRAC
            n_drop = int((~keep).sum())
            if n_drop:
                logger.info(f"行级缺失清洗: 剔除 {n_drop:,} 行 (特征缺失率>{Config.HIGH_NAN_ROW_FRAC:.0%})")
                for attr in ('full_dates', 'full_symbols', 'full_y', 'full_amount', 'full_turn', 'full_open', 'full_close'):
                    val = getattr(self, attr)
                    if val is not None:
                        setattr(self, attr, val[keep])
                X_clean = X_clean[keep]

        effective_pool_size = min(Config.TARGET_FACTOR_POOL_SIZE, len(valid_cols))
        X_clean = np.nan_to_num(X_clean, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

        if not Config.CLUSTER_FEATURE_POOL:
            logger.info("跳过聚类正交化筛选，使用全量门禁因子池。")
            self.valid_features = valid_cols
            self.full_X = X_clean
        else:
            logger.info(f"启动层次聚类正交化筛选 (目标池: {effective_pool_size})...")
            try:
                corr_matrix = np.abs(spearmanr(X_clean)[0])
                distance_matrix = (1 - corr_matrix).copy()
                np.fill_diagonal(distance_matrix, 0)
                distance_matrix = (distance_matrix + distance_matrix.T) / 2
                condensed_dist = squareform(distance_matrix)
                linkage_matrix = linkage(condensed_dist, method='complete')
                cluster_labels = fcluster(linkage_matrix, effective_pool_size, criterion='maxclust')

                ic_scores = {}
                for i, col in enumerate(valid_cols):
                    ic = spearmanr(X_clean[:, i], self.full_y)[0]
                    ic_scores[col] = abs(ic) if np.isfinite(ic) else 0.0

                selected_factors = []
                for cluster_id in range(1, effective_pool_size + 1):
                    cluster_cols = [valid_cols[j] for j, label in enumerate(cluster_labels) if label == cluster_id]
                    cluster_cols.sort(key=lambda x: ic_scores.get(x, 0), reverse=True)
                    if cluster_cols:
                        selected_factors.append(cluster_cols[0])

                self.valid_features = selected_factors
                final_indices = [valid_cols.index(f) for f in self.valid_features]
                self.full_X = X_clean[:, final_indices]
                logger.info(f"聚类筛选完成！最终核心因子池: {len(self.valid_features)}")
            except Exception as e:
                logger.warning(f"聚类筛选失败 ({e})，回退至全量因子。")
                self.valid_features = valid_cols
                self.full_X = X_clean

        self.feature_to_idx = {f: i for i, f in enumerate(self.valid_features)}
        Config.FINAL_FEATURE_POOL = self.valid_features

        self.full_X_raw = self.full_X.copy()
        self.full_y_raw = self.full_y.copy()

        if Config.SPLIT_MODE == 'walkforward':
            self.wf_splitter = WalkForwardSplitter(
                self.full_dates,
                train_months=Config.WF_TRAIN_MONTHS,
                val_months=Config.WF_VAL_MONTHS,
                test_months=Config.WF_TEST_MONTHS,
            )
            self.n_windows = len(self.wf_splitter.windows() or [])
            logger.info(f"Walk-Forward 滚动窗口: {self.n_windows} 个 "
                        f"({Config.WF_TRAIN_MONTHS}:{Config.WF_VAL_MONTHS}:{Config.WF_TEST_MONTHS} 月)")
            if self.n_windows > 0:
                self.set_window(0)
        else:
            if Config.SPLIT_MODE == 'month':
                self._set_month_split()
            else:
                self._set_ratio_split()
            self._standardize()

    def _month_anchor(self, min_day: np.datetime64) -> int:
        """锚定月份 = 数据内首个 07-01 所在月 (YYYY-MM), 作为月偏移 off 基准.

        锚定前的行 (off<0, 如 06-30 结尾文件首月) 不进入任何 tr/va/te 掩码.
        """
        year = pd.Timestamp(min_day).year
        july = np.datetime64(f'{year}-07', 'M')
        if min_day.astype('datetime64[M]') > july:
            july = july + np.timedelta64(1, 'Y')
        return int(july.astype(int))

    def _set_month_split(self):
        """按 SPLIT_WARMUP / SPLIT_MONTH_ANCHOR 分派月度切分:
        warmup → 12月回溯 + 36:12:12; 否则 data_start 或 first_07_01."""
        if getattr(Config, 'SPLIT_WARMUP', False):
            self._set_warmup_split()
        elif getattr(Config, 'SPLIT_MONTH_ANCHOR', 'data_start') == 'data_start':
            self._set_month_split_data_start()
        else:
            self._set_month_split_first_july()

    def _set_warmup_split(self):
        """warmup 切分: 前 WARMUP_MONTHS 月仅作特征回溯缓冲 (不进 tr/va/te、不进标准化),
        其后与 data_start 相同: train/+36/val/+12/test/+12 (含尾).

        warmup: [S, S+wa)     train: [S+wa, S+wa+36]     val: (.., +48]     test: (.., +60]
        """
        tr_m, va_m, te_m = Config.TRAIN_MONTHS, Config.VAL_MONTHS, Config.TEST_MONTHS
        wa_m = getattr(Config, 'WARMUP_MONTHS', 12)
        full_days = self.full_dates.astype('datetime64[D]')
        days = np.sort(np.unique(full_days))
        start = self._calendar_anchor(days)
        b_tr = start + pd.DateOffset(months=wa_m)
        b_va = b_tr + pd.DateOffset(months=tr_m)
        b_te = b_va + pd.DateOffset(months=va_m)
        b_end = b_te + pd.DateOffset(months=te_m)

        dates = pd.DatetimeIndex(pd.to_datetime(full_days))
        warm = dates < b_tr
        tr = (dates >= b_tr) & (dates <= b_va)
        va = (dates > b_va) & (dates <= b_te)
        te = (dates > b_te) & (dates <= b_end)

        for name, m in (('train', tr), ('val', va), ('test', te)):
            if m.sum() == 0:
                raise ValueError(f"warmup切分失败: {name} 段为空.")

        self.full_warmup_mask = warm
        self.tr_mask, self.val_mask, self.te_mask = tr, va, te

        def _span(m):
            d = np.sort(np.unique(full_days[m]))
            return f"{pd.Timestamp(d.min()).date()} ~ {pd.Timestamp(d.max()).date()} ({len(d)}日)"

        logger.info(f"warmup切分 {wa_m}:{tr_m}:{va_m}:{te_m} (起点 {start.date()}) | "
                    f"边界: wa<{b_tr.date()}, tr<{b_va.date()}, va<{b_te.date()}, te<{b_end.date()}")
        logger.info(f"  warmup: {_span(warm)} | {int(warm.sum()):,} 行")
        logger.info(f"  train : {_span(tr)} | {int(tr.sum()):,} 行")
        logger.info(f"  val   : {_span(va)} | {int(va.sum()):,} 行")
        logger.info(f"  test  : {_span(te)} | {int(te.sum()):,} 行")

    def _set_month_split_first_july(self):
        """07-01 锚定月切分 (TRAIN_MONTHS:VAL_MONTHS:TEST_MONTHS).

        锚月 = 数据首个 07-01 所在月; 月偏移 off 从锚月起算, 锚定前行剔除.
        train: off∈[0,36)     val: off∈[36,48)     test: off∈[48,60)
        """
        tr_m, va_m, te_m = Config.TRAIN_MONTHS, Config.VAL_MONTHS, Config.TEST_MONTHS
        full_days = self.full_dates.astype('datetime64[D]')
        days = np.sort(np.unique(full_days))
        anchor_int = self._month_anchor(days.min())
        off = (full_days.astype('datetime64[M]').astype(int) - anchor_int).astype(int)
        keep = off >= 0

        tr = keep & (off < tr_m)
        va = keep & (off >= tr_m) & (off < tr_m + va_m)
        te = keep & (off >= tr_m + va_m) & (off < tr_m + va_m + te_m)

        for name, m in (('train', tr), ('val', va), ('test', te)):
            if m.sum() == 0:
                raise ValueError(f"07-01锚定切分失败: {name} 段为空 (数据不足 {tr_m + va_m + te_m} 个月).")

        self.tr_mask, self.val_mask, self.te_mask = tr, va, te

        def _span(m):
            d = np.sort(np.unique(full_days[m]))
            return f"{pd.Timestamp(d.min()).date()} ~ {pd.Timestamp(d.max()).date()} ({len(d)}日)"

        logger.info(f"07-01锚定切分 {tr_m}:{va_m}:{te_m} | 锚月={pd.Timestamp(np.datetime64(anchor_int, 'M')).date()} "
                    f"| 共 {int(off.max()) + 1} 个月 (锚定前剔除 {int((~keep).sum()):,} 行)")
        logger.info(f"  train : {_span(tr)} | {int(tr.sum()):,} 行")
        logger.info(f"  val   : {_span(va)} | {int(va.sum()):,} 行")
        logger.info(f"  test  : {_span(te)} | {int(te.sum()):,} 行")

    def _calendar_anchor(self, days: np.ndarray) -> pd.Timestamp:
        """日历锚点: 文件名义起始日 (06-30). 若数据首个交易日为 7 月上旬
        (即 06-30 为周末/休市), 锚点回退到同年 06-30; 否则取首交易日本身."""
        t = pd.Timestamp(days.min())
        if t.month == 7 and t.day <= 10:
            return pd.Timestamp(t.year, 6, 30)
        return pd.Timestamp(days.min())

    def _set_month_split_data_start(self):
        """数据首日 36:48:60 月边界切分, 全保留 (含首日).

        边界日 = 数据首日 + N 个月 (calendar), 区间半开含尾:
        train: [start, +36mo]     val: (+36mo, +48mo]     test: (+48mo, +60mo]
        """
        tr_m, va_m, te_m = Config.TRAIN_MONTHS, Config.VAL_MONTHS, Config.TEST_MONTHS
        full_days = self.full_dates.astype('datetime64[D]')
        days = np.sort(np.unique(full_days))
        start = self._calendar_anchor(days)
        b_tr = start + pd.DateOffset(months=tr_m)
        b_va = start + pd.DateOffset(months=tr_m + va_m)
        b_te = start + pd.DateOffset(months=tr_m + va_m + te_m)

        dates = pd.DatetimeIndex(pd.to_datetime(full_days))
        tr = dates <= b_tr
        va = (dates > b_tr) & (dates <= b_va)
        te = (dates > b_va) & (dates <= b_te)

        for name, m in (('train', tr), ('val', va), ('test', te)):
            if m.sum() == 0:
                raise ValueError(f"数据首日切分失败: {name} 段为空 (数据不足 {tr_m + va_m + te_m} 个月).")

        self.tr_mask, self.val_mask, self.te_mask = tr, va, te

        def _span(m):
            d = np.sort(np.unique(full_days[m]))
            return f"{pd.Timestamp(d.min()).date()} ~ {pd.Timestamp(d.max()).date()} ({len(d)}日)"

        logger.info(f"数据首日切分 {tr_m}:{va_m}:{te_m} (起点 {start.date()}) | "
                    f"边界: tr<{b_tr.date()}, va<{b_va.date()}, te<{b_te.date()}")
        logger.info(f"  train : {_span(tr)} | {int(tr.sum()):,} 行")
        logger.info(f"  val   : {_span(va)} | {int(va.sum()):,} 行")
        logger.info(f"  test  : {_span(te)} | {int(te.sum()):,} 行")

    def _set_ratio_split(self):
        logger.info("按【自然交易日】划分数据集 (6:2:2)...")
        full_days = pd.to_datetime(self.full_dates).normalize().values
        unique_days = np.sort(np.unique(full_days))
        n_days = len(unique_days)

        t_end_day = unique_days[int(n_days * Config.TRAIN_RATIO)]
        v_end_day = unique_days[int(n_days * (Config.TRAIN_RATIO + Config.VAL_RATIO))]

        self.tr_mask = full_days <= t_end_day
        self.val_mask = (full_days > t_end_day) & (full_days <= v_end_day)
        self.te_mask = full_days > v_end_day

    def set_window(self, window_idx: int) -> Dict[str, str]:
        """切换到指定 walk-forward 窗口: 设置行级掩码并按该窗口 train 重新标准化."""
        row_masks = self.wf_splitter.to_row_masks(window_idx, self.full_dates)
        self.tr_mask = row_masks['train']
        self.val_mask = row_masks['val']
        self.te_mask = row_masks['test']
        self._standardize()
        return self.wf_splitter.window_dates(window_idx)

    def _standardize(self):
        """基于当前 tr_mask 拟合标准化 (均值/方差) 与 y 截断 (1-99pct), 避免跨窗口泄漏."""
        self.full_X = self.full_X_raw.copy()
        self.full_y = self.full_y_raw.copy()

        tr_X = self.full_X[self.tr_mask]
        mean_vals = np.mean(tr_X, axis=0)
        std_vals = np.std(tr_X, axis=0)
        std_vals[std_vals < 1e-8] = 1.0
        self.full_X = (self.full_X - mean_vals) / std_vals

        tr_y = self.full_y[self.tr_mask]
        lo, hi = np.percentile(tr_y, [1, 99])
        self.full_y = np.clip(self.full_y, lo, hi)

        logger.info(
            f"数据集就绪 | 总样本: {len(self.full_X):,} | "
            f"Train/Val/Test: {self.tr_mask.sum():,}/{self.val_mask.sum():,}/{self.te_mask.sum():,}")

    def get_subset(self, trial_features: List[str]) -> Dict[str, Tuple]:
        idx = [self.feature_to_idx[f] for f in trial_features if f in self.feature_to_idx]
        if not idx:
            raise ValueError(f"无匹配的有效特征")
        X_sub = self.full_X[:, idx]
        return {
            'train': (X_sub[self.tr_mask], self.full_y[self.tr_mask], self.full_dates[self.tr_mask]),
            'val': (X_sub[self.val_mask], self.full_y[self.val_mask], self.full_dates[self.val_mask]),
            'test': (X_sub[self.te_mask], self.full_y[self.te_mask],
                     self.full_dates[self.te_mask], self.full_symbols[self.te_mask],
                     self.full_amount[self.te_mask]),
            'all': (X_sub, self.full_y, self.full_dates, self.full_symbols, self.full_amount)
        }

    def prepare_timing_dataset(self, df_raw: pd.DataFrame):
        """时序择时数据准备 — 从成分股数据构建因子 + 标签.

        Args:
            df_raw: 成分股日线数据, 需含 date/symbol/open/high/low/close/volume/amount.
        """
        logger.info("构建时序择时数据集 (含截面因子)...")

        df_raw = df_raw.copy()
        df_raw['date'] = pd.to_datetime(df_raw['date'])

        full_factors = TimingFeatureRegistry.build_timing_factors(df_raw)

        # 等权指数用于 target
        agg = df_raw.groupby('date').agg(
            close=('close', 'mean'),
        ).reset_index().sort_values('date').reset_index(drop=True)

        full_factors = full_factors.merge(agg[['date', 'close']], on='date', how='left')
        full_factors['forward_ret'] = full_factors['close'].pct_change(Config.HORIZON).shift(-Config.HORIZON).values
        full_factors.drop(columns=['close'], inplace=True)

        if 'amount' in df_raw.columns:
            amt_daily = df_raw.groupby('date')['amount'].mean().reset_index()
            full_factors = full_factors.merge(amt_daily, on='date', how='left')
        if 'volume' in df_raw.columns:
            vol_daily = df_raw.groupby('date')['volume'].mean().reset_index()
            full_factors = full_factors.merge(vol_daily, on='date', how='left')

        full_factors = full_factors.dropna(subset=['forward_ret', 'date'])

        self.full_dates = full_factors['date'].values
        self.full_symbols = np.array(['index'] * len(full_factors))
        self.full_y = full_factors['forward_ret'].values.astype(np.float32)
        self.full_amount = full_factors['amount'].values.astype(np.float64) if 'amount' in full_factors.columns else np.ones(len(full_factors), dtype=np.float64)
        self.full_turn = np.zeros(len(full_factors), dtype=np.float64)
        self.full_open = None
        self.full_close = agg.set_index('date').reindex(full_factors['date'])['close'].values.astype(np.float64) if 'close' in agg.columns else None

        valid_cols = [c for c in full_factors.columns if c not in ('date', 'forward_ret', 'amount', 'volume')]
        X_raw = full_factors[valid_cols].values.astype(np.float32)

        logger.info("执行时序因子质量门禁...")
        valid_indices = []
        for i, feat in enumerate(valid_cols):
            col_data = X_raw[:, i]
            nan_ratio = np.isnan(col_data).mean()
            if nan_ratio > Config.NaN_RATIO_CAP:
                logger.info(f"  剔除因子 {feat}: 缺失率 {nan_ratio:.1%} > {Config.NaN_RATIO_CAP:.0%}")
                continue
            if np.nanvar(col_data) < 1e-6:
                continue
            valid_indices.append(i)

        X_clean = X_raw[:, valid_indices]
        valid_cols = [valid_cols[i] for i in valid_indices]
        logger.info(f"质量门禁后保留 {len(valid_cols)} 个因子。")

        X_clean = np.nan_to_num(X_clean, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

        self.valid_features = valid_cols
        self.full_X = X_clean
        self.feature_to_idx = {f: i for i, f in enumerate(self.valid_features)}
        Config.FINAL_FEATURE_POOL = self.valid_features
        Config.INITIAL_FEATURE_COLS = self.valid_features

        self.full_X_raw = self.full_X.copy()
        self.full_y_raw = self.full_y.copy()

        if Config.SPLIT_MODE == 'walkforward':
            self.wf_splitter = WalkForwardSplitter(
                self.full_dates,
                train_months=Config.WF_TRAIN_MONTHS,
                val_months=Config.WF_VAL_MONTHS,
                test_months=Config.WF_TEST_MONTHS,
            )
            self.n_windows = len(self.wf_splitter.windows() or [])
            logger.info(f"Walk-Forward 滚动窗口: {self.n_windows} 个")
            if self.n_windows > 0:
                self.set_window(0)
        else:
            if Config.SPLIT_MODE == 'month':
                self._set_month_split()
            else:
                self._set_ratio_split()
            self._standardize()
