import logging
import warnings
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from alpha.config import Config
from alpha.features.feature_registry import UltimateDailyFeatureRegistry, LargeCapFeatureRegistry

logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        self.full_X = self.full_y = None
        self.full_dates = self.full_symbols = None
        self.tr_mask = self.val_mask = self.te_mask = None
        self.valid_features = []
        self.feature_to_idx = {}
        self.full_amount = None

    def prepare_full_pool(self, df_raw: pd.DataFrame):
        logger.info("构建日频数据集并执行特征质量门禁...")
        if Config.MARKET == 'hs300':
            full_factors = LargeCapFeatureRegistry.build_large_cap_factors(df_raw)
        else:
            full_factors = UltimateDailyFeatureRegistry.build_ultimate_factors(df_raw)

        df_raw = df_raw.sort_values(['symbol', 'date'])
        labels = df_raw.groupby('symbol')['close'].pct_change(Config.HORIZON).shift(-Config.HORIZON)

        full_factors['forward_ret'] = labels.values
        if 'amount' in df_raw.columns:
            amt_map = df_raw[['symbol', 'date', 'amount']].drop_duplicates(subset=['symbol', 'date'])
            full_factors = full_factors.merge(amt_map, on=['symbol', 'date'], how='left')
        full_factors = full_factors.dropna(subset=['forward_ret', 'date', 'symbol'])

        self.full_dates = full_factors['date'].values
        self.full_symbols = full_factors['symbol'].values
        self.full_y = full_factors['forward_ret'].values.astype(np.float32)
        self.full_amount = full_factors['amount'].values.astype(np.float64) if 'amount' in full_factors.columns else np.zeros(len(full_factors), dtype=np.float64)

        X_raw = full_factors[Config.INITIAL_FEATURE_COLS].values.astype(np.float32)

        logger.info("执行特征质量门禁...")
        warnings.filterwarnings('ignore', message='invalid value encountered in reduce')
        X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        valid_indices = []
        for i, feat in enumerate(Config.INITIAL_FEATURE_COLS):
            col_data = X_raw[:, i]
            if np.isnan(col_data).mean() > 0.30:
                continue
            if np.nanvar(col_data) < 1e-6:
                continue
            valid_indices.append(i)

        X_clean = X_raw[:, valid_indices]
        valid_cols = [Config.INITIAL_FEATURE_COLS[i] for i in valid_indices]
        logger.info(f"质量门禁后保留 {len(valid_cols)} 个因子。")

        effective_pool_size = min(Config.TARGET_FACTOR_POOL_SIZE, len(valid_cols))
        logger.info(f"启动层次聚类正交化筛选 (目标池: {effective_pool_size})...")
        X_clean = np.nan_to_num(X_clean, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

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

        logger.info("按【自然交易日】划分数据集...")
        full_days = pd.to_datetime(self.full_dates).normalize().values
        unique_days = np.sort(np.unique(full_days))
        n_days = len(unique_days)

        t_end_day = unique_days[int(n_days * Config.TRAIN_RATIO)]
        v_end_day = unique_days[int(n_days * (Config.TRAIN_RATIO + Config.VAL_RATIO))]

        self.tr_mask = full_days <= t_end_day
        self.val_mask = (full_days > t_end_day) & (full_days <= v_end_day)
        self.te_mask = full_days > v_end_day

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
