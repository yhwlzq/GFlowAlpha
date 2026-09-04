import logging
import os
import random
import numpy as np
import torch

logger = logging.getLogger(__name__)

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
)

class Config:
    INITIAL_FEATURE_COLS = []
    FINAL_FEATURE_POOL = []

    TRAIN_RATIO = 0.6
    VAL_RATIO = 0.2

    SPLIT_MODE = 'ratio'  # 'ratio' (默认 6:2:2) / 'month' (月度锚定或 warmup) / 'walkforward'
    SPLIT_MONTH_ANCHOR = 'data_start'  # 'data_start'(冷启动: 数据首日含首日, 默认) / 'first_07_01'(07-01锚定, 剔锚前行)
    SPLIT_WARMUP = False               # True = warmup 模式: 前 WARMUP_MONTHS 月仅作特征回溯, 不进 tr/va/te/标准化
    WARMUP_MONTHS = 12
    TRAIN_MONTHS = 36
    VAL_MONTHS = 12
    TEST_MONTHS = 12
    WF_TRAIN_MONTHS = 27
    WF_VAL_MONTHS = 12
    WF_TEST_MONTHS = 3
    WF_PURGE_DAYS = 1
    WF_EMBARGO_DAYS = 2

    MIN_FEATURES = 3
    MAX_FEATURES = 5
    TARGET_FACTOR_POOL_SIZE = 999
    CLUSTER_FEATURE_POOL = True   # False = 跳过聚类正交化，使用全量门禁池

    # 聚类前近重复去重: 仅剔除 Spearman 正相关 > 阈值的镜像对 (按 |IC| 保最高),
    # 负相关 (如 rev_* vs ret_* 反转/动量镜像对) 不作为重复, 全部保留。
    DEDUP_NEAR_DUPLICATES = True
    NEAR_DUPLICATE_CORR = 0.95

    # 特征质量门禁: 缺失率超过该阈值的因子列直接剔除 (须在 NaN 填充前评估)
    NaN_RATIO_CAP = 0.30
    # 行级缺失清洗: 特征缺失率超过 HIGH_NAN_ROW_FRAC 的行剔除 (warmup/短史污染行)
    DROP_HIGH_NAN_ROWS = True
    HIGH_NAN_ROW_FRAC = 0.50

    HORIZON = 1
    BATCH_SIZE = 50000

    TSTAT_THRESHOLD = 2.0
    TSTAT_STRONG = 3.0
    ICIR_MIN = 0.05
    IC_THRESHOLD = 0.016
    PRODUCT_FMT_THRESHOLD = 3.0
    PRODUCT_IC_THRESHOLD = 0.03
    PRODUCT_ICIR_THRESHOLD = 0.3
    REGISTER_IC_MIN = 0.01
    REGISTER_ICIR_MIN = 0.10
    NW_LAGS = 3

    # === 公式选型加固 (top-K 稳定性择优) ===
    VAL_SELECT_TOP_K = 3            # 参与稳定性排序的候选数 (按 PySR 训练侧评分排序后取前 K); 0=关闭(legacy best)
    FORMULA_MAX_CAND_CMPLX = 12     # 候选复杂度上限, 过滤过拟合长公式
    STABILITY_ENTRANCE_T = 2.0      # 候选进入扰动排序的原始 val |t| 门槛 (与 val 门 ≥2 一致)
    STABILITY_T_MIN = 1.0           # 单扰动态 val |t| 达标阈值
    STABILITY_MIN_HITS = 2          # 至少几个扰动态达标(且符号一致)视为稳定
    TEST_TSTAT_MIN = 2.0            # 注册最低 test |t| (与 val 门一致)
    TEST_TSTAT_STRONG = 2.5         # 生产底线: test |t| 低于此值强制标 WEAK

    REWARD_HARD = 0.01
    REWARD_SOFT_FLOOR = 0.3
    REWARD_TOP = 10.0
    SIGN_MISMATCH_PENALTY = 0.1
    EXCLUDE_WEAK_FROM_PRODUCTION = True
    USE_CONTROLS = False

    PY_SR_ITERATIONS = 60
    PY_SR_POPULATIONS = 30
    PY_SR_MAXSIZE = 15
    PY_SR_MAXDEPTH = 8
    PY_SR_PARSIMONY = 0.001

    GFN_HIDDEN_DIM = 64
    GFN_LR = 1e-3
    GFN_NUM_HEADS = 4

    OUTPUT_DIR = "factor_output_academic_v81"
    REGISTRY_FILE = "registry_academic.json"
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    REBALANCE_FREQ_DAYS = 20

    # === 随机种子统一配置 (所有 seed 在此修改) ===
    SEED = 42              # 全局基础种子: torch/numpy/random 及 CLI 默认
    PYSR_SEED = 42         # PySR 符号回归
    GP_SEED = 42           # gplearn 遗传规划基线
    LGBM_SEED = 42         # LightGBM 黑盒基线 (第 i 个独立 seed = LGBM_SEED + i)

    USE_RESIDUAL_NEUTRALIZATION = True
    USE_MONOTONICITY_GATE = True
    USE_DYNAMIC_DIVERSITY = True
    USE_RANK_FEATURES = False

    NEUTRALIZE_CONTROLS = ["amount"]

    # 跨子期稳健 reward: 默认关 (ratio 等入口走原逻辑), walkforward 脚本里按需开启
    USE_SUBPERIOD_ROBUST_REWARD = False
    SUBPERIOD_ROBUST_SPLITS = 3

    REWARD_CONFIG = {
        'zs500': {
            't_stat':    {'slope': 2.0, 'threshold': TSTAT_THRESHOLD},
            'icir':      {'slope': 25,  'threshold': 0.10},
            'ic':        {'slope': 150, 'threshold': 0.03},
            'weight':    {'t_stat': 0.20, 'icir': 1.0, 'ic': 1.0},
        },
        'hs300': {
            't_stat':    {'slope': 3.0, 'threshold': TSTAT_THRESHOLD},
            'icir':      {'slope': 20,  'threshold': 0.08},
            'ic':        {'slope': 100, 'threshold': 0.02},
            'weight':    {'t_stat': 0.35, 'icir': 1.0, 'ic': 1.0},
        },
    }

    # === 市场维度配置: FM 回归最低股票数 & 稳定性入口 t 统计量 ===
    MARKET_CONFIG = {
        'zs500': {
            'fm_min_stocks': 30,           # FM 回归每日期最低股票数 (500只的6%)
            'stability_entrance_t': 2.0,   # 候选进入稳定性排序的 val|t| 门槛
        },
        'hs300': {
            'fm_min_stocks': 20,           # HS300 共298只，20只≈7%（OLS 2参数最低要求）
            'stability_entrance_t': 0.8,   # 大盘股 alpha 弱，进一步放宽
        },
    }

    MARKET = 'zs500'

    # === 运行模式: 'cross_sectional' (截面选股) / 'timing' (时序择时) ===
    MODE = 'cross_sectional'

    # === 时序择时专用配置 ===
    TIMING_DIRECTION_THRESHOLD = 0.52   # 方向准确率门禁 (val集)
    TIMING_SHARPE_THRESHOLD = 0.3       # 择时 Sharpe 门禁 (test集)
    TIMING_HIT_RATE_MIN = 0.52          # 最低命中率 (reward sigmoid 中心)
    TIMING_TEST_TSTAT_MIN = 0.8         # test集最低 |t|
    TIMING_TEST_DIR_MIN = 0.48          # test集最低方向准确率
    TIMING_STRONG_SHARPE = 1.0          # STRONG 门槛: |sharpe|
    TIMING_STRONG_DIR = 0.55            # STRONG 门槛: 方向准确率

    # 特征池选择: 'buildin' = 自建语义因子池 (UltimateDaily/LargeCap registry),
    #            'alpha158' = Qlib Alpha158 工程化因子池
    FEATURE_POOL = 'buildin'

    @classmethod
    def get_reward_config(cls) -> dict:
        return cls.REWARD_CONFIG[cls.MARKET]

    @classmethod
    def get_market_config(cls) -> dict:
        return cls.MARKET_CONFIG.get(cls.MARKET, cls.MARKET_CONFIG['zs500'])


def apply_split_mode(mode: str = 'warmup'):
    """切分模式设置, 与 transform_pysr_primary.console_main 的逻辑完全一致.

    mode: 'warmup'(前12月回溯+36:12:12, 默认) / 'cold'(冷启动 36:12:12) /
          'ratio'(自然交易日 6:2:2) / 'month'(月度锚定 data_start, 等同 cold).
    """
    if mode == 'ratio':
        Config.SPLIT_MODE = 'ratio'
        Config.SPLIT_WARMUP = False
    elif mode == 'warmup':
        Config.SPLIT_MODE = 'month'
        Config.SPLIT_MONTH_ANCHOR = 'data_start'
        Config.SPLIT_WARMUP = True
    else:
        Config.SPLIT_MODE = 'month'
        Config.SPLIT_MONTH_ANCHOR = 'data_start'
        Config.SPLIT_WARMUP = False


def set_global_seed(seed=None):
    if seed is None:
        seed = Config.SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Seed={seed}")
