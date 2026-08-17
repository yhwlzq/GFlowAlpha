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

    SPLIT_MODE = 'ratio'  # 'ratio' (默认 6:2:2) 或 'walkforward'
    WF_TRAIN_MONTHS = 27
    WF_VAL_MONTHS = 12
    WF_TEST_MONTHS = 3
    WF_PURGE_DAYS = 1
    WF_EMBARGO_DAYS = 2

    MIN_FEATURES = 3
    MAX_FEATURES = 5
    TARGET_FACTOR_POOL_SIZE = 85
    CLUSTER_FEATURE_POOL = True   # False = 跳过聚类正交化，使用全量门禁池

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

    MARKET = 'zs500'

    # 特征池选择: 'buildin' = 自建语义因子池 (UltimateDaily/LargeCap registry),
    #            'alpha158' = Qlib Alpha158 工程化因子池
    FEATURE_POOL = 'buildin'

    @classmethod
    def get_reward_config(cls) -> dict:
        return cls.REWARD_CONFIG[cls.MARKET]

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
