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

    MIN_FEATURES = 3
    MAX_FEATURES = 5
    TARGET_FACTOR_POOL_SIZE = 85

    HORIZON = 1
    BATCH_SIZE = 50000

    TSTAT_THRESHOLD = 2.0
    TSTAT_STRONG = 3.0
    ICIR_MIN = 0.05
    IC_THRESHOLD = 0.02
    NW_LAGS = 3
    USE_CONTROLS = False

    PY_SR_ITERATIONS = 60
    PY_SR_POPULATIONS = 30
    PY_SR_MAXSIZE = 15
    PY_SR_MAXDEPTH = 8
    PY_SR_PARSIMONY = 0.001

    GFN_HIDDEN_DIM = 64
    GFN_LR = 1e-3
    GFN_NUM_HEADS = 4

    OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "outputs")
    OUTPUT_DIR = OUTPUT_ROOT
    REGISTRY_FILE = "registry_academic.json"
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    REBALANCE_FREQ_DAYS = 20

    USE_RESIDUAL_NEUTRALIZATION = True
    USE_MONOTONICITY_GATE = True
    USE_DYNAMIC_DIVERSITY = True
    USE_RANK_FEATURES = False

    REWARD_CONFIG = {
        'zs500': {
            't_stat':    {'slope': 2.0, 'threshold': 2.0},
            'icir':      {'slope': 8,   'threshold': 0.10},
            'ic':        {'slope': 80,  'threshold': 0.03},
            'weight':    {'t_stat': 0.20, 'icir': 0.40, 'ic': 0.40},
        },
        'hs300': {
            't_stat':    {'slope': 3.0, 'threshold': 2.0},
            'icir':      {'slope': 6,   'threshold': 0.08},
            'ic':        {'slope': 40,  'threshold': 0.02},
            'weight':    {'t_stat': 0.35, 'icir': 0.35, 'ic': 0.30},
        },
    }

    MARKET = 'zs500'

    @classmethod
    def get_reward_config(cls) -> dict:
        return cls.REWARD_CONFIG[cls.MARKET]

def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Seed={seed}")
