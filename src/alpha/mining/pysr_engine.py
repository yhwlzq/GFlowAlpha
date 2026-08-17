import logging
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from scipy.stats import rankdata, norm
from alpha.config import Config

logger = logging.getLogger(__name__)

try:
    from pysr import PySRRegressor
    HAS_PYSR = True
except Exception:
    HAS_PYSR = False
    logger.warning("PySR 无法初始化（Julia 环境可能缺失）")


class PySRMiningEngine:
    def __init__(self):
        self.un_ops = ["square", "abs", "log1p", "sign", "inv"]
        self.bin_ops = ["+", "-", "*", "/"]

    def run(self, X_train: pd.DataFrame, y_train: np.ndarray) -> Optional[Tuple[str, int]]:
        if not HAS_PYSR or X_train.empty:
            return None
        X_tr_s = np.nan_to_num(np.asarray(X_train, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        X_tr_s = np.clip(X_tr_s, -10.0, 10.0)

        y_tr_s = np.asarray(y_train, dtype=np.float32)
        y_ranks = rankdata(y_tr_s, method='average')
        y_tr_s = norm.ppf(y_ranks / (len(y_ranks) + 1))
        y_tr_s = np.nan_to_num(y_tr_s, nan=0.0, posinf=3.0, neginf=-3.0)
        y_tr_s = np.clip(y_tr_s, -3.0, 3.0)

        model = PySRRegressor(
            niterations=Config.PY_SR_ITERATIONS, populations=Config.PY_SR_POPULATIONS,
            maxsize=Config.PY_SR_MAXSIZE, maxdepth=Config.PY_SR_MAXDEPTH,
            binary_operators=self.bin_ops,
            unary_operators=self.un_ops, elementwise_loss="L2DistLoss()",
            parsimony=Config.PY_SR_PARSIMONY,
            deterministic=True, parallelism='serial', random_state=Config.PYSR_SEED, procs=0,
            batching=True, batch_size=10000, warmup_maxsize_by=2,
            timeout_in_seconds=600, verbosity=1)
        try:
            logger.info("PySR 开始训练...")
            model.fit(X_tr_s, y_tr_s, variable_names=list(X_train.columns))

            equations_df = model.equations_
            if equations_df is not None and len(equations_df) > 0:
                eq_col = 'equation' if 'equation' in equations_df.columns else 'Equation'
                cmp_col = next((c for c in ['complexity', 'Complexity'] if c in equations_df.columns), None)
                loss_col = next((c for c in ['loss', 'Loss', 'MSE'] if c in equations_df.columns), None)
                score_col = next((c for c in ['score', 'Score'] if c in equations_df.columns), None)

                logger.info(f"PySR 训练完成 | Hall of Fame: {len(equations_df)} 个候选公式")
                if cmp_col and loss_col:
                    logger.info(f"  {'复杂度':<8} {'Loss':<14} {'Score':<12}  公式预览")
                    logger.info("-" * 70)
                    for _, row in equations_df.iterrows():
                        eq_str = str(row.get(eq_col, ''))[:50]
                        logger.info(f"  {int(row[cmp_col]):<8} {float(row[loss_col]):<14.6f} {float(row.get(score_col, 0)):<12.6f}  {eq_str}")

                    if len(equations_df) >= 3:
                        best_loss = float(equations_df.iloc[0][loss_col])
                        mid_loss = float(equations_df.iloc[len(equations_df)//2][loss_col])
                        logger.info(f"  Best Loss: {best_loss:.6f}  |  Median Loss: {mid_loss:.6f}")
                        if mid_loss > 0 and best_loss / mid_loss > 0.9:
                            logger.warning("复杂度对 loss 改善有限，可能存在过拟合风险")

            best = model.get_best()
            if best is None:
                return None
            best_row = best if isinstance(best, pd.Series) else best.iloc[0]
            eq = str(best_row['equation']).replace(" ", "")
            complexity = int(best_row.get('complexity', 0))
            if not eq or len(eq) > 150:
                return None
            logger.info(f"最佳公式复杂度: {complexity}")
            return eq, complexity
        except Exception as e:
            logger.warning(f"PySR执行失败: {e}")
        return None
