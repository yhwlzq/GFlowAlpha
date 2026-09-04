import logging
import os
import time
import re
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
import sympy as sp
from scipy.stats import spearmanr
from alpha.config import Config
from .safe_ops import SafeOps
from .preprocessor import DataPreprocessor
from .fm_regression import FMRegressionResult, FamaMacBethRegressor, TimingResult, TimingEvaluator
from .pysr_engine import PySRMiningEngine
from .gflownet import FeatureMetadataExtractor, GFlowNetPolicy_Transformer, TrajectorySampler_Transformer, GFlowNetTrainerInternal
from .registry import FactorRegistry

logger = logging.getLogger(__name__)


class MiningOrchestrator:
    def __init__(self, preprocessor: DataPreprocessor,
                 name: str = "V9.1",
                 report_title: str = "因子挖掘汇总报告",
                 registry_version: str = "9.1_ICIR_Prior"):
        self.prep = preprocessor
        self.name = name
        self.report_title = report_title
        self.fm_regressor = FamaMacBethRegressor(nw_lags=Config.NW_LAGS)
        self.timing_evaluator = TimingEvaluator(nw_lags=Config.NW_LAGS)
        self.registry = FactorRegistry(version=registry_version)
        self.device = Config.DEVICE

        self.metadata_extractor = FeatureMetadataExtractor(preprocessor)
        feat_attrs_tensor = self.metadata_extractor.build_metadata_matrix()

        self.gfn_model = GFlowNetPolicy_Transformer(
            num_feats=len(Config.FINAL_FEATURE_POOL),
            attr_dim=feat_attrs_tensor.shape[1],
            hidden_dim=Config.GFN_HIDDEN_DIM,
            num_heads=Config.GFN_NUM_HEADS
        ).to(self.device)

        self.gfn_sampler = TrajectorySampler_Transformer(
            model=self.gfn_model, feature_pool=Config.FINAL_FEATURE_POOL,
            feat_attrs=feat_attrs_tensor, min_select=Config.MIN_FEATURES,
            max_select=Config.MAX_FEATURES, device=self.device)

        self.gfn_trainer = GFlowNetTrainerInternal(self.gfn_model, Config.GFN_LR)

        self.feature_usage = {}
        self.trial_feature_counter = {}
        self.trial_count = 0
        self.total_passed = 0
        self.registered_formulas = []
        self.stable_candidates = 0      # 统计诚实: 进入 val FM 评分的候选数
        self.gate_passed = 0            # 统计诚实: 通过 TSTAT+单调性等门禁的候选数
        self.register_attempts = 0      # 统计诚实: 进入 test 注册判定的候选数
        self.soft_mask = {}
        self.hard_mask = {}
        self.pass_counter = 0
        self.structural_fingerprints = set()
        self.structural_signatures = set()

        self.pysr_ns = {
            "square": np.square, "sigmoid": SafeOps.safe_sigmoid,
            "log1p": SafeOps.safe_log1p, "inv": SafeOps.safe_inv,
            "/": SafeOps.protected_div,
            "sign": np.sign, "abs": np.abs,
        }
        logger.info(f"Orchestrator {self.name} 就绪 | 核心因子池: {len(Config.FINAL_FEATURE_POOL)} | "
                     f"设备: {self.device.upper()}")

    @staticmethod
    def _normalize_structure(formula: str, feature_names: List[str]) -> str:
        float_re = re.compile(r'(?<![a-zA-Z])([-]?\d+\.?\d*(?:[eE][-+]?\d+)?)(?![a-zA-Z])')
        ns = {name: sp.Symbol(name) for name in feature_names}
        _C_ = sp.Symbol('_C_')
        ns.update({
            '_C_': _C_,
            'square': lambda x: x ** 2,
            'abs': sp.Abs,
        })
        try:
            s = float_re.sub('_C_', formula)
            expr = sp.sympify(s, locals=ns)
            expr = sp.expand(expr)

            def _squash_and_sort(e):
                if isinstance(e, sp.Symbol):
                    return e
                if isinstance(e, (sp.Float, sp.Integer, sp.Rational)):
                    return _C_
                args = [_squash_and_sort(a) for a in e.args]
                if isinstance(e, sp.Mul):
                    non_c = [a for a in args if a != _C_]
                    if non_c:
                        return sp.Mul(*sorted(non_c, key=str))
                    return _C_
                if isinstance(e, sp.Add):
                    return sp.Add(*sorted(args, key=str))
                if isinstance(e, sp.Pow):
                    base, exp = args
                    if base == _C_ and exp == _C_:
                        return _C_
                    if exp == _C_:
                        return sp.Pow(base, exp)
                    return sp.Pow(base, exp)
                return e.func(*args)

            norm = _squash_and_sort(expr)
            return str(norm).replace(' ', '').replace('_C_', 'K')
        except Exception:
            s = float_re.sub('K', formula)
            return s.replace(' ', '')

    @staticmethod
    def _canonical_signature(formula: str, feature_names: List[str]) -> str:
        norm = MiningOrchestrator._normalize_structure(formula, feature_names)
        for feat in sorted(set(feature_names), key=lambda x: -len(x)):
            norm = norm.replace(feat, 'Feat')
        return norm

    @staticmethod
    def _check_monotonicity(quantile_returns, t_stat: float = 0.0) -> bool:
        if len(quantile_returns) < 5:
            return False
        direction = 1 if t_stat >= 0 else -1
        spread = quantile_returns[-1] - quantile_returns[0]
        passed = direction * spread > 0
        logger.info(f"单调性 | spread={spread:.6f} dir={direction} {'✓' if passed else '✗'}")
        return passed

    def _compute_formula_similarity(self, feats1: List[str], feats2: List[str]) -> float:
        set1, set2 = set(feats1), set(feats2)
        if not set1 or not set2:
            return 0.0
        return len(set1 & set2) / len(set1 | set2)

    def _compute_formula_effective_features(self, formula: str, used_features: List[str]) -> int:
        count = 0
        for feat in used_features:
            if feat in formula:
                count += 1
        return count

    def _compute_reward(self, fm_result: FMRegressionResult, used_features: List[str],
                        formula: str = "", robustness: float = 1.0) -> Tuple[float, Dict]:
        debug = {}

        abs_t = abs(fm_result.t_stat)
        raw_icir = fm_result.rank_icir
        abs_icir = abs(raw_icir)
        abs_ic = abs(fm_result.rank_ic_mean)

        rc = Config.get_reward_config()
        t_score    = 1.0 / (1.0 + np.exp(-rc['t_stat']['slope'] * (abs_t - rc['t_stat']['threshold'])))
        icir_score = 1.0 / (1.0 + np.exp(-rc['icir']['slope']   * (abs_icir - rc['icir']['threshold'])))
        ic_score   = 1.0 / (1.0 + np.exp(-rc['ic']['slope']     * (abs_ic - rc['ic']['threshold'])))
        debug['t_ctrl_score'] = t_score
        debug['icir_score'] = icir_score
        debug['ic_score'] = ic_score

        reward = (Config.REWARD_SOFT_FLOOR +
                  (Config.REWARD_TOP - Config.REWARD_SOFT_FLOOR) *
                  (t_score ** rc['weight']['t_stat']) *
                  (icir_score ** rc['weight']['icir']) *
                  (ic_score ** rc['weight']['ic']))

        raw_ic = fm_result.rank_ic_mean
        if np.sign(fm_result.t_stat) * np.sign(raw_ic) < 0 or \
           np.sign(fm_result.t_stat) * np.sign(raw_icir) < 0:
            reward *= Config.SIGN_MISMATCH_PENALTY
            debug['sign_penalty'] = Config.SIGN_MISMATCH_PENALTY
            debug['reason'] = 'sign_mismatch'

        if formula:
            eff_feats = self._compute_formula_effective_features(formula, used_features)
            if eff_feats <= 1:
                reward *= 0.2
                debug['single_feat_penalty'] = 0.2

        if Config.USE_DYNAMIC_DIVERSITY and used_features:
            novel_count = 0
            for feat in used_features:
                usage = self.feature_usage.get(feat, 0)
                if usage <= max(1, int(self.total_passed * 0.15)):
                    novel_count += 1
            diversity_mult = min(1.5, 1.0 + 0.15 * novel_count)
            reward *= diversity_mult
            debug['div_mult'] = diversity_mult

            sim_penalty = 1.0
            for _, _, reg_feats in self.registered_formulas:
                sim = self._compute_formula_similarity(used_features, reg_feats)
                if sim > 0.75:
                    sim_penalty = 0.1
                    break
            if sim_penalty < 1.0:
                reward *= sim_penalty
                debug['sim_penalty'] = sim_penalty

        if Config.USE_SUBPERIOD_ROBUST_REWARD:
            robust_mult = 0.3 + 0.7 * robustness
            reward *= robust_mult
            debug['robustness'] = robustness
            debug['robust_mult'] = robust_mult

        final_reward = max(reward, Config.REWARD_SOFT_FLOOR)
        debug['final_reward'] = final_reward
        return final_reward, debug

    def _compute_timing_reward(self, timing_result: TimingResult, used_features: List[str],
                                formula: str = "", robustness: float = 1.0) -> Tuple[float, Dict]:
        """时序择时模式的 reward 计算."""
        debug = {}

        abs_sharpe = abs(timing_result.timing_sharpe)
        dir_acc = timing_result.direction_accuracy
        abs_t = abs(timing_result.t_stat)

        # Sigmoid scoring for each component
        sharpe_score = 1.0 / (1.0 + np.exp(-2.0 * (abs_sharpe - Config.TIMING_SHARPE_THRESHOLD)))
        dir_score = 1.0 / (1.0 + np.exp(-10.0 * (dir_acc - Config.TIMING_HIT_RATE_MIN)))
        t_score = 1.0 / (1.0 + np.exp(-1.5 * (abs_t - 1.5)))

        debug['sharpe_score'] = sharpe_score
        debug['dir_score'] = dir_score
        debug['t_ctrl_score'] = t_score

        reward = (Config.REWARD_SOFT_FLOOR +
                  (Config.REWARD_TOP - Config.REWARD_SOFT_FLOOR) *
                  (sharpe_score ** 0.5) *
                  (dir_score ** 1.0) *
                  (t_score ** 0.3))

        if formula:
            eff_feats = self._compute_formula_effective_features(formula, used_features)
            if eff_feats <= 1:
                reward *= 0.2
                debug['single_feat_penalty'] = 0.2

        if Config.USE_DYNAMIC_DIVERSITY and used_features:
            novel_count = 0
            for feat in used_features:
                usage = self.feature_usage.get(feat, 0)
                if usage <= max(1, int(self.total_passed * 0.15)):
                    novel_count += 1
            diversity_mult = min(1.5, 1.0 + 0.15 * novel_count)
            reward *= diversity_mult
            debug['div_mult'] = diversity_mult

            sim_penalty = 1.0
            for _, _, reg_feats in self.registered_formulas:
                sim = self._compute_formula_similarity(used_features, reg_feats)
                if sim > 0.75:
                    sim_penalty = 0.1
                    break
            if sim_penalty < 1.0:
                reward *= sim_penalty
                debug['sim_penalty'] = sim_penalty

        if Config.USE_SUBPERIOD_ROBUST_REWARD:
            robust_mult = 0.3 + 0.7 * robustness
            reward *= robust_mult
            debug['robustness'] = robustness
            debug['robust_mult'] = robust_mult

        final_reward = max(reward, Config.REWARD_SOFT_FLOOR)
        debug['final_reward'] = final_reward
        return final_reward, debug

    @staticmethod
    def _timing_registration_verdict(timing_result_test) -> Tuple[bool, str, Optional[str]]:
        """时序择时模式的 test 集注册门禁."""
        t_te = timing_result_test.t_stat
        dir_acc = timing_result_test.direction_accuracy
        sharpe = timing_result_test.timing_sharpe

        t_min = Config.TIMING_TEST_TSTAT_MIN
        dir_min = Config.TIMING_TEST_DIR_MIN
        if abs(t_te) < t_min:
            return False, f"test |t| 过低 ({t_te:.3f} < {t_min})", None
        if dir_acc < dir_min:
            return False, f"test 方向准确率 < {dir_min:.0%} ({dir_acc:.2%})", None
        if abs(sharpe) < Config.TIMING_SHARPE_THRESHOLD:
            return False, f"test 择时 Sharpe 不达标 ({sharpe:.3f} < {Config.TIMING_SHARPE_THRESHOLD})", None

        gate_status = 'STRONG' if (abs(sharpe) >= Config.TIMING_STRONG_SHARPE and dir_acc >= Config.TIMING_STRONG_DIR) else 'WEAK'
        return True, "", gate_status

    @staticmethod
    def _registration_verdict(fm_result_test) -> Tuple[bool, str, Optional[str]]:
        """test 集注册门禁 (统一入库判定尺): 符号一致性 + |t|/|IC|/|ICIR| 最低门槛.

        依次判定: 符号一致 (t·IC>0 且 t·ICIR>0)、|t|>=TEST_TSTAT_MIN、
        |IC|>=REGISTER_IC_MIN、|ICIR|>=REGISTER_ICIR_MIN; 通过则算 STRONG/WEAK。
        返回 (是否入库, 拒绝原因或'', gate_status STRONG/WEAK 或 None)。
        primary run() 与消融/基线实验共用同一套判定逻辑。
        """
        t_te = fm_result_test.t_stat
        ic_te = fm_result_test.rank_ic_mean
        icir_te = fm_result_test.rank_icir
        if not ((t_te * ic_te > 0) and (t_te * icir_te > 0)):
            return False, f"符号不一致(混号) t={t_te:.3f} IC={ic_te:.4f} ICIR={icir_te:.3f}", None
        if abs(t_te) < Config.TEST_TSTAT_MIN:
            return False, f"test |t| 不达标 ({t_te:.3f} < {Config.TEST_TSTAT_MIN})", None
        if abs(ic_te) < Config.REGISTER_IC_MIN:
            return False, f"test |IC| 过小 ({ic_te:.4f} < {Config.REGISTER_IC_MIN})", None
        if abs(icir_te) < Config.REGISTER_ICIR_MIN:
            return False, f"test |ICIR| 过小 ({icir_te:.3f} < {Config.REGISTER_ICIR_MIN})", None
        rc = Config.get_reward_config()
        strong_icir = abs(icir_te) >= rc['icir']['threshold']
        strong_ic = abs(ic_te) >= rc['ic']['threshold']
        gate_status = 'STRONG' if (strong_icir and strong_ic) else 'WEAK'
        if abs(t_te) < Config.TEST_TSTAT_STRONG:
            gate_status = 'WEAK'  # 生产底线: test |t| < 2.5 强制降级
        return True, "", gate_status

    def _extra_controls(self, mask: np.ndarray):
        """按 Config.NEUTRALIZE_CONTROLS 构建额外横截面控制列 (已按 mask 切片).

        amount 永远作为基础控制进入 fwl_neutralize (ln(amount)),
        这里只返回 amount 之外 (如 turn/vol_20d) 的控制列; 无则 None."""
        cols = []
        for name in Config.NEUTRALIZE_CONTROLS:
            if name == 'amount':
                continue
            attr = {'turn': 'full_turn'}.get(name)
            arr = getattr(self.prep, attr, None) if attr else None
            if arr is None:
                continue
            v = np.asarray(arr, dtype=np.float64)[mask]
            if np.all(v == 0):          # 数据缺该列 (占位 0) 时跳过, 防奇异设计矩阵
                continue
            cols.append(v)
        return np.column_stack(cols) if cols else None

    def _neutralize(self, pred: np.ndarray, ret: np.ndarray, dates: np.ndarray,
                    amount: np.ndarray, mask: np.ndarray):
        from .neutralize import fwl_neutralize
        return fwl_neutralize(pred, ret, dates, np.zeros(len(pred)), amount,
                              controls=self._extra_controls(mask))

    def _eval_pred(self, formula: str, feats: List[str], ds: Dict) -> Optional[np.ndarray]:
        """对候选公式在全量标准化特征上 eval 预测列 (含所有行). 失败返回 None."""
        try:
            X_all = ds['all'][0]
            ns = {f: X_all[:, i] for i, f in enumerate(feats)}
            ns.update(self.pysr_ns)
            pred = eval(formula, {"__builtins__": {}}, ns)
            return np.where(np.isfinite(pred), pred, np.nan)
        except Exception:
            return None

    def _fm_val_t(self, pred_val: np.ndarray, ds: Dict, vm: np.ndarray) -> Optional[float]:
        """FWL 中性化后对 val 段算 FM t-stat (带符号). 无效返回 None."""
        try:
            if np.isnan(pred_val).all():
                return None
            val_ret = ds['val'][1]
            val_dates = ds['val'][2]
            if Config.MODE == 'timing':
                result = self.timing_evaluator.run(
                    factor_values=pred_val, returns=val_ret, dates=val_dates)
                return result.t_stat if np.isfinite(result.t_stat) else 0.0
            amt_val = ds['all'][4][vm]
            s_val = ds['all'][3][vm]
            if Config.USE_RESIDUAL_NEUTRALIZATION:
                pure_pred, pure_ret = self._neutralize(pred_val, val_ret, val_dates, amt_val, vm)
            else:
                pure_pred, pure_ret = pred_val, val_ret
            fm = self.fm_regressor.run(factor_values=pure_pred, returns=pure_ret,
                                       dates=val_dates, symbols=s_val)
            t = fm.t_stat
            return t if np.isfinite(t) else 0.0
        except Exception:
            return None

    def _candidate_t(self, formula: str, feats: List[str], ds: Dict) -> Optional[float]:
        """候选公式按原样(不重拟合)在 val 上的带符号 FM t."""
        pred = self._eval_pred(formula, feats, ds)
        if pred is None:
            return None
        return self._fm_val_t(pred[self.prep.val_mask], ds, self.prep.val_mask)

    def _scale_candidate_t(self, formula: str, feats: List[str], ds: Dict,
                           tr_variant: np.ndarray) -> Optional[float]:
        """在扰动 train 掩码上重拟合候选幅度参数 a (OLS), 回 val 求带符号 FM t.

        将候选表达式视为 信号列 raw = expr(X), 在扰动 train 上求 a = argmin ||y - a*raw||²,
        pred_val = a * raw[val]. 用于检验候选信号是否对 train 边界敏感."""
        try:
            raw = self._eval_pred(formula, feats, ds)
            if raw is None:
                return None
            y_all = ds['all'][1]
            raw_tr = raw[tr_variant]
            y_tr = y_all[tr_variant]
            denom = np.nansum(raw_tr * raw_tr)
            if denom <= 1e-12:
                return None
            a = np.nansum(y_tr * raw_tr) / denom
            if not np.isfinite(a) or abs(a) < 1e-12:
                return 0.0
            pred_val = a * raw[self.prep.val_mask]
            return self._fm_val_t(pred_val, ds, self.prep.val_mask)
        except Exception:
            return None

    def _tr_variants(self) -> Tuple[str, np.ndarray, np.ndarray]:
        """train 边界扰动掩码: 去掉首/尾部各 min(5, 1/2) 个交易日."""
        fd = self.prep.full_dates.astype('M8[D]')
        tr_days = np.sort(np.unique(fd[self.prep.tr_mask]))
        k = min(5, max(1, len(tr_days) // 2))
        head = set(tr_days[:k])
        tail = set(tr_days[-k:])
        head_mask = self.prep.tr_mask & ~np.isin(fd, np.array(list(head), dtype='M8[D]'))
        tail_mask = self.prep.tr_mask & ~np.isin(fd, np.array(list(tail), dtype='M8[D]'))
        return head_mask, tail_mask

    def _select_stable_candidate(self, cands: List[Dict], feats: List[str], ds: Dict):
        """top-K 候选 → 扰动稳定性排序 → 稳定分最高者.

        稳定性 = (原始 val|t| ≥ ENTRANCE) 且 每个扰动态(去首/去尾5日重拟合)与原态
        符号一致且 |t| ≥ STABILITY_T_MIN; 达标数 ≥ STABILITY_MIN_HITS 才算稳定.
        按 (稳定比↓, 原val|t|↓, 复杂度↑) 排序. 返回 (formula, complexity, orig|t|) 或 None."""
        head_mask, tail_mask = self._tr_variants()
        variants = [('head', head_mask), ('tail', tail_mask)]
        valid_cnt = 0
        best = None
        for c in cands:
            formula = c['formula']
            used = [f for f in feats if f in formula]
            if len(used) <= 1 and '(' not in formula:
                logger.info(f"  跳过 {formula[:40]} | 单特征线性 (used={used})")
                continue
            orig = self._candidate_t(formula, feats, ds)
            if orig is None:
                logger.info(f"  跳过 {formula[:40]} | FM评估失败 (orig=None)")
                continue
            orig_abs = abs(orig)
            entrance_t = Config.get_market_config().get('stability_entrance_t', Config.STABILITY_ENTRANCE_T)
            if orig_abs < entrance_t:
                logger.info(f"  跳过 {formula[:40]} | val|t|={orig_abs:.3f} < {entrance_t}")
                continue
            valid_cnt += 1
            base_sign = np.sign(orig)
            conds = [(base_sign, orig_abs)]
            for _, tm in variants:
                st = self._scale_candidate_t(formula, feats, ds, tm)
                if st is not None:
                    conds.append((np.sign(st), abs(st)))
            hits = sum(1 for s, t in conds if s == base_sign and t >= Config.STABILITY_T_MIN)
            ratio = hits / len(conds)
            key = (ratio, orig_abs, -c['complexity'], -c['loss'])
            logger.info(f"  候选 {c['formula'][:50]} | 原val|t|={orig_abs:.2f} "
                        f"稳定{hits}/{len(conds)} (比率{ratio:.0%})")
            if (best is None or key > best[3]) and hits >= Config.STABILITY_MIN_HITS:
                best = (formula, c['complexity'], orig_abs, key)
        if best is None:
            if valid_cnt:
                logger.info(f"  ({valid_cnt} 条候选入围但均不稳定，跳过)")
            return None
        return best[0], best[1], best[2]

    def _subperiod_robustness(self, pred_all: np.ndarray, ds: Dict,
                              fm_val: FMRegressionResult) -> float:
        """跨子期方向一致性: 把 train 按日切 SUBPERIOD_ROBUST_SPLITS 段 + val 段,
        统计"方向一致(t·IC>0)且 |t|>=1"的段占比. 段内显式排除 test 行, 无泄漏."""
        d_all = ds['all'][2]
        tr_dates = np.sort(np.unique(d_all[self.prep.tr_mask]))
        chunks = np.array_split(tr_dates, Config.SUBPERIOD_ROBUST_SPLITS)
        consistent, valid = 0, 0
        for ch in chunks:
            seg = self.prep.tr_mask & np.isin(d_all, ch)
            assert (seg & self.prep.te_mask).sum() == 0, "子期段内不得包含 test 行"
            dates_s = d_all[seg]
            if np.unique(dates_s).size < 30:
                continue
            valid += 1
            pp, pr = pred_all[seg], ds['all'][1][seg]
            if Config.USE_RESIDUAL_NEUTRALIZATION:
                pp, pr = self._neutralize(pp, pr, dates_s, ds['all'][4][seg], seg)
            try:
                fm_s = self.fm_regressor.run(
                    factor_values=pp, returns=pr,
                    dates=dates_s, symbols=ds['all'][3][seg])
            except Exception as e:
                logger.warning(f"子期 FM 失败: {e}")
                continue
            if fm_s.t_stat * fm_s.rank_ic_mean > 0 and abs(fm_s.t_stat) >= 1.0:
                consistent += 1
        if fm_val.t_stat * fm_val.rank_ic_mean > 0 and abs(fm_val.t_stat) >= 1.0:
            consistent += 1
        return consistent / (valid + 1) if valid > 0 else 0.0

    def run(self, max_trials: int = 50, time_limit_min: int = 60):
        start = time.time()
        best_tstat, trial = 0.0, 0
        logger.info(f"Orchestrator {self.name} 启动 | 上限:{max_trials}轮/{time_limit_min}分钟")

        while trial < max_trials:
            if (time.time() - start) / 60 > time_limit_min:
                break

            if Config.USE_DYNAMIC_DIVERSITY:
                if self.soft_mask:
                    expired = [f for f, t in self.soft_mask.items() if t <= 0]
                    for f in expired:
                        del self.soft_mask[f]
                    for feat in list(self.soft_mask.keys()):
                        self.soft_mask[feat] -= 1
                if self.hard_mask:
                    expired = [f for f, t in self.hard_mask.items() if t <= 0]
                    for f in expired:
                        del self.hard_mask[f]
                    for feat in list(self.hard_mask.keys()):
                        self.hard_mask[feat] -= 1
                penalty_dict = {}
                for f in self.soft_mask:
                    if f in self.prep.feature_to_idx:
                        penalty_dict[self.prep.feature_to_idx[f]] = 5.0
                for f in self.hard_mask:
                    if f in self.prep.feature_to_idx:
                        penalty_dict[self.prep.feature_to_idx[f]] = 50.0
                self.gfn_sampler.set_feature_penalties(penalty_dict)

            logger.info(f"\n{'=' * 50}\nTrial #{trial}\n{'=' * 50}")

            traj = self.gfn_sampler.sample(batch_size=1)[0]
            feats = traj.features
            self.trial_count += 1
            for f in feats:
                self.trial_feature_counter[f] = self.trial_feature_counter.get(f, 0) + 1

            if Config.USE_DYNAMIC_DIVERSITY and self.trial_count > 15:
                for f, cnt in list(self.trial_feature_counter.items()):
                    ratio = cnt / self.trial_count
                    if ratio > 0.50 and f not in self.hard_mask and f not in self.soft_mask:
                        self.hard_mask[f] = 10
                        logger.info(f"硬屏蔽 {f} (出现率{ratio:.0%})")

            if len(feats) < Config.MIN_FEATURES:
                self.gfn_trainer.train_step(traj.log_pf, traj.log_pb, reward=Config.REWARD_HARD)
                trial += 1
                continue

            ds = self.prep.get_subset(feats)
            X_tr, y_tr = ds['train'][0], ds['train'][1]

            engine = PySRMiningEngine()
            pysr_result = engine.run(pd.DataFrame(X_tr, columns=feats), y_tr,
                                     top_k=Config.VAL_SELECT_TOP_K)
            if not pysr_result:
                self.gfn_trainer.train_step(traj.log_pf, traj.log_pb, reward=Config.REWARD_HARD)
                trial += 1
                continue

            if isinstance(pysr_result, list):
                chosen = self._select_stable_candidate(pysr_result, feats, ds)
                if not chosen:
                    logger.info(f"Trial #{trial} 候选稳定性不足，跳过")
                    self.gfn_trainer.train_step(traj.log_pf, traj.log_pb, reward=Config.REWARD_HARD)
                    trial += 1
                    continue
                formula, complexity, sel_t = chosen
                logger.info(f"Trial #{trial} 稳定择优选定 (val|t|={sel_t:.3f}): {formula}")
            else:
                formula, complexity = pysr_result

            used_feats = [f for f in feats if f in formula]
            if len(used_feats) <= 1 and '(' not in formula:
                logger.info(f"Trial #{trial} 单特征线性公式，跳过")
                self.gfn_trainer.train_step(traj.log_pf, traj.log_pb, reward=Config.REWARD_HARD)
                trial += 1
                continue

            try:
                X_all, y_all, d_all, s_all, amt_all = ds['all']
                ns = {f: X_all[:, i] for i, f in enumerate(feats)}
                ns.update(self.pysr_ns)
                pred_all = eval(formula, {"__builtins__": {}}, ns)
                pred_all = np.where(np.isfinite(pred_all), pred_all, np.nan)

                pred_val = pred_all[self.prep.val_mask]
                val_ret = ds['val'][1]
                val_dates = ds['val'][2]
                val_symbols = s_all[self.prep.val_mask]
                val_amount = amt_all[self.prep.val_mask]

            except Exception as e:
                logger.warning(f"Trial #{trial} 执行失败: {e}")
                self.gfn_trainer.train_step(traj.log_pf, traj.log_pb, reward=Config.REWARD_HARD)
                trial += 1
                continue

            if Config.MODE == 'timing':
                # 时序择时模式: 不需要中性化
                pure_pred, pure_ret = pred_val, val_ret
                timing_result = self.timing_evaluator.run(
                    factor_values=pure_pred,
                    returns=pure_ret,
                    dates=val_dates,
                )
                self.stable_candidates += 1
                fid = f"ACAD_{trial:03d}"
                self.timing_evaluator.print_report(timing_result, factor_name=fid)

                if abs(timing_result.direction_accuracy) < Config.TIMING_DIRECTION_THRESHOLD:
                    logger.warning(f"{fid} 方向准确率不达标 ({timing_result.direction_accuracy:.3f} < {Config.TIMING_DIRECTION_THRESHOLD})")
                    self.gfn_trainer.train_step(traj.log_pf, traj.log_pb, reward=Config.REWARD_HARD)
                    trial += 1
                    continue
                self.gate_passed += 1

                structure_fp = self._normalize_structure(formula, feats)
                if structure_fp in self.structural_fingerprints:
                    logger.warning(f"{fid} 结构重复 ({structure_fp[:60]}...)")
                    self.gfn_trainer.train_step(traj.log_pf, traj.log_pb, reward=Config.REWARD_HARD)
                    trial += 1
                    continue

                canon_sig = self._canonical_signature(formula, feats)
                canon_dup_penalty = 1.0
                if canon_sig in self.structural_signatures:
                    logger.info(f"{fid} 特征无关结构重复 ({canon_sig[:50]}...)，施加软惩罚")
                    canon_dup_penalty = 0.3

                reward, reward_debug = self._compute_timing_reward(
                    timing_result, feats, formula=formula, robustness=1.0)
                if canon_dup_penalty < 1.0:
                    reward *= canon_dup_penalty
                    reward_debug['canon_dup_penalty'] = canon_dup_penalty

                logger.info(
                    f"{fid} | Reward:{reward:.3f} Cpx:{complexity} | "
                    f"sharpe:{reward_debug.get('sharpe_score', 0):.3f} "
                    f"dir:{reward_debug.get('dir_score', 0):.3f} "
                    f"t:{reward_debug.get('t_ctrl_score', 0):.3f}"
                )

                self.gfn_trainer.train_step(traj.log_pf, traj.log_pb, reward=reward)

                # Phase B: Test set evaluation
                pred_te = pred_all[self.prep.te_mask]
                test_ret = ds['test'][1]
                test_dates = ds['test'][2]

                timing_result_test = self.timing_evaluator.run(
                    factor_values=pred_te,
                    returns=test_ret,
                    dates=test_dates,
                )
                self.register_attempts += 1

                ok, reason, gate_status = self._timing_registration_verdict(timing_result_test)
                if not ok:
                    logger.warning(f"{fid} {reason} 不入库")
                    trial += 1
                    continue

                self.timing_evaluator.print_report(timing_result_test, factor_name=f"{fid} (Test集样本外)")

                timing_dict = timing_result_test.to_dict()
                timing_dict['formula'] = formula
                timing_dict['features'] = feats
                timing_dict['complexity'] = complexity
                self.registry.register(fid, formula, timing_dict, feats, gate_status=gate_status)
                self.structural_fingerprints.add(structure_fp)
                self.structural_signatures.add(canon_sig)

                for feat in feats:
                    self.feature_usage[feat] = self.feature_usage.get(feat, 0) + 1
                self.total_passed += 1
                self.pass_counter += 1
                self.registered_formulas.append((fid, formula, feats))

                if abs(timing_result.timing_sharpe) > best_tstat:
                    best_tstat = abs(timing_result.timing_sharpe)

            else:
                # 截面选股模式 (原有逻辑)
                if Config.USE_RESIDUAL_NEUTRALIZATION:
                    pure_pred, pure_ret = self._neutralize(
                        pred_val, val_ret, val_dates, val_amount, self.prep.val_mask)
                else:
                    pure_pred, pure_ret = pred_val, val_ret

                fm_result = self.fm_regressor.run(
                    factor_values=pure_pred,
                    returns=pure_ret,
                    dates=val_dates,
                    symbols=val_symbols
                )
                self.stable_candidates += 1

                fid = f"ACAD_{trial:03d}"
                self.fm_regressor.print_report(fm_result, factor_name=fid)

                robustness = 1.0
                if Config.USE_SUBPERIOD_ROBUST_REWARD:
                    robustness = self._subperiod_robustness(pred_all, ds, fm_result)
                    logger.info(f"{fid} robustness: {robustness:.3f} "
                                f"({Config.SUBPERIOD_ROBUST_SPLITS} train段 + val)")

                if abs(fm_result.t_stat) < Config.TSTAT_THRESHOLD:
                    logger.warning(f"{fid} t-stat不达标 ({fm_result.t_stat:.3f} < {Config.TSTAT_THRESHOLD})")
                    self.gfn_trainer.train_step(traj.log_pf, traj.log_pb, reward=Config.REWARD_HARD)
                    trial += 1
                    continue

                if Config.USE_MONOTONICITY_GATE:
                    if not self._check_monotonicity(fm_result.quantile_returns, fm_result.t_stat):
                        logger.warning(f"{fid} 单调性不达标")
                        self.gfn_trainer.train_step(traj.log_pf, traj.log_pb, reward=Config.REWARD_HARD)
                        trial += 1
                        continue
                self.gate_passed += 1

                structure_fp = self._normalize_structure(formula, feats)
                if structure_fp in self.structural_fingerprints:
                    logger.warning(f"{fid} 结构重复 ({structure_fp[:60]}...)")
                    self.gfn_trainer.train_step(traj.log_pf, traj.log_pb, reward=Config.REWARD_HARD)
                    trial += 1
                    continue

                canon_sig = self._canonical_signature(formula, feats)
                canon_dup_penalty = 1.0
                if canon_sig in self.structural_signatures:
                    logger.info(f"{fid} 特征无关结构重复 ({canon_sig[:50]}...)，施加软惩罚")
                    canon_dup_penalty = 0.3

                reward, reward_debug = self._compute_reward(
                    fm_result, feats, formula=formula, robustness=robustness)
                if canon_dup_penalty < 1.0:
                    reward *= canon_dup_penalty
                    reward_debug['canon_dup_penalty'] = canon_dup_penalty

                diversity_part = f" div:{reward_debug.get('div_mult', 1.0):.2f}x" if 'div_mult' in reward_debug else ""
                sim_part = " SIM-PENALTY" if 'sim_penalty' in reward_debug else ""
                canon_part = f" canon:{reward_debug.get('canon_dup_penalty', 1.0):.2f}x" if reward_debug.get('canon_dup_penalty', 1.0) < 1.0 else ""
                robust_part = ""
                if Config.USE_SUBPERIOD_ROBUST_REWARD:
                    robust_part = f" robust:{reward_debug.get('robustness', 0.0):.3f}"
                if not Config.USE_DYNAMIC_DIVERSITY:
                    diversity_part = ""
                    sim_part = ""
                logger.info(
                    f"{fid} | Reward:{reward:.3f} Cpx:{complexity} | "
                    f"t:{reward_debug['t_ctrl_score']:.3f} "
                    f"icir:{reward_debug['icir_score']:.3f} "
                    f"ic:{reward_debug['ic_score']:.3f}{diversity_part}{sim_part}{canon_part}{robust_part}"
                )

                # Train GFlowNet on VAL set reward
                self.gfn_trainer.train_step(traj.log_pf, traj.log_pb, reward=reward)

                # === Phase B: Test set evaluation for registry ===
                pred_te = pred_all[self.prep.te_mask]
                test_ret = ds['test'][1]
                test_dates = ds['test'][2]
                test_symbols = ds['test'][3]
                test_amount = ds['test'][4]

                if Config.USE_RESIDUAL_NEUTRALIZATION:
                    pure_pred_te, pure_ret_te = self._neutralize(
                        pred_te, test_ret, test_dates, test_amount, self.prep.te_mask
                    )
                else:
                    pure_pred_te, pure_ret_te = pred_te, test_ret

                fm_result_test = self.fm_regressor.run(
                    factor_values=pure_pred_te,
                    returns=pure_ret_te,
                    dates=test_dates,
                    symbols=test_symbols
                )
                self.register_attempts += 1

                ok, reason, gate_status = self._registration_verdict(fm_result_test)
                if not ok:
                    logger.warning(f"{fid} {reason} 不入库")
                    trial += 1
                    continue

                self.fm_regressor.print_report(fm_result_test, factor_name=f"{fid} (Test集样本外)")

                fm_dict = fm_result_test.to_dict()
                fm_dict['formula'] = formula
                fm_dict['features'] = feats
                fm_dict['complexity'] = complexity
                self.registry.register(fid, formula, fm_dict, feats, gate_status=gate_status)
                self.structural_fingerprints.add(structure_fp)
                self.structural_signatures.add(canon_sig)

                for feat in feats:
                    self.feature_usage[feat] = self.feature_usage.get(feat, 0) + 1
                self.total_passed += 1
                self.pass_counter += 1
                self.registered_formulas.append((fid, formula, feats))

                if Config.USE_DYNAMIC_DIVERSITY:
                    if self.pass_counter >= 1:
                        self.pass_counter = 0
                        formula_feats = [f for f in feats if f in formula]
                        if formula_feats:
                            for feat in formula_feats:
                                if feat not in self.soft_mask:
                                    self.soft_mask[feat] = 10
                                    logger.info(f"软屏蔽 {feat} (公式核心特征)")
                        else:
                            sorted_feats = sorted(self.feature_usage.items(), key=lambda x: -x[1])
                            for feat, _ in sorted_feats[:2]:
                                if feat not in self.soft_mask:
                                    self.soft_mask[feat] = 10
                                    logger.info(f"软屏蔽 {feat} (出现{self.feature_usage[feat]}次)")

                if abs(fm_result.t_stat) > best_tstat:
                    best_tstat = abs(fm_result.t_stat)
                    logger.info(f"新纪录! (Val集) |t-stat| = {best_tstat:.4f}")

            trial += 1

        prod_factors = self.registry.get_production_ready()
        logger.info(f"达标因子: {len(prod_factors)} 个")
        self._generate_summary_report(prod_factors)
        return prod_factors

    def _generate_summary_report(self, prod_factors: List[Dict]):
        report_lines = [
            "=" * 90,
            self.report_title,
            "=" * 90,
        ]
        # 统计诚实: 候选筛选漏斗 (无条件输出, 即使 0 注册)
        n_trial = self.trial_count
        n_reg = len(self.registered_formulas)
        ret_rate = n_reg / n_trial if n_trial else 0.0
        report_lines.append(
            f"[统计诚实] 总 trial={n_trial} 稳定候选={self.stable_candidates} "
            f"过MLQC门={self.gate_passed} test判定={self.register_attempts} "
            f"注册={n_reg} 保留率={ret_rate:.3%} | "
            f"NEUTRALIZE_CONTROLS={Config.NEUTRALIZE_CONTROLS}"
        )
        report_lines.append("-" * 90)

        if not prod_factors:
            report_lines.append("(无达标因子)")
            report_text = "\n".join(report_lines)
            logger.info(f"\n{report_text}")
            report_path = os.path.join(Config.OUTPUT_DIR, "fm_summary_report.txt")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report_text)
            return

        report_lines.append(
            f"{'ID':<12} {'t-stat':>8} {'p-value':>10} {'Coef':>10} {'NW-SE':>10} "
            f"{'Rank_IC':>8} {'ICIR':>8} {'L-S':>10} {'R²':>8} {'公式'}"
        )
        report_lines.append("-" * 90)
        for f in sorted(prod_factors, key=lambda x: abs(x['metrics'].get('FM_tstat', 0)), reverse=True):
            m = f['metrics']
            sig = "***" if abs(m.get('FM_tstat', 0)) >= 3.0 else (
                "**" if abs(m.get('FM_tstat', 0)) >= 2.0 else "")
            line = (
                f"{f['id']:<12} {m.get('FM_tstat', 0):>7.3f}{sig:<3} "
                f"{m.get('FM_pvalue', 1):>10.6f} {m.get('FM_coef', 0):>10.6f} "
                f"{m.get('FM_se', 0):>10.6f} {m.get('Rank_IC', 0):>8.4f} "
                f"{m.get('ICIR', 0):>8.4f} {m.get('LS_spread', 0):>10.6f} "
                f"{m.get('FM_R2_avg', 0):>8.4f} {m.get('formula', '')[:50]}"
            )
            report_lines.append(line)

        report_text = "\n".join(report_lines)
        logger.info(f"\n{report_text}")

        report_path = os.path.join(Config.OUTPUT_DIR, "fm_summary_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        logger.info(f"报告已保存至: {report_path}")
