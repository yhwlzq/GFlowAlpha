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
from alpha.mining.safe_ops import SafeOps
from alpha.mining.preprocessor import DataPreprocessor
from alpha.mining.fm_regression import FMRegressionResult, FamaMacBethRegressor
from alpha.mining.pysr_engine import PySRMiningEngine
from alpha.mining.gflownet import FeatureMetadataExtractor, GFlowNetPolicy_Transformer, TrajectorySampler_Transformer, GFlowNetTrainerInternal
from alpha.mining.registry import FactorRegistry

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
                        formula: str = "") -> Tuple[float, Dict]:
        debug = {}

        abs_t = abs(fm_result.t_stat)
        raw_icir = fm_result.rank_icir
        abs_icir = abs(raw_icir)
        abs_ic = abs(fm_result.rank_ic_mean)

        if abs_icir < 0.08:
            debug['t_ctrl_score'] = 0.0
            debug['icir_score'] = 0.0
            debug['ic_score'] = 0.0
            debug['reason'] = 'ICIR_too_low'
            final_reward = 0.01
            debug['final_reward'] = final_reward
            return final_reward, debug

        if abs_ic < 0.005:
            debug['t_ctrl_score'] = 0.0
            debug['icir_score'] = 0.0
            debug['ic_score'] = 0.0
            debug['reason'] = 'IC_too_low'
            final_reward = 0.01
            debug['final_reward'] = final_reward
            return final_reward, debug

        raw_ic = fm_result.rank_ic_mean
        if np.sign(fm_result.t_stat) * np.sign(raw_ic) < 0 or \
           np.sign(fm_result.t_stat) * np.sign(raw_icir) < 0:
            debug['t_ctrl_score'] = 0.0
            debug['icir_score'] = 0.0
            debug['ic_score'] = 0.0
            debug['reason'] = 'sign_mismatch'
            final_reward = 0.01
            debug['final_reward'] = final_reward
            return final_reward, debug

        rc = Config.get_reward_config()
        t_score    = 1.0 / (1.0 + np.exp(-rc['t_stat']['slope'] * (abs_t - rc['t_stat']['threshold'])))
        icir_score = 1.0 / (1.0 + np.exp(-rc['icir']['slope']   * (abs_icir - rc['icir']['threshold'])))
        ic_score   = 1.0 / (1.0 + np.exp(-rc['ic']['slope']     * (abs_ic - rc['ic']['threshold'])))
        debug['t_ctrl_score'] = t_score
        debug['icir_score'] = icir_score
        debug['ic_score'] = ic_score

        reward = (t_score ** rc['weight']['t_stat']) * (icir_score ** rc['weight']['icir']) * (ic_score ** rc['weight']['ic']) * 10.0
        
   
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

        final_reward = max(reward, 0.01)
        debug['final_reward'] = final_reward
        return final_reward, debug

    @staticmethod
    def _neutralize(pred: np.ndarray, ret: np.ndarray, dates: np.ndarray, amount: np.ndarray):
        from .neutralize import fwl_neutralize
        return fwl_neutralize(pred, ret, dates, np.zeros(len(pred)), amount)

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
                self.gfn_trainer.train_step(traj.log_pf, traj.log_pb, reward=0.01)
                trial += 1
                continue

            ds = self.prep.get_subset(feats)
            X_tr, y_tr = ds['train'][0], ds['train'][1]

            engine = PySRMiningEngine()
            pysr_result = engine.run(pd.DataFrame(X_tr, columns=feats), y_tr)
            if not pysr_result:
                self.gfn_trainer.train_step(traj.log_pf, traj.log_pb, reward=0.01)
                trial += 1
                continue
            formula, complexity = pysr_result

            used_feats = [f for f in feats if f in formula]
            if len(used_feats) <= 1 and '(' not in formula:
                logger.info(f"Trial #{trial} 单特征线性公式，跳过")
                self.gfn_trainer.train_step(traj.log_pf, traj.log_pb, reward=0.01)
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
                self.gfn_trainer.train_step(traj.log_pf, traj.log_pb, reward=0.01)
                trial += 1
                continue

            if Config.USE_RESIDUAL_NEUTRALIZATION:
                pure_pred, pure_ret = self._neutralize(pred_val, val_ret, val_dates, val_amount)
            else:
                pure_pred, pure_ret = pred_val, val_ret

            fm_result = self.fm_regressor.run(
                factor_values=pure_pred,
                returns=pure_ret,
                dates=val_dates,
                symbols=val_symbols
            )

            fid = f"ACAD_{trial:03d}"
            self.fm_regressor.print_report(fm_result, factor_name=fid)

            if abs(fm_result.t_stat) < Config.TSTAT_THRESHOLD:
                logger.warning(f"{fid} t-stat不达标 ({fm_result.t_stat:.3f} < {Config.TSTAT_THRESHOLD})")
                self.gfn_trainer.train_step(traj.log_pf, traj.log_pb, reward=0.01)
                trial += 1
                continue

            if Config.USE_MONOTONICITY_GATE:
                if not self._check_monotonicity(fm_result.quantile_returns, fm_result.t_stat):
                    logger.warning(f"{fid} 单调性不达标")
                    self.gfn_trainer.train_step(traj.log_pf, traj.log_pb, reward=0.01)
                    trial += 1
                    continue

            structure_fp = self._normalize_structure(formula, feats)
            if structure_fp in self.structural_fingerprints:
                logger.warning(f"{fid} 结构重复 ({structure_fp[:60]}...)")
                self.gfn_trainer.train_step(traj.log_pf, traj.log_pb, reward=0.005)
                trial += 1
                continue

            canon_sig = self._canonical_signature(formula, feats)
            canon_dup_penalty = 1.0
            if canon_sig in self.structural_signatures:
                logger.info(f"{fid} 特征无关结构重复 ({canon_sig[:50]}...)，施加软惩罚")
                canon_dup_penalty = 0.3

            reward, reward_debug = self._compute_reward(fm_result, feats, formula=formula)
            if canon_dup_penalty < 1.0:
                reward *= canon_dup_penalty
                reward_debug['canon_dup_penalty'] = canon_dup_penalty

            diversity_part = f" div:{reward_debug.get('div_mult', 1.0):.2f}x" if 'div_mult' in reward_debug else ""
            sim_part = " SIM-PENALTY" if 'sim_penalty' in reward_debug else ""
            canon_part = f" canon:{reward_debug.get('canon_dup_penalty', 1.0):.2f}x" if reward_debug.get('canon_dup_penalty', 1.0) < 1.0 else ""
            if not Config.USE_DYNAMIC_DIVERSITY:
                diversity_part = ""
                sim_part = ""
            logger.info(
                f"{fid} | Reward:{reward:.3f} Cpx:{complexity} | "
                f"t:{reward_debug['t_ctrl_score']:.3f} "
                f"icir:{reward_debug['icir_score']:.3f} "
                f"ic:{reward_debug['ic_score']:.3f}{diversity_part}{sim_part}{canon_part}"
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
                    pred_te, test_ret, test_dates, test_amount
                )
            else:
                pure_pred_te, pure_ret_te = pred_te, test_ret

            fm_result_test = self.fm_regressor.run(
                factor_values=pure_pred_te,
                returns=pure_ret_te,
                dates=test_dates,
                symbols=test_symbols
            )

            self.fm_regressor.print_report(fm_result_test, factor_name=f"{fid} (Test集样本外)")

            fm_dict = fm_result_test.to_dict()
            fm_dict['formula'] = formula
            fm_dict['features'] = feats
            fm_dict['complexity'] = complexity
            self.registry.register(fid, formula, fm_dict, feats)
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
        if not prod_factors:
            return
        report_lines = [
            "=" * 90,
            self.report_title,
            "=" * 90,
            f"{'ID':<12} {'t-stat':>8} {'p-value':>10} {'Coef':>10} {'NW-SE':>10} "
            f"{'Rank_IC':>8} {'ICIR':>8} {'L-S':>10} {'R²':>8} {'公式'}",
            "-" * 90
        ]
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
