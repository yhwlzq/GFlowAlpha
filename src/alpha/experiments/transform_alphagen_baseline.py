#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
学术级日频单因子挖掘系统 — AlphaGen (PPO) 外部 RL 基线

对比对象: transform_pysr_primary.py (GFlowNet + PySR + MLQC)

方法学 (训练用官方实现, 评测与 GFlowAlpha 完全同口径):
  1. 数据: 自研 `alphagen_qlib.stock_data.ParquetStockData` (qlib-free drop-in)
     直接吃 parquet面板, 布局对齐上游 [T, n_features, n_stocks] (回溯暖身 +
     窗口), 零 qlib 依赖。
  2. 训练: 复用 vendored `alphagen` (官方 AlphaSAGE) 的 AlphaPool + AlphaEnv +
     sb3 `MaskablePPO` + `LSTMSharedNet`; target 统一改为 1 日收益
     `Ref(close, -1) / close - 1` (与评测 horizon 对齐, 只用历史, 无未来泄漏)。
  3. 入库: 最终 pool 表达式按 |weight| 排序全量入库, 不做 MLQC 门禁过滤。
  4. 评估: 每个候选在测试集做 FWL 中性化 + Fama-MacBeth, 指标写入 registry,
     与主流程同口径比较 (FWL + FM 为统一评估通道)。
  5. 产出: registry_academic.json / fm_summary_report.txt / 回测 top-3 /
     与 primary 及其它 registry 的对比表 (comparison_*.csv / .md)。

用法示例:
  python src/alpha/experiments/transform_alphagen_baseline.py --data data/...parquet --steps 300 --pool 3
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import time
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_VENDOR_DIR = (_THIS_DIR.parents[1] / 'vendor')
if str(_VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(_VENDOR_DIR))

import argparse
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from alpha.config import Config, REPO_ROOT, apply_split_mode, set_global_seed
from alpha.data.data_loader import CSI500Loader
from alpha.mining.preprocessor import DataPreprocessor
from alpha.mining.fm_regression import FamaMacBethRegressor
from alpha.mining.registry import FactorRegistry, is_production_ready, format_fm_table
from alpha.mining.orchestrator import MiningOrchestrator

from alphagen.config import MAX_EXPR_LENGTH
from alphagen.data.expression import Feature, Ref
from alphagen.models.alpha_pool import AlphaPool
from alphagen.rl.env.wrapper import AlphaEnv
from alphagen.rl.policy import LSTMSharedNet
from alphagen.utils.random import reseed_everything
from alphagen_qlib.stock_data import ParquetStockData, FeatureType

logger = logging.getLogger("AlphaGenBaseline")

FEATURE_COLS = ('open', 'high', 'low', 'close', 'volume', 'amount')
FEATURE_LABELS = ('OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'VWAP')

_MAX_BACKTRACK = 100
_MAX_FUTURE = 1  # target=Ref(close,-1)/close-1 (前向1日) 需在未来区读1行


def _build_panel(
    df: pd.DataFrame,
    cols: Tuple[str, ...] = FEATURE_COLS,
) -> Tuple[np.ndarray, pd.DatetimeIndex, pd.Index]:
    """把长表面板转成 [T, n_features, n_stocks] numpy 矩阵 (行=交易日旧->新),
    顺序对齐 FeatureType 枚举; 缺失 (停牌/未上市) 为 NaN。"""
    df = df.copy()
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    df['symbol'] = df['symbol'].astype(str)
    cal = pd.DatetimeIndex(sorted(df['date'].unique()))
    syms = pd.Index(sorted(df['symbol'].unique()))
    mats = []
    for c in cols:
        p = df.pivot_table(index='date', columns='symbol', values=c, aggfunc='last')
        mats.append(p.reindex(index=cal, columns=syms).values.astype(np.float64))
    vwap = np.divide(mats[5], mats[4], out=np.full_like(mats[4], np.nan), where=mats[4] != 0)
    feat_vals = np.stack(mats[:5] + [vwap], axis=1)  # [T, 6, S]
    return feat_vals, cal, syms


def _make_stock_data(
    feat_vals: np.ndarray,
    cal: pd.DatetimeIndex,
    syms: pd.Index,
    win_start: int,
    win_end: int,
    device: torch.device,
    max_backtrack_days: int = _MAX_BACKTRACK,
    max_future_days: int = 0,
) -> ParquetStockData:
    """从全面板切出窗口 [win_start, win_end) 并前补回溯暖身区 (NaN),
    末尾补 max_future_days 个真实未来日 (不足则 NaN)。"""
    n = win_end - win_start
    warm = np.full((max_backtrack_days, feat_vals.shape[1], feat_vals.shape[2]),
                   np.nan, dtype=np.float32)
    real_future = feat_vals[win_end:win_end + max_future_days].astype(np.float32)
    if real_future.shape[0] < max_future_days:
        pad_future = np.full((max_future_days - real_future.shape[0],
                              feat_vals.shape[1], feat_vals.shape[2]), np.nan, dtype=np.float32)
        real_future = np.concatenate([real_future, pad_future], axis=0)
    sub = np.concatenate([warm, feat_vals[win_start:win_end].astype(np.float32),
                          real_future], axis=0)
    warm_dates = pd.DatetimeIndex([pd.NaT] * max_backtrack_days)
    future_cal = cal[win_end:win_end + max_future_days]
    if len(future_cal) < max_future_days:
        future_cal = future_cal.append(pd.DatetimeIndex([pd.NaT] * (max_future_days - len(future_cal))))
    dates = warm_dates.append(cal[win_start:win_end].append(pd.DatetimeIndex(future_cal)))
    tensor = torch.tensor(sub, dtype=torch.float, device=device)
    return ParquetStockData(tensor, dates, syms,
                            max_backtrack_days=max_backtrack_days,
                            max_future_days=max_future_days,
                            device=device)


def _cross_day_zscore(arr: np.ndarray) -> np.ndarray:
    """逐日横截面 zscore (mask 掉 NaN), 与 pool 的标准化语义一致。"""
    mu = np.nanmean(arr, axis=1, keepdims=True)
    sd = np.nanstd(arr, axis=1, keepdims=True)
    sd[sd == 0] = np.nan
    z = (arr - mu) / sd
    return np.where(np.isfinite(z), z, np.nan)


def _factor_long_series(
    expr,
    stock_data: ParquetStockData,
) -> pd.Series:
    """表达式在全样本窗口上的因子值 -> (date, symbol) 长表 Series, 逐日 zscore。"""
    with torch.no_grad():
        val = expr.evaluate(stock_data)
    vals = val.detach().cpu().numpy()  # [n_days, n_stocks]
    z = _cross_day_zscore(vals)
    mb = stock_data.max_backtrack_days
    cal = stock_data._dates[mb:mb + stock_data.n_days]  # type: ignore
    idx = pd.MultiIndex.from_product([cal, stock_data._stock_ids], names=['date', 'symbol'])
    return pd.Series(z.reshape(-1), index=idx, name='factor')


from stable_baselines3.common.callbacks import BaseCallback


class _PoolSnapshotCallback(BaseCallback):
    """每个 rollout 结束记录 pool 状态快照 (候选收集 + 进度打印)。"""

    def __init__(self, show_freq: int = 50, verbose: int = 0):
        super().__init__(verbose)
        self.show_freq = show_freq
        self.snapshots: List[Dict] = []
        self._last_size = -1

    def _pool(self):
        return self.training_env.envs[0].unwrapped.pool  # type: ignore

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self):
        pool = self._pool()
        st = pool.state
        if pool.size != self._last_size:
            self._last_size = pool.size
            self.snapshots.append({
                'step': self.num_timesteps,
                'size': pool.size,
                'best_ic_ret': st['best_ic_ret'],
                'exprs': [str(e) for e in st['exprs']],
                'weights': list(st['weights']),
                'ics_ret': list(st['ics_ret']),
            })
        if self.num_timesteps % max(1, self.show_freq) == 0:
            print(f"[RL] step={self.num_timesteps} pool_size={pool.size} "
                  f"best_ic_ret={st['best_ic_ret']:.5f}", flush=True)
        return True


class _TimeBudgetCallback(BaseCallback):
    """超出时间预算即中止训练 (sb3 StopTrainingFlag)。"""

    def __init__(self, budget_sec: int, verbose: int = 0):
        super().__init__(verbose)
        self.budget_sec = budget_sec
        self.t0 = time.time()

    def _on_step(self):
        if self.budget_sec and (time.time() - self.t0) > self.budget_sec:
            self.model.stop_training = True  # type: ignore
            return False
        return True


def _neutralize(pred, ret, dates, symbols, amount):
    keep = np.isfinite(pred) & np.isfinite(ret)
    pred = pred[keep]
    ret = ret[keep]
    dates = np.asarray(dates)[keep]
    symbols = np.asarray(symbols)[keep]
    amount = np.asarray(amount)[keep]
    pure_pred, pure_ret = MiningOrchestrator._neutralize(pred, ret, dates, amount)
    return pure_pred, pure_ret, dates, symbols


def _production_metrics(registry: FactorRegistry) -> Dict:
    factors = registry.data.get("factors", {})
    prod = registry.get_production_ready()
    stats = {
        "n_registered": len(factors),
        "n_production": len(prod),
        "mean_abs_t": np.nan, "max_abs_t": np.nan,
        "mean_abs_icir": np.nan, "mean_abs_ic": np.nan,
        "feature_union": 0, "structure_unique": 0,
    }
    if not factors:
        return stats
    t_vals, icir_vals, ic_vals = [], [], []
    formulas, feats = set(), set()
    for fid, info in factors.items():
        m = info.get("metrics", {})
        t_vals.append(abs(m.get("FM_tstat", 0)))
        icir_vals.append(abs(m.get("ICIR", 0)))
        ic_vals.append(abs(m.get("Rank_IC", 0)))
        formulas.add(str(m.get("formula", "")))
        for f in info.get("features", []) or []:
            feats.add(f)
    stats.update({
        "mean_abs_t": float(np.mean(t_vals)) if t_vals else np.nan,
        "max_abs_t": float(np.max(t_vals)) if t_vals else np.nan,
        "mean_abs_icir": float(np.mean(icir_vals)) if icir_vals else np.nan,
        "mean_abs_ic": float(np.mean(ic_vals)) if ic_vals else np.nan,
        "feature_union": len(feats), "structure_unique": len(formulas),
    })
    return stats


def _load_registry_stats(path: str, tag: str) -> Dict:
    if not os.path.exists(path):
        logger.warning(f"对比 registry 不存在: {path}")
        return {"tag": tag, "path": path, "n_registered": 0, "n_production": 0}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    factors = data.get("factors", {})
    t_vals, icir_vals, ic_vals = [], [], []
    feats, formulas = set(), set()
    n_prod = 0
    for _, info in factors.items():
        m = info.get("metrics", info)
        t = abs(m.get("FM_tstat", 0))
        icir = abs(m.get("ICIR", 0))
        ic = abs(m.get("Rank_IC", 0))
        t_vals.append(t); icir_vals.append(icir); ic_vals.append(ic)
        formulas.add(str(m.get("formula", "")))
        for f in info.get("features", []) or []:
            feats.add(f)
        if is_production_ready(m, info.get('gate_status')):
            n_prod += 1
    return {"tag": tag, "path": path, "n_registered": len(factors),
            "n_production": n_prod,
            "mean_abs_t": float(np.mean(t_vals)) if t_vals else np.nan,
            "max_abs_t": float(np.max(t_vals)) if t_vals else np.nan,
            "mean_abs_icir": float(np.mean(icir_vals)) if icir_vals else np.nan,
            "mean_abs_ic": float(np.mean(ic_vals)) if ic_vals else np.nan,
            "feature_union": len(feats), "structure_unique": len(formulas)}


def write_comparison(rows: List[Dict], output_dir: str):
    cols = ["tag", "path", "n_registered", "n_production",
            "mean_abs_t", "max_abs_t", "mean_abs_icir", "mean_abs_ic",
            "feature_union", "structure_unique"]
    df = pd.DataFrame(rows, columns=cols)
    csv_path = os.path.join(output_dir, "comparison_baseline.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8")
    md_path = os.path.join(output_dir, "comparison_baseline.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "---|" * len(cols) + "\n")
        for _, row in df.iterrows():
            f.write("| " + " | ".join(str(row[c]) for c in cols) + " |\n")
    logger.info(f"\n对比表已保存: {csv_path} / {md_path}")
    print(df.to_string(index=False))


def run_alphagen_baseline(
    data_path: str,
    steps: int = 20000,
    pool_capacity: int = 20,
    max_time_min: int = 0,
    seed: int = 42,
    n_layers: int = 2,
    d_model: int = 128,
    n_steps_ppo: int = 512,
    batch_size: int = 128,
    backtest_top: int = 3,
    compare: Optional[List[Tuple[str, str]]] = None,
):
    set_global_seed()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    Config.OUTPUT_DIR = os.path.join("factor_output_academic_v81",
                                     f"alphagen_ppo_baseline_{timestamp}")
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-7s | %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(Config.OUTPUT_DIR, 'mining.log'), mode='w', encoding='utf-8')
        ],
        force=True
    )

    device = torch.device(Config.DEVICE)
    logger.info(f"设备: {device}")

    logger.info(f"加载日频数据: {data_path}")
    data_util = CSI500Loader(path=data_path)
    df = data_util.load()

    required_cols = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'amount']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"数据必须包含列: {required_cols}")
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=['close', 'volume']).sort_values(['symbol', 'date']).reset_index(drop=True)

    # ---- 统一评测通道所需的面板/mask (复用主流程 DataPreprocessor) ----
    prep = DataPreprocessor()
    prep.prepare_full_pool(df)
    logger.info(f"数据集就绪 | Train/Val/Test: "
                f"{prep.tr_mask.sum():,}/{prep.val_mask.sum():,}/{prep.te_mask.sum():,}")

    # ---- AlphaGen 原始 OHLCV 面板 (自研 drop-in StockData) ----
    feat_vals, cal, syms = _build_panel(df)
    t_days = len(cal)
    logger.info(f"交易日面板: {t_days} 天 x {len(syms)} 股 x {feat_vals.shape[1]} 特征")

    data_eval = _make_stock_data(feat_vals, cal, syms, 0, t_days, device,
                                 max_future_days=_MAX_FUTURE)

    tr_days = sorted(pd.to_datetime(prep.full_dates[prep.tr_mask]).normalize().unique())
    tr_cal = pd.DatetimeIndex(tr_days)
    win_start = int(cal.get_indexer([tr_cal[0]])[0])
    win_end = int(cal.get_indexer([tr_cal[-1]])[0]) + 1
    n_tr = len(tr_cal)
    logger.info(f"PPO 训练窗口: {pd.Timestamp(tr_cal[0]).date()} ~ "
                f"{pd.Timestamp(tr_cal[-1]).date()} ({n_tr} 个交易日)")
    data_train = _make_stock_data(feat_vals, cal, syms, win_start, win_end, device,
                                  max_future_days=_MAX_FUTURE)

    # ---- 训练: 官方 pool/env + MaskablePPO, target = 1 日收益 (仅用历史) ----
    reseed_everything(seed)
    close = Feature(FeatureType.CLOSE)
    target = Ref(close, -1) / close - 1

    pool = AlphaPool(capacity=pool_capacity, stock_data=data_train,
                     target=target, ic_lower_bound=None)
    env = AlphaEnv(pool=pool, device=device, print_expr=False)

    from sb3_contrib.ppo_mask import MaskablePPO

    model = MaskablePPO(
        'MlpPolicy', env,
        policy_kwargs=dict(
            features_extractor_class=LSTMSharedNet,
            features_extractor_kwargs=dict(
                n_layers=n_layers,
                d_model=d_model,
                dropout=0.1,
                device=device,
            ),
        ),
        gamma=1.0,
        ent_coef=0.01,
        n_steps=n_steps_ppo,
        batch_size=batch_size,
        tensorboard_log=None,
        device=device,
        verbose=0,
    )

    budget_sec = max_time_min * 60 if max_time_min > 0 else 0
    logger.info(f"PPO 训练开始: steps={steps}, pool_capacity={pool_capacity}, "
                f"time_budget={budget_sec or 'unlimited'}s")
    t0 = time.time()
    cb_snap = _PoolSnapshotCallback(show_freq=max(50, n_steps_ppo))
    model.learn(total_timesteps=steps,
                callback=[cb_snap, _TimeBudgetCallback(budget_sec)],
                progress_bar=False)
    train_sec = time.time() - t0
    logger.info(f"PPO 训练完成 | 耗时 {train_sec:.1f}s | pool_size={pool.size} "
                f"| best_ic_ret={pool.best_ic_ret:.5f}")

    # ---- 候选: 最终 pool 按 |weight| 排序, 去重 ----
    st = pool.state
    candidates = sorted(
        zip(st['exprs'], st['weights'], st['ics_ret']),
        key=lambda t: -abs(t[1]) if t[1] is not None else 0)
    seen, cands = set(), []
    for expr, w, icr in candidates:
        key = str(expr)
        if key in seen:
            continue
        seen.add(key)
        cands.append((expr, float(w), float(icr)))
    logger.info(f"候选表达式: {len(cands)} 个 (pool_size={pool.size}, capacity={pool_capacity})")

    pool_json = os.path.join(Config.OUTPUT_DIR, "alphagen_pool.json")
    with open(pool_json, "w", encoding="utf-8") as f:
        json.dump(pool.to_dict(), f, ensure_ascii=False, indent=2)
    snap_json = os.path.join(Config.OUTPUT_DIR, "alphagen_pool_snapshots.json")
    with open(snap_json, "w", encoding="utf-8") as f:
        json.dump(cb_snap.snapshots, f, ensure_ascii=False, indent=2)

    # ---- 全量入库 (无 MLQC 门禁) + Test 集统一评测 ----
    fm = FamaMacBethRegressor(nw_lags=Config.NW_LAGS)
    registry = FactorRegistry(version="AlphaGen_PPO_baseline")

    midx_rows = pd.MultiIndex.from_arrays([
        pd.to_datetime(prep.full_dates).normalize(),
        prep.full_symbols.astype(str)])

    total_passed = 0
    for i, (expr, weight, icr) in enumerate(cands):
        formula = str(expr)
        used_feats = [lab.lower() for lab in FEATURE_LABELS if f'${lab.lower()}' in formula]
        complexity = len(formula)

        try:
            fac_series = _factor_long_series(expr, data_eval)
        except Exception as exc:
            logger.warning(f"候选 #{i} 计算失败: {exc}")
            continue
        pred_all = fac_series.reindex(midx_rows).to_numpy(dtype=np.float64)
        pred_all = np.where(np.isfinite(pred_all), pred_all, np.nan)

        pred_te = pred_all[prep.te_mask]
        test_ret = prep.full_y[prep.te_mask]
        test_dates = prep.full_dates[prep.te_mask]
        test_symbols = prep.full_symbols[prep.te_mask]
        test_amount = prep.full_amount[prep.te_mask]
        pure_pred_te, pure_ret_te, test_dates, test_symbols = _neutralize(
            pred_te, test_ret, test_dates, test_symbols, test_amount)
        if len(pure_pred_te) < 500:
            logger.warning(f"候选 #{i} 测试集有效样本过少，跳过入库")
            continue

        fid = f"AG_{total_passed:03d}"
        fm_test = fm.run(pure_pred_te, pure_ret_te, test_dates, test_symbols)
        fm.print_report(fm_test, factor_name=f"{fid} (Test集样本外)")

        fm_dict = fm_test.to_dict()
        fm_dict['formula'] = formula
        fm_dict['features'] = used_feats
        fm_dict['complexity'] = complexity
        fm_dict['AlphaGen_weight'] = round(weight, 6)
        fm_dict['AlphaGen_ic_ret'] = round(icr, 6)
        registry.register(fid, formula, fm_dict, used_feats)

        logger.info(f"{fid} 入库 | w={weight:.3f} | t={fm_test.t_stat:.3f} | "
                    f"特征: {len(used_feats)} | {formula[:60]}")
        total_passed += 1

    prod_factors = registry.get_production_ready()
    logger.info(f"AlphaGen 全量入库: {total_passed} 个 | 达生产标准: {len(prod_factors)} 个")

    # ---- 汇总报告 (两段式: 达标/未达标, 与 primary 同口径) ----
    passed_factors = registry.get_production_ready()
    passed_ids = {f['id'] for f in passed_factors}
    failed_entries = [{"id": fid, **info}
                      for fid, info in registry.data['factors'].items()
                      if fid not in passed_ids]
    report_lines = [
        "=" * 90,
        "AlphaGen (PPO) 外部 RL 基线因子挖掘汇总报告 (Test集样本外)",
        "=" * 90,
        f"统一达标标准: |t|≥{Config.PRODUCT_FMT_THRESHOLD}, "
        f"|ICIR|≥{Config.PRODUCT_ICIR_THRESHOLD}, |Rank_IC|≥{Config.PRODUCT_IC_THRESHOLD}, "
        f"截面期数≥30",
        "-" * 90,
        f"[1/2] 达标因子 ({len(passed_factors)} 个)",
        format_fm_table(passed_factors),
        "-" * 90,
        f"[2/2] 未达标因子 ({len(failed_entries)} 个)",
        format_fm_table(failed_entries),
    ]
    report_text = "\n".join(report_lines)
    logger.info(f"\n{report_text}")
    report_path = os.path.join(Config.OUTPUT_DIR, "fm_summary_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    logger.info(f"报告已保存至: {report_path}")

    # ---- 单因子回测 top-N (用回测通道复算, 保持口径) ----
    if prod_factors:
        logger.info("启动单因子回测...")
        from alpha.evaluation.backtester import AcademicBacktester
        backtester = AcademicBacktester()
        formula_to_fac = {}
        for expr, _, _ in cands:
            formula_to_fac[str(expr)] = _factor_long_series(expr, data_eval).reindex(midx_rows).to_numpy(dtype=np.float64)
        for factor in prod_factors[:backtest_top]:
            fid = factor['id']
            formula = factor['metrics']['formula']
            pred_all = formula_to_fac.get(formula)
            if pred_all is None:
                logger.warning(f"{fid} 回测: 无法重建因子值, 跳过")
                continue
            pred_all = np.where(np.isfinite(pred_all), pred_all, np.nan)
            try:
                backtester.run(pred_all[prep.te_mask],
                               prep.full_y[prep.te_mask],
                               prep.full_dates[prep.te_mask],
                               prep.full_symbols[prep.te_mask],
                               fid,
                               open_prices=prep.full_open[prep.te_mask] if prep.full_open is not None else None,
                               close_prices=prep.full_close[prep.te_mask] if prep.full_close is not None else None)
            except Exception as e:
                logger.warning(f"{fid} 回测失败: {e}")

    # ---- 对比表 ----
    rows = [{"tag": "AlphaGen-PPO (无MLQC)", "path": registry.path,
             **_production_metrics(registry)}]
    for tag, path in (compare or []):
        rows.append(_load_registry_stats(path, tag))
    write_comparison(rows, Config.OUTPUT_DIR)

    cfg_path = os.path.join(Config.OUTPUT_DIR, "alphagen_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump({
            "data": data_path, "steps": steps, "pool_capacity": pool_capacity,
            "seed": seed, "n_layers": n_layers, "d_model": d_model,
            "n_steps_ppo": n_steps_ppo, "batch_size": batch_size,
            "target": "Ref(close,-1)/close-1 (前向1日收益)",
            "max_backtrack_days": _MAX_BACKTRACK,
            "max_future_days": _MAX_FUTURE,
            "device": str(device),
            "train_seconds": round(train_sec, 1),
        }, f, indent=2, ensure_ascii=False)

    print(f"\nAlphaGen (PPO) 外部 RL 基线完成！产出物: {os.path.abspath(Config.OUTPUT_DIR)}")
    return prod_factors, registry, pool


def console_main(argv=None):
    parser = argparse.ArgumentParser(
        description="AlphaGen (PPO) 外部 RL 基线 (官方 alphagen 训练 + 统一评测)")
    parser.add_argument('--data', type=str,
                        default=os.path.join(REPO_ROOT, 'data', 'warmup', 'csi500_daily_2020-06-30_to_2026-06-30.parquet'))
    parser.add_argument('--steps', type=int, default=20000,
                        help='PPO 总时间步数 (默认 20000; 冒烟用 300)')
    parser.add_argument('--time', type=int, default=0,
                        help='训练时间预算 (分钟, 0=不限制)')
    parser.add_argument('--pool', type=int, default=20, help='AlphaPool 容量')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--market', type=str, default='zs500', choices=['zs500', 'hs300'])
    parser.add_argument('--feature-pool', type=str, default='buildin', choices=['buildin', 'alpha158'])
    parser.add_argument('--mode', type=str, default='warmup', choices=['cold', 'warmup', 'ratio', 'month'],
                        help='切分模式(与主线一致): warmup(前12月回溯+36:12:12, 默认), cold/ratio/month')
    parser.add_argument('--compare', type=str, default=None,
                        help='逗号分隔 tag:path 列表, 与其它方法 registry 对比')
    args = parser.parse_args(argv)

    Config.MARKET = args.market
    Config.FEATURE_POOL = args.feature_pool
    apply_split_mode(args.mode)
    compare = None
    if args.compare:
        compare = []
        for item in args.compare.split(','):
            if ':' in item:
                tag, path = item.split(':', 1)
            else:
                tag, path = os.path.basename(item), item
            compare.append((tag, path))

    run_alphagen_baseline(
        data_path=args.data,
        steps=args.steps,
        pool_capacity=args.pool,
        max_time_min=args.time,
        seed=args.seed,
        compare=compare,
        backtest_top=3,
    )


if __name__ == "__main__":
    console_main()