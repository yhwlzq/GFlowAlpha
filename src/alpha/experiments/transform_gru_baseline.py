#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
学术级日频单因子挖掘系统 — GRU 黑盒基线

与 transform_pysr_primary.py 同特征池、同切分、同评估口径 (FWL+FM),
唯一区别: 用 GRU 序列模型替代 GFlowNet+PySR 符号搜索。
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import logging
import sys
import json
import time
import argparse
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from alpha.config import REPO_ROOT, Config, set_global_seed
from alpha.data.data_loader import CSI500Loader
from alpha.mining.preprocessor import DataPreprocessor
from alpha.mining.fm_regression import FamaMacBethRegressor
from alpha.mining.neutralize import fwl_neutralize

logger = logging.getLogger('GRUBaseline')


# ============================================================
# 模型定义
# ============================================================

class GRUBaselineModel(nn.Module):
    """轻量 GRU 回归: GRU(85, 64, 1) → Linear(64, 1)."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, input_dim) → (batch,)."""
        _, h_n = self.gru(x)              # h_n: (num_layers, batch, hidden)
        last = h_n[-1]                     # (batch, hidden) — 取最后一层
        return self.head(last).squeeze(-1) # (batch,)


# ============================================================
# 滑窗序列数据集
# ============================================================

class StockSequenceDataset(Dataset):
    """按股票分组、滑窗构建 (seq_len, n_features) 序列.

    对于 train/val 集: 只返回 mask 内的 (序列, 标签) 对.
    序列窗口允许跨越 mask 边界 (用 warmup 区特征, 不泄漏标签).
    """

    def __init__(self, full_X: np.ndarray, full_y: np.ndarray,
                 full_symbols: np.ndarray, mask: np.ndarray,
                 seq_len: int = 20):
        self.seq_len = seq_len
        self.samples: List[Tuple[np.ndarray, float]] = []

        symbols = full_symbols[mask]
        unique_syms = np.unique(symbols)

        for sym in unique_syms:
            sym_idx = np.where((full_symbols == sym) & mask)[0]
            if len(sym_idx) < seq_len:
                continue
            # 需要完整的 seq_len 历史, 包括 mask 之前的 warmup 区
            first_in_mask = sym_idx[0]
            sym_all = np.where(full_symbols == sym)[0]
            pos_in_all = np.where(sym_all == first_in_mask)[0][0]

            for i in range(pos_in_all, len(sym_all)):
                global_idx = sym_all[i]
                if not mask[global_idx]:
                    continue
                start = sym_all[max(0, i - seq_len + 1)]
                window = full_X[start: global_idx + 1]
                if window.shape[0] < seq_len:
                    pad_len = seq_len - window.shape[0]
                    window = np.pad(window, ((pad_len, 0), (0, 0)), mode='edge')
                self.samples.append((window.astype(np.float32), float(full_y[global_idx])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)


# ============================================================
# 训练 / 评估主流程
# ============================================================

def run_gru_baseline(data_path: str,
                     seq_len: int = 20,
                     hidden_dim: int = 64,
                     num_layers: int = 1,
                     dropout: float = 0.1,
                     lr: float = 1e-3,
                     epochs: int = 200,
                     es_patience: int = 20,
                     batch_size: int = 1024,
                     seed: int = 42,
                     run_mode: str = 'cross_sectional',
                     compare: Optional[List[Tuple[str, str]]] = None):
    set_global_seed(seed)
    Config.MODE = run_mode
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    Config.OUTPUT_DIR = os.path.join("factor_output_academic_v81", f"run_{timestamp}_s{seed}")
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

    logger.info(f"加载日频数据: {data_path}")
    prep = DataPreprocessor()

    if run_mode == 'timing':
        data_util = CSI500Loader(path=data_path)
        df = data_util.load()
        df['date'] = pd.to_datetime(df['date'])
        df = df.dropna(subset=['close', 'volume']).sort_values(['symbol', 'date']).reset_index(drop=True)
        logger.info(f"时序择时模式 | 成分股数据: {len(df)} 行, {df['symbol'].nunique()} 只股票")
        prep.prepare_timing_dataset(df)
    else:
        data_util = CSI500Loader(path=data_path)
        df = data_util.load()
        required_cols = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'amount']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"数据必须包含列: {required_cols}")
        df['date'] = pd.to_datetime(df['date'])
        df = df.dropna(subset=['close', 'volume']).sort_values(['symbol', 'date']).reset_index(drop=True)
        prep.prepare_full_pool(df)

    feature_names = list(Config.FINAL_FEATURE_POOL)
    n_features = len(feature_names)
    logger.info(f"特征池: {n_features} 个原子因子 (与符号搜索相同)")

    # --- 构建序列数据集 ---
    logger.info(f"构建滑窗序列 (seq_len={seq_len}) ...")
    ds_train = StockSequenceDataset(prep.full_X, prep.full_y, prep.full_symbols, prep.tr_mask, seq_len)
    ds_val = StockSequenceDataset(prep.full_X, prep.full_y, prep.full_symbols, prep.val_mask, seq_len)
    logger.info(f"序列样本: train={len(ds_train):,} | val={len(ds_val):,}")

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=0)

    # --- 模型 & 训练 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GRUBaselineModel(n_features, hidden_dim, num_layers, dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float('inf')
    best_state = None
    no_improve = 0

    logger.info(f"GRU 训练开始 | hidden={hidden_dim} layers={num_layers} epochs={epochs} "
                f"es_patience={es_patience} device={device}")
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_batch = 0
        for xb, yb in dl_train:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            n_batch += 1
        scheduler.step()
        train_loss /= max(n_batch, 1)

        # val
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for xb, yb in dl_val:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss += criterion(pred, yb).item() * len(xb)
                n_val += len(xb)
        val_loss /= max(n_val, 1)

        if epoch % 10 == 0 or epoch == 1:
            logger.info(f"  Epoch {epoch:>3d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= es_patience:
                logger.info(f"  Early stopping at epoch {epoch} (patience={es_patience})")
                break

    elapsed = time.time() - t0
    logger.info(f"训练完成 | {elapsed:.1f}s | best_val_loss={best_val_loss:.6f}")
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    # --- Test 集预测 ---
    logger.info("Test 集预测 ...")
    test_dates = prep.full_dates[prep.te_mask]
    test_symbols = prep.full_symbols[prep.te_mask]
    test_amount = prep.full_amount[prep.te_mask]
    test_ret = prep.full_y[prep.te_mask]

    # 逐股票滑窗预测
    pred_te = np.full(prep.te_mask.sum(), np.nan)
    unique_test_syms = np.unique(test_symbols)

    for sym in unique_test_syms:
        sym_all_idx = np.where(prep.full_symbols == sym)[0]
        sym_test_idx = np.where((prep.full_symbols == sym) & prep.te_mask)[0]
        if len(sym_test_idx) == 0:
            continue
        windows = []
        for gi in sym_test_idx:
            pos_in_all = np.where(sym_all_idx == gi)[0][0]
            start = sym_all_idx[max(0, pos_in_all - seq_len + 1)]
            window = prep.full_X[start: gi + 1]
            if window.shape[0] < seq_len:
                pad_len = seq_len - window.shape[0]
                window = np.pad(window, ((pad_len, 0), (0, 0)), mode='edge')
            windows.append(window.astype(np.float32))
        if not windows:
            continue
        batch_x = torch.from_numpy(np.stack(windows)).to(device)
        with torch.no_grad():
            preds = model(batch_x).cpu().numpy()
        local_idx = sym_test_idx - prep.te_mask.nonzero()[0][0]
        # sym_test_idx 是 prep.full_X 中的位置, 需要映射到 te 子集的 0-based 索引
        te_positions = np.where(prep.te_mask)[0]
        for gi, p in zip(sym_test_idx, preds):
            pos_in_te = np.searchsorted(te_positions, gi)
            if pos_in_te < len(pred_te):
                pred_te[pos_in_te] = p

    pred_te = np.where(np.isfinite(pred_te), pred_te, np.nan)

    # --- FWL 中性化 + FM 评估 ---
    fm = FamaMacBethRegressor(nw_lags=Config.NW_LAGS)

    # 原始 FM (未中性化, 参照)
    ok_raw = np.isfinite(pred_te) & np.isfinite(test_ret)
    fm_raw = fm.run(pred_te[ok_raw], test_ret[ok_raw], test_dates[ok_raw], test_symbols[ok_raw])

    # FWL 中性化 + FM (统一评估通道)
    pure_pred, pure_ret = fwl_neutralize(pred_te, test_ret, test_dates, test_symbols, test_amount)
    ok = np.isfinite(pure_pred) & np.isfinite(pure_ret)
    fm_test = fm.run(pure_pred[ok], pure_ret[ok], test_dates[ok], test_symbols[ok])

    fid = f"GRU_{seed:03d}"
    fm.print_report(fm_test, factor_name=f"{fid} (Test集样本外, FWL+FM)")
    logger.info(f"{fid} | raw_IC={fm_raw.rank_ic_mean:.4f} raw_ICIR={fm_raw.rank_icir:.4f} | "
                f"FWL IC={fm_test.rank_ic_mean:.4f} ICIR={fm_test.rank_icir:.4f} "
                f"t={fm_test.t_stat:.3f}")

    # --- 汇总报告 ---
    report_lines = [
        "=" * 90,
        "GRU 黑盒基线因子挖掘汇总报告 (Test集样本外, FWL+FM)",
        "=" * 90,
        f"特征池: {n_features} 个原子因子 (FINAL_FEATURE_POOL, 与符号搜索相同)",
        f"模型: GRU | hidden={hidden_dim} layers={num_layers} seq_len={seq_len} | "
        f"lr={lr} epochs={epochs} es_patience={es_patience} | seed={seed}",
        f"训练耗时: {elapsed:.1f}s | 训练样本: {len(ds_train):,} | 验证样本: {len(ds_val):,}",
        "-" * 90,
        f"{'ID':<12} {'t-stat':>8} {'p-value':>10} {'Coef':>10} {'NW-SE':>10} "
        f"{'Rank_IC':>8} {'ICIR':>8} {'L-S':>10} {'R²':>8} {'原始IC':>8}",
        "-" * 90,
    ]
    sig = "***" if abs(fm_test.t_stat) >= 3.0 else ("**" if abs(fm_test.t_stat) >= 2.0 else "")
    report_lines.append(
        f"{fid:<12} {fm_test.t_stat:>7.3f}{sig:<3} {fm_test.p_value:>10.6f} "
        f"{fm_test.coefficient:>10.6f} {fm_test.std_error:>10.6f} {fm_test.rank_ic_mean:>8.4f} "
        f"{fm_test.rank_icir:>8.4f} {fm_test.long_short_spread:>10.6f} {fm_test.avg_r_squared:>8.4f} "
        f"{fm_raw.rank_ic_mean:>8.4f}"
    )
    report_text = "\n".join(report_lines)
    logger.info(f"\n{report_text}")
    report_path = os.path.join(Config.OUTPUT_DIR, "fm_summary_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    logger.info(f"报告已保存至: {report_path}")

    # --- 对比表 ---
    rows = []
    t = abs(fm_test.t_stat)
    icir = abs(fm_test.rank_icir)
    ic = abs(fm_test.rank_ic_mean)
    n_prod = int(t >= Config.TSTAT_THRESHOLD and icir >= Config.ICIR_MIN
                 and ic >= Config.IC_THRESHOLD and fm_test.n_periods >= 30)
    rows.append({
        "tag": "GRU+全部原子因子 (黑盒, 序列模型)",
        "path": Config.OUTPUT_DIR,
        "n_registered": 1, "n_production": n_prod,
        "mean_abs_t": round(t, 4), "max_abs_t": round(t, 4),
        "mean_abs_icir": round(icir, 4), "mean_abs_ic": round(ic, 4),
        "feature_union": n_features, "structure_unique": 1,
    })
    for tag, path in (compare or []):
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            factors = data.get("factors", {})
            t_vals, icir_vals, ic_vals = [], [], []
            for fid_, info in factors.items():
                m = info.get("metrics", info)
                t_vals.append(abs(m.get("FM_tstat", 0)))
                icir_vals.append(abs(m.get("ICIR", 0)))
                ic_vals.append(abs(m.get("Rank_IC", 0)))
            rows.append({
                "tag": tag, "path": path,
                "n_registered": len(factors),
                "n_production": sum(1 for tv, iv, icv in zip(t_vals, icir_vals, ic_vals)
                                    if tv >= Config.TSTAT_THRESHOLD and iv >= Config.ICIR_MIN and icv >= Config.IC_THRESHOLD),
                "mean_abs_t": float(np.mean(t_vals)) if t_vals else 0,
                "max_abs_t": float(np.max(t_vals)) if t_vals else 0,
                "mean_abs_icir": float(np.mean(icir_vals)) if icir_vals else 0,
                "mean_abs_ic": float(np.mean(ic_vals)) if ic_vals else 0,
                "feature_union": 0, "structure_unique": len(factors),
            })
    cols = ["tag", "path", "n_registered", "n_production",
            "mean_abs_t", "max_abs_t", "mean_abs_icir", "mean_abs_ic",
            "feature_union", "structure_unique"]
    cmp_df = pd.DataFrame(rows, columns=cols)
    csv_path = os.path.join(Config.OUTPUT_DIR, "comparison_baseline.csv")
    cmp_df.to_csv(csv_path, index=False, encoding="utf-8")
    md_path = os.path.join(Config.OUTPUT_DIR, "comparison_baseline.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "---|" * len(cols) + "\n")
        for _, row in cmp_df.iterrows():
            f.write("| " + " | ".join(str(row[c]) for c in cols) + " |\n")
    logger.info(f"对比表已保存: {csv_path}")
    print(cmp_df.to_string(index=False))

    # --- 保存配置 ---
    cfg_path = os.path.join(Config.OUTPUT_DIR, "gru_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump({
            "data": data_path,
            "seed": seed,
            "run_mode": run_mode,
            "feature_pool_size": n_features,
            "feature_pool": feature_names,
            "model": {
                "type": "GRU",
                "input_dim": n_features,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "dropout": dropout,
                "seq_len": seq_len,
            },
            "training": {
                "lr": lr, "epochs": epochs, "es_patience": es_patience,
                "batch_size": batch_size, "optimizer": "AdamW",
                "best_val_loss": float(best_val_loss),
                "elapsed_sec": round(elapsed, 1),
            },
            "metrics": {
                "FM_tstat": fm_test.t_stat, "FM_pvalue": fm_test.p_value,
                "FM_coef": fm_test.coefficient, "FM_se": fm_test.std_error,
                "Rank_IC": fm_test.rank_ic_mean, "ICIR": fm_test.rank_icir,
                "LS_spread": fm_test.long_short_spread, "R2_avg": fm_test.avg_r_squared,
                "FM_n_periods": fm_test.n_periods,
                "raw_Rank_IC": fm_raw.rank_ic_mean, "raw_ICIR": fm_raw.rank_icir,
            },
            "compare": compare or [],
        }, f, indent=2, ensure_ascii=False)

    print(f"\nGRU 基线完成！产出物: {os.path.abspath(Config.OUTPUT_DIR)}")
    return model, fm_test


# ============================================================
# CLI 入口 (对齐 transform_pysr_primary.py)
# ============================================================

def console_main(argv=None):
    parser = argparse.ArgumentParser(
        description="GRU 黑盒基线 (85 原子因子, 与符号搜索同特征/同测试集, FWL+FM 评估)")
    parser.add_argument('--data', type=str,
                        default=os.path.join(REPO_ROOT, 'data', 'warmup',
                                             'csi500_daily_2020-06-30_to_2026-06-30.parquet'))
    parser.add_argument('--market', type=str, default='zs500', choices=['zs500', 'hs300'],
                        help='市场类型: zs500 (中证500) 或 hs300 (沪深300)')
    parser.add_argument('--pool', type=str, default='buildin', choices=['buildin', 'alpha158'],
                        help='特征池: buildin (自建语义因子池) 或 alpha158 (Qlib Alpha158)')
    parser.add_argument('--mode', type=str, default='warmup', choices=['cold', 'warmup', 'ratio', 'month'],
                        help='切分模式: warmup(前12月回溯+36:12:12, 默认)')
    parser.add_argument('--run_mode', type=str, default='cross_sectional',
                        choices=['cross_sectional', 'timing'],
                        help='运行模式: cross_sectional (截面选股, 默认) 或 timing (时序择时)')
    parser.add_argument('--anchor', type=str, default='data_start', choices=['first_07_01', 'data_start'],
                        help='month 模式的锚点')
    parser.add_argument('--seed', type=int, default=Config.SEED,
                        help='随机种子 (默认 42)')
    # GRU 专属参数
    parser.add_argument('--seq-len', type=int, default=20,
                        help='回看窗口 (交易日, 默认 20)')
    parser.add_argument('--hidden', type=int, default=64,
                        help='GRU hidden_dim (默认 64)')
    parser.add_argument('--layers', type=int, default=1,
                        help='GRU 层数 (默认 1)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout (默认 0.1)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='学习率 (默认 1e-3)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='最大训练轮数 (默认 200)')
    parser.add_argument('--es-patience', type=int, default=20,
                        help='Early stopping patience (默认 20)')
    parser.add_argument('--batch-size', type=int, default=1024,
                        help='批大小 (默认 1024)')
    parser.add_argument('--compare', type=str, default=None,
                        help='逗号分隔 tag:path 列表, 与其它方法 registry 对比')
    args = parser.parse_args(argv)

    Config.MARKET = args.market
    Config.FEATURE_POOL = args.pool

    if args.mode == 'ratio':
        Config.SPLIT_MODE = 'ratio'
        Config.SPLIT_WARMUP = False
    elif args.mode == 'warmup':
        Config.SPLIT_MODE = 'month'
        Config.SPLIT_MONTH_ANCHOR = 'data_start'
        Config.SPLIT_WARMUP = True
    else:
        Config.SPLIT_MODE = 'month'
        Config.SPLIT_MONTH_ANCHOR = 'data_start'
        Config.SPLIT_WARMUP = False

    # 时序择时模式自动切换默认数据路径
    data_path = args.data
    if args.run_mode == 'timing' and 'csi500' in data_path:
        timing_default = os.path.join(REPO_ROOT, 'data', 'warmup',
                                      'hs300_daily_2020-06-30_to_2026-06-30.parquet')
        if os.path.exists(timing_default):
            data_path = timing_default
            logger.info(f"时序择时模式: 自动切换数据至 {data_path}")

    Config.SEED = args.seed

    compare = None
    if args.compare:
        compare = []
        for item in args.compare.split(','):
            if ':' in item:
                tag, path = item.split(':', 1)
            else:
                tag, path = os.path.basename(item), item
            compare.append((tag, path))

    run_gru_baseline(
        data_path=data_path,
        seq_len=args.seq_len,
        hidden_dim=args.hidden,
        num_layers=args.layers,
        dropout=args.dropout,
        lr=args.lr,
        epochs=args.epochs,
        es_patience=args.es_patience,
        batch_size=args.batch_size,
        seed=args.seed,
        run_mode=args.run_mode,
        compare=compare,
    )


if __name__ == "__main__":
    console_main()
