#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GP 基线公式解码: 将 gp_baseline registry 公式中的 X<idx> 占位变量
替换为实际特征列名 (如 X6 -> vol_20d)。

思路: 公式里 X<n> 是 gplearn 按特征列下标生成的占位符, 下标对应
DataPreprocessor.valid_features 的顺序。该池由孵化时的
seed / 数据 / 特征构建与聚类流程完全决定 (确定性), 因此用
与原始 run 相同的调用顺序复现可得 X_idx -> feature_name 的映射。

用法示例:
    python src/alpha/evaluation/gp_formula_decode.py \\
        --registry factor_output_academic_v81/gp_baseline_20260815_155058/registry_academic.json \\
        --data data/csi500_daily_2021-06-30_to_2026-06-30.parquet

默认原位改写并生成 .bak 备份; 可用 --dry-run 只打印不落盘。
"""
import os
import re
import sys
import json
import argparse
import logging

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from alpha.config import Config, set_global_seed
from alpha.data.data_loader import CSI500Loader
from alpha.mining.preprocessor import DataPreprocessor

logger = logging.getLogger('gp_formula_decode')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)-7s | %(message)s')
from alpha.config import REPO_ROOT, Config, set_global_seed
_REPO_ROOT = REPO_ROOT
_LEGACY_REGISTRY = os.path.join(_REPO_ROOT, 'factor_output_academic_v81', 'gp_baseline_20260815_155058', 'registry_academic.json')

_X_RE = re.compile(r'(?<!\w)X(\d+)(?!\w)')


def rebuild_feature_mapping(data_path: str) -> dict:
    """复现原始 run 的特征池, 返回 {X下标: 特征名}。"""
    set_global_seed()
    loader = CSI500Loader(path=data_path)
    df_raw = loader.load()

    prep = DataPreprocessor()
    prep.prepare_full_pool(df_raw)

    pool_size = len(prep.valid_features)
    logger.info(f"复现特征池: 共 {pool_size} 个")
    if pool_size < 1:
        raise RuntimeError("特征池为空, 无法构建映射")
    mapping = {i: name for i, name in enumerate(prep.valid_features)}
    return mapping


def decode_formula(formula: str, mapping: dict) -> str:
    def _repl(m):
        idx = int(m.group(1))
        name = mapping.get(idx)
        if name is None:
            raise ValueError(f"公式引用了不存在的特征 X{idx} (池大小 {len(mapping)})")
        return name
    return _X_RE.sub(_repl, formula)


def decode_registry(registry_path: str, mapping: dict, dry_run: bool = False):
    with open(registry_path, 'r', encoding='utf-8') as f:
        reg = json.load(f)

    decoded = 0
    for fid, entry in reg.get('factors', {}).items():
        for key in ('formula',):
            if key in entry and entry[key]:
                entry[key] = decode_formula(entry[key], mapping)
                decoded += 1
        metrics = entry.get('metrics') or {}
        if isinstance(metrics, dict) and metrics.get('formula'):
            metrics['formula'] = decode_formula(metrics['formula'], mapping)
            decoded += 1

    logger.info(f"已解码 {decoded} 处公式字段 (覆盖 {len(reg.get('factors', {}))} 个因子)")

    if dry_run:
        for fid, entry in reg['factors'].items():
            logger.info(f"{fid} -> {entry['formula']}")
        return

    backup_path = registry_path + '.bak'
    if not os.path.exists(backup_path):
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(reg, f, indent=2, ensure_ascii=False)
        logger.info(f"备份已保存: {backup_path}")

    with open(registry_path, 'w', encoding='utf-8') as f:
        json.dump(reg, f, indent=2, ensure_ascii=False)
    logger.info(f"已原位改写: {registry_path}")


def main():
    parser = argparse.ArgumentParser(description="GP baseline 公式 X<idx> -> 特征名解码")
    parser.add_argument('--registry', type=str, default=_LEGACY_REGISTRY,
                        help='registry_academic.json 路径')

    parser.add_argument('--data', default=os.path.join(_REPO_ROOT, 'data', 'csi500_daily_2021-06-30_to_2026-06-30.parquet'), help='数据路径 (CSV/Parquet)')
   
    parser.add_argument('--dry-run', action='store_true',
                        help='只解码并打印, 不改写文件')
    args = parser.parse_args()

    mapping = rebuild_feature_mapping(args.data)
    decode_registry(args.registry, mapping, dry_run=args.dry_run)


if __name__ == '__main__':
    main()