#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从已存在的 registry_academic.json 重新生成 fm_summary_report.txt，不重跑挖掘.

会把该 registry 下全部已注册因子写入报告（含 WEAK，Gate 列标注 PROD/WEAK），
保证文件始终有内容；生产达标数在日志中单独打印.

用法:
    python3 src/alpha/analysis/regen_fm_report.py --registry <path/to/registry_academic.json>
"""
import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-7s | %(message)s',
                    stream=sys.stdout)
logger = logging.getLogger('RegenFMReport')

try:
    from alpha.config import Config
    from alpha.mining.registry import is_production_ready
except Exception as e:
    logger.warning(f"alpha 导入失败 ({e!r})")

    def is_production_ready(metrics: Dict, gate_status: Optional[str] = None) -> bool:
        return (abs(metrics.get('FM_tstat', 0)) >= 3.0 and
                abs(metrics.get('ICIR', 0)) >= 0.30 and
                abs(metrics.get('Rank_IC', 0)) >= 0.03 and
                metrics.get('FM_n_periods', 0) >= 30)


def build_fm_report(factors: List[Dict], title: str = "因子挖掘汇总报告") -> str:
    """与 MiningOrchestrator._generate_summary_report 同格式; 始终输出全部因子,
    Gate 列标注 PROD (生产达标) / WEAK / '-'."""
    if not factors:
        return f"{title}\n\n未登记因子\n"
    lines = [
        "=" * 90,
        title,
        "=" * 90,
        f"{'ID':<12} {'t-stat':>8} {'p-value':>10} {'Coef':>10} {'NW-SE':>10} "
        f"{'Rank_IC':>8} {'ICIR':>8} {'L-S':>10} {'R²':>8} {'Gate':<6} 公式",
        "-" * 90,
    ]
    for f in sorted(factors, key=lambda x: abs(x['metrics'].get('FM_tstat', 0)), reverse=True):
        m = f['metrics']
        sig = "***" if abs(m.get('FM_tstat', 0)) >= 3.0 else (
            "**" if abs(m.get('FM_tstat', 0)) >= 2.0 else "")
        gate = f.get('gate_status', '')
        if is_production_ready(m, gate):
            gate_mark = "PROD"
        elif gate == 'WEAK':
            gate_mark = "WEAK"
        else:
            gate_mark = "-"
        line = (
            f"{f['id']:<12} {m.get('FM_tstat', 0):>7.3f}{sig:<3} "
            f"{m.get('FM_pvalue', 1):>10.6f} {m.get('FM_coef', 0):>10.6f} "
            f"{m.get('FM_se', 0):>10.6f} {m.get('Rank_IC', 0):>8.4f} "
            f"{m.get('ICIR', 0):>8.4f} {m.get('LS_spread', 0):>10.6f} "
            f"{m.get('FM_R2_avg', 0):>8.4f} {gate_mark:<6} {m.get('formula', '')[:50]}"
        )
        lines.append(line)
    return "\n".join(lines) + "\n"


def load_registry(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    factors = data.get('factors', {})
    entries = []
    for fid, info in factors.items():
        entries.append({'id': fid, **info})
    return entries, data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从 registry_academic.json 重新生成 fm_summary_report.txt")
    parser.add_argument('--registry', type=str, required=True,
                        help='registry_academic.json 路径')
    parser.add_argument('--title', type=str, default="因子挖掘汇总报告",
                        help='报告标题 (默认: 因子挖掘汇总报告)')
    args = parser.parse_args()

    if not os.path.exists(args.registry):
        logger.error(f"registry 不存在: {args.registry}")
        sys.exit(1)

    out_dir = os.path.dirname(os.path.abspath(args.registry))
    entries, data = load_registry(args.registry)

    n_prod = sum(1 for e in entries if is_production_ready(
        e.get('metrics', {}), e.get('gate_status')))
    logger.info(f"已注册因子: {len(entries)} 个 | 生产达标: {n_prod} 个 | "
                f"registry 版本: {data.get('meta', {}).get('version', '?')}")

    report_text = build_fm_report(entries, title=args.title)
    report_path = os.path.join(out_dir, 'fm_summary_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    logger.info(f"报告已写入: {report_path}")
    logger.info(f"\n{report_text}")


if __name__ == "__main__":
    main()