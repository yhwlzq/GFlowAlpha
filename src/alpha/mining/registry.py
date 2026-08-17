import logging
import os
import json
from datetime import datetime
from typing import List, Dict, Optional
from alpha.config import Config

logger = logging.getLogger(__name__)


def is_production_ready(metrics: Dict, gate_status: Optional[str] = None) -> bool:
    """统一"生产达标"口径 (所有基线/报告/对比表共用同一把尺)。

    判定: 非 WEAK + |t|>=PRODUCT_FMT + |ICIR|>=PRODUCT_ICIR + |IC|>=PRODUCT_IC + 截面期数>=30。
    """
    if Config.EXCLUDE_WEAK_FROM_PRODUCTION and gate_status == 'WEAK':
        return False
    return (abs(metrics.get('FM_tstat', 0)) >= Config.PRODUCT_FMT_THRESHOLD and
            abs(metrics.get('ICIR', 0)) >= Config.PRODUCT_ICIR_THRESHOLD and
            abs(metrics.get('Rank_IC', 0)) >= Config.PRODUCT_IC_THRESHOLD and
            metrics.get('FM_n_periods', 0) >= 30)


def format_fm_table(entries: List[Dict]) -> str:
    """生成与 MiningOrchestrator._generate_summary_report 相同的 12 列表格文本.

    entries: [{'id': str, 'metrics': {FM_*}}, ...], 内部按 |t| 降序。
    """
    header = (f"{'ID':<12} {'t-stat':>8} {'p-value':>10} {'Coef':>10} {'NW-SE':>10} "
              f"{'Rank_IC':>8} {'ICIR':>8} {'L-S':>10} {'R²':>8} {'公式'}")
    lines = [header, "-" * 90]
    for e in sorted(entries, key=lambda x: -abs(x['metrics'].get('FM_tstat', 0))):
        m = e['metrics']
        sig = "***" if abs(m.get('FM_tstat', 0)) >= 3.0 else (
            "**" if abs(m.get('FM_tstat', 0)) >= 2.0 else "")
        lines.append(
            f"{e.get('id', ''):<12} {m.get('FM_tstat', 0):>7.3f}{sig:<3} "
            f"{m.get('FM_pvalue', 1):>10.6f} {m.get('FM_coef', 0):>10.6f} "
            f"{m.get('FM_se', 0):>10.6f} {m.get('Rank_IC', 0):>8.4f} "
            f"{m.get('ICIR', 0):>8.4f} {m.get('LS_spread', 0):>10.6f} "
            f"{m.get('FM_R2_avg', 0):>8.4f} {m.get('formula', '')[:50]}"
        )
    return "\n".join(lines)


class FactorRegistry:
    def __init__(self, version: str = "9.1_ICIR_Prior"):
        self.path = os.path.join(Config.OUTPUT_DIR, Config.REGISTRY_FILE)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.data = {"factors": {}, "meta": {"version": version}}
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    self.data = json.load(f)
            except:
                pass

    def register(self, fid: str, formula: str, metrics: Dict, feats: List[str] = None,
                 gate_status: Optional[str] = None) -> bool:
        entry = {
            "formula": formula,
            "gate_status": gate_status,
            "metrics": metrics,
            "registered_at": datetime.now().isoformat()
        }
        if feats is not None:
            entry["features"] = feats
        self.data["factors"][fid] = entry
        with open(self.path, "w", encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
        return True

    def get_production_ready(self) -> List[Dict]:
        results = []
        for fid, info in self.data["factors"].items():
            m = info.get("metrics", {})
            if not is_production_ready(m, info.get('gate_status')):
                continue
            results.append({"id": fid, **info})
        return results
