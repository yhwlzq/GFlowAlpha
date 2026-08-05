import logging
import os
import json
from datetime import datetime
from typing import List, Dict
from alpha.config import Config

logger = logging.getLogger(__name__)


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

    def register(self, fid: str, formula: str, metrics: Dict, feats: List[str] = None) -> bool:
        entry = {
            "formula": formula,
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
            if (abs(m.get('FM_tstat', 0)) >= Config.TSTAT_THRESHOLD and
                    abs(m.get('ICIR', 0)) >= Config.ICIR_MIN and
                    abs(m.get('Rank_IC', 0)) >= Config.IC_THRESHOLD and
                    m.get('FM_n_periods', 0) >= 30):
                results.append({"id": fid, **info})
        return results
