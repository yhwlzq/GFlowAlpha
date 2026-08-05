import json
import os
import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class AuditLogger:
    """JSONL 审计日志: 记录每窗每 trial 的搜索/去重/筛选过程, 供审稿透明性核查."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._fh = open(path, 'a', encoding='utf-8')

    def log(self, entry: Dict[str, Any]):
        row = dict(entry)
        row.setdefault('ts', datetime.now().isoformat(timespec='seconds'))
        self._fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._fh.flush()

    def log_window(self, window_idx: int, meta: Dict[str, Any]):
        self.log({'event': 'window_start', 'window': window_idx, **meta})

    def log_window_end(self, idx: int, n_trials: int, n_passed: int, n_registered: int):
        self.log({'event': 'window_end', 'window': idx, 'n_trials': n_trials,
                  'n_passed': n_passed, 'n_registered': n_registered})

    def log_trial(self, window: int, trial: int, **fields):
        self.log({'event': 'trial', 'window': window, 'trial': trial, **fields})

    def close(self):
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()