"""
Registro append-only de invocaciones (JSONL).

Variables:
    COPILOT_EVAL_LOG_PATH — ruta absoluta o relativa al fichero (default: ``data/eval/runs.jsonl``
    bajo la raíz del repo ``clinical-ai-copilot``).
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

_lock = threading.Lock()

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _default_log_path() -> Path:
    return _REPO_ROOT / "data" / "eval" / "runs.jsonl"


def _log_path() -> Path:
    raw = os.getenv("COPILOT_EVAL_LOG_PATH", "").strip()
    if "#" in raw:
        raw = raw.split("#", 1)[0].strip()
    if raw:
        p = Path(raw).expanduser()
        return p.resolve() if p.is_absolute() else (_REPO_ROOT / p).resolve()
    return _default_log_path()


def log_eval_event(record: Dict[str, Any]) -> None:
    """
    Añade una línea JSON al log de evaluación (thread-safe).

    Campos típicos: ``session_id``, ``query``, ``route``, ``route_reason``,
    ``pmids``, ``latency_ms``, ``ok``, ``error``, ``failure_kind``.
    """
    path = _log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    line_obj = {
        "ts": datetime.now(timezone.utc).isoformat(),
        **record,
    }
    line = json.dumps(line_obj, ensure_ascii=False, default=str) + "\n"
    with _lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(line)
