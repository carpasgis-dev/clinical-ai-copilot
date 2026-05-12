"""Resuelve ``ClinicalCapability`` SQLite según ``CLINICAL_DB_PATH``."""
from __future__ import annotations

import os
from pathlib import Path


def _repo_root() -> Path:
    """Raíz del repo ``clinical-ai-copilot`` (donde está ``.env``)."""
    return Path(__file__).resolve().parent.parent.parent


def clinical_cache_token() -> str:
    """
    Token para ``lru_cache`` del grafo: ruta absoluta del SQLite si existe, si no ``none``.

    Las rutas relativas se resuelven respecto a la **raíz del repo**, no a ``os.getcwd()``
    (evita stub SQL si uvicorn se lanza desde otra carpeta, p. ej. PyCharm).
    """
    raw = (os.getenv("CLINICAL_DB_PATH") or "").strip()
    if not raw:
        return "none"
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (_repo_root() / path).resolve()
    else:
        path = path.resolve()
    if not path.is_file():
        return "none"
    return str(path)


def build_sqlite_clinical_if_configured(db_path: str):
    """Instancia ``SqliteClinicalCapability`` solo si ``db_path`` es fichero válido."""
    from app.capabilities.clinical_sql import SqliteClinicalCapability

    cap = SqliteClinicalCapability(db_path=db_path)
    return cap if cap.health_check() else None
