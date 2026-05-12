"""Tests resolución ``CLINICAL_DB_PATH`` / token de caché del grafo."""
from __future__ import annotations

import sqlite3

from app.api import clinical_factory


def test_clinical_cache_token_resolves_relative_to_repo(tmp_path, monkeypatch) -> None:
    """Rutas relativas deben encontrar el .db aunque ``cwd`` sea otra carpeta."""
    root = clinical_factory._repo_root()
    db = root / "_pytest_clinical_token.db"
    try:
        conn = sqlite3.connect(str(db))
        conn.execute("CREATE TABLE t (x INT)")
        conn.commit()
        conn.close()
        monkeypatch.setenv("CLINICAL_DB_PATH", "_pytest_clinical_token.db")
        monkeypatch.chdir(tmp_path)
        tok = clinical_factory.clinical_cache_token()
        assert tok != "none"
        assert tok.endswith("_pytest_clinical_token.db")
    finally:
        db.unlink(missing_ok=True)
