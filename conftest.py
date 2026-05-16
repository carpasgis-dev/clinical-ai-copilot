"""
Configuración global de pytest.
Añade la raíz del proyecto al sys.path para que los imports
`from app.schemas...` y `from app.orchestration...` funcionen
sin necesidad de instalar el paquete.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))


@pytest.fixture(autouse=True)
def _reset_session_memory_between_tests() -> None:
    from app.session.memory import clear_session_memory_store

    clear_session_memory_store()
    yield
    clear_session_memory_store()


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: pruebas que requieren red o APIs externas (p. ej. NCBI).",
    )
