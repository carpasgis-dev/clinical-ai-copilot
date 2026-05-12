"""
Configuración global de pytest.
Añade la raíz del proyecto al sys.path para que los imports
`from app.schemas...` y `from app.orchestration...` funcionen
sin necesidad de instalar el paquete.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: pruebas que requieren red o APIs externas (p. ej. NCBI).",
    )
