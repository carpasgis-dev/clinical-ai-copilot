"""Punto de entrada ASGI (FastAPI)."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values, load_dotenv

# Cargar `.env` de la raíz del repo antes de cualquier import que lea ``os.environ``
# (planner LLM, NCBI, backend de evidencia, caché del grafo).
_root = Path(__file__).resolve().parent.parent
_env_file = _root / ".env"
load_dotenv(_env_file, override=False)


def _force_dotenv_planner_and_llm(path: Path) -> None:
    """
    ``load_dotenv(override=False)`` no pisa variables ya definidas en el sistema (Windows
    a veces deja ``COPILOT_QUERY_PLANNER=heuristic``). Para planner y LLM, el fichero
    ``.env`` del repo debe mandar si la clave está definida ahí.
    """
    if not path.is_file():
        return
    data = dotenv_values(path)
    for key in ("COPILOT_QUERY_PLANNER", "LLM_BASE_URL", "LLM_MODEL", "OPENAI_API_KEY"):
        raw = data.get(key)
        if raw is None:
            continue
        s = str(raw).strip()
        if not s:
            continue
        os.environ[key] = s


_force_dotenv_planner_and_llm(_env_file)

from fastapi import FastAPI

from app.api.routes import router as query_router

app = FastAPI(
    title="Clinical Evidence Copilot",
    version="0.2.0",
    description=(
        "API mínima del copiloto: ``POST /query`` devuelve ruta, respuesta, traza y citas. "
        "Variables (ver ``.env.example``): ``COPILOT_EVIDENCE_BACKEND``, "
        "``COPILOT_QUERY_PLANNER`` (heuristic|llm|llm_only), ``LLM_*``, "
        "``COPILOT_EVAL_LOG_PATH``, ``NCBI_EMAIL``. El fichero ``.env`` se carga al arrancar.\n\n"
        "**Swagger /docs:** en respuestas 200, la pestaña *Example value* muestra **placeholders** "
        "generados a partir de los tipos del schema (p. ej. la cadena literal ``string``), **no** "
        "una respuesta del servidor. Los datos reales solo aparecen tras *Execute* en *Try it out*."
    ),
    swagger_ui_parameters={
        "defaultModelsExpandDepth": 3,
        "displayRequestDuration": True,
    },
)
app.include_router(query_router, tags=["query"])


@app.get("/", tags=["meta"])
def root() -> dict[str, str]:
    """Raíz: evita 404 al abrir el puerto en el navegador."""
    return {
        "service": "Clinical Evidence Copilot",
        "docs": "/docs",
        "health": "/health",
        "query": "POST /query",
    }


@app.get("/health", tags=["meta"])
def health_root() -> dict[str, str]:
    """Incluye lectura no sensible de config (útil para comprobar que el proceso ve el ``.env``)."""
    from urllib.parse import urlparse

    from app.api.clinical_factory import clinical_cache_token

    llm_url = (os.getenv("LLM_BASE_URL") or "").strip()
    host = urlparse(llm_url).netloc if llm_url else ""
    ctok = clinical_cache_token()
    return {
        "status": "ok",
        "copilot_query_planner": os.getenv("COPILOT_QUERY_PLANNER", "heuristic").lower().strip(),
        "copilot_evidence_backend": os.getenv("COPILOT_EVIDENCE_BACKEND", "ncbi").lower().strip(),
        "llm_base_url_host": host or "(unset)",
        "openai_api_key_set": "true" if (os.getenv("OPENAI_API_KEY") or "").strip() else "false",
        "clinical_db_loaded": "true" if ctok != "none" else "false",
        "clinical_db_path": ctok if ctok != "none" else "(unset or missing file)",
    }
