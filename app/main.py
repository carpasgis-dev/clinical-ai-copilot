"""Punto de entrada ASGI (FastAPI)."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values, load_dotenv  # pyright: ignore[reportMissingImports]

# Cargar `.env` de la raíz del repo antes de cualquier import que lea ``os.environ``
# (planner LLM, NCBI, backend de evidencia, caché del grafo).
_root = Path(__file__).resolve().parent.parent
_env_file = _root / ".env"
load_dotenv(_env_file, override=False)


def _force_dotenv_planner_and_llm(path: Path) -> None:
    """
    ``load_dotenv(override=False)`` no pisa variables ya definidas en el sistema.

    Para estas claves, el fichero ``.env`` del repo debe mandar si está definido ahí
    (evita ``CLINICAL_DB_PATH`` vacío en el shell → stub SQL eterno; mismo criterio
    que planner/LLM en Windows).
    """
    if not path.is_file():
        return
    data = dotenv_values(path)
    for key in (
        "COPILOT_QUERY_PLANNER",
        "COPILOT_LLM_PROFILE",
        "LLM_BASE_URL",
        "LLM_MODEL",
        "OPENAI_API_KEY",
        "CLINICAL_DB_PATH",
        "SYNTHEA_CSV_DIR",
    ):
        raw = data.get(key)
        if raw is None:
            continue
        s = str(raw).strip()
        if not s:
            continue
        os.environ[key] = s


_force_dotenv_planner_and_llm(_env_file)

from app.config.llm_env import apply_copilot_llm_profile_from_dotenv

apply_copilot_llm_profile_from_dotenv(_env_file)

from fastapi import FastAPI # pyright: ignore[reportMissingImports]
from fastapi.responses import Response # pyright: ignore[reportMissingImports]

from app.api.routes import router as query_router

app = FastAPI(
    title="Clinical Evidence Copilot",
    version="0.2.0",
    description=(
        "API mínima del copiloto: ``POST /query`` devuelve ruta, respuesta, traza y citas. "
        "Variables (ver ``.env.example``): ``COPILOT_EVIDENCE_BACKEND``, "
        "``COPILOT_QUERY_PLANNER`` (heuristic|llm|llm_only), ``COPILOT_LLM_PROFILE`` "
        "(custom|off|openai|llamacpp), ``LLM_*``, "
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


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    """El navegador pide /favicon.ico al abrir ``/``; sin ruta genera 404 en logs."""
    return Response(status_code=204)


@app.get("/health", tags=["meta"])
def health_root() -> dict[str, str]:
    """Incluye lectura no sensible de config (útil para comprobar que el proceso ve el ``.env``)."""
    from urllib.parse import urlparse

    from app.api.clinical_factory import build_sqlite_clinical_if_configured, clinical_cache_token

    llm_url = (os.getenv("LLM_BASE_URL") or "").strip()
    host = urlparse(llm_url).netloc if llm_url else ""
    ctok = clinical_cache_token()
    cap_ok = False
    if ctok != "none":
        cap = build_sqlite_clinical_if_configured(ctok)
        cap_ok = cap is not None
    return {
        "status": "ok",
        "copilot_query_planner": os.getenv("COPILOT_QUERY_PLANNER", "heuristic").lower().strip(),
        "copilot_llm_profile": os.getenv("COPILOT_LLM_PROFILE", "custom").lower().strip(),
        "copilot_evidence_backend": os.getenv("COPILOT_EVIDENCE_BACKEND", "ncbi").lower().strip(),
        "llm_base_url_host": host or "(unset)",
        "openai_api_key_set": "true" if (os.getenv("OPENAI_API_KEY") or "").strip() else "false",
        "clinical_db_loaded": "true" if ctok != "none" else "false",
        "clinical_db_path": ctok if ctok != "none" else "(unset or missing file)",
        "clinical_capability_ready": "true" if cap_ok else "false",
    }
