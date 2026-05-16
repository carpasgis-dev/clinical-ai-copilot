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
        # NCBI: sin forzar, un export erróneo en el shell pisa el .env (override=False arriba).
        "NCBI_API_KEY",
        "NCBI_EMAIL",
        "COPILOT_SYNTHESIS",
        "COPILOT_SYNTHESIS_TIMEOUT",
        "COPILOT_SYNTHESIS_MAX_TOKENS",
        "COPILOT_SYNTHESIS_TEMPERATURE",
        "COPILOT_PUBMED_PLANNER_MAX_TOKENS",
    ):
        raw = data.get(key)
        if raw is None:
            continue
        s = str(raw).strip()
        if not s:
            continue
        os.environ[key] = s

    # Perfil ``llamacpp``: a veces solo hay ``LLAMACPP_LLM_*`` (sin ``LLM_*`` en el fichero).
    # Forzar aquí evita que ``LlmQueryPlanner`` arranque sin variables si ``apply_copilot_llm_profile_from_dotenv``
    # no se ha ejecutado aún (imports alternativos) o el proceso no recargó el .env.
    prof = str(data.get("COPILOT_LLM_PROFILE") or "").strip().lower()
    if prof in ("llamacpp", "llama_cpp", "ollama", "local"):

        def _pick_llm(keys: tuple[str, ...]) -> str:
            for k in keys:
                raw = data.get(k)
                if raw is None:
                    continue
                s = str(raw).strip()
                if s:
                    return s
            return ""

        bump_b = _pick_llm(("LLAMACPP_LLM_BASE_URL", "LLAMA_CPP_OPENAI_BASE_URL", "LLM_BASE_URL"))
        bump_m = _pick_llm(("LLAMACPP_LLM_MODEL", "LLAMA_CPP_MODEL", "LLM_MODEL"))
        if bump_b:
            os.environ["LLM_BASE_URL"] = bump_b.rstrip("/")
        if bump_m:
            os.environ["LLM_MODEL"] = bump_m


_force_dotenv_planner_and_llm(_env_file)

from app.config.llm_env import apply_copilot_llm_profile_from_dotenv

apply_copilot_llm_profile_from_dotenv(_env_file)

from fastapi import FastAPI # pyright: ignore[reportMissingImports]
from fastapi.responses import Response # pyright: ignore[reportMissingImports]

from app.api.routes import router as query_router
from app.api.schemas import HealthOut, RootOut

app = FastAPI(
    title="Clinical Evidence Copilot",
    version="0.2.0",
    description=(
        "API mínima del copiloto: ``POST /query`` devuelve ruta, respuesta, traza y citas. "
        "Variables (ver ``.env.example``): ``COPILOT_EVIDENCE_BACKEND``, "
        "``COPILOT_QUERY_PLANNER`` (heuristic|llm|llm_only), ``COPILOT_LLM_PROFILE`` "
        "(custom|off|openai|llamacpp), ``LLM_*``, "
        "``COPILOT_EVAL_LOG_PATH``, ``NCBI_EMAIL``. El fichero ``.env`` se carga al arrancar.\n\n"
        "``COPILOT_SYNTHESIS`` (``deterministic`` | ``llm``): con ``llm`` y ``LLM_BASE_URL`` + ``LLM_MODEL``, "
        "la narrativa de ``final_answer`` puede generarse vía chat (p. ej. llama-server); si falla, se mantiene la síntesis por reglas. "
        "``COPILOT_SYNTHESIS_TIMEOUT`` (default 90 s, máx. 600) controla la espera HTTP de lectura; si ``LLM_BASE_URL`` es solo host:puerto, se añade ``/v1`` automáticamente (Ollama, etc.).\n\n"
        "**Swagger /docs:** *Example value* del 200 se genera desde los **tipos** del schema "
        "(p. ej. ``string``, ``0``); no es una respuesta real (usa *Execute*). "
        "Los sub-objetos tipados (p. ej. ``operator_counts`` con claves ``and``/``or``, ``Http422Response``, "
        "``GET /`` → ``RootOut``, ``GET /health`` → ``HealthOut``) evitan placeholders ``additionalProp*`` "
        "donde aplica; ``sql_result.rows`` solo expone columnas declaradas (p. ej. ``cohort_size``)."
    ),
    swagger_ui_parameters={
        "defaultModelsExpandDepth": 3,
        "displayRequestDuration": True,
    },
)
app.include_router(query_router, tags=["query"])


@app.on_event("startup")
async def _startup_preload_semantic_models() -> None:
    """Precarga bi-encoder + cross-encoder en GPU al arrancar (evita latencia en primera petición)."""
    import asyncio

    from app.capabilities.evidence_rag.semantic_ranking import preload_models

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, preload_models)


@app.get("/", tags=["meta"], response_model=RootOut)
def root() -> RootOut:
    """Raíz: evita 404 al abrir el puerto en el navegador."""
    return RootOut(
        service="Clinical Evidence Copilot",
        docs="/docs",
        health="/health",
        query="POST /query",
    )


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    """El navegador pide /favicon.ico al abrir ``/``; sin ruta genera 404 en logs."""
    return Response(status_code=204)


@app.get("/health", tags=["meta"], response_model=HealthOut)
def health_root() -> HealthOut:
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
    return HealthOut(
        status="ok",
        copilot_query_planner=os.getenv("COPILOT_QUERY_PLANNER", "heuristic").lower().strip(),
        copilot_llm_profile=os.getenv("COPILOT_LLM_PROFILE", "custom").lower().strip(),
        copilot_synthesis=(os.getenv("COPILOT_SYNTHESIS") or "deterministic").lower().strip(),
        copilot_evidence_backend=os.getenv("COPILOT_EVIDENCE_BACKEND", "ncbi").lower().strip(),
        llm_base_url_host=host or "(unset)",
        openai_api_key_set="true" if (os.getenv("OPENAI_API_KEY") or "").strip() else "false",
        clinical_db_loaded="true" if ctok != "none" else "false",
        clinical_db_path=ctok if ctok != "none" else "(unset or missing file)",
        clinical_capability_ready="true" if cap_ok else "false",
    )
