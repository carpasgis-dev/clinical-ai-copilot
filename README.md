# Clinical Evidence Copilot

> Copiloto de IA clínica que combina datos estructurados (SQL/FHIR/Synthea)
> con evidencia biomédica fundamentada (PubMed / Europe PMC, PMIDs verificables).

## Visión

Un copiloto healthcare-AI que decide de forma **determinista** si una consulta requiere
**datos del paciente/centro** (SQL), **evidencia bibliográfica** o **ambas**,
y orquesta el pipeline con **trazabilidad** (`trace`).

**Diferenciador:** routing sin LLM en el router; evidencia recuperada con APIs reales
y citas con PMID; planificación de query PubMed **heurística y/o LLM** (configurable).

---

## Arquitectura

```
Usuario
  │
  ▼
POST /query  →  LangGraph (router determinista)
  │
  ├─ SQL ──────────────► nodo SQL (stub en v0.2)
  │
  ├─ Evidence ─────────► build_pubmed_query (planner) → retrieve_evidence
  │
  └─ Hybrid ───────────► resumen clínico (stub) → planner → evidencia → síntesis (stub) → safety
```

- **Planner** (`EvidenceQueryPlanner`): solo construye `pubmed_query` (heurística, LLM o composite con fallback).
- **Capability** (`EvidenceCapability`): ejecuta búsqueda (NCBI, Europe PMC, multi-fuente o stub en tests).
- El grafo solo conoce `CopilotState` y los **Protocols** de `app/capabilities/contracts.py`.

---

## Capabilities

| Capability | Implementación v0.2 | Notas |
|------------|----------------------|--------|
| **A — Clinical SQL** | `SqliteClinicalCapability` (`app/capabilities/clinical_sql/`) | SQLite vía `CLINICAL_DB_PATH`; el nodo clínico del grafo sigue en stub hasta cablearlo. Ver `docs/SYNTHEA.md`. |
| **B — Evidence** | `NcbiEvidenceCapability`, `EuropePmcCapability`, `MultiSourceEvidenceCapability`, `StubEvidenceCapability` | E-utilities alineadas con PRSN; query compartida con Europe PMC |

---

## API (FastAPI)

Desde la raíz del repo `clinical-ai-copilot`:

```bash
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

| Ruta | Descripción |
|------|-------------|
| `POST /query` | Cuerpo: `{ "query": "...", "session_id": "opcional" }`. Respuesta: ruta, `pubmed_query`, `final_answer`, `trace`, `pmids`, `citations`, etc. |
| `GET /health` | Estado + lectura no sensible de config (`copilot_query_planner`, `copilot_evidence_backend`, host LLM, si hay API key). |
| `GET /` | Enlaces a docs y health |
| `GET /docs` | Swagger UI |

En Swagger, la pestaña **Example value** del código 200 muestra placeholders típicos del schema; los datos reales aparecen tras **Execute** en *Try it out*.

---

## Variables de entorno

Copiar `.env.example` → `.env` y revisar al menos:

| Variable | Rol |
|----------|-----|
| `COPILOT_EVIDENCE_BACKEND` | `ncbi` (default), `stub`, `epmc`, `multi` |
| `COPILOT_QUERY_PLANNER` | `heuristic`, `llm` (LLM + fallback heurístico), `llm_only` |
| `LLM_BASE_URL`, `LLM_MODEL`, `OPENAI_API_KEY` | Necesarios si `COPILOT_QUERY_PLANNER=llm` o `llm_only` (p. ej. `https://api.openai.com/v1` + modelo OpenAI). |
| `NCBI_EMAIL` | Recomendado para cuotas E-utilities |
| `COPILOT_EVAL_LOG_PATH` | Opcional; log JSONL de evaluación |
| `CLINICAL_DB_PATH` | Ruta al SQLite de datos clínicos (p. ej. `data/clinical/synthea.db` tras tu ETL). Usado por `SqliteClinicalCapability`. |

Al arrancar, `app/main.py` carga `.env` y fuerza desde fichero las claves del planner/LLM para evitar que variables del sistema las pisen.

---

## Datos Synthea y SQLite (capability A)

1. Clonar y ejecutar Synthea (Java 17+). En Windows usa `.\gradlew.bat run -Params="['-p','100']"` en lugar de `./run_synthea`.
2. Synthea **no** genera por defecto un `synthea.db`: normalmente FHIR/CCDA bajo `output/`; el CSV hay que activarlo en `synthea.properties`. Un `.db` SQLite es un paso **posterior** (import / ETL) al esquema que definas.
3. Apunta `CLINICAL_DB_PATH` en `.env` a ese fichero.
4. Detalle paso a paso: **[`docs/SYNTHEA.md`](docs/SYNTHEA.md)**.

---

## Instalación rápida

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1     # Windows
# source .venv/bin/activate    # Linux / macOS

pip install -r requirements.txt
copy .env.example .env         # Rellenar según tabla anterior
```

## Tests

```bash
pytest tests/ -q
```

Incluye grafo, API, PubMed (parser/integration opcional con `RUN_NCBI_INTEGRATION`), planificadores de query, etc.

---

## Principios de diseño

**Evitar:** monolitos con dumps de BD; inventar PMIDs; subir `.env` con secretos.

**Priorizar:** routing determinista; límites en DTOs (`copilot_state`); **separación planner vs retrieval**; trazas auditables; fallback heurístico cuando el LLM falle.

---

## Estado y roadmap

### Hecho (≈ v0.2.x)

- Grafo LangGraph con nodos stub + evidencia inyectable
- `POST /query`, `GET /health`, carga de `.env` + caché de grafo por backend y planner
- PubMed (NCBI), Europe PMC, multi-fuente, stub; planificador heurístico + LLM + composite
- Log de evaluación JSONL opcional

### Siguiente

- Cablear `SqliteClinicalCapability` en el grafo (sustituir stub clínico / SQL) cuando `CLINICAL_DB_PATH` exista
- Síntesis con LLM (sustituir stub)
- Capability A SQL real sobre datos de demo
- UI / evaluación sistemática
