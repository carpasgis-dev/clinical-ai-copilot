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
  ├─ SQL ──────────────► cohorte SQLite (NL heurístico → SQL seguro o conteo simple)
  │
  ├─ Evidence ─────────► build_pubmed_query (planner) → retrieve_evidence
  │
  └─ Hybrid ───────────► resumen clínico (SQLite o stub) → planner → evidencia → síntesis (stub) → safety
```

- **Planner** (`EvidenceQueryPlanner`): solo construye `pubmed_query` (heurística, LLM o composite con fallback).
- **Capability** (`EvidenceCapability`): ejecuta búsqueda (NCBI, Europe PMC, multi-fuente o stub en tests).
- El grafo solo conoce `CopilotState` y los **Protocols** de `app/capabilities/contracts.py`.

---

## Capabilities

| Capability | Implementación v0.2 | Notas |
|------------|----------------------|--------|
| **A — Clinical SQL** | `SqliteClinicalCapability`, ETL Synthea (`scripts/synthea_csv_to_sqlite.py`), `cohort_nl` (NL → `WHERE` / `EXISTS` acotados) | Ver `docs/SYNTHEA.md` y `CLINICAL_DB_PATH`. |
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
| `GET /health` | Estado + config no sensible (`copilot_query_planner`, `copilot_evidence_backend`, host LLM, API key, BD clínica resuelta). |
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
| `CLINICAL_DB_PATH` | Ruta al SQLite de datos clínicos (p. ej. `data/clinical/synthea.db` tras tu ETL). Usado por `SqliteClinicalCapability`. **Rutas relativas** se resuelven desde la **raíz del repo** (no desde el directorio de trabajo de uvicorn). |

Al arrancar, `app/main.py` carga `.env` y fuerza desde fichero las claves del planner/LLM para evitar que variables del sistema las pisen.

---

## Datos Synthea y SQLite (capability A)

1. Clonar y ejecutar Synthea (Java 17+). En Windows usa `.\gradlew.bat run -Params="['-p','100']"` en lugar de `./run_synthea`.
2. Con CSV activados, importar a SQLite: `python scripts/synthea_csv_to_sqlite.py` (ver `docs/SYNTHEA.md`).
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

### Hecho (≈ v0.3)

- Grafo LangGraph: router determinista, evidencia inyectable, trazas (`trace`)
- `POST /query`, `GET /health`, caché de grafo (evidencia + planner + BD clínica)
- PubMed (NCBI), Europe PMC, multi-fuente, stub; planificador heurístico + LLM + composite
- `SqliteClinicalCapability` + ETL CSV→SQLite + ruta SQL con **NL heurístico → SQL seguro** (`app/capabilities/clinical_sql/cohort_nl.py`)
- Log de evaluación JSONL opcional

### Próximos pasos (cohorte SQL y analizador)

Inspiración parcial en el proyecto hermano `sina_mcp/sqlite-analyzer` (p. ej. introspección de esquema y agente SQL en `FHire.py`), manteniendo aquí **SQL solo vía plantillas / builder** y `run_safe_query`, sin texto SQL arbitrario del modelo.

| Ahora | Siguiente |
|--------|-----------|
| Heurística fija + sinónimos | **LLM o NER clínico** que rellene un `CohortNLSpec` (o JSON schema) con límites y validación |
| Solo `COUNT(DISTINCT id)` | **`SELECT` con agregados / desglose**, siempre plantillas + validación |
| Agente SQL abierto tipo `FHire.py` | Reutilizar ideas de **schema tool**, sin dejar al modelo escribir SQL arbitrario sin pasar por un **builder blanco** |

### Más adelante

- Síntesis con LLM (sustituir stub de síntesis)
- UI / evaluación sistemática
- Enriquecer cohorte (fechas Synthea, códigos, tablas adicionales del export)
