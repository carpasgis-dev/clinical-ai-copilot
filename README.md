# Clinical Evidence Copilot

> Copiloto de IA clínica que combina datos estructurados (SQL/FHIR/Synthea)
> con evidencia biomédica fundamentada (PubMed / Europe PMC, PMIDs verificables).

## Visión

Un copiloto healthcare-AI que decide de forma **determinista** si una consulta requiere
**datos del paciente/centro** (SQL), **evidencia bibliográfica** o **ambas**,
y orquesta el pipeline con **trazabilidad** (`trace`).

**Diferenciador:** routing sin LLM en el router; evidencia recuperada con APIs reales
y citas con PMID; **recuperación PubMed amplia (heurística multi-etapa)** + **re-ranking clínico**
(`ClinicalIntent` + alineación por eje PICO); planificación LLM opcional en traza (`pubmed_query`).

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
  ├─ Evidence ─────────► pubmed_query (LLM, traza) + retrieve (heurística multi-etapa) → rerank → síntesis
  │
  └─ Hybrid ───────────► cohorte SQL → clinical_context → pubmed_query (traza) → retrieve → rerank → síntesis
```

### Pipeline de evidencia (PubMed)

Diseño **recall amplio → precisión en rerank** (no filtrar todo en la query booleana):

```
pregunta + clinical_context
        │
        ▼
extract_clinical_intent()     → ClinicalIntent (población, intervención, comparador, outcomes, priority_axis)
        │
        ▼
build_evidence_search_queries()   ← heurística (lo que llama NCBI)
        │   Plan adaptativo A→B→C (parada temprana en executor si hay ≥12 PMIDs):
        │   A estricto (MACE/CVOT + edad MeSH si ≥65) → B moderado (cardiovascular benefit) → C amplio (solo si <5 PMIDs)
        │   Temas: CV, safety, renal, glycemic (bloques distintos por `primary_outcome_theme`)
        ▼
NCBI esearch + efetch (por stage) → merge PMIDs → pool pre-rerank (hasta ~200, ``COPILOT_PUBMED_RETRIEVAL_POOL_MAX``)
        │
        ▼
cheap filter: noise_suppression (temas laterales ×0.35–0.65, sin exclusión dura)
        │
        ▼
rerank_article_dicts(..., clinical_intent)
        │   heurística PICO + clinical_alignment + diseño de evidencia
        │   query semántica: build_intent_semantic_query (inglés clínico, no texto ES crudo)
        │   opcional (COPILOT_SEMANTIC_RERANK=full): bi-encoder top-K → cross-encoder → fusión
        ▼
top 6 → evidence_bundle → reasoning → síntesis
```

- **PubMed retrieval** (`heuristic_evidence_query.py`, multi-etapa en `executor`): stages A→B→C + `cv_evidence_hierarchy` (PubType suave) + `cvot_landmark`; ontología de desenlaces (`outcome_ontology.py`, MACE + componentes); memoria landmark (`clinical_knowledge.py`); supresión negativa intent-aware (`noise_suppression.py`). Tras ejecutar, **`pubmed_query` = query canónica real** (`pubmed_queries_executed` lista todas). Refinamiento LLM opcional: `COPILOT_PUBMED_LLM_REFINE=1`.
- **Heurística** (`heuristic_evidence_query.py`): MeSH + `[tiab]` en población/intervención; comparador y edad no son AND duros en PubMed (van al rerank).
- **Alineación** (`clinical_intent.py`, `clinical_alignment.py`): scores 0–1 por eje; penalización suave de off-topic (p. ej. depresión sin CV) solo sin señal CV; comparador «vs metformina» con frases explícitas.
- **Deduplicación** (`evidence_dedup.py`): PMIDs/títulos/`evidence_statements` antes de síntesis.
- **Capability** (`EvidenceCapability`): NCBI / Europe PMC / multi / stub.
- **Síntesis** (`build_stub_medical_answer` + `llm_synthesis` opcional): `MedicalAnswer` estructurado; con `COPILOT_SYNTHESIS=llm`, `final_answer` narrativo desde JSON de hechos (incl. `intencion_clinica` y `alineacion_clinica` por PMID).
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
| `POST /query` | Cuerpo: `{ "query": "...", "session_id": "opcional" }`. Respuesta: ruta, `pubmed_query` (planner LLM), `clinical_intent`, `final_answer`, `trace`, `pmids`, `citations`, `retrieval_debug` (p. ej. `multi_stage_queries`), etc. |
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
| `COPILOT_QUERY_PLANNER` | Legacy en `.env` / `GET /health` (default `heuristic`). Retrieval = heurística multi-etapa; no depende del LLM salvo refinamiento opcional. |
| `COPILOT_PUBMED_LLM_REFINE` | `1` / `true`: añade etapa `llm_refine` extra tras el plan heurístico (requiere `LLM_BASE_URL` + `LLM_MODEL`). Off por defecto. |
| `COPILOT_LLM_PROFILE` | `custom` (default), `openai`, `llamacpp`, `off` — ver `.env.example` y `app/config/llm_env.py`. |
| `COPILOT_SYNTHESIS` | `deterministic` (default): `final_answer` = render de `MedicalAnswer` por reglas. `llm`: `final_answer` = solo narrativa LLM (JSON de hechos + interpretación por PMID; sin pegar el borrador determinista); `medical_answer` sigue siendo estructurado. Si el LLM falla, fallback al render determinista. Opcionales: `COPILOT_SYNTHESIS_MAX_TOKENS` (default **1536**), `COPILOT_SYNTHESIS_TIMEOUT` (default **120** s, máx. 600), `COPILOT_SYNTHESIS_TEMPERATURE`. |
| `COPILOT_SEMANTIC_RERANK` | `off` (default). `embeddings` \| `cross_encoder` \| `full`: rerank con `sentence-transformers` (ver `requirements-semantic.txt`). Modelos: `COPILOT_EMBEDDING_MODEL`, `COPILOT_RERANKER_MODEL`. |
| `LLM_BASE_URL`, `LLM_MODEL`, `OPENAI_API_KEY` | Necesarios si `COPILOT_PUBMED_LLM_REFINE=1` o `COPILOT_SYNTHESIS=llm` (p. ej. `https://api.openai.com/v1` o `http://127.0.0.1:8080/v1`). |
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

**Priorizar:** routing determinista; límites en DTOs (`copilot_state`); **separación retrieval (recall) vs rerank (precisión clínica)**; trazas auditables; errores con `error_code` (sin sustituir silenciosamente LLM ni PubMed).

---

## Estado y roadmap

### Hecho (≈ v0.3–v0.4)

- Grafo LangGraph: router determinista, evidencia inyectable, trazas (`trace`)
- `POST /query`, `GET /health`, caché de grafo (evidencia + planner + BD clínica)
- PubMed (NCBI), Europe PMC, multi-fuente, stub; **recuperación heurística multi-etapa** + `pubmed_query` canónica auditada; LLM PubMed solo con `COPILOT_PUBMED_LLM_REFINE`
- **`ClinicalIntent`** + **`clinical_alignment`** (rerank por PICO; `priority_axis`; outcome gradual; comparador)
- **`evidence_dedup`**: deduplicación de PMIDs/citas antes de síntesis
- `MedicalAnswer` determinista + síntesis narrativa opcional (`COPILOT_SYNTHESIS=llm`)
- `SqliteClinicalCapability` + ETL CSV→SQLite + ruta SQL con **NL heurístico → SQL seguro** (`cohort_nl.py`)
- Log de evaluación JSONL opcional
- Tests: `test_clinical_intent_alignment`, `test_pubmed_retrieval_broad`, `test_evidence_dedup`

### Próximos pasos (cohorte SQL y analizador)

Inspiración parcial en el proyecto hermano `sina_mcp/sqlite-analyzer` (p. ej. introspección de esquema y agente SQL en `FHire.py`), manteniendo aquí **SQL solo vía plantillas / builder** y `run_safe_query`, sin texto SQL arbitrario del modelo.

| Ahora | Siguiente |
|--------|-----------|
| Heurística fija + sinónimos | **LLM o NER clínico** que rellene un `CohortNLSpec` (o JSON schema) con límites y validación |
| Solo `COUNT(DISTINCT id)` | **`SELECT` con agregados / desglose**, siempre plantillas + validación |
| Agente SQL abierto tipo `FHire.py` | Reutilizar ideas de **schema tool**, sin dejar al modelo escribir SQL arbitrario sin pasar por un **builder blanco** |

### Más adelante

- UI / evaluación sistemática (gold sets de ranking)
- Boost guías (ADA/ESC/EASD) moderado en rerank; detección de tensiones entre PMIDs
- Outcome/comparator con más cobertura semántica (embeddings) si la heurística lexical no basta
- Enriquecer cohorte (fechas Synthea, códigos, tablas adicionales del export)
