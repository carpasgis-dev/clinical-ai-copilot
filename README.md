# Clinical Evidence Copilot

> Copiloto de IA clínica que combina datos estructurados (SQL/FHIR/Synthea)
> con evidencia biomédica fundamentada (PubMed + RAG + PMIDs verificables).

## Visión

Un copiloto healthcare-AI que decide de forma autónoma si una consulta requiere
**datos del paciente/centro** (SQL), **evidencia bibliográfica** (PubMed) o **ambas**,
y orquesta el pipeline adecuado de forma trazable y explicable.

**Diferenciador:** combinación de razonamiento clínico estructurado
con recuperación de evidencia biomédica verificable.

---

## Arquitectura

```
Usuario
  │
  ▼
[Router Node]          ← reglas deterministas, sin LLM
  │
  ├─ SQL ──────────────► [SQL Node]
  │                          │ SqlResult (query ejecutada, filas limitadas)
  │
  ├─ Evidence ─────────► [Evidence Node]
  │                          │ EvidenceBundle (PMIDs, abstracts acotados)
  │
  └─ Hybrid ───────────► [ClinicalSummary] → [PubMedQueryBuilder]
                              └──────────────► [EvidenceRetrieval]
                                                    │
                                              ▼
                                        [Synthesis Node]
                                              │
                                        [Safety & Disclaimer]
                                              │
                                        Respuesta Final
                              (JSON estructurado + markdown + trace)
```

El **orquestador** (LangGraph) solo conoce `CopilotState` y los contratos de capability.
Nunca importa SQL, schemas de BD ni APIs de PubMed directamente.

---

## Capabilities

| Capability | Origen | Responsabilidad |
|------------|--------|-----------------|
| **A — Clinical SQL** | sqlite-analyzer-mcp | SQL seguro, esquema, perfiles/cohortes |
| **B — Evidence RAG** | PRSN 3.0 | PubMed, RAG, PMIDs, Open Access |

---

## Estado del proyecto

### v0.1.0 — PR #1: contratos de orquestación

- [x] `app/schemas/copilot_state.py` — estado del grafo + DTOs con límites de contexto
- [x] `app/capabilities/contracts.py` — `ClinicalCapability` / `EvidenceCapability` (Protocols)
- [x] `app/orchestration/router.py` — clasificador determinista con señales clínicas
- [x] `tests/test_router.py` — golden paths + caso hero + tests de propiedades
- [ ] Grafo LangGraph (nodos, flujo completo)
- [ ] Capability A — implementación SQL
- [ ] Capability B — implementación PubMed
- [ ] FastAPI (`POST /query`) + UI

---

## Instalación rápida

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1     # Windows
# source .venv/bin/activate    # Linux / macOS

pip install -r requirements.txt
copy .env.example .env         # Rellenar credenciales
```

## Ejecutar tests

```bash
pytest tests/ -v
```

Salida esperada tras PR #1:

```
tests/test_router.py::test_hero_case PASSED
tests/test_router.py::test_sql_route[...] PASSED  (×6)
tests/test_router.py::test_evidence_route[...] PASSED  (×7)
tests/test_router.py::test_hybrid_route[...] PASSED  (×6)
tests/test_router.py::test_unknown_route[...] PASSED  (×5)
tests/test_router.py::test_router_is_pure PASSED
...
```

---

## Principios de diseño

**NO hacer:**
- Monolitos gigantes ni prompts con dumps completos de BD
- Dependencia total del razonamiento LLM
- Inventar PMIDs o citas no recuperadas
- Cometer credenciales reales

**HACER:**
- Routing determinista primero, LLM opcional después
- Límites de contexto documentados y enforced en los DTOs
- Citas verificables (PMID en cada artículo del bundle)
- Trazabilidad completa (`trace: list[TraceStep]`) para healthcare governance

---

## Roadmap

| Milestone | Descripción |
|-----------|-------------|
| `v0.1` | Contratos, router, tests (este PR) |
| `v0.2` | Grafo LangGraph con nodos stub |
| `v0.3` | Capability B real (PubMed E-utilities) |
| `v0.4` | Capability A real (SQL sobre Synthea) |
| `v0.5` | FastAPI + UI básica con trace visible |
| `v0.6` | Caso hero E2E funcional |
| `v1.0` | Evaluación, guardrails, memoria de sesión |
