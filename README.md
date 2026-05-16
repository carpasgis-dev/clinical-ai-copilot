# Clinical Evidence Copilot 🩺🤖

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-async-green.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-orchestration-orange.svg)
![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)
![Status](https://img.shields.io/badge/status-experimental-yellow.svg)

Copiloto clínico experimental orientado a **evidencia biomédica real**. 

Combina contexto estructurado de pacientes (SQL/FHIR/Synthea) con recuperación semántica sobre PubMed y Europe PMC, utilizando un **motor de ranking epistémico** que prioriza Ensayos Clínicos Aleatorizados (RCTs), Meta-Análisis y Guías Clínicas sobre evidencia mecanicista o preclínica.

Diseñado estrictamente para minimizar alucinaciones y mantener trazabilidad completa mediante PMIDs verificables.

> **La similitud semántica no equivale a calidad clínica.** <br>
> No inventa papers. No mezcla evidencia débil con fuerte. No responde sin trazabilidad.

---

## ⚡ Quick Demo (Time to Insight)

**Request** (`POST /query`):
```json
{
  "query": "Paciente diabético mayor de 65 años con ERC. ¿Qué evidencia reciente existe sobre reducción de riesgo cardiovascular?",
  "session_id": "demo-1"
}
```

**Response**:
```json
{
  "route": "hybrid",
  "execution_plan": ["pubmed_query", "evidence_retrieval", "synthesis"],
  "pmids": ["37212345", "38190012"],
  "reasoning_state": {
    "evidence_quality": "alta",
    "evidence_assessments": [
      {
        "pmid": "37212345",
        "study_type": "rct",
        "applicability": "Alineación positiva: cohorte adulta mayor con comorbilidades concurrentes."
      }
    ]
  },
  "final_answer": "Los análisis clínicos recientes respaldan el uso de inhibidores de SGLT2... [1]",
  "latency_ms": 842.3
}
```

---

## 🧠 ¿Por qué existe este proyecto?

La inmensa mayoría de los sistemas RAG (Retrieval-Augmented Generation) médicos optimizan simplemente la proximidad de sus *embeddings* vectoriales. Sin embargo, el razonamiento clínico no funciona así.

Un paper mecanicista basado en modelos murinos (ratones) **nunca** debería posicionar por encima de un ensayo clínico aleatorizado fase III con 10,000 pacientes, incluso si sus embeddings comparten una similitud semántica altísima con la pregunta.

Este proyecto explora arquitecturas RAG alineadas con la **jerarquía de la Medicina Basada en Evidencia (MBE)** en lugar de limitar el retrieval a la proximidad matemática pura.

---

## ✨ ¿Qué hace a este copiloto diferente?

Este proyecto no es otro *wrapper* de ChatGPT. Su arquitectura está diseñada desde cero para el rigor clínico, con varias capas de control que no encontrarás en sistemas RAG genéricos:

1.  **Búsqueda Guiada por Relevancia Clínica (PICO)**: En lugar de una simple búsqueda semántica, el sistema primero descompone la pregunta en un esquema PICO (Paciente, Intervención, Comparación, Desenlace). Esto permite construir consultas a PubMed mucho más precisas y ejecutar "escaleras de evidencia" automáticas, que van de lo más específico a lo más general si no se encuentran resultados, manteniendo siempre el control.

2.  **Ranking Epistémico: La Calidad de la Evidencia Importa**: El núcleo del sistema es un motor de ranking que entiende la jerarquía de la evidencia médica. Un Ensayo Clínico Aleatorizado (RCT) siempre tendrá más peso que un estudio preclínico en ratones, incluso si este último es semánticamente más cercano a la pregunta. Esto se implementa con un multiplicador de puntuación basado en el tipo de estudio, previniendo que evidencia débil contamine la respuesta final.

3.  **Filtro de Aplicabilidad Demográfica**: El sistema cruza automáticamente los datos del paciente (ej: "mujer de 72 años") con los de los estudios recuperados. Si un artículo trata sobre pediatría o una población completamente diferente, su puntuación se penaliza, evitando recomendaciones basadas en evidencia no aplicable.

4.  **Síntesis Anti-Alucinaciones**: La generación de la respuesta final está fuertemente restringida. El LLM opera con `temperature=0.0` (máximo determinismo) y sobre un conjunto de "hechos" extraídos de los papers. Un post-proceso verifica que cada afirmación en la respuesta esté directamente respaldada por una cita (PMID), eliminando la invención de información.

5.  **Enrutamiento Híbrido (SQL + RAG)**: El sistema es capaz de decidir de forma autónoma si la pregunta se responde mejor consultando una base de datos de pacientes locales (vía SQL) o buscando en la literatura médica externa (vía RAG), o una combinación de ambas. Esto permite respuestas que integran el contexto local con la evidencia global.

---

## 🩺 Casos de Uso y Consultas de Ejemplo

El sistema está diseñado para resolver escenarios clínicos complejos que combinan el estado de una cohorte local (vía SQL) con la literatura biomédica mundial (vía PubMed/Europe PMC). Algunos ejemplos de consultas que el copiloto comprende y resuelve óptimamente:

- **Riesgo Cardiometabólico (SGLT2 / GLP-1):** 
  > *"En pacientes con diabetes e hipertensión ≥65 años, ¿qué evidencia hay sobre inhibidores SGLT2 o agonistas GLP-1 frente a solo metformina para reducir eventos cardiovasculares (MACE)?"*
- **Anticoagulación y Fibrilación Auricular:** 
  > *"Paciente de nuestra cohorte con diabetes, hipertensión y fibrilación auricular ≥75 años. ¿Qué evidencia existe sobre anticoagulantes orales directos (DOAC) frente a warfarina para prevención de ictus?"*
- **Protección Renal e Insuficiencia Cardíaca:** 
  > *"Diabetes y enfermedad renal crónica en la cohorte local: ¿cuál es la evidencia de uso de iSGLT2 en desenlaces renales y cardiovasculares prospectivos?"*
- **Seguridad y Efectos Adversos:** 
  > *"En pacientes diabéticos de nuestra cohorte, ¿qué evidencia de alta calidad existe sobre la seguridad clínica de semaglutide respecto a episodios de hipoglucemia?"*
- **Intersección Híbrida Explícita (SQL + Literatura):** 
  > *"Usando los pacientes de nuestra base de datos biomédica local y la bibliografía externa actual, ¿qué opciones terapéuticas tienen mejor evidencia para reducir MACE en diabetes tipo 2 con hipertensión?"*

---

## 🏗️ Arquitectura del Flujo (Epistemic RAG)

El pipeline opera como un embudo de precisión rigurosa:

```text
Patient Context (SQL/FHIR) & Free-text Query
       ↓
Clinical Intent Extraction (PICO & Desired Outcomes)
       ↓
Adaptive PubMed Retrieval (High-Recall Stage)
       ↓
Semantic Cross-Encoder Reranking 
       ↓
Epistemic Hierarchy Modeling (RCTs > Preclinical)
       ↓
Demographic Applicability Adjustment
       ↓
Hallucination-Constrained Synthesis
```

---

## 🔬 Epistemic & Applicability Scoring Math

El ranking final no es un simple producto del LLM o modelo semántico. Inyecta reglas clínicas en cada cálculo:

| Tipo de Evidencia | Epistemic Multiplier |
|-------------------|----------------------|
| Meta-Analysis / Guideline / RCT | `1.0 + 0.15 boost` |
| Observational / Target Trial | `1.0` |
| Case Report | `0.45` |
| Mechanistic / Pathophysiology | `0.25` |
| Preclinical / Animal model | `0.25` |

El núcleo matemático del reranker sigue esta constante:
```python
final_score = (semantic_score) * applicability_score * epistemic_multiplier * noise_suppression
```

**Applicability Scoring:**
El pipeline penaliza automáticamente evidencia desconectada del perfil clínico del paciente. Por ejemplo:
- Estudios de cohorte obstétrica/embarazo ante un paciente no gestacional (`× 0.7`).
- Papers de pediatría (`adolescents`, `neonatal`) arrastrados accidentalmente para cohortes locales estructuradas sobre pacientes `≥ 65` años (`× 0.5`).

---

## 🔌 Empezar (Quickstart)

Requiere **Python 3.11+**. Soporta APIs compatibles con OpenAI o inferencia local (ej. `llama.cpp` + `Llama-3-8b`).

```bash
# Entorno
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-semantic.txt

# Config (Rellenar LLM_BASE_URL, keys y perfil)
cp .env.example .env

# Arrancar la API local
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
Swagger UI disponible en `http://127.0.0.1:8000/docs`.

---

## ⚖️ Limitaciones y Honestidad Metodológica

- **No es un dispositivo médico**: Proyecto puramente de *research engineering* / portafolio.
- **Heurísticas léxicas**: El scoring de aplicabilidad poblacional está fundamentado actualmente en reglas de matching léxico robustas, pero no constituye una red neuronal clínica supervisada (Clinical NER).
- **No hay validación formal**: Carece de un dataset de *gold standards* (evaluación sistémica) puntuado por un tribunal médico para métricas tipo nDCG.
- **Safety over Creativity**: La síntesis usa guardarraíles fuertes que cortan o bloquean texto, pero no reemplaza, ni por asomo, la inferencia médica humana.

---

## 🔭 Research Directions (Roadmap)

- Clinical NER fundamentado directamente sobre *ontology embeddings*.
- Modelado de peso para inferencia epidemiológica causal (Causal Evidence Weighting).
- Extracción e ingestión estructurada de guías médicas de la ADA/ESC.
- Razonamiento sobre cohortes (Agentic SQL Constrained Execution).
- Modelado longitudinal de la trayectoria del paciente basado en perfiles complejos.
