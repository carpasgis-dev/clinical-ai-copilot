"""
Síntesis narrativa opcional vía LLM (API OpenAI-compatible: llama.cpp, Ollama, cloud).

Los datos estructurados (cohorte, PMIDs, citas) los construye ``build_stub_medical_answer``;
el LLM redacta un texto legible en español sin inventar números ni identificadores, apoyándose
en extractos de PubMed y en el resto del JSON de hechos (sin bloques de guías incrustados por heurística).
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from app.capabilities.evidence_rag.query_planning.llm_planner import _openai_chat_completion_text
from app.capabilities.evidence_rag.query_planning.llm_postprocess import strip_code_fence
from app.orchestration.evidence_dedup import (
    deduplicate_citations,
    deduplicate_medical_answer_evidence,
)

_SYSTEM_SYNTHESIS = """Eres un médico experto que redacta respuestas clínicas orientativas para profesionales sanitarios.
Recibes un JSON con hechos verificados por el sistema (cohorte SQL, artículos PubMed con extractos).
Tu respuesta DEBE tener SIEMPRE estas dos secciones, en este orden:

## Datos locales (SQL)
Describe el resultado de la consulta SQL sobre la base de datos clínica local:
- Número de pacientes encontrados y criterios aplicados (condiciones, medicación, edad, sexo).
- Si «cohorte_vacia_por_filtros_sql» es true: DEBES explicar que 0 pacientes se debe a criterios
  combinados estrictos y/o tamaño de la BD disponible, y que esto NO significa que no haya
  evidencia científica sobre la pregunta. NUNCA digas "no se encontraron evidencias" si hay
  entradas en «citas_pubmed».
- CRÍTICO: si «num_referencias_pubmed» > 0, está PROHIBIDO escribir frases como
  «no hay evidencia científica», «no existe evidencia», «no se dispone de estudios» o equivalentes.
  Si los estudios recuperados no son los ensayos landmark ideales, di: «Los estudios recuperados
  no incluyen ensayos CVOT de referencia; se recomienda buscar EMPA-REG, LEADER, DECLARE-TIMI 58
  directamente en PubMed.»

## Evidencia bibliográfica (PubMed)
Recorre «citas_pubmed» en el orden del JSON. Por cada PMID distinto, escribe **exactamente un**
subapartado (nunca repitas el mismo PMID; «num_pmids_unicos» indica cuántos deben salir).

Encabezado obligatorio (copia literal de pmid y title del JSON; sin texto extra):

### PMID {pmid} — {titulo}

PROHIBIDO: «(repetido)», «otra vez», numerar «Estudio N» o repetir el mismo PMID dos veces.
Tras el último PMID, escribe solo el párrafo de cierre (A o B); no generes más encabezados ### PMID.

Debajo de cada encabezado, 2-4 frases en español según solo «titulo» y «extracto_del_resumen_indexado»:
- Describe estrictamente de qué trata el artículo sin inventar eficacia. Si 'tipo_estudio_o_evidencia' es 'mechanistic' o 'preclinical', explicita que es investigación de mecanismos biológicos y NO debes afirmar que el fármaco reduce eventos o tiene eficacia clínica.
- IMPORTANTE: Nunca concluyas 'alineación clínica baja', 'alta', 'media' ni adjetivos subjetivos parecidos; el sistema determinista ya proporciona las métricas. Limítate a verbalizar estrictamente lo contenido en 'extracto_del_resumen_indexado'..
- Si la población difiere de la cohorte (p. ej. VIH, FOP, solo ERC), indícalo como limitación.
- No afirmes «reducción significativa», superioridad ni cifras si el extracto no las menciona;
  usa formulaciones prudentes («según el resumen indexado…», «el abstract sugiere…»).
- No mezcles contenido de un estudio con otro (p. ej. no cites survodutide en el bloque de SMARTEST).
- Si «extracto_del_resumen_indexado» está vacío, dilo y comenta solo con «titulo»; no dejes el bloque vacío.

Cierre de la sección PubMed (elige UNO según el JSON):

A) Si «pregunta_pide_comparacion_terapeutica_directa» es **true**:
   Párrafo titulado exactamente **Síntesis sobre la comparación solicitada**. Resume si algún
   PMID compara directamente las opciones de la pregunta (X vs Y). Si ninguno lo hace, dilo
   explícitamente sin inferir superioridad de una opción.

B) Si «pregunta_pide_comparacion_terapeutica_directa» es **false**:
   Párrafo titulado exactamente **Síntesis sobre la pregunta clínica**. Si existe
   «claims_clinicos_resumen», úsalo como eje PRINCIPAL (afirmaciones clínicas agregadas).
   Si no, usa «hallazgos_agregados_terapeutica». Responde por eje clínico (SGLT2, GLP-1, DOAC…),
   NO repitas paper por paper. Si hay «claims_clinicos» con contradicting, menciona matiz explícito.
   Los bloques ### PMID son soporte; el cierre debe sintetizar por terapia/outcome.
   Si predominan estudios mechanistic/preclínicos, dilo. No uses «Síntesis sobre la comparación
   solicitada» salvo que la pregunta pida comparación directa.

CALIBRACIÓN Y TIERS (JSON «calibracion_sintesis» y «citas_pubmed»):
- Si «calibracion_sintesis.retrieval_outcome» es «partial_primary_miss», indica que la búsqueda PubMed
  estricta (PICO) no recuperó suficientes candidatos y que parte de la evidencia viene de etapas ampliadas.
- Si «nivel_epistemico_busqueda» ≥ 4 en una cita, PROHIBIDO afirmar eficacia clínica directa o reducción
  de eventos; usa formulaciones prudentes y menciona «expansión de búsqueda».
- Si «nivel_epistemico_busqueda» es 2 y existe «landmark_conocido», puedes citar el ensayo como referencia
  contextual, sin extrapolar cifras no presentes en el extracto.
- La cohorte local (SQL) es CONTEXTO descriptivo; NO es fuente de evidencia de eficacia terapéutica.
- Si «calibracion_sintesis.retrieval_confidence» < 0.5, evita tono de certeza absoluta en la síntesis final.
- Si «nota_procedencia_recuperacion» está presente en una cita, inclúyela al inicio del bloque ### PMID.

REGLAS ABSOLUTAS:
- Usa ÚNICAMENTE el JSON del mensaje del usuario; no recibes ni debes reproducir otro borrador.
- PROHIBIDO copiar listas tipo «Hallazgos clave», «Referencias PubMed», extractos entre comillas
  largas o secciones «Cohorte local» / «Evidencia» preformateadas del pipeline.
- No inventes PMIDs, cifras ni fármacos que no aparezcan en el JSON; no inventes hallazgos clínicos.
- No traduzcas de forma absurda ni añadas patologías o mecanismos que no figuren en título/extracto.
- No des diagnósticos individuales ni prescripciones; orienta a revisar fuentes y protocolos del centro.
- No uses tablas markdown; párrafos y viñetas simples bajo cada sección ##.
- Salida: solo las dos secciones ## (sin prefacio tipo "Aquí tienes")."""

# Versión corta para modelos locales con ventana pequeña (p. ej. Llama 3.2 3B en llama-server).
_SYSTEM_SYNTHESIS_CLAIM_FIRST = """Eres un médico experto que redacta respuestas clínicas orientativas en español.
Recibes un JSON con cohorte SQL (si aplica) y «claims_clinicos»: afirmaciones clínicas agregadas
con soporte, landmarks y contradicciones explícitas. NO recibes abstracts por paper.

Tu respuesta DEBE tener SIEMPRE estas dos secciones, en este orden:

## Datos locales (SQL)
Igual que en modo estándar: cohorte, filtros, 0 pacientes ≠ ausencia de literatura si hay PMIDs en índice.

## Evidencia bibliográfica (PubMed)
PROHIBIDO escribir bloques «### PMID …» por artículo.
Estructura OBLIGATORIA: un subapartado por cada entrada en «claims_clinicos.claims»:

### Claim: {usa axis_label del JSON}
- Redacta la afirmación clínica en 2-3 frases (solo lo que dice el claim).
- Indica fuerza de evidencia, consistencia y aplicabilidad del JSON.
- Cita landmarks de «landmark_support» y PMIDs de «support» (no inventes otros).
- Si «contradicting» no está vacío, párrafo breve de matiz (no reconcilies en silencio).

Al final, párrafo titulado exactamente **Síntesis sobre la pregunta clínica** (o **Síntesis sobre la
comparación solicitada** si «pregunta_pide_comparacion_terapeutica_directa» es true), integrando
los claims como razonamiento clínico — NO enumeres papers uno a uno.

REGLAS:
- Usa SOLO «claims_clinicos» e «indice_pmids» para trazabilidad; no inventes cifras ni PMIDs.
- «indice_pmids» es solo referencia; no redactes un bloque por cada ítem del índice.
- Respeta «calibracion_sintesis» (cautela si retrieval_confidence baja).
- No prescripciones individuales; orienta a protocolos y fuentes primarias."""

_SYSTEM_SYNTHESIS_COMPACT = """Eres un médico que redacta respuestas orientativas en español.
Usa SOLO el JSON del usuario. Salida: exactamente dos secciones ## en este orden:

## Datos locales (SQL)
Resume cohorte SQL (n, filtros). Si hay PMIDs en el JSON, NO digas que no hay evidencia.

## Evidencia bibliográfica (PubMed)
Por cada cita en «citas_pubmed», un bloque:
### PMID {pmid} — {titulo}
2-4 frases según título y extracto; no inventes cifras ni PMIDs.
Respeta «calibracion_sintesis» y «nivel_epistemico_busqueda» (tier alto = cautela).
Cierra con «Síntesis sobre la pregunta clínica» o «Síntesis sobre la comparación solicitada»
según «pregunta_pide_comparacion_terapeutica_directa». Prioriza «claims_clinicos_resumen»;
si no, «hallazgos_agregados_terapeutica». Cierre por eje clínico, no paper por paper.
Sin tablas ni prefacios."""

# Prompts mínimos para síntesis en ventanas (una llamada ≈ un fragmento).
_SYSTEM_WINDOW_SQL = (
    "Redacta SOLO la sección «## Datos locales (SQL)» en español, usando únicamente el JSON. "
    "Sin sección PubMed. Si no hay cohorte SQL, indícalo en una frase."
)
_SYSTEM_WINDOW_PMID = (
    "Redacta SOLO bloques «### PMID … — título» (uno por cita del JSON), 2-4 frases cada uno. "
    "Sin sección ## ni párrafo de cierre. No inventes cifras."
)
_SYSTEM_WINDOW_CLOSING = (
    "Redacta SOLO un párrafo de cierre en español (sin encabezado ##). "
    "Usa el título exacto indicado en el mensaje del usuario. "
    "Si el JSON incluye claims_clinicos_resumen, sintetiza por afirmaciones clínicas (eje); "
    "si no, hallazgos_agregados_terapeutica. No enumeres papers uno a uno. "
    "Menciona contradicting si existe."
)


def _synthesis_windowed_mode() -> str:
    """
    ``off`` | ``auto`` (default) | ``on``.

    - auto: ventanas solo tras error de contexto en llamada monolítica.
    - on: siempre ventanas (útil en llama.cpp con n_ctx pequeño).
    """
    return (os.getenv("COPILOT_SYNTHESIS_WINDOWED") or "auto").strip().lower()


def _synthesis_use_windowed_proactively() -> bool:
    return _synthesis_windowed_mode() in ("1", "true", "yes", "on", "always")


def _synthesis_pmids_per_window() -> int:
    return max(1, min(4, _env_int("COPILOT_SYNTHESIS_PMIDS_PER_WINDOW", 1)))


def _synthesis_compact_prompt_default() -> bool:
    """Perfil llamacpp o flag explícito → system prompt compacto (menos tokens)."""
    flag = (os.getenv("COPILOT_SYNTHESIS_COMPACT_PROMPT") or "").strip().lower()
    if flag in ("1", "true", "yes", "on"):
        return True
    if flag in ("0", "false", "no", "off"):
        return False
    prof = (os.getenv("COPILOT_LLM_PROFILE") or "").strip().lower()
    return prof in ("llamacpp", "llama_cpp", "ollama", "local")


def synthesis_system_prompt(
    *, compact: bool | None = None, claim_first: bool = False
) -> str:
    if claim_first:
        return _SYSTEM_SYNTHESIS_CLAIM_FIRST
    use_compact = _synthesis_compact_prompt_default() if compact is None else compact
    return _SYSTEM_SYNTHESIS_COMPACT if use_compact else _SYSTEM_SYNTHESIS


def facts_use_claim_first(facts: dict[str, Any]) -> bool:
    if facts.get("sintesis_modo") == "claim_first":
        return True
    cb = facts.get("claims_clinicos")
    return isinstance(cb, dict) and bool(cb.get("claims"))


def _is_context_size_exceeded(exc: BaseException) -> bool:
    blob = str(exc).lower()
    if "context size" in blob or "context length" in blob:
        return True
    resp = getattr(exc, "response", None)
    if resp is not None:
        try:
            blob += " " + (resp.text or "").lower()
        except Exception:
            pass
    return "context size" in blob or "context length" in blob


from app.capabilities.evidence_rag.clinical_knowledge import landmark_synthesis_hint  # noqa: E402
from app.capabilities.evidence_rag.retrieval_tiers import tier_retrieval_provenance_line  # noqa: E402
from app.orchestration.synthesis_calibration import (  # noqa: E402
    SynthesisCalibration,
    calibration_from_state,
    tier_aware_evidence_leadin,
)

def synthesis_uses_llm() -> bool:
    """``True`` si ``COPILOT_SYNTHESIS=llm`` y hay ``LLM_BASE_URL`` + ``LLM_MODEL``."""
    mode = (os.getenv("COPILOT_SYNTHESIS") or "deterministic").strip().lower()
    if mode != "llm":
        return False
    base = (os.getenv("LLM_BASE_URL") or "").strip()
    model = (os.getenv("LLM_MODEL") or "").strip()
    return bool(base and model)


def _env_int(name: str, default: int) -> int:
    try:
        return int((os.getenv(name) or str(default)).strip())
    except ValueError:
        return default


def question_requests_direct_therapeutic_comparison(user_query: str) -> bool:
    """
    True si la pregunta en NL sugiere comparar tratamientos u opciones (A vs B, frente a, etc.).
    """
    u = (user_query or "").strip().lower()
    if len(u) < 12:
        return False
    patterns = (
        r"\bvs\.?\b",
        r"\bversus\b",
        r"\bfrente\s+a\b",
        r"\bcompar",
        r"\belegir\b",
        r"\bpreferencia\b",
        r"\bmejor\b.+\b(que|frente|vs)",
        r"\b(uno|una)\s+frente\b",
        r"\brespecto\s+a\b",
        r"\bentre\b.+\b(y|o)\b.+\b(cual|cuál|mejor)",
    )
    return any(re.search(p, u) for p in patterns)


def _env_float(name: str, default: float) -> float:
    try:
        return float((os.getenv(name) or str(default)).strip())
    except ValueError:
        return default


from app.capabilities.evidence_rag.clinical_intent import ClinicalIntent
from app.capabilities.evidence_rag.epistemic_ranking import infer_epistemic_profile

def _clinical_intent_from_state(intent_raw: Any) -> ClinicalIntent | None:
    if intent_raw is None:
        return None
    if isinstance(intent_raw, ClinicalIntent):
        return intent_raw
    if isinstance(intent_raw, dict):
        return ClinicalIntent.from_dict(intent_raw)
    return None


def _build_facts_dict(state: Dict[str, Any], medical_answer: Dict[str, Any]) -> dict[str, Any]:
    uq = (state.get("user_query") or "").strip()
    if len(uq) > 600:
        uq = uq[:597].rstrip() + "…"
    route = str(state.get("route") or "")
    ma_clean = deduplicate_medical_answer_evidence(dict(medical_answer))
    cites = ma_clean.get("citations") or []
    intent_raw = state.get("clinical_intent")
    clinical_intent = _clinical_intent_from_state(intent_raw)
    pubmed_q = (state.get("pubmed_query") or "").strip()
    pubmed_executed = state.get("pubmed_queries_executed")
    if len(pubmed_q) > 400:
        pubmed_q = pubmed_q[:397].rstrip() + "…"

    by_pmid_snip: dict[str, str] = {}
    eb0 = state.get("evidence_bundle")
    if isinstance(eb0, dict):
        for ar in eb0.get("articles") or []:
            if not isinstance(ar, dict):
                continue
            p = str(ar.get("pmid") or "").strip()
            sn = str(ar.get("abstract_snippet") or "").strip()
            if p and sn:
                by_pmid_snip[p] = sn
    align_by_pmid: dict[str, dict[str, float]] = {}
    eb_align = state.get("evidence_bundle")
    if isinstance(eb_align, dict):
        for ar in eb_align.get("articles") or []:
            if not isinstance(ar, dict):
                continue
            pm_a = str(ar.get("pmid") or "").strip()
            sc_a = ar.get("alignment_scores")
            if pm_a and isinstance(sc_a, dict):
                align_by_pmid[pm_a] = sc_a

    cite_rows: list[dict[str, Any]] = []
    if isinstance(cites, list):
        # Truncamiento brutal de ventana de contexto: Max 4 PMIDs, Abstracts cortos
        for c in deduplicate_citations([x for x in cites[:4] if isinstance(x, dict)]):
            pm = str(c.get("pmid") or "").strip()
            if not pm:
                continue
            title_c = (str(c.get("title") or "").strip())[:150]
            snip_c = (by_pmid_snip.get(pm, "")[:350])
            ep = infer_epistemic_profile(title_c, snip_c, intent=clinical_intent)
            
            # Buscar info de la etapa (incluyendo el tier) en el objeto article completo original (desde evidence_bundle)
            art_obj = {}
            if isinstance(eb_align, dict):
                art_obj = next((ar for ar in (eb_align.get("articles") or []) if isinstance(ar, dict) and str(ar.get("pmid") or "").strip() == pm), {})
            tier_val = art_obj.get("retrieval_tier") or 99
            stage_str = art_obj.get("retrieval_stage") or "primary"
            prov_line = tier_retrieval_provenance_line(int(tier_val), str(stage_str))

            row: dict[str, Any] = {
                "pmid": pm,
                "title": title_c,
                "extracto_del_resumen_indexado": snip_c,
                "tipo_estudio_o_evidencia": ep.evidence_type,
                "solo_mecanismos_o_preclinico": ep.evidence_type in (
                    "mechanistic",
                    "preclinical",
                ),
                "puede_afirmar_eficacia_clinica_directa": ep.evidence_type
                in ("rct", "meta_analysis", "guideline", "target_trial")
                and int(tier_val) <= 3,
                "nivel_epistemico_busqueda": tier_val,
                "fase_recuperacion": stage_str,
                "nota_procedencia_recuperacion": prov_line,
            }
            
            hint = landmark_synthesis_hint(title_c, snip_c)
            if hint:
                row["landmark_conocido"] = hint
                
            if pm in align_by_pmid:
                row["alineacion_clinica"] = align_by_pmid[pm]
            cite_rows.append(row)

    cohort_size = medical_answer.get("cohort_size")
    cohort_zero_no_evidence = (
        cohort_size == 0
        and route in ("hybrid", "sql")
        and bool(cite_rows)
    )

    summary_sql = (str(medical_answer.get("summary") or "").strip())[:480]
    asks_comparison = question_requests_direct_therapeutic_comparison(uq)

    cal = calibration_from_state(state)
    cal_dict: dict[str, Any] | None = cal.to_dict() if cal else None

    agg_block = medical_answer.get("aggregated_findings")
    if not isinstance(agg_block, str) or not agg_block.strip():
        from app.capabilities.evidence_rag.evidence_aggregation import (
            aggregate_therapeutic_findings_from_state,
        )

        agg_block = aggregate_therapeutic_findings_from_state(state)

    claim_bundle = medical_answer.get("clinical_claim_bundle")
    claims_summary = medical_answer.get("clinical_claims_summary")
    if not isinstance(claims_summary, str) or not claims_summary.strip():
        from app.capabilities.evidence_rag.claim_extraction import (
            extract_claims_from_state,
            render_claim_bundle_markdown,
        )

        _b = extract_claims_from_state(state)
        if _b.claims:
            claim_bundle = _b.to_dict()
            claims_summary = render_claim_bundle_markdown(_b)

    claim_first = bool(
        isinstance(claim_bundle, dict) and (claim_bundle or {}).get("claims")
    )
    indice_pmids = [
        {"pmid": str(r.get("pmid") or ""), "titulo": str(r.get("titulo") or "")[:200]}
        for r in cite_rows
        if r.get("pmid")
    ]

    facts: dict[str, Any] = {
        "pregunta_usuario": uq,
        "pregunta_pide_comparacion_terapeutica_directa": asks_comparison,
        "sintesis_modo": "claim_first" if claim_first else "paper_centric",
        "predominan_estudios_mecanisticos": bool(cite_rows)
        and all(
            r.get("solo_mecanismos_o_preclinico") for r in cite_rows if isinstance(r, dict)
        ),
        "ruta_pipeline": route,
        "consulta_pubmed_usada": pubmed_q or None,
        "mensaje_cohorte_sql": summary_sql or None,
        "cohorte_texto": medical_answer.get("cohort_summary"),
        "tamano_cohorte": cohort_size,
        "cohorte_vacia_por_filtros_sql": cohort_zero_no_evidence,
        "num_referencias_pubmed": len(cite_rows),
        "num_pmids_unicos": len(cite_rows),
        "incertidumbre": [
            str(x).strip()
            for x in (medical_answer.get("uncertainty_notes") or [])[:8]
            if str(x).strip()
        ],
        "aplicabilidad": [
            str(x).strip()
            for x in (medical_answer.get("applicability_notes") or [])[:8]
            if str(x).strip()
        ],
        "calibracion_sintesis": cal_dict,
        "cohorte_local_solo_contexto": route in ("hybrid", "sql"),
        "hallazgos_agregados_terapeutica": (agg_block or None),
        "claims_clinicos": claim_bundle,
        "claims_clinicos_resumen": (claims_summary or None),
        "indice_pmids": indice_pmids,
    }
    if claim_first:
        facts["citas_pubmed"] = []
        facts["instruccion_sintesis"] = (
            "Modo claim-first: redacta por afirmaciones clínicas en claims_clinicos; "
            "no uses bloques ### PMID por paper."
        )
    else:
        facts["citas_pubmed"] = cite_rows

    return facts


def _compact_facts_json(state: Dict[str, Any], medical_answer: Dict[str, Any]) -> str:
    return json.dumps(_build_facts_dict(state, medical_answer), ensure_ascii=False, indent=2)


def _llm_synthesis_chat(
    *,
    base_url: str,
    api_key: str | None,
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    temperature: float,
    timeout_s: float,
) -> str:
    return _openai_chat_completion_text(
        base_url=base_url,
        api_key=api_key,
        model=model,
        system=system,
        user=user,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout_s=timeout_s,
    )


def _extract_markdown_section(text: str, heading: str) -> str:
    """Fragmento desde ``heading`` hasta el siguiente ``## `` o fin."""
    body = strip_code_fence(text).strip()
    if heading not in body:
        return body
    start = body.find(heading)
    chunk = body[start:]
    nxt = re.search(r"\n##(?!\#)", chunk[len(heading) :])
    if nxt:
        chunk = chunk[: len(heading) + nxt.start()]
    return chunk.strip()


def _extract_pmid_blocks_only(text: str) -> str:
    """Conserva solo subapartados ``### PMID …``."""
    body = strip_code_fence(text).strip()
    parts = re.split(r"(?=^### PMID \d+\b)", body, flags=re.MULTILINE)
    blocks = [p.strip() for p in parts if p.strip().startswith("### PMID")]
    return "\n\n".join(blocks)


def _closing_title(asks_comparison: bool) -> str:
    if asks_comparison:
        return "Síntesis sobre la comparación solicitada"
    return "Síntesis sobre la pregunta clínica"


def _synthesis_narrative_claim_first(
    facts: dict[str, Any],
    *,
    base_url: str,
    api_key: str | None,
    model: str,
    max_tokens: int,
    temperature: float,
    timeout_s: float,
) -> tuple[str | None, list[str]]:
    """Una sola llamada: SQL + claims (sin ventanas PMID)."""
    warns: list[str] = ["synthesis_llm: modo claim-first (sin abstracts por paper)"]
    uq = str(facts.get("pregunta_usuario") or "")
    asks_cmp = bool(facts.get("pregunta_pide_comparacion_terapeutica_directa"))
    cierre = (
        "Cierra con «Síntesis sobre la comparación solicitada»."
        if asks_cmp
        else "Cierra con «Síntesis sobre la pregunta clínica»."
    )
    payload = {
        k: facts.get(k)
        for k in (
            "pregunta_usuario",
            "pregunta_pide_comparacion_terapeutica_directa",
            "sintesis_modo",
            "claims_clinicos",
            "claims_clinicos_resumen",
            "indice_pmids",
            "calibracion_sintesis",
            "mensaje_cohorte_sql",
            "cohorte_texto",
            "tamano_cohorte",
            "cohorte_vacia_por_filtros_sql",
            "num_referencias_pubmed",
            "instruccion_sintesis",
        )
    }
    user_body = (
        "Redacta la respuesta con exactamente dos secciones ##:\n"
        "## Datos locales (SQL)\n"
        "## Evidencia bibliográfica (PubMed)\n\n"
        "Usa SOLO claims_clinicos (un ### Claim por entrada). "
        f"PROHIBIDO ### PMID por paper. {cierre}\n\n"
        "HECHOS_JSON:\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )
    try:
        raw = _llm_synthesis_chat(
            base_url=base_url,
            api_key=api_key,
            model=model,
            system=synthesis_system_prompt(claim_first=True),
            user=user_body,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_s=timeout_s,
        )
        return strip_code_fence(raw).strip() or None, warns
    except Exception as exc:  # noqa: BLE001
        warns.append(f"synthesis_llm claim-first: {exc}")
        summary = facts.get("claims_clinicos_resumen")
        if isinstance(summary, str) and summary.strip():
            sql = (facts.get("mensaje_cohorte_sql") or facts.get("cohorte_texto") or "").strip()
            sql_block = "## Datos locales (SQL)\n\n" + (
                sql or "Cohorte local no disponible."
            )
            return f"{sql_block}\n\n{summary.strip()}", warns
        return None, warns


def _synthesis_narrative_windowed(
    facts: dict[str, Any],
    *,
    base_url: str,
    api_key: str | None,
    model: str,
    max_tokens: int,
    temperature: float,
    timeout_s: float,
) -> tuple[str | None, list[str]]:
    """
    Síntesis en varias llamadas pequeñas: SQL → PMIDs (ventanas) → párrafo de cierre.
    """
    if facts_use_claim_first(facts):
        return _synthesis_narrative_claim_first(
            facts,
            base_url=base_url,
            api_key=api_key,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_s=timeout_s,
        )

    warns: list[str] = [
        "synthesis_llm: modo ventanas (varias llamadas) por límite de contexto del servidor"
    ]
    route = str(facts.get("ruta_pipeline") or "")
    asks_cmp = bool(facts.get("pregunta_pide_comparacion_terapeutica_directa"))
    cites: list[dict[str, Any]] = [
        c for c in (facts.get("citas_pubmed") or []) if isinstance(c, dict)
    ]
    sub_max = max(128, min(768, max_tokens // 2))

    sql_section = ""
    if route in ("hybrid", "sql"):
        sql_payload = {
            k: facts.get(k)
            for k in (
                "pregunta_usuario",
                "mensaje_cohorte_sql",
                "cohorte_texto",
                "tamano_cohorte",
                "cohorte_vacia_por_filtros_sql",
                "cohorte_local_solo_contexto",
                "num_referencias_pubmed",
            )
        }
        try:
            raw_sql = _llm_synthesis_chat(
                base_url=base_url,
                api_key=api_key,
                model=model,
                system=_SYSTEM_WINDOW_SQL,
                user="JSON:\n" + json.dumps(sql_payload, ensure_ascii=False),
                max_tokens=sub_max,
                temperature=temperature,
                timeout_s=timeout_s,
            )
            sql_section = _extract_markdown_section(raw_sql, "## Datos locales (SQL)")
        except Exception as exc:  # noqa: BLE001
            warns.append(f"synthesis_llm ventana SQL: {exc}")
            summary = (facts.get("mensaje_cohorte_sql") or facts.get("cohorte_texto") or "").strip()
            sql_section = "## Datos locales (SQL)\n\n" + (
                summary or "Cohorte local no disponible en esta respuesta."
            )
    else:
        sql_section = (
            "## Datos locales (SQL)\n\n"
            "No aplica: la consulta se resolvió solo con evidencia bibliográfica (PubMed)."
        )

    pmid_parts: list[str] = []
    per_win = _synthesis_pmids_per_window()
    if not cites:
        pmid_parts.append(
            "_No se recuperaron referencias PubMed indexadas para esta pregunta._"
        )
    else:
        for batch in _chunk_list(cites, per_win):
            mini = {
                "pregunta_usuario": facts.get("pregunta_usuario"),
                "calibracion_sintesis": facts.get("calibracion_sintesis"),
                "citas_pubmed": batch,
            }
            try:
                raw_p = _llm_synthesis_chat(
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    system=_SYSTEM_WINDOW_PMID,
                    user="JSON:\n" + json.dumps(mini, ensure_ascii=False),
                    max_tokens=sub_max,
                    temperature=temperature,
                    timeout_s=timeout_s,
                )
                block = _extract_pmid_blocks_only(raw_p)
                if block:
                    pmid_parts.append(block)
            except Exception as exc:  # noqa: BLE001
                warns.append(f"synthesis_llm ventana PMID: {exc}")
                for row in batch:
                    pm = str(row.get("pmid") or "").strip()
                    title = str(row.get("title") or "").strip()
                    snip = str(row.get("extracto_del_resumen_indexado") or "").strip()[:280]
                    if pm:
                        pmid_parts.append(
                            f"### PMID {pm} — {title}\n"
                            f"{snip or 'Extracto no disponible; revisar PubMed.'}"
                        )

    closing_title = _closing_title(asks_cmp)
    study_lines = [
        f"- PMID {c.get('pmid')}: {c.get('title')}"
        for c in cites
        if c.get("pmid")
    ]
    closing_payload = {
        "pregunta_usuario": facts.get("pregunta_usuario"),
        "calibracion_sintesis": facts.get("calibracion_sintesis"),
        "pregunta_pide_comparacion_terapeutica_directa": asks_cmp,
        "hallazgos_agregados_terapeutica": facts.get("hallazgos_agregados_terapeutica"),
        "claims_clinicos_resumen": facts.get("claims_clinicos_resumen"),
        "claims_clinicos": facts.get("claims_clinicos"),
        "estudios_ya_redactados": study_lines[:8],
    }
    closing_user = (
        f"Escribe el párrafo titulado exactamente «{closing_title}».\n\n"
        "JSON:\n"
        + json.dumps(closing_payload, ensure_ascii=False)
    )
    closing_text = ""
    try:
        raw_c = _llm_synthesis_chat(
            base_url=base_url,
            api_key=api_key,
            model=model,
            system=_SYSTEM_WINDOW_CLOSING,
            user=closing_user,
            max_tokens=sub_max,
            temperature=temperature,
            timeout_s=timeout_s,
        )
        closing_body = strip_code_fence(raw_c).strip()
        if closing_title in closing_body:
            idx = closing_body.find(closing_title)
            closing_text = closing_body[idx:].strip()
        else:
            closing_text = f"**{closing_title}**\n\n{closing_body}"
    except Exception as exc:  # noqa: BLE001
        warns.append(f"synthesis_llm ventana cierre: {exc}")
        closing_text = (
            f"**{closing_title}**\n\n"
            "Revise los PMIDs citados y la calibración de recuperación antes de aplicar "
            "conclusiones clínicas."
        )

    pubmed_body = "\n\n".join(p for p in pmid_parts if p.strip())
    assembled = (
        f"{sql_section.rstrip()}\n\n"
        f"## Evidencia bibliográfica (PubMed)\n\n"
        f"{pubmed_body.rstrip()}\n\n"
        f"{closing_text.rstrip()}"
    ).strip()
    return assembled if assembled else None, warns


def _chunk_list(items: list[Any], size: int) -> list[list[Any]]:
    n = max(1, size)
    return [items[i : i + n] for i in range(0, len(items), n)]


_PMID_SECTION_HEAD = re.compile(r"^### PMID (\d+)\b", re.MULTILINE)


def dedupe_pmid_sections(text: str) -> str:
    """Conserva la primera aparición de cada ### PMID … (cualquier sufijo tras el número)."""
    if not text.strip():
        return text
    parts = re.split(r"(?=^### PMID \d+\b)", text, flags=re.MULTILINE)
    if len(parts) <= 1:
        return text
    kept: list[str] = []
    seen: set[str] = set()
    closing_text = ""
    for part in parts:
        m = _PMID_SECTION_HEAD.match(part)
        if not m:
            kept.append(part)
            continue
        pmid = m.group(1)
        
        # Extract closing text if it falls in a duplicate section
        for marker in (
            "**Síntesis sobre la comparación solicitada**",
            "**Síntesis sobre la pregunta clínica**",
            "Síntesis sobre la comparación solicitada",
            "Síntesis sobre la pregunta clínica",
        ):
            idx = part.find(marker)
            if idx != -1:
                closing_text = part[idx:]
                part = part[:idx]
                break
                
        if pmid in seen:
            continue
        seen.add(pmid)
        kept.append(part)
        
    kept.append("\n\n" + closing_text)
    out = "".join(kept)
    # Conservar párrafo de cierre si aparece tras repeticiones espurias del modelo.
    for marker in (
        "**Síntesis sobre la comparación solicitada**",
        "**Síntesis sobre la pregunta clínica**",
        "Síntesis sobre la comparación solicitada",
        "Síntesis sobre la pregunta clínica",
    ):
        idx = out.find(marker)
        if idx != -1:
            return out[:idx].rstrip() + "\n\n" + out[idx:].lstrip()
    return out


def try_llm_synthesis_narrative(
    state: Dict[str, Any],
    medical_answer: Dict[str, Any],
) -> Tuple[str | None, List[str]]:
    """
    Devuelve ``(texto_narrativo_o_None, warnings)``.

    Si el LLM falla o devuelve vacío, ``None`` y el caller debe usar el render determinista.
    """
    warns: list[str] = []
    if not synthesis_uses_llm():
        return None, warns

    base = (os.getenv("LLM_BASE_URL") or "").strip().rstrip("/")
    model = (os.getenv("LLM_MODEL") or "").strip()
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip() or None

    max_tokens = max(128, min(4096, _env_int("COPILOT_SYNTHESIS_MAX_TOKENS", 1536)))
    # Por defecto 90 s: modelos locales (llama.cpp) suelen tardar más que 35 s en el primer token.
    timeout_s = max(5.0, min(600.0, _env_float("COPILOT_SYNTHESIS_TIMEOUT", 120.0)))
    temperature = 0.0

    facts = _build_facts_dict(state, medical_answer)
    facts_json = json.dumps(facts, ensure_ascii=False, indent=2)
    uq = (state.get("user_query") or "").strip()
    asks_cmp = question_requests_direct_therapeutic_comparison(uq)
    claim_first = facts_use_claim_first(facts)

    if claim_first:
        cierre = (
            "Cierra con «Síntesis sobre la comparación solicitada»."
            if asks_cmp
            else "Cierra con «Síntesis sobre la pregunta clínica»."
        )
        user_body = (
            "Redacta la respuesta con exactamente dos secciones markdown:\n"
            "## Datos locales (SQL)\n"
            "## Evidencia bibliográfica (PubMed)\n\n"
            "Modo claim-first: un ### Claim por cada entrada en claims_clinicos.claims. "
            f"PROHIBIDO bloques ### PMID por paper. {cierre}\n\n"
            "HECHOS_JSON:\n"
            f"{facts_json}"
        )
    else:
        cierre = (
            "Cierra PubMed con el párrafo «Síntesis sobre la comparación solicitada» "
            "(comparación directa entre opciones de la pregunta)."
            if asks_cmp
            else "Cierra PubMed con el párrafo «Síntesis sobre la pregunta clínica» "
            "(no uses el título de comparación directa)."
        )
        user_body = (
            "Redacta la respuesta con exactamente dos secciones markdown:\n"
            "## Datos locales (SQL)\n"
            "## Evidencia bibliográfica (PubMed)\n\n"
            "Usa SOLO el JSON. Escribe exactamente «num_pmids_unicos» bloques ### PMID (uno por "
            f"cita), sin repetir ningún PMID ni añadir bloques extra. {cierre} "
            "Parafrasea; no copies extractos largos.\n\n"
            "HECHOS_JSON:\n"
            f"{facts_json}"
        )

    chat_kw = dict(
        base_url=base,
        api_key=api_key,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout_s=timeout_s,
    )

    def _finalize(text: str) -> str:
        text = dedupe_pmid_sections(text)
        text = _inject_tier_provenance_in_pmid_sections(text, facts_json)
        text = _sanitize_prompt_leakage(text)
        text = _sanitize_no_evidence_claim(text, medical_answer)
        text = _sanitize_extrapolated_efficacy(text, facts_json)
        cal = calibration_from_state(state)
        if cal is not None:
            text = _sanitize_overconfident_synthesis(text, cal)
        if len(text) > 14_000:
            text = text[:13_997].rstrip() + "…"
        return text

    if _synthesis_use_windowed_proactively():
        text_w, w_win = _synthesis_narrative_windowed(facts, **chat_kw)
        if text_w:
            warns.extend(w_win)
            return _finalize(text_w), warns
        warns.extend(w_win)

    system = synthesis_system_prompt(claim_first=claim_first)
    raw: str | None = None
    try:
        raw = _llm_synthesis_chat(system=system, user=user_body, **chat_kw)
    except Exception as exc:  # noqa: BLE001 — síntesis: fallback silencioso
        if system != _SYSTEM_SYNTHESIS_COMPACT and _is_context_size_exceeded(exc):
            try:
                raw = _llm_synthesis_chat(
                    system=_SYSTEM_SYNTHESIS_COMPACT,
                    user=user_body,
                    **chat_kw,
                )
                warns.append(
                    "synthesis_llm: prompt compacto por límite de contexto del servidor local"
                )
            except Exception as exc2:  # noqa: BLE001
                exc = exc2
                raw = None
        if raw is None and _is_context_size_exceeded(exc) and _synthesis_windowed_mode() != "off":
            text_w, w_win = _synthesis_narrative_windowed(facts, **chat_kw)
            if text_w:
                warns.extend(w_win)
                return _finalize(text_w), warns
            warns.extend(w_win)
            warns.append(f"synthesis_llm: ventanas fallaron tras contexto excedido ({exc})")
            return None, warns
        if raw is None:
            warns.append(f"synthesis_llm: fallo de red o servidor ({exc})")
            return None, warns

    text = strip_code_fence(raw or "")
    if not text:
        warns.append("synthesis_llm: respuesta vacía del modelo")
        return None, warns

    return _finalize(text), warns


_NO_EVIDENCE_PATTERNS = re.compile(
    r"(no\s+hay\s+evidencia\s+cient[ií]fica\s+sobre\s+la\s+pregunta"
    r"|no\s+se\s+encontr[oó]\s+evidencia\s+cient[ií]fica"
    r"|no\s+existe\s+evidencia\s+cient[ií]fica"
    r"|la\s+evidencia\s+cient[ií]fica\s+no\s+existe"
    r"|no\s+hay\s+estudios\s+disponibles\s+sobre\s+este\s+tema"
    r"|no\s+se\s+dispone\s+de\s+evidencia)",
    re.I,
)
_NO_EVIDENCE_REPLACEMENT = (
    "Los estudios recuperados en esta búsqueda no incluyen los ensayos CVOT de referencia "
    "(p. ej. EMPA-REG, LEADER, DECLARE-TIMI 58); se recomienda revisar directamente "
    "en PubMed usando los PMIDs citados y buscar ensayos landmark por nombre de molécula."
)


_EFFICACY_EXTRAPOLATION = re.compile(
    r"(pueden?\s+reducir|reduce[n]?\s+(el\s+)?riesgo|disminuyen?\s+los\s+eventos|"
    r"disminuye\s+el\s+riesgo|eficacia\s+en\s+reducir)"
    r"[^.]{0,80}(cardiovascul|MACE|eventos\s+cardiovasculares)",
    re.I,
)


def _sanitize_extrapolated_efficacy(text: str, facts_json: str) -> str:
    """
    Si el lote es mayoritariamente mechanistic/preclinical, atenúa afirmaciones de eficacia CV
    no ancladas en los extractos.
    """
    try:
        facts = json.loads(facts_json)
    except json.JSONDecodeError:
        return text
    cites = facts.get("citas_pubmed") or []
    if not isinstance(cites, list) or not cites:
        return text
    mech_n = sum(
        1 for c in cites if isinstance(c, dict) and c.get("solo_mecanismos_o_preclinico")
    )
    if mech_n < len(cites) or mech_n == 0:
        return text
    replacement = (
        "Los resúmenes indexados recuperados describen sobre todo mecanismos o vías "
        "(p. ej. fibrosis, patofisiología) y no reportan por sí mismos reducción de eventos "
        "cardiovasculares clínicos; no debe inferirse eficacia terapéutica directa desde estos PMIDs."
    )

    def _repl(m: re.Match[str]) -> str:
        return replacement

    return _EFFICACY_EXTRAPOLATION.sub(_repl, text, count=3)


_OVERCONFIDENT_PATTERNS = re.compile(
    r"(demuestra|demostró|reduce\s+de\s+forma\s+significativa|"
    r"superioridad\s+clara|eficacia\s+comprobada|evidencia\s+contundente)",
    re.I,
)


_PROMPT_LEAK_MARKERS = (
    re.compile(r"Resume cohorte SQL.*NO diga que no hay evidencia", re.I | re.S),
    re.compile(r"Usa SOLO el JSON del usuario", re.I),
    re.compile(r"Redacta SOLO la sección", re.I),
    re.compile(r"Redacta SOLO bloques", re.I),
    re.compile(r"HECHOS_JSON\s*:", re.I),
    re.compile(r"exactamente dos secciones\s*##", re.I),
    re.compile(r"PROHIBIDO copiar listas", re.I),
    re.compile(r"Sin sección PubMed", re.I),
)


def _sanitize_prompt_leakage(text: str) -> str:
    """Elimina fragmentos del system prompt que el LLM local a veces copia."""
    if not text or not text.strip():
        return text
    out = text
    for pat in _PROMPT_LEAK_MARKERS:
        out = pat.sub("", out)
    # Líneas sueltas con instrucciones internas bajo ## Datos locales
    cleaned: list[str] = []
    for line in out.splitlines():
        low = line.lower().strip()
        if "resume cohorte sql" in low and "no diga" in low:
            continue
        if low.startswith("usa solo el json"):
            continue
        cleaned.append(line)
    out = "\n".join(cleaned)
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    if "## Datos locales (SQL)" in out:
        parts = out.split("## Datos locales (SQL)", 1)
        if len(parts) == 2:
            body = parts[1].split("##", 1)[0].strip()
            if not body or len(body) < 12:
                out = (
                    out.replace(
                        "## Datos locales (SQL)",
                        "## Datos locales (SQL)\n\n"
                        "Resumen de cohorte local no disponible en esta respuesta.",
                        1,
                    )
                )
    return out


def _sanitize_overconfident_synthesis(text: str, cal: SynthesisCalibration) -> str:
    """Atenúa lenguaje de certeza cuando la calibración indica recuperación débil o tier alto."""
    if cal.retrieval_confidence >= 0.5 and cal.dominant_retrieval_tier < 3:
        return text
    caveat = (
        " (Nota: la recuperación bibliográfica fue parcial o de baja especificidad; "
        "formulación prudente recomendada.)"
    )

    def _repl(m: re.Match[str]) -> str:
        return m.group(0) + caveat

    return _OVERCONFIDENT_PATTERNS.sub(_repl, text, count=2)


def _inject_tier_provenance_in_pmid_sections(text: str, facts_json: str) -> str:
    """Inserta nota de procedencia bajo cada ### PMID si tier > 1 y el modelo no la puso."""
    try:
        facts = json.loads(facts_json)
    except json.JSONDecodeError:
        return text
    cites = facts.get("citas_pubmed") or []
    if not isinstance(cites, list):
        return text
    out = text
    for c in cites:
        if not isinstance(c, dict):
            continue
        pm = str(c.get("pmid") or "").strip()
        note = c.get("nota_procedencia_recuperacion")
        if not pm or not isinstance(note, str) or not note.strip():
            continue
        if note in out:
            continue
        pat = re.compile(
            rf"(^### PMID {re.escape(pm)}\b[^\n]*\n)(?!\*Recuperado)",
            re.MULTILINE,
        )
        out = pat.sub(rf"\1{note.strip()}\n\n", out, count=1)
    return out


def apply_tier_aware_evidence_summary(
    medical_answer: Dict[str, Any],
    state: Dict[str, Any],
) -> Dict[str, Any]:
    """Prefija evidence_summary con párrafo determinista si la calibración lo exige."""
    cal = calibration_from_state(state)
    if cal is None:
        return medical_answer
    lead = tier_aware_evidence_leadin(cal)
    if not lead:
        return medical_answer
    out = dict(medical_answer)
    es = (out.get("evidence_summary") or "").strip()
    if lead not in es:
        out["evidence_summary"] = f"{lead}\n\n{es}".strip() if es else lead
    un = list(out.get("uncertainty_notes") or [])
    if not any("Calibración" in str(x) for x in un):
        un.append(
            f"Calibración: confianza recuperación={cal.retrieval_confidence:.2f}, "
            f"tier dominante={cal.dominant_retrieval_tier}, outcome={cal.retrieval_outcome}."
        )
    out["uncertainty_notes"] = un
    return out


def _sanitize_no_evidence_claim(text: str, medical_answer: Dict[str, Any]) -> str:
    """
    Si el LLM genera frases tipo «no hay evidencia científica sobre la pregunta»
    cuando SÍ hay PMIDs recuperados, sustituye por un aviso de retrieval limitado.

    Es seguro en medical AI: nunca decir que evidencia no existe cuando la pipeline
    sí devolvió citas — el problema es el retrieval, no la ausencia de literatura.
    """
    pmids = medical_answer.get("citations") or []
    if not pmids:
        return text
    return _NO_EVIDENCE_PATTERNS.sub(_NO_EVIDENCE_REPLACEMENT, text)


def medical_answer_after_llm_synthesis(medical_answer: Dict[str, Any]) -> Dict[str, Any]:
    """Ajusta limitaciones: sustituye la línea 'solo determinista' por aviso de LLM narrativo."""
    out = dict(medical_answer)
    lim = [str(x).strip() for x in (out.get("limitations") or []) if str(x).strip()]
    det = (
        "Síntesis generada con reglas deterministas (no LLM clínico); no sustituye juicio clínico."
    )
    lim = [x for x in lim if x != det]
    llm_line = (
        "Texto narrativo principal generado con LLM (API compatible con OpenAI, p. ej. llama.cpp); "
        "cifras de cohorte, SQL y PMIDs provienen de la pipeline determinista. No sustituye juicio clínico."
    )
    if llm_line not in lim:
        lim.insert(0, llm_line)
    out["limitations"] = lim
    return out
