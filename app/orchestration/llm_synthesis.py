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
   Párrafo titulado exactamente **Síntesis sobre la pregunta clínica**. Resume estrictamente la temática de los PMIDs aportados. Si predominan estudios 'mechanistic' o preclínicos, aclara fuertemente que la evidencia aportada explica mecanismos o vías y NO demuestra eficacia clínica directa para la pregunta. No extrapoles conclusiones. Señala si faltan en el lote estudios de eficacia/RCT centrados en la intervención preguntada. No uses el
   título «Síntesis sobre la comparación solicitada» ni hables de «comparación directa»
   salvo que la pregunta lo pida.

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


def _compact_facts_json(state: Dict[str, Any], medical_answer: Dict[str, Any]) -> str:
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

    facts: dict[str, Any] = {
        "pregunta_usuario": uq,
        "pregunta_pide_comparacion_terapeutica_directa": asks_comparison,
        "predominan_estudios_mecanisticos": bool(cite_rows)
        and all(
            r.get("solo_mecanismos_o_preclinico") for r in cite_rows if isinstance(r, dict)
        ),
        "ruta_pipeline": route,
        "consulta_pubmed_usada": pubmed_q or None,
        "mensaje_cohorte_sql": summary_sql or None,
        "cohorte_texto": medical_answer.get("cohort_summary"),
        "tamano_cohorte": cohort_size,
        # True = 0 pacientes por criterios SQL/BD, no por ausencia de literatura.
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
        "citas_pubmed": cite_rows,
        "calibracion_sintesis": cal_dict,
        "cohorte_local_solo_contexto": route in ("hybrid", "sql"),
    }

    return json.dumps(facts, ensure_ascii=False, indent=2)


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

    facts_json = _compact_facts_json(state, medical_answer)
    uq = (state.get("user_query") or "").strip()
    asks_cmp = question_requests_direct_therapeutic_comparison(uq)
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

    try:
        raw = _openai_chat_completion_text(
            base_url=base,
            api_key=api_key,
            model=model,
            system=_SYSTEM_SYNTHESIS,
            user=user_body,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_s=timeout_s,
        )
    except Exception as exc:  # noqa: BLE001 — síntesis: fallback silencioso
        warns.append(f"synthesis_llm: fallo de red o servidor ({exc})")
        return None, warns

    text = strip_code_fence(raw)
    if not text:
        warns.append("synthesis_llm: respuesta vacía del modelo")
        return None, warns

    text = dedupe_pmid_sections(text)
    text = _inject_tier_provenance_in_pmid_sections(text, facts_json)
    text = _sanitize_no_evidence_claim(text, medical_answer)
    text = _sanitize_extrapolated_efficacy(text, facts_json)
    cal = calibration_from_state(state)
    if cal is not None:
        text = _sanitize_overconfident_synthesis(text, cal)
    if len(text) > 14_000:
        text = text[:13_997].rstrip() + "…"
    return text, warns


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
