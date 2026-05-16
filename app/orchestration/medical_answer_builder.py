"""
Fase 3.4 — ``MedicalAnswer`` como fuente de verdad; ``final_answer`` = render legado.

Síntesis determinista (sin LLM): estable, testeable; narrativa opcional vía ``llm_synthesis`` si ``COPILOT_SYNTHESIS=llm``.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from app.api.schemas import CitationOut, EvidenceStrength
from app.orchestration.evidence_dedup import (
    deduplicate_evidence_bundle_dict,
    deduplicate_medical_answer_evidence,
    deduplicate_pmids,
    state_with_deduped_evidence,
)
from app.capabilities.evidence_rag.evidence_rerank import (
    infer_applicability_line,
    infer_study_type_from_title,
    is_weak_for_clinical_synthesis,
    lexical_relevance_score,
    MIN_HEADLINE_LEXICAL_RELEVANCE,
)
from app.schemas.copilot_state import ClinicalContext, Route


def sql_cohort_size(sql_result: object) -> int | None:
    """Extrae ``cohort_size`` del primer resultado de un conteo SQL. Compartido por reasoning y nodes."""
    if not isinstance(sql_result, dict):
        return None
    rows = sql_result.get("rows") or []
    if rows and isinstance(rows[0], dict):
        raw = rows[0].get("cohort_size")
        if raw is not None:
            try:
                return int(raw)
            except (TypeError, ValueError):
                pass
    return None


def _clinical_ctx(state: Dict[str, Any]) -> ClinicalContext | None:
    raw = state.get("clinical_context")
    if not raw:
        return None
    if isinstance(raw, ClinicalContext):
        return raw
    try:
        return ClinicalContext.model_validate(raw)
    except Exception:
        return None


def _route(state: Dict[str, Any]) -> Route:
    r = state.get("route")
    if isinstance(r, Route):
        return r
    try:
        return Route(str(r))
    except Exception:
        return Route.UNKNOWN


def _cohort_filter_description(ctx: ClinicalContext | None, cohort_size: int | None) -> Optional[str]:
    if ctx is None:
        if cohort_size is not None:
            return "Cohorte local acotada a la consulta (criterios desde NL estructurado → SQL)."
        return None
    bits: list[str] = []
    if ctx.population_conditions:
        bits.append("condiciones: " + ", ".join(ctx.population_conditions))
    if ctx.population_medications:
        bits.append("medicación: " + ", ".join(ctx.population_medications))
    if ctx.population_age_min is not None:
        bits.append(f"edad ≥ {ctx.population_age_min} años")
    if ctx.population_age_max is not None:
        bits.append(f"edad < {ctx.population_age_max} años")
    if ctx.population_sex:
        bits.append(f"sexo {ctx.population_sex}")
    if not bits:
        if cohort_size is not None:
            return "Cohorte local acotada a la consulta (criterios desde NL estructurado → SQL)."
        return None
    head = "Cohorte local filtrada según la consulta"
    head += " (" + "; ".join(bits) + ")."
    return head


def _cohort_summary_sql_only(sql_r: object, n: int | None) -> Optional[str]:
    if not sql_r:
        return None
    if n is not None:
        return (
            f"Conteo de cohorte en datos locales: {n} pacientes "
            "(ver traza para la sentencia SQL ejecutada)."
        )
    return "Se ejecutó consulta SQL sobre datos locales (ver traza para detalle)."


def _infer_evidence_strength(title: str) -> EvidenceStrength | None:
    st = infer_study_type_from_title(title)
    if st in ("meta-analysis", "network-meta-analysis", "systematic-review", "umbrella-review", "guideline"):
        return EvidenceStrength.HIGH
    if st in ("rct", "clinical-trial", "cohort", "epidemiology"):
        return EvidenceStrength.MODERATE
    if st in ("review", "cross-sectional", "pilot-trial"):
        return EvidenceStrength.LOW
    if st in ("case-report", "case-series", "editorial", "preclinical", "basic-research"):
        return EvidenceStrength.LOW
    return None


def _build_synthesis_orientation(
    state: Dict[str, Any], arts: list[dict[str, Any]], user_query: str
) -> str | None:
    """
    Párrafo de conclusión orientativa: prioriza cabecera de la lista y advierte sobre diseños débiles.
    El marco de guías específicas queda fuera: la síntesis clínica rica debe venir del LLM (si está habilitado)
    y del contenido de los abstracts recuperados en PubMed, no de reglas deterministas prefijadas aquí.
    """
    if not arts:
        return None
    route = _route(state)
    if route not in (Route.HYBRID, Route.EVIDENCE):
        return None
    ctx = _clinical_ctx(state)

    def _headline_arts(a_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
        amin = ctx.population_age_min if ctx else None
        conds = list(ctx.population_conditions) if ctx and ctx.population_conditions else None
        meds = list(ctx.population_medications) if ctx and ctx.population_medications else None
        good: list[dict[str, Any]] = []
        bad: list[dict[str, Any]] = []
        for a in a_list:
            tit = str(a.get("title") or "").strip()
            line = infer_applicability_line(
                tit,
                population_age_min=amin,
                population_conditions=conds,
                population_medications=meds,
                user_query=user_query,
                abstract_snippet=str(a.get("abstract_snippet") or ""),
            )
            if line and line.startswith("Limitada"):
                bad.append(a)
            else:
                good.append(a)

        def _rank_score(a: dict[str, Any]) -> float:
            align = a.get("alignment_scores")
            if isinstance(align, dict) and align:
                pop = float(align.get("population_score") or 0)
                iv = float(align.get("intervention_score") or 0)
                out = float(align.get("outcome_score") or 0)
                return 0.4 * pop + 0.35 * iv + 0.25 * out
            return lexical_relevance_score(
                user_query,
                str(a.get("title") or ""),
                str(a.get("abstract_snippet") or ""),
            )

        good_ranked = [a for a in good if _rank_score(a) >= MIN_HEADLINE_LEXICAL_RELEVANCE]
        good_rest = [a for a in good if _rank_score(a) < MIN_HEADLINE_LEXICAL_RELEVANCE]
        good_sorted = sorted(good_ranked, key=_rank_score, reverse=True) + good_rest
        merged = good_sorted + bad
        return merged[:2] if merged else a_list[:2]

    parts: list[str] = []
    labels: list[str] = []
    for a in _headline_arts(arts):
        tit = str(a.get("title") or "").strip()
        pmid = str(a.get("pmid") or "").strip()
        if not pmid:
            continue
        st = infer_study_type_from_title(tit)
        short = tit if len(tit) <= 120 else tit[:117].rstrip() + "…"
        lab = f"PMID {pmid}" + (f" [{st}]" if st else "")
        labels.append(f"{lab}: «{short}»")
    if labels:
        parts.append(
            "Conclusión orientativa (priorización automática; no sustituye leer el abstract): "
            "las referencias en cabeza combinan mayor alineación con la pregunta y, heurísticamente, "
            "diseños más informativos para síntesis (revisiones, meta-análisis, ECA). "
            "Referencias destacadas: " + " · ".join(labels) + "."
        )
    weak_n = sum(
        1
        for a in arts
        if is_weak_for_clinical_synthesis(
            infer_study_type_from_title(str(a.get("title") or "")),
            user_query,
        )
    )
    if weak_n:
        parts.append(
            f"Incluye {weak_n} trabajo(s) con menor peso para inferencias sobre tratamiento "
            "(p. ej. caso clínico, editorial o preclínico); permanecen por trazabilidad."
        )
    return "\n\n".join(parts) if parts else None


def _snippet_usable(snip: str) -> bool:
    s = (snip or "").strip()
    if len(s) < 12:
        return False
    low = s.lower()
    if "(sin abstract" in low:
        return False
    return True


def _evidence_statement_from_article(a: dict[str, Any]) -> str:
    """
    Texto para ``evidence_statements`` sin LLM: prioriza un recorte del abstract
    indexado; si no hay snippet útil, deja claro que solo se expone el título.
    """
    pmid = str(a.get("pmid") or "").strip()
    title = str(a.get("title") or "").strip()
    snip = re.sub(r"\s+", " ", str(a.get("abstract_snippet") or "").strip())
    tshort = (title[:180] + "…") if len(title) > 180 else (title or "(sin título)")
    if _snippet_usable(snip):
        cap = 340
        frag = snip if len(snip) <= cap else snip[: cap - 1].rsplit(" ", 1)[0] + "…"
        return (
            f"PMID {pmid} — extracto orientativo del resumen indexado en PubMed "
            f"(no es síntesis clínica ni conclusión causal): «{frag}»"
        )
    return (
        f"PMID {pmid} — en esta vista solo consta el título indexado; "
        f"abrir PubMed para leer el abstract completo: «{tshort}»"
    )


def _base_limitations() -> List[str]:
    return [
        "Síntesis generada con reglas deterministas (no LLM clínico); no sustituye juicio clínico.",
        "Verificar siempre frente a fuentes primarias y protocolos del centro.",
    ]


def build_unknown_medical_answer(message: str) -> Dict[str, Any]:
    """Respuesta estructurada mínima para ruta ``UNKNOWN``."""
    lim = list(_base_limitations())
    lim.append("El sistema no identificó cohorte SQL ni búsqueda de evidencia en esta petición.")
    return {
        "summary": message,
        "cohort_summary": None,
        "evidence_summary": None,
        "cohort_size": None,
        "key_findings": [],
        "recommendations": [],
        "limitations": lim,
        "citations": [],
        "evidence_statements": [],
        "uncertainty_notes": [],
        "applicability_notes": [],
    }


def citations_from_state(state: Dict[str, Any]) -> tuple[List[str], List[CitationOut]]:
    """Extrae PMIDs y citas tipadas desde ``evidence_bundle`` del estado del grafo."""
    eb = state.get("evidence_bundle")
    if eb is None:
        return [], []
    if hasattr(eb, "model_dump"):
        eb = eb.model_dump(mode="json")
    if not isinstance(eb, dict):
        return [], []
    eb = deduplicate_evidence_bundle_dict(eb)
    pmids_top = deduplicate_pmids(str(p) for p in (eb.get("pmids") or []))
    cites: List[CitationOut] = []
    for art in eb.get("articles") or []:
        if not isinstance(art, dict):
            continue
        pmid = str(art.get("pmid") or "").strip()
        if not pmid:
            continue
        year = art.get("year")
        y_int: Optional[int] = None
        if year is not None and str(year).strip().isdigit():
            y_int = int(str(year)[:4])
        doi = art.get("doi")
        cites.append(
            CitationOut(
                pmid=pmid,
                title=str(art.get("title") or ""),
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                year=y_int,
                doi=str(doi) if doi else None,
            )
        )
    pmids = pmids_top or [c.pmid for c in cites]
    return pmids, cites


def build_stub_medical_answer(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Construye ``MedicalAnswer`` serializable (dict) desde el estado del grafo.

    Las citas coinciden con ``citations_from_state`` (misma fuente que la API).
    """
    state = state_with_deduped_evidence(state)
    route = _route(state)
    ctx = _clinical_ctx(state)
    sql_r = state.get("sql_result")
    cohort_n = sql_cohort_size(sql_r) if sql_r else None

    cohort_summary: str | None = None
    if route == Route.HYBRID and ctx is not None:
        cohort_summary = _cohort_filter_description(ctx, cohort_n)
    elif route == Route.SQL and sql_r:
        cohort_summary = _cohort_summary_sql_only(sql_r, cohort_n)
    elif route == Route.HYBRID and sql_r and cohort_summary is None:
        cohort_summary = _cohort_summary_sql_only(sql_r, cohort_n)

    _, citations = citations_from_state(state)
    cites_json: List[Dict[str, Any]] = [c.model_dump(mode="json") for c in citations]

    eb = state.get("evidence_bundle")
    arts: list[dict[str, Any]] = []
    if isinstance(eb, dict):
        raw_arts = eb.get("articles") or []
        arts = [a for a in raw_arts if isinstance(a, dict)]

    uq_full = (state.get("user_query") or "").strip()

    refine_suffix = ""
    eb_rd0 = state.get("evidence_bundle")
    if isinstance(eb_rd0, dict):
        rd0 = eb_rd0.get("retrieval_debug")
        if isinstance(rd0, dict) and (rd0.get("synthesis_pubtype_refine") or {}).get("applied"):
            refine_suffix = (
                " Tras un primer listado muy amplio en PubMed, se aplicó un segundo filtro por "
                "tipo de publicación (revisión, meta-análisis o ensayo controlado aleatorizado)."
            )

    evidence_statements: List[Dict[str, Any]] = []
    for a in arts[:6]:
        pmid = str(a.get("pmid") or "").strip()
        title = str(a.get("title") or "").strip()
        if not pmid:
            continue
        design = infer_study_type_from_title(title)
        st_body = _evidence_statement_from_article(a)
        st = (f"[Diseño inferido del título: {design}] " if design else "") + st_body
        row: Dict[str, Any] = {"statement": st, "citation_pmids": [pmid]}
        strength = _infer_evidence_strength(title)
        if strength is not None:
            row["strength"] = strength.value
        evidence_statements.append(row)

    if arts:
        n = len(arts)
        if uq_full:
            qref = uq_full if len(uq_full) <= 160 else uq_full[:157].rstrip() + "…"
            extra_order = ""
            if route in (Route.HYBRID, Route.EVIDENCE):
                extra_order = (
                    " El orden mostrado prioriza coincidencia con la pregunta y tipo de estudio "
                    "inferido del título (heurística)."
                )
            evidence_summary = (
                f"Se recuperaron {n} referencias en PubMed ligadas a la petición "
                f"(«{qref}»). Los bloques siguientes citan recortes del resumen indexado "
                "cuando el copiloto los tiene disponibles; no sustituyen leer el artículo completo."
                + extra_order
                + refine_suffix
            )
        else:
            evidence_summary = (
                f"Se recuperaron {n} referencias indexadas; los extractos orientan sobre "
                "el abstract cuando está disponible en este turno."
                + refine_suffix
            )
    else:
        evidence_summary = None

    key_findings: List[str] = []
    if cohort_n == 0 and route == Route.HYBRID and arts:
        key_findings.append(
            "Cohorte local: 0 pacientes con los criterios SQL aplicados (BD disponible). "
            "La evidencia PubMed recuperada es independiente del conteo local y sigue siendo válida."
        )
    elif cohort_n is not None:
        key_findings.append(
            f"En la cohorte local acotada a la consulta se contabilizan {cohort_n} pacientes."
        )
    elif route in (Route.SQL, Route.HYBRID) and sql_r:
        key_findings.append("Se obtuvo resultado de consulta SQL sobre datos locales.")
    if ctx and ctx.population_conditions:
        key_findings.append(
            "Perfil de condición en cohorte: " + ", ".join(ctx.population_conditions[:5]) + "."
        )
    if ctx and ctx.population_medications:
        key_findings.append(
            "Medicación considerada en cohorte: " + ", ".join(ctx.population_medications[:5]) + "."
        )
    if arts:
        key_findings.append(
            f"Literatura: {len(arts)} artículo(s) recuperado(s) para apoyar la consulta de evidencia."
        )

    ori = _build_synthesis_orientation(state, arts, uq_full)
    if ori:
        key_findings.insert(0, ori)

    recommendations: List[str] = []
    if cohort_n is not None or route == Route.SQL:
        recommendations.append(
            "Revisar la consulta SQL y las tablas usadas en la traza de auditoría antes de decisiones clínicas."
        )
    if arts:
        recommendations.append(
            "Contrastar los hallazgos con el abstract completo en PubMed usando los PMIDs citados."
        )
    if route == Route.EVIDENCE and not arts:
        recommendations.append(
            "Reformular la pregunta o ampliar criterios de búsqueda si no hubo artículos recuperados."
        )

    summary_parts: list[str] = []
    if cohort_n == 0 and route == Route.HYBRID and arts:
        summary_parts.append(
            f"La cohorte local con los criterios aplicados (SQL) contabiliza 0 pacientes "
            f"en la base de datos disponible. Esto refleja criterios combinados estrictos "
            f"y/o el tamaño de la BD, no implica ausencia de evidencia científica sobre "
            f"esta población: se recuperaron {len(arts)} referencia(s) en PubMed que sí "
            f"abordan la pregunta y se analizan a continuación."
        )
    elif cohort_n is not None:
        summary_parts.append(
            f"En la cohorte local se identificaron {cohort_n} pacientes acotados a la consulta."
        )
    elif cohort_summary and route == Route.HYBRID:
        summary_parts.append(
            "La consulta se orientó a cohorte local y evidencia científica (ver detalles estructurados)."
        )
    # ``evidence_summary`` no se concatena aquí: ``render_medical_answer_to_text`` ya añade la sección «Evidencia».
    if route == Route.EVIDENCE and not arts:
        summary_parts.append(
            "Consulta orientada a evidencia científica; no se recuperaron artículos en este turno."
        )
    if not summary_parts:
        if evidence_summary and arts:
            summary_parts.append(
                "Consulta orientada a evidencia en PubMed; el detalle de la petición y los extractos "
                "figuran en la sección «Evidencia»."
            )
        else:
            summary_parts.append(
                "Respuesta generada sin conteo de cohorte ni referencias recuperadas en este turno."
            )
    summary = " ".join(summary_parts).strip()

    result: Dict[str, Any] = {
        "summary": summary,
        "cohort_summary": cohort_summary,
        "evidence_summary": evidence_summary,
        "cohort_size": cohort_n,
        "key_findings": key_findings,
        "recommendations": recommendations,
        "limitations": _base_limitations(),
        "citations": cites_json,
        "evidence_statements": evidence_statements,
        "uncertainty_notes": [],
        "applicability_notes": [],
    }

    rs_raw = state.get("reasoning_state")
    if isinstance(rs_raw, dict):
        un = [str(x) for x in (rs_raw.get("uncertainty_notes") or []) if str(x).strip()]
        an = [str(x) for x in (rs_raw.get("applicability_notes") or []) if str(x).strip()]
        if un:
            result["uncertainty_notes"] = un
        if an:
            result["applicability_notes"] = an
        for c in rs_raw.get("conflicts") or []:
            cs = str(c).strip()
            if cs and cs not in result["limitations"]:
                result["limitations"].append(cs)
        eq = rs_raw.get("evidence_quality")
        if eq and isinstance(eq, str) and eq.strip():
            line = f"Calidad/agrupación de evidencia (heurística): {eq.strip()}."
            if line not in result["limitations"]:
                result["limitations"].append(line)

    return deduplicate_medical_answer_evidence(result)


def render_medical_answer_to_text(answer: Dict[str, Any]) -> str:
    """
    Vista legado humana derivada 100 % de ``MedicalAnswer`` (sin ``[synthesis stub]``).

    El nodo Safety añade el disclaimer al pie de ``final_answer``.
    """
    blocks: list[str] = []
    s = (answer.get("summary") or "").strip()
    if s:
        blocks.append(s)
    cs = answer.get("cohort_summary")
    if isinstance(cs, str) and cs.strip():
        blocks.append("Cohorte local\n" + cs.strip())
    es = answer.get("evidence_summary")
    if isinstance(es, str) and es.strip():
        blocks.append("Evidencia\n" + es.strip())
    kf = answer.get("key_findings") or []
    if isinstance(kf, list) and kf:
        lines = "\n".join(f"• {str(x).strip()}" for x in kf if str(x).strip())
        if lines:
            blocks.append("Hallazgos clave\n" + lines)
    rec = answer.get("recommendations") or []
    if isinstance(rec, list) and rec:
        lines = "\n".join(f"• {str(x).strip()}" for x in rec if str(x).strip())
        if lines:
            blocks.append("Recomendaciones (orientativas)\n" + lines)
    est = answer.get("evidence_statements") or []
    if isinstance(est, list) and est:
        sub: list[str] = []
        for item in est:
            if not isinstance(item, dict):
                continue
            st = str(item.get("statement") or "").strip()
            if not st:
                continue
            pm = item.get("citation_pmids") or []
            pm_s = ", ".join(str(p) for p in pm if p)
            line = st
            if pm_s and not re.search(r"\bPMID\s*\d+", st, re.I):
                line += f"\n  PMIDs: {pm_s}"
            strg = item.get("strength")
            if strg:
                line += f"\n  Fuerza de evidencia (heurística): {strg}"
            sub.append(line)
        if sub:
            blocks.append(
                "Referencias PubMed (extractos del resumen; orientativas)\n" + "\n\n".join(sub)
            )
    un = answer.get("uncertainty_notes") or []
    if isinstance(un, list) and un:
        lines = "\n".join(f"• {str(x).strip()}" for x in un if str(x).strip())
        if lines:
            blocks.append("Incertidumbre\n" + lines)
    an = answer.get("applicability_notes") or []
    if isinstance(an, list) and an:
        lines = "\n".join(f"• {str(x).strip()}" for x in an if str(x).strip())
        if lines:
            blocks.append("Aplicabilidad (cohorte ↔ evidencia)\n" + lines)
    lim = answer.get("limitations") or []
    if isinstance(lim, list) and lim:
        lines = "\n".join(f"• {str(x).strip()}" for x in lim if str(x).strip())
        if lines:
            blocks.append("Limitaciones\n" + lines)
    return "\n\n".join(blocks).strip()
