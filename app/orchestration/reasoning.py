"""
Fase 3.7.a — estado de razonamiento clínico determinista (sin LLM).

Construye ``ReasoningState`` a partir de ``sql_result``, ``clinical_context`` y
``evidence_bundle`` ya presentes en el estado del grafo.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from app.capabilities.evidence_rag.evidence_rerank import infer_applicability_line, infer_study_type_from_title
from app.orchestration.medical_answer_builder import sql_cohort_size
from app.schemas.copilot_state import Route


@dataclass
class EvidenceAssessment:
    pmid: str
    relevance_score: float
    study_type: str | None = None
    applicability: str | None = None


@dataclass
class ReasoningState:
    cohort_summary: str | None = None
    evidence_assessments: list[EvidenceAssessment] = field(default_factory=list)
    uncertainty_notes: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    applicability_notes: list[str] = field(default_factory=list)
    evidence_quality: str | None = None
    synthesis_calibration: dict[str, Any] | None = None


def reasoning_state_to_dict(rs: ReasoningState) -> dict[str, Any]:
    d = asdict(rs)
    d["evidence_assessments"] = [asdict(e) for e in rs.evidence_assessments]
    return d


def _ctx_age_signals(ctx: dict[str, Any] | None) -> tuple[bool, str | None]:
    """True si la cohorte/resumen apunta a población mayor (heurística simple)."""
    if not isinstance(ctx, dict):
        return False, None
    amin = ctx.get("population_age_min")
    try:
        if amin is not None and int(amin) >= 65:
            return True, f"edad mínima cohorte ≥{int(amin)}"
    except (TypeError, ValueError):
        pass
    ar = (ctx.get("age_range") or "").strip().lower()
    if ar and ("65" in ar or ">60" in ar or ">70" in ar or "≥65" in ar or ">=" in ar):
        return True, f"rango de edad en contexto: {ctx.get('age_range')}"
    return False, None


def _as_jsonable_dict(obj: Any) -> dict[str, Any] | None:
    """LangGraph puede conservar modelos Pydantic; el razonamiento espera dict."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj
    md = getattr(obj, "model_dump", None)
    if callable(md):
        try:
            return md(mode="json")
        except TypeError:
            return md()
    return None


def _effective_user_query(state: dict[str, Any]) -> str:
    """Misma lógica que ``_graph_effective_user_query`` (evita import circular con nodes)."""
    r = state.get("resolved_user_query")
    if isinstance(r, str) and r.strip():
        return r.strip()
    return str(state.get("user_query") or "").strip()


def build_reasoning_state(state: dict[str, Any]) -> ReasoningState:
    route = state.get("route")
    if isinstance(route, str):
        try:
            route = Route(route)
        except ValueError:
            route = Route.UNKNOWN
    elif not isinstance(route, Route):
        route = Route.UNKNOWN

    sql_r = state.get("sql_result")
    cohort_n = sql_cohort_size(sql_r)
    ctx_raw = state.get("clinical_context")
    ctx = _as_jsonable_dict(ctx_raw)
    cohort_text: str | None = None
    if isinstance(ctx, dict) and ctx:
        bits: list[str] = []
        if cohort_n is not None:
            bits.append(f"n≈{cohort_n}")
        pconds = [str(x).strip() for x in (ctx.get("population_conditions") or []) if str(x).strip()]
        if pconds:
            bits.append("condiciones: " + ", ".join(pconds[:8]))
        pmeds = [str(x).strip() for x in (ctx.get("population_medications") or []) if str(x).strip()]
        if pmeds:
            bits.append("medicación: " + ", ".join(pmeds[:6]))
        amin = ctx.get("population_age_min")
        amax = ctx.get("population_age_max")
        try:
            if amin is not None:
                bits.append(f"edad ≥ {int(amin)} años")
        except (TypeError, ValueError):
            pass
        try:
            if amax is not None:
                bits.append(f"edad < {int(amax)} años")
        except (TypeError, ValueError):
            pass
        sex = (ctx.get("population_sex") or "").strip().upper()
        if sex in ("F", "M"):
            bits.append(f"sexo {sex}")
        if bits:
            cohort_text = (
                "Cohorte local (" + "; ".join(bits) + ") con contexto clínico estructurado para acotar evidencia."
            )
        else:
            cohort_text = "Contexto clínico estructurado disponible para acotar evidencia."
    elif cohort_n is not None:
        cohort_text = f"Cohorte local contada: n={cohort_n}."
    else:
        cohort_text = None

    eb_raw = state.get("evidence_bundle")
    eb = _as_jsonable_dict(eb_raw)
    arts: list[dict[str, Any]] = []
    pmids: list[str] = []
    if isinstance(eb, dict):
        pmids = [str(p).strip() for p in (eb.get("pmids") or []) if str(p).strip()]
        raw_arts = eb.get("articles") or []
        arts = [a for a in raw_arts if isinstance(a, dict)]

    notes: list[str] = []
    conflicts: list[str] = []
    assessments: list[EvidenceAssessment] = []

    if cohort_n is not None and cohort_n < 5:
        notes.append("La cohorte local es pequeña (n<5); interpretar conteos con cautela.")

    old_years = [a.get("year") for a in arts if isinstance(a.get("year"), int) and a["year"] < 2020]
    if old_years:
        notes.append(
            "Parte de la evidencia recuperada no es reciente (al menos un artículo anterior a 2020)."
        )

    expects_evidence = route in (Route.EVIDENCE, Route.HYBRID)
    pq = (state.get("pubmed_query") or "").strip()
    if expects_evidence and not pmids and not arts:
        if pq:
            notes.append(
                "PubMed se consultó con un término construido, pero no se recuperaron artículos "
                "en este turno (p. ej. filtro de fechas en esearch, formulación del término, "
                "respuesta vacía o error silenciado en la API de NCBI)."
            )
        else:
            conflicts.append(
                "No se encontró evidencia indexada recuperable en este turno "
                "(no se generó término de búsqueda PubMed)."
            )

    cohort_old, _age_reason = _ctx_age_signals(ctx if isinstance(ctx, dict) else None)
    pop_age_min: int | None = None
    pop_conds: list[str] = []
    if isinstance(ctx, dict):
        raw_amin = ctx.get("population_age_min")
        try:
            if raw_amin is not None:
                pop_age_min = int(raw_amin)
        except (TypeError, ValueError):
            pop_age_min = None
        pop_conds = [str(x).strip() for x in (ctx.get("population_conditions") or []) if str(x).strip()]
        pop_meds = [str(x).strip() for x in (ctx.get("population_medications") or []) if str(x).strip()]
    else:
        pop_meds = []

    uq_eff = _effective_user_query(state)

    if cohort_old and arts:
        notes.append(
            "La cohorte local incluye edad ≥65 años: conviene verificar en cada abstract la edad "
            "de inclusión del estudio (puede ser más amplia que la cohorte SQL)."
        )

    scored_arts: list[tuple[float, dict[str, Any]]] = []
    for a in arts:
        if not isinstance(a, dict):
            continue
        fr = a.get("final_rank_score")
        if fr is not None:
            score = float(fr)
        else:
            dbg = a.get("rank_score_debug")
            if isinstance(dbg, dict) and dbg.get("final_fusion") is not None:
                score = float(dbg["final_fusion"])
            else:
                sem = a.get("semantic_scores")
                if isinstance(sem, dict) and sem.get("fused") is not None:
                    score = float(sem["fused"])
                else:
                    score = 0.35
        scored_arts.append((score, a))
    scored_arts.sort(key=lambda x: x[0], reverse=True)

    for score, a in scored_arts[:6]:
        pmid = str(a.get("pmid") or "").strip()
        title = str(a.get("title") or "")
        year = a.get("year")
        y_int: int | None = None
        if isinstance(year, int):
            y_int = year
        elif year is not None and str(year).strip().isdigit():
            y_int = int(str(year)[:4])

        score = round(score, 3)
        study = infer_study_type_from_title(title)

        appl = infer_applicability_line(
            title,
            population_age_min=pop_age_min,
            population_conditions=pop_conds if pop_conds else None,
            population_medications=pop_meds if pop_meds else None,
            user_query=uq_eff,
            abstract_snippet=str(a.get("abstract_snippet") or ""),
        )

        if pmid:
            assessments.append(
                EvidenceAssessment(
                    pmid=pmid,
                    relevance_score=round(score, 3),
                    study_type=study,
                    applicability=appl,
                )
            )

    appl_notes: list[str] = []
    if assessments:
        lim_n = sum(1 for e in assessments if (e.applicability or "").startswith("Limitada"))
        par_n = sum(1 for e in assessments if (e.applicability or "").startswith("Parcial"))
        if lim_n:
            appl_notes.append(
                f"{lim_n} referencia(s) con aplicabilidad probablemente limitada a la cohorte local "
                "(p. ej. población de estudio distinta); priorizar lectura crítica del abstract."
            )
        elif par_n:
            appl_notes.append(
                f"{par_n} referencia(s) con trasladabilidad parcial frente al perfil de cohorte; "
                "verificar criterios de inclusión en el abstract."
            )
            
    rd_check = eb.get("retrieval_debug") if isinstance(eb, dict) else None
    if isinstance(rd_check, dict) and rd_check.get("outcome") == "partial_primary_miss":
        notes.append(
            "La recuperación principal estricta no obtuvo resultados. "
            "Se amplió la métrica apoyándose en etapas adicionales (ej. ensayos clínicos clásicos o relajación de términos)."
        )

    if expects_evidence:
        if not arts and not pmids:
            rd = eb.get("retrieval_debug") if isinstance(eb, dict) else None
            if isinstance(rd, dict) and rd.get("outcome") and str(rd.get("outcome")) not in ("success", "partial_primary_miss"):
                eq = str(rd["outcome"])
            elif pq:
                eq = "sin_resultados_pubmed"
            else:
                eq = "sin_referencias"
        elif old_years and any(
            isinstance(a.get("year"), int) and a["year"] >= 2020 for a in arts
        ):
            eq = "mixta"
        elif arts and not old_years:
            eq = "reciente"
        elif old_years:
            eq = "predominantemente_historica"
        else:
            eq = None
    else:
        eq = None

    from app.orchestration.synthesis_calibration import calibration_from_state

    cal = calibration_from_state(state)
    cal_dict: dict[str, Any] | None = cal.to_dict() if cal else None
    if cal is not None:
        cal_line = (
            f"Calibración síntesis: confianza recuperación={cal.retrieval_confidence:.2f}, "
            f"utilidad respuesta={cal.clinical_answer_confidence:.2f}, "
            f"tier dominante={cal.dominant_retrieval_tier}, outcome={cal.retrieval_outcome}."
        )
        if cal_line not in notes:
            notes.append(cal_line)
        if cal.retrieval_confidence < 0.5 and cal.dominant_retrieval_tier >= 3:
            weak = (
                "Evidencia recuperada con baja confianza o tier epistémico alto (≥3); "
                "evitar conclusiones de eficacia directa sin revisar abstracts."
            )
            if weak not in notes:
                notes.append(weak)

    return ReasoningState(
        cohort_summary=cohort_text,
        evidence_assessments=assessments,
        uncertainty_notes=notes,
        conflicts=conflicts,
        applicability_notes=appl_notes,
        evidence_quality=eq,
        synthesis_calibration=cal_dict,
    )
