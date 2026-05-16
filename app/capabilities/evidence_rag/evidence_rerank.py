"""
Re-ranking post-retrieval: mayor especificidad frente a la pregunta y jerarquía por diseño.

PubMed devuelve candidatos en orden de ranking interno; aquí se reordenan con señal léxica
(pregunta ↔ título/abstract) y con tipo de estudio inferido del título, priorizando síntesis
(meta-análisis, revisiones, ECA) frente a casos clínicos o modelos experimentales cuando la
pregunta pide tratamiento o evidencia clínica.

Además: ajuste por **cohorte estructurada** (condiciones/medicación), **nichos clínicos** no
invocados en la petición (ver ``population_context_alignment``) y, si se activa por env,
**shingles** pregunta↔texto (``semantic_rerank``) como sustituto ligero de embeddings.
"""
from __future__ import annotations

import re
from typing import Any, Iterable

from app.capabilities.evidence_rag.clinical_alignment import (
    alignment_composite,
    score_paper_alignment,
)
from app.capabilities.evidence_rag.clinical_knowledge import landmark_rerank_boost
from app.capabilities.evidence_rag.clinical_intent import ClinicalIntent, intent_asks_cv_outcomes
from app.capabilities.evidence_rag.population_context_alignment import (
    niche_applicability_limitada_line,
    niche_mismatch_penalty,
)
from app.capabilities.evidence_rag.intent_semantic_query import build_intent_semantic_query
from app.capabilities.evidence_rag.semantic_config import semantic_rerank_mode
from app.capabilities.evidence_rag.epistemic_ranking import finalize_rank_score
from app.capabilities.evidence_rag.semantic_ranking import semantic_rank_articles

_STOP = frozenset(
    {
        "que",
        "qué",
        "para",
        "por",
        "como",
        "cómo",
        "sobre",
        "con",
        "una",
        "unos",
        "este",
        "esta",
        "existe",
        "existen",
        "hay",
        "the",
        "and",
        "with",
        "from",
        "this",
        "that",
        "patients",
        "patient",
        "years",
        "year",
        "mayor",
        "mayores",
        "menor",
        "menores",
        "años",
    }
)


def _fold(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


_CVOT_TRIAL_NAMES = re.compile(
    r"\b(leader|sustain[-\s]?6|empa[-\s]?reg|declare[-\s]?timi|rewind|select|"
    r"canvas|credence|vertis[-\s]?cv|harmony|elixir|figure|bexagliflozin)\b",
    re.I,
)
_MECHANISTIC_REVIEW_MARKERS = re.compile(
    r"\b(mechanistic|pathophysiology|gut[-\s]?heart|antiatherosclerotic|"
    r"oxidative stress|experimental model|steatosis|fibrosis|nafld|nash)\b",
    re.I,
)
_OFF_TOPIC_CV_NOISE = re.compile(
    r"\b(depress|steatosis|nafld|nash|fibrosis|hepat|liver fat|gut microbiome)\b",
    re.I,
)


def infer_study_type_from_title(title: str) -> str | None:
    """
    Diseño inferido solo del título (heurística). Orden: patrones específicos antes que «review».
    """
    t = _fold(title)
    if re.search(r"\bcvot\b|cardiovascular outcome[s]?\s+trial|outcomes trial", t, re.I):
        return "cvot-outcomes-trial"
    if _CVOT_TRIAL_NAMES.search(t):
        return "cvot-outcomes-trial"
    if re.search(r"network\s+meta[-\s]?analysis", t) or "network meta-analysis" in t:
        return "network-meta-analysis"
    if "umbrella review" in t or ("umbrella" in t and "review" in t):
        return "umbrella-review"
    if re.search(r"meta[-\s]?analysis", t) or "metaanalysis" in t:
        return "meta-analysis"
    if "systematic review" in t or ("systematic" in t and "scoping" not in t and "review" in t):
        return "systematic-review"
    if re.search(r"randomi[sz]ed", t) or " rct " in t or t.startswith("rct "):
        return "rct"
    if re.search(r"\bcase report\b", t) or ": a case report" in t or " case report." in t:
        return "case-report"
    if "case series" in t:
        return "case-series"
    if "pilot" in t and ("trial" in t or "study" in t):
        return "pilot-trial"
    if "cross-sectional" in t or "cross sectional" in t:
        return "cross-sectional"
    if re.search(r"\b(guideline|consensus statement|position statement)\b", t):
        return "guideline"
    if re.search(r"\b(editorial|letter to the editor|corrigendum)\b", t):
        return "editorial"
    if re.search(r"\b(in vitro|animal model|murine|\brats?\b|\bmice\b|zebrafish)\b", t):
        return "preclinical"
    if "experimental model" in t and ("pathophysiology" in t or "oxidative stress" in t):
        return "basic-research"
    if re.search(r"\b(prevalence|epidemiologic|epidemiological)\b", t) and (
        "study" in t or "survey" in t or "burden" in t
    ):
        return "epidemiology"
    if "cohort" in t or "observational" in t or "prospective study" in t:
        return "cohort"
    if "narrative review" in t or ("narrative" in t and "review" in t):
        return "narrative-review"
    if re.search(r"\breview\b", t) or "critical review" in t:
        if _MECHANISTIC_REVIEW_MARKERS.search(t):
            return "mechanistic-review"
        return "review"
    return None


_DEFAULT_DESIGN_WEIGHTS: dict[str, float] = {
    "cvot-outcomes-trial": 1.0,
    "meta-analysis": 1.0,
    "network-meta-analysis": 1.0,
    "systematic-review": 0.95,
    "umbrella-review": 0.94,
    "guideline": 0.92,
    "rct": 0.88,
    "clinical-trial": 0.72,
    "pilot-trial": 0.55,
    "cohort": 0.5,
    "epidemiology": 0.48,
    "cross-sectional": 0.4,
    "review": 0.58,
    "mechanistic-review": 0.55,
    "narrative-review": 0.45,
    "case-series": 0.22,
    "case-report": 0.15,
    "editorial": 0.12,
    "preclinical": 0.1,
    "basic-research": 0.12,
}

_CV_OUTCOME_DESIGN_WEIGHTS: dict[str, float] = {
    **{k: v for k, v in _DEFAULT_DESIGN_WEIGHTS.items()},
    "cvot-outcomes-trial": 1.0,
    "rct": 0.95,
    "meta-analysis": 0.95,
    "network-meta-analysis": 0.95,
    "systematic-review": 0.80,
    "umbrella-review": 0.78,
    "review": 0.55,
    "mechanistic-review": 0.55,
    "narrative-review": 0.45,
    "cohort": 0.48,
    "epidemiology": 0.42,
    "cross-sectional": 0.38,
}


def study_design_weight(
    study_type: str | None,
    *,
    clinical_intent: ClinicalIntent | None = None,
) -> float:
    """Peso 0–1 para jerarquía de evidencia (mayor = más útil para síntesis clínica agregada)."""
    if not study_type:
        return 0.42
    table = (
        _CV_OUTCOME_DESIGN_WEIGHTS
        if clinical_intent is not None and intent_asks_cv_outcomes(clinical_intent)
        else _DEFAULT_DESIGN_WEIGHTS
    )
    return table.get(study_type, 0.4)


_WEAK_DESIGNS_FOR_REFINE = frozenset(
    {
        "case-report",
        "case-series",
        "editorial",
        "preclinical",
        "basic-research",
        "mechanistic-review",
        "narrative-review",
    }
)

_WEAK_SEMANTIC_FOR_CV = frozenset(
    {
        "mechanistic-review",
        "narrative-review",
        "review",
        "cross-sectional",
        "epidemiology",
        "preclinical",
        "basic-research",
        "case-report",
        "case-series",
        "editorial",
    }
)


def _semantic_weak_for_cv_ask(title: str, abstract_snippet: str) -> bool:
    """Ruido frecuente en preguntas CV que el tipo de estudio no captura bien."""
    blob = _fold(f"{title} {abstract_snippet}")
    if _OFF_TOPIC_CV_NOISE.search(blob) and not re.search(
        r"\b(mace|cardiovascular outcome|cvot|heart failure hospital|"
        r"myocardial infarction|cardiovascular death)\b",
        blob,
        re.I,
    ):
        return True
    if re.search(r"\btype\s*1\s*diabet|\bt1dm\b", blob, re.I) and not re.search(
        r"\btype\s*2\s*diabet|\bt2dm\b", blob, re.I
    ):
        return True
    return False


def weak_design_share_from_titles(titles: Iterable[str]) -> float:
    """
    Fracción de títulos cuyo diseño inferido es «débil» (caso clínico, serie, editorial, preclínico).

    Sirve para decidir una segunda búsqueda PubMed con filtros de tipo de publicación sin
    depender del texto de la pregunta del usuario.
    """
    ts = [str(t or "").strip() for t in titles if str(t or "").strip()]
    if not ts:
        return 0.0
    weak = 0
    for title in ts:
        st = infer_study_type_from_title(title)
        if st in _WEAK_DESIGNS_FOR_REFINE:
            weak += 1
    return weak / len(ts)


def clinical_weak_evidence_share(
    rows: Iterable[tuple[str, str]],
    *,
    clinical_intent: ClinicalIntent | None = None,
) -> float:
    """
    Fracción de candidatos «débiles» para síntesis CV (diseño + ruido semántico).

    Complementa ``weak_design_share_from_titles`` cuando ``weak_design_share_primary_page`` es 0
    pero la página mezcla revisiones mecanísticas, T1DM o temas off-topic.
    """
    items = [(str(t or "").strip(), str(s or "")) for t, s in rows if str(t or "").strip()]
    if not items:
        return 0.0
    weak = 0
    cv_ask = clinical_intent is not None and intent_asks_cv_outcomes(clinical_intent)
    for title, snip in items:
        st = infer_study_type_from_title(title)
        is_weak = st in _WEAK_DESIGNS_FOR_REFINE
        if cv_ask and st in _WEAK_SEMANTIC_FOR_CV and st not in (
            "cvot-outcomes-trial",
            "rct",
            "meta-analysis",
            "systematic-review",
        ):
            if st in ("mechanistic-review", "narrative-review", "review"):
                is_weak = True
        if cv_ask and _semantic_weak_for_cv_ask(title, snip):
            is_weak = True
        if is_weak:
            weak += 1
    return weak / len(items)


def infer_applicability_line(
    title: str,
    *,
    population_age_min: int | None = None,
    population_conditions: list[str] | None = None,
    population_medications: list[str] | None = None,
    user_query: str | None = None,
    abstract_snippet: str | None = None,
) -> str | None:
    """
    Una línea de aplicabilidad cohorte ↔ estudio (solo heurística de título + contexto estructurado).

    No sustituye leer el abstract; sirve para ``ReasoningState`` y avisos en UI.
    """
    if not (title or "").strip():
        return None

    snip = abstract_snippet or ""
    niche_lim = niche_applicability_limitada_line(
        title,
        snip,
        user_query=user_query or "",
        population_conditions=population_conditions,
        population_medications=population_medications,
    )
    if niche_lim:
        return niche_lim

    t = _fold(title)
    cohort_old = False
    try:
        cohort_old = population_age_min is not None and int(population_age_min) >= 65
    except (TypeError, ValueError):
        cohort_old = False
    conds = [_fold(str(c)) for c in (population_conditions or []) if str(c).strip()]

    if cohort_old:
        young = bool(
            re.search(
                r"\b(adolescents?|pediatric|paediatric|children|childhood|"
                r"school-?age|infants?|teens?|teenage|youth)\b",
                t,
            )
        )
        if young:
            return "Limitada: población joven/pediátrica frente a cohorte local ≥65 años."
        if re.search(r"\b(pregnan|gestation|prenatal|obstetric)\b", t):
            return "Limitada: población obstétrica; revisar trasladabilidad a la cohorte acotada."
        if ("middle-aged" in t or "middle aged" in t) and not re.search(
            r"\b(older adult|elderly|aged\s+65|≥\s*65|65\s*\+)\b", t
        ):
            return "Parcial: verificar inclusión de adultos mayores frente a cohorte ≥65 años."

    if re.search(r"\b(pcos|polycystic ovary)\b", t):
        if cohort_old or any("diabet" in c for c in conds):
            return "Parcial: foco SOP/metabolismo; contrastar con diabetes/hipertensión y edad de la cohorte."

    if cohort_old:
        return "Revisar en abstract edad de inclusión y representación de ≥65 años frente a la cohorte local."
    if conds:
        return "Verificar en abstract alineación con criterios de la cohorte local (condiciones/medicación)."
    return None


def is_weak_for_clinical_synthesis(study_type: str | None, user_query: str) -> bool:
    """True si el diseño suele aportar poco a una pregunta de tratamiento / evidencia clínica."""
    u = _fold(user_query)
    clinical_ask = bool(
        re.search(
            r"(tratamient|terapia|evidenc|f[aá]rmac|recomend|reduc(e|ir)|riesgo|prevenci[oó]n)",
            u,
        )
    )
    if not clinical_ask:
        return False
    return study_type in {
        "case-report",
        "case-series",
        "editorial",
        "preclinical",
        "basic-research",
    }


def _token_keywords(text: str) -> set[str]:
    t = re.sub(r"[^\w\sáéíóúüñ]", " ", _fold(text))
    return {w for w in t.split() if len(w) > 3 and w not in _STOP}


def lexical_relevance_score(user_query: str, title: str, abstract_snippet: str) -> float:
    """Solapamiento simple pregunta ↔ título+snippet (0–1)."""
    kq = _token_keywords(user_query)
    if not kq:
        return 0.35
    kt = _token_keywords(title) | _token_keywords(abstract_snippet)
    if not kt:
        return 0.2
    inter = len(kq & kt)
    union = len(kq | kt) or 1
    return min(1.0, 0.25 + 0.75 * (inter / max(len(kq), 4)))


# Umbral mínimo pregunta↔(título+snippet) para «referencias destacadas» en texto orientativo.
MIN_HEADLINE_LEXICAL_RELEVANCE = 0.36


def cohort_lexical_adjustment(
    title: str,
    abstract_snippet: str,
    *,
    population_conditions: list[str] | None,
    population_medications: list[str] | None = None,
) -> float:
    """
    Bonus/penalty acotado por solape título+abstract con condiciones/medicación estructuradas de cohorte.

    Complementa ``infer_applicability_line`` (edad/SOP): aquí se premia mencionar diabetes/HTA/etc.
    y se penaliza suavemente si la cohorte exige varios ejes y el artículo no refleja ninguno,
    o si el foco del título es claramente ajeno (p. ej. sarcoidosis sin diabetes cuando la cohorte es DM).
    """
    conds = [str(c).strip() for c in (population_conditions or []) if str(c).strip()]
    meds = [str(m).strip() for m in (population_medications or []) if str(m).strip()]
    if not conds and not meds:
        return 0.0

    blob = _fold(f"{title} {abstract_snippet}")
    adj = 0.0

    want_dm = any(re.search(r"diabet|glucos|insulin", _fold(c)) for c in conds)
    want_htn = any(re.search(r"hipertens|hta|tensi|pressure", _fold(c), re.I) for c in conds)
    has_dm = bool(
        re.search(
            r"\b(diabet|glucose|glycemic|glycaemic|insulin|t2d|t1d|t2dm|t1dm|hyperglyc)\b",
            blob,
        )
    )
    has_htn = bool(
        re.search(r"\b(hypertens|hipertens|blood\s+pressure|arterial\s+pressure)\b", blob)
    )

    for c in conds[:8]:
        cf = _fold(c)
        stems: list[str] = []
        if re.search(r"diabet|glucos|insulin", cf):
            stems.extend(["diabet", "glucose", "glycemic", "insulin", "t2d", "t1d", "t2dm", "t1dm"])
        if re.search(r"hipertens|hta|tensi", cf):
            stems.extend(["hypertens", "hipertens", "blood pressure", "pressure"])
        if not stems and len(cf) >= 4:
            stems = [w for w in cf.split() if len(w) > 4][:3]
        for s in stems:
            if len(s) > 3 and s in blob:
                adj += 0.055
                break
    adj = min(0.16, adj)

    for m in meds[:6]:
        mf = _fold(m)
        if len(mf) >= 4 and mf in blob:
            adj += 0.045
    adj = min(0.2, adj)

    if want_dm and want_htn and not has_dm and not has_htn:
        adj -= 0.17
    elif want_dm and want_htn and (not has_dm or not has_htn):
        adj -= 0.05

    if want_dm and re.search(r"\bsarcoid", blob) and not has_dm:
        adj -= 0.13
    if want_dm and re.search(r"\bkidney\s+cancer\b|\brenal\s+cell\b", blob) and not has_dm:
        adj -= 0.09

    return max(-0.28, min(0.22, adj))


def composite_relevance_score(
    *,
    user_query: str,
    title: str,
    abstract_snippet: str,
    study_type: str | None,
    position_index: int,
    population_age_min: int | None = None,
    population_conditions: list[str] | None = None,
    population_medications: list[str] | None = None,
    clinical_intent: ClinicalIntent | None = None,
    clinical_context: ClinicalContext | None = None,
) -> tuple[float, str]:
    """
    Puntuación interna para ordenar: combina diseño, relevancia léxica y penalización por
    diseños débiles cuando la pregunta es clínica.
    """
    lex = lexical_relevance_score(user_query, title, abstract_snippet)
    des = study_design_weight(study_type, clinical_intent=clinical_intent)
    weak = is_weak_for_clinical_synthesis(study_type, user_query)
    penalty = 0.35 if weak else 0.0
    # Calculate new applicability score
    applicability = calculate_applicability(title, abstract_snippet, clinical_context)
    app_line = applicability.explanation
    app_multiplier = applicability.applicability_score

    clin = cohort_lexical_adjustment(
        title,
        abstract_snippet,
        population_conditions=population_conditions,
        population_medications=population_medications,
    )
    niche_pen = niche_mismatch_penalty(
        title,
        abstract_snippet,
        user_query=user_query,
        population_conditions=population_conditions or [],
        population_medications=population_medications or [],
    )
    recency = max(0.0, 0.06 - position_index * 0.01)
    align_term = 0.0
    if clinical_intent is not None:
        align_scores = score_paper_alignment(clinical_intent, title, abstract_snippet)
        axis = getattr(clinical_intent, "priority_axis", "intervention") or "intervention"
        align_term = 0.38 * alignment_composite(
            align_scores, priority_axis=axis, intent=clinical_intent
        )
    landmark = landmark_rerank_boost(
        title, abstract_snippet, clinical_intent=clinical_intent
    )
    raw = (
        0.30 * des
        + 0.28 * lex
        + align_term
        + recency
        + clin
        + niche_pen
        + landmark
        - penalty
    )
    
    raw = max(0.01, raw * app_multiplier)
    
    return raw, app_line or "" 


def rerank_article_dicts(
    articles: Iterable[dict[str, Any]],
    user_query: str,
    *,
    cap: int = 6,
    population_age_min: int | None = None,
    population_conditions: list[str] | None = None,
    population_medications: list[str] | None = None,
    clinical_intent: ClinicalIntent | None = None,
    clinical_context: ClinicalContext | None = None,
) -> list[dict[str, Any]]:
    """
    Reordena artículos (dicts con pmid, title, abstract_snippet, …) y devuelve los ``cap`` primeros.
    """
    rows: list[dict[str, Any]] = [dict(a) for a in articles if isinstance(a, dict)]
    if not rows:
        return []

    rank_query = build_intent_semantic_query(clinical_intent, user_query)

    heuristic_scores: list[float] = []
    enriched: list[dict[str, Any]] = []
    for i, a in enumerate(rows):
        title = str(a.get("title") or "")
        snip = str(a.get("abstract_snippet") or "")
        st = infer_study_type_from_title(title)
        raw_sc, applicability_line = composite_relevance_score(
            user_query=rank_query,
            title=title,
            abstract_snippet=snip,
            study_type=st,
            position_index=i,
            population_age_min=population_age_min,
            population_conditions=population_conditions,
            population_medications=population_medications,
            clinical_intent=clinical_intent,
            clinical_context=clinical_context,
        )
        if applicability_line:
            a["applicability_line"] = applicability_line
        sc, rank_meta = finalize_rank_score(
            raw_sc,
            title=title,
            abstract=snip,
            clinical_intent=clinical_intent,
        )
        row = dict(a)
        row["rank_score_debug"] = rank_meta
        if clinical_intent is not None:
            row["alignment_scores"] = score_paper_alignment(
                clinical_intent, title, snip
            ).to_dict()
        heuristic_scores.append(sc)
        enriched.append(row)

    if semantic_rerank_mode() != "off":
        out, sem_dbg = semantic_rank_articles(
            enriched,
            user_query=user_query,
            clinical_intent=clinical_intent,
            heuristic_scores=heuristic_scores,
            cap=cap,
        )
        for art in out:
            art["semantic_ranking_debug"] = sem_dbg
        return out

    pairs = list(zip(heuristic_scores, enriched))
    pairs.sort(key=lambda x: x[0], reverse=True)
    return [a for _, a in pairs[: max(1, cap)]]
