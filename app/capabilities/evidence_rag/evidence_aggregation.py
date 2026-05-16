"""
Agregación cross-paper orientada a la pregunta terapéutica (sin LLM).

Produce un bloque «Hallazgos principales» por clase terapéutica cuando
``question_type`` pide eficacia/selección/comparación.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from app.capabilities.clinical_sql.terminology import fold_ascii
from app.capabilities.evidence_rag.clinical_answerability import (
    clinical_answerability_score,
    passes_answerability_gate,
)
from app.capabilities.evidence_rag.evidence_rerank import infer_study_type_from_title
from app.capabilities.evidence_rag.landmark_registry import match_landmark_trial

from app.capabilities.evidence_rag.clinical_semantics import (
    classify_intervention_in_text,
    get_concept,
    intervention_concepts,
    policy_for_question_type,
)

_OUTCOME_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("MACE / eventos CV mayores", re.compile(r"\b(mace|major adverse cardiovascular)\b", re.I)),
    (
        "Muerte cardiovascular / mortalidad CV",
        re.compile(r"\b(cardiovascular death|cardiovascular mortality)\b", re.I),
    ),
    (
        "Hospitalización por insuficiencia cardíaca",
        re.compile(r"\b(heart failure hospitalization|hospitalization for heart failure)\b", re.I),
    ),
    (
        "Ictus / prevención embólica",
        re.compile(r"\b(stroke|systemic embolism|embolic stroke)\b", re.I),
    ),
    (
        "Hemorragia mayor / intracraneal",
        re.compile(r"\b(major bleeding|intracranial hemorrhage)\b", re.I),
    ),
)

_STRONG_DESIGNS = frozenset(
    {
        "cvot-outcomes-trial",
        "rct",
        "meta-analysis",
        "network-meta-analysis",
        "systematic-review",
        "umbrella-review",
        "guideline",
    }
)

_TOPICAL_ONLY = re.compile(
    r"\b("
    r"summit report|conference report|workshop report|"
    r"state of the science|year in review|expert panel report|"
    r"position paper(?!.*randomi)|"
    r"annual report|scientific statement(?!.*trial)"
    r")\b",
    re.I,
)


@dataclass
class _TherapyBucket:
    label: str
    pmids: list[str] = field(default_factory=list)
    designs: set[str] = field(default_factory=set)
    outcomes: set[str] = field(default_factory=set)
    landmarks: set[str] = field(default_factory=set)
    answerability_sum: float = 0.0


def _outcomes_in_blob(blob: str) -> set[str]:
    found: set[str] = set()
    for label, pat in _OUTCOME_PATTERNS:
        if pat.search(blob):
            found.add(label)
    return found


def _is_topical_report(title: str, abstract: str) -> bool:
    blob = fold_ascii(f"{title} {abstract}")
    if not _TOPICAL_ONLY.search(blob):
        return False
    if re.search(
        r"\b(randomi[sz]ed|controlled trial|cvot|meta[-\s]?analysis|"
        r"primary outcome|hazard ratio)\b",
        blob,
        re.I,
    ):
        return False
    return True


def _should_aggregate(question_type: str | None) -> bool:
    return policy_for_question_type(question_type).therapeutic_objective


def _evidence_level_label(buckets: dict[str, _TherapyBucket]) -> str:
    strong_n = 0
    landmark_n = 0
    for b in buckets.values():
        if b.designs & _STRONG_DESIGNS:
            strong_n += 1
        if b.landmarks:
            landmark_n += 1
    if landmark_n >= 2 or strong_n >= 2:
        return "alto para desenlaces cardiovasculares mayores (CVOTs, meta-análisis o revisiones sistemáticas en el pool)"
    if strong_n >= 1:
        return "moderado-alto (hay al menos un diseño fuerte por clase terapéutica identificada)"
    return "limitado (predominan diseños débiles o reportes contextuales en el pool recuperado)"


def aggregate_therapeutic_findings(
    articles: list[dict[str, Any]],
    *,
    question_type: str | None = None,
    user_query: str = "",
) -> str | None:
    """
    Bloque markdown con hallazgos agrupados por clase terapéutica.
    """
    if not _should_aggregate(question_type):
        return None
    if not articles:
        return None

    from app.capabilities.evidence_rag.clinical_intent import ClinicalIntent

    intent = ClinicalIntent(question_type=question_type or "general")
    buckets: dict[str, _TherapyBucket] = {}

    for art in articles:
        if not isinstance(art, dict):
            continue
        title = str(art.get("title") or "")
        snip = str(art.get("abstract_snippet") or "")
        pmid = str(art.get("pmid") or "").strip()
        if not pmid:
            continue
        if _is_topical_report(title, snip):
            continue
        st = infer_study_type_from_title(title)
        if not passes_answerability_gate(title, snip, study_type=st, intent=intent, question_type=question_type):
            continue
        blob = f"{title} {snip}"
        iv_ids = classify_intervention_in_text(blob)
        if not iv_ids:
            continue
        tid = iv_ids[0]
        concept = get_concept(tid)
        label = concept.label_es if concept else tid
        b = buckets.get(tid)
        if b is None:
            b = _TherapyBucket(label=label)
            buckets[tid] = b
        if pmid not in b.pmids:
            b.pmids.append(pmid)
        if st:
            b.designs.add(st)
        b.outcomes |= _outcomes_in_blob(blob)
        trial = match_landmark_trial(title, snip)
        if trial:
            b.landmarks.add(trial.acronym)
        b.answerability_sum += clinical_answerability_score(
            st, title, snip, intent=intent, question_type=question_type
        )

    if not buckets:
        return None

    priority = [c.id for c in intervention_concepts()]
    ordered_ids = [x for x in priority if x in buckets] + [
        x for x in buckets if x not in priority
    ]

    lines: list[str] = ["**Hallazgos principales (agregación automática por clase terapéutica):**", ""]
    for i, tid in enumerate(ordered_ids, start=1):
        b = buckets[tid]
        bullets: list[str] = []
        if b.landmarks:
            bullets.append(
                "Ensayos landmark detectados: " + ", ".join(sorted(b.landmarks)[:4]) + "."
            )
        strong = [d for d in b.designs if d in _STRONG_DESIGNS]
        if strong:
            bullets.append(
                "Diseños fuertes en el pool: " + ", ".join(sorted(strong)[:4]).replace("-", " ") + "."
            )
        if b.outcomes:
            bullets.append(
                "Desenlaces mencionados en títulos/resúmenes: "
                + "; ".join(sorted(b.outcomes)[:4])
                + "."
            )
        if not bullets:
            bullets.append(
                f"{len(b.pmids)} referencia(s) indexada(s); revisar abstracts (diseño no clasificado como fuerte)."
            )
        lines.append(f"{i}. **{b.label}**")
        for bl in bullets:
            lines.append(f"   - {bl}")
        lines.append(f"   - Soporte: PMID {', '.join(b.pmids[:5])}" + ("…" if len(b.pmids) > 5 else "") + ".")
        lines.append("")

    lines.append(
        f"**Nivel de evidencia en el pool recuperado (heurística):** {_evidence_level_label(buckets)}."
    )
    lines.append(
        "Esta agregación no sustituye meta-análisis ni guías; orienta la lectura frente a la pregunta terapéutica."
    )
    return "\n".join(lines).strip()


def aggregate_therapeutic_findings_from_state(state: dict[str, Any]) -> str | None:
    """Conveniencia: artículos + question_type desde estado del grafo."""
    from app.capabilities.evidence_rag.clinical_semantics import frame_from_state

    eb = state.get("evidence_bundle")
    if not isinstance(eb, dict):
        return None
    arts = [a for a in (eb.get("articles") or []) if isinstance(a, dict)]
    frame = frame_from_state(state)
    qtype = frame.question_type if frame else None
    if not qtype:
        raw = state.get("clinical_intent")
        if isinstance(raw, dict):
            qtype = raw.get("question_type")
    return aggregate_therapeutic_findings(
        arts,
        question_type=qtype,
        user_query=str(state.get("user_query") or ""),
    )
