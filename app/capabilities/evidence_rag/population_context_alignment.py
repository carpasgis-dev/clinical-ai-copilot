"""
Alineación cohorte/pregunta ↔ nicho poblacional del artículo (marco general, ampliable).

Si el título o el extracto sugieren un **contexto asistencial especial** (p. ej. infección de
transmisión sexual tratada de forma crónica, embarazo, trasplante) y **ni la pregunta ni la
cohorte estructurada** invocan ese contexto, se aplica una penalización suave en el re-ranking.

No sustituye revisar el abstract; solo reduce prioridad cuando la señal es clara y la petición
no la pedía explícitamente.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from app.capabilities.clinical_sql.terminology import fold_ascii


def _fold(s: str) -> str:
    return fold_ascii((s or "").lower())


@dataclass(frozen=True, slots=True)
class _ClinicalNicheSpec:
    """Un nicho: señales en el artículo y señales que deben aparecer en pregunta/cohorte para legitimarlo."""

    article_signals: re.Pattern[str]
    user_or_cohort_signals: re.Pattern[str]
    # Si el artículo encaja en el nicho y el contexto no lo «invita», texto opcional para aplicabilidad (p. ej. rerank).
    applicability_limitada_line: str | None = None


# Penalización por nicho «no invitado»; acumulable con techo para no aplastar todo el lote.
_PENALTY_EACH = 0.13
_MAX_PENALTY = 0.32

# Tabla declarativa: añadir filas = nuevos contextos sin tocar la fórmula del score.
_NICHES: tuple[_ClinicalNicheSpec, ...] = (
    _ClinicalNicheSpec(
        article_signals=re.compile(
            r"("
            r"\b(hiv|aids|pwh|plwh|plhiv)\b|"
            r"\bpeople with hiv\b|\bpeople living with hiv\b|"
            r"\bantiretrovir\w*\b|\banti[-\s]?retrovir\w*\b|"
            r"\bart[-\s]?naive\b|\bart naive\b|\bon haart\b|\bhaart\b|"
            r"\bintegrase[-\s]?based\b|\bintegrase strand transfer\b|"
            r"\bintegrase inhibitor\w*\b|\binsti\b|"
            r"\bstrand transfer inhibitor\w*\b|"
            r"\bcd4\b|\bviral load\b|\bvirologic\w*\b|"
            r"\bart\b"
            r")",
            re.I,
        ),
        user_or_cohort_signals=re.compile(
            r"\b("
            r"hiv|aids|vih|sida|antiretrovir|inmunodefici|pwh|"
            r"vir[oó]log|viral load|cd4"
            r")\b",
            re.I,
        ),
        applicability_limitada_line=(
            "Limitada: foco en población con VIH o terapia antirretroviral; "
            "trasladabilidad muy limitada si la petición y la cohorte no lo mencionan."
        ),
    ),
    _ClinicalNicheSpec(
        article_signals=re.compile(
            r"\b("
            r"pregnant|pregnancy|gestation|prenatal|obstetric|"
            r"fetal|trimester|gravida"
            r")\b",
            re.I,
        ),
        user_or_cohort_signals=re.compile(
            r"\b("
            r"embaraz|gestaci[oó]n|prenatal|obst[eé]tr|fetal|parto|gravida|"
            r"gestational|gestacional|pregnan"
            r")\b",
            re.I,
        ),
    ),
    _ClinicalNicheSpec(
        article_signals=re.compile(
            r"\b("
            r"kidney transplant|liver transplant|heart transplant|lung transplant|"
            r"transplant recipient|graft rejection|donor organ|immunosuppressive regimen"
            r")\b",
            re.I,
        ),
        user_or_cohort_signals=re.compile(
            r"\b("
            r"trasplante|transplant|donante|inmunosup|injerto|rechazo de injerto"
            r")\b",
            re.I,
        ),
    ),
    _ClinicalNicheSpec(
        article_signals=re.compile(
            r"\b("
            r"chemotherapy|radiotherapy|oncology trial|solid tumor|malignancy|"
            r"metastatic carcinoma|anti[- ]?pd-1|anti[- ]?pd-l1|immunotherapy for cancer"
            r")\b",
            re.I,
        ),
        user_or_cohort_signals=re.compile(
            r"\b("
            r"cancer|tumor|oncolog|neoplasia|malign|quimioterap|radioterap|metasta|carcinoma"
            r")\b",
            re.I,
        ),
    ),
    _ClinicalNicheSpec(
        article_signals=re.compile(
            r"\b("
            r"hemodialysis|peritoneal dialysis|dialysis patient|on dialysis|"
            r"esrd|eskd|end[- ]stage kidney|stage 5 ckd|kidney replacement therapy"
            r")\b",
            re.I,
        ),
        user_or_cohort_signals=re.compile(
            r"\b("
            r"renal|riñon|rinon|dialisis|d[ií]alisis|insuficiencia renal|nefrop|ckd\b|erc"
            r")\b",
            re.I,
        ),
    ),
)


def niche_applicability_limitada_line(
    title: str,
    abstract_snippet: str,
    *,
    user_query: str,
    population_conditions: Sequence[str] | None,
    population_medications: Sequence[str] | None,
) -> str | None:
    """
    Primera línea de aplicabilidad «Limitada» asociada a un nicho declarado en ``_NICHES``,
    solo para filas que definan ``applicability_limitada_line`` y solo si el artículo activa
    el nicho sin que pregunta/cohorte lo legitimen.

    Mecanismo genérico: ampliar cobertura añadiendo texto en la tabla, no funciones por nicho.
    """
    uq = (user_query or "").strip()
    conds = [str(c).strip() for c in (population_conditions or []) if str(c).strip()]
    meds = [str(m).strip() for m in (population_medications or []) if str(m).strip()]
    if not uq and not conds and not meds:
        return None
    blob = _fold(f"{title} {abstract_snippet}")
    ctx_blob = _fold(" ".join([uq, *conds, *meds]))
    for spec in _NICHES:
        line = spec.applicability_limitada_line
        if not line:
            continue
        if spec.article_signals.search(blob) and not spec.user_or_cohort_signals.search(ctx_blob):
            return line
    return None


def niche_mismatch_penalty(
    title: str,
    abstract_snippet: str,
    *,
    user_query: str,
    population_conditions: Sequence[str] | None,
    population_medications: Sequence[str] | None,
) -> float:
    """
    Valor ≤ 0: penalización si el artículo activa nichos no reflejados en pregunta/cohorte.

    Si no hay texto de usuario ni cohorte estructurada, no se penaliza (no hay referencia).
    """
    uq = (user_query or "").strip()
    conds = [str(c).strip() for c in (population_conditions or []) if str(c).strip()]
    meds = [str(m).strip() for m in (population_medications or []) if str(m).strip()]
    if not uq and not conds and not meds:
        return 0.0

    blob = _fold(f"{title} {abstract_snippet}")
    ctx_blob = _fold(" ".join([uq, *conds, *meds]))

    pen = 0.0
    for spec in _NICHES:
        if spec.article_signals.search(blob) and not spec.user_or_cohort_signals.search(ctx_blob):
            pen -= _PENALTY_EACH
    return max(-_MAX_PENALTY, pen)
