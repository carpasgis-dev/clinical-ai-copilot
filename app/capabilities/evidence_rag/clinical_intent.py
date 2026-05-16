"""
Extracción heurística de intención clínica (PICO-lite) desde pregunta NL + cohorte estructurada.

Sin LLM por defecto: estable, testeable. El resultado alimenta re-ranking, query PubMed y síntesis.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Optional, Union

IntentPriorityAxis = Literal["population", "intervention", "outcome"]

from app.capabilities.clinical_sql.terminology import fold_ascii
from app.schemas.copilot_state import ClinicalContext


def _fold(s: str) -> str:
    return fold_ascii((s or "").lower())


def _as_clinical_context(
    clinical_context: Optional[Union[ClinicalContext, dict, Mapping[str, Any]]],
) -> ClinicalContext | None:
    if clinical_context is None:
        return None
    if isinstance(clinical_context, ClinicalContext):
        return clinical_context
    return ClinicalContext.model_validate(clinical_context)


@dataclass
class ClinicalIntent:
    """Intención clínica normalizada para alineación dinámica (no exclusión rígida)."""

    population: list[str] = field(default_factory=list)
    interventions: list[str] = field(default_factory=list)
    comparator: list[str] = field(default_factory=list)
    outcomes: list[str] = field(default_factory=list)
    evidence_preference: list[str] = field(default_factory=list)
    age_min: int | None = None
    age_max: int | None = None
    # Poblaciones que el artículo puede mencionar pero la pregunta no invita (penalización suave).
    population_noise: list[str] = field(default_factory=list)
    # Eje dominante de la pregunta (afecta recuperación PubMed vs rerank).
    priority_axis: IntentPriorityAxis = "intervention"

    def to_dict(self) -> dict[str, Any]:
        return {
            "population": list(self.population),
            "interventions": list(self.interventions),
            "comparator": list(self.comparator),
            "outcomes": list(self.outcomes),
            "evidence_preference": list(self.evidence_preference),
            "age_min": self.age_min,
            "age_max": self.age_max,
            "population_noise": list(self.population_noise),
            "priority_axis": self.priority_axis,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> ClinicalIntent:
        if not raw:
            return cls()
        return cls(
            population=[str(x) for x in (raw.get("population") or []) if str(x).strip()],
            interventions=[str(x) for x in (raw.get("interventions") or []) if str(x).strip()],
            comparator=[str(x) for x in (raw.get("comparator") or []) if str(x).strip()],
            outcomes=[str(x) for x in (raw.get("outcomes") or []) if str(x).strip()],
            evidence_preference=[
                str(x) for x in (raw.get("evidence_preference") or []) if str(x).strip()
            ],
            age_min=_safe_int(raw.get("age_min")),
            age_max=_safe_int(raw.get("age_max")),
            population_noise=[
                str(x) for x in (raw.get("population_noise") or []) if str(x).strip()
            ],
            priority_axis=_coerce_priority_axis(raw.get("priority_axis")),
        )


def _safe_int(v: Any) -> int | None:
    try:
        if v is None:
            return None
        return int(v)
    except (TypeError, ValueError):
        return None


def _dedupe(seq: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in seq:
        k = x.strip().lower()
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(x)
    return out


def _has_any(blob: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(p, blob, re.I) for p in patterns)


def _age_from_query(q: str) -> tuple[int | None, int | None]:
    fl = _fold(q)
    age_min: int | None = None
    age_max: int | None = None
    m = re.search(r"(?:>=|≥|mayor(?:es)?\s+de|>\s*)(\d{2,3})\s*a", fl)
    if m:
        try:
            age_min = int(m.group(1))
        except ValueError:
            pass
    m2 = re.search(r"(\d{2,3})\s*a[nñ]os\s+o\s+mas", fl)
    if m2 and age_min is None:
        try:
            age_min = int(m2.group(1))
        except ValueError:
            pass
    if re.search(r"\b(ancian|elderly|older adult|geriatr)\b", fl) and age_min is None:
        age_min = 65
    m3 = re.search(r"(?:<=|≤|menor(?:es)?\s+de|<\s*)(\d{2})\s*a", fl)
    if m3:
        try:
            age_max = int(m3.group(1))
        except ValueError:
            pass
    if re.search(r"\b(pediatr|adolesc|infant|nino|niño)\b", fl):
        age_max = 18
    return age_min, age_max


def _population_from_text(fl: str) -> list[str]:
    pop: list[str] = []
    t1 = _has_any(
        fl,
        (
            r"\b(t1dm|t1d|dm1|type\s*1\s*diabet|diabetes\s*tipo\s*1|insulin[-\s]?dependent)\b",
        ),
    )
    t2 = _has_any(
        fl,
        (
            r"\b(t2dm|t2d|dm2|type\s*2\s*diabet|diabetes\s*tipo\s*2)\b",
            r"\bdiabet",
        ),
    )
    if t1 and not t2:
        pop.append("type 1 diabetes")
    elif t2 or _has_any(fl, (r"\bdiabet", r"\bglucos", r"\bglucem")):
        pop.append("type 2 diabetes")
    if _has_any(fl, (r"\bhipertens", r"\bhta\b", r"\bblood\s+pressure", r"\bantihypertens")):
        pop.append("hypertension")
    if _has_any(fl, (r"\bpcos\b", r"\bpolycystic\s+ovary", r"\bsop\b")):
        pop.append("pcos")
    if _has_any(fl, (r"\bobes", r"\boverweight", r"\bbmi\b")):
        pop.append("obesity")
    if _has_any(fl, (r"\bhiv\b", r"\baids\b", r"\bvih\b", r"\bsida\b", r"\bantiretrovir")):
        pop.append("hiv")
    if _has_any(fl, (r"\bembaraz", r"\bpregnan", r"\bgestation")):
        pop.append("pregnancy")
    if age_min := (_age_from_query(fl)[0]):
        if age_min >= 65:
            pop.append("older adults")
    return _dedupe(pop)


def _interventions_from_text(fl: str) -> list[str]:
    iv: list[str] = []
    if _has_any(
        fl,
        (
            r"\bsglt2\b",
            r"\bsglt-2\b",
            r"\bgliflozin",
            r"\bempagliflozin",
            r"\bdapagliflozin",
            r"\bcanagliflozin",
            r"\binhibitor(?:es)?\s+sglt",
        ),
    ):
        iv.append("sglt2")
    if _has_any(
        fl,
        (
            r"\bglp-?1\b",
            r"\bsemaglutide",
            r"\bliraglutide",
            r"\btirzepatide",
            r"\bdulaglutide",
            r"\bagonist(?:a)?\s+glp",
        ),
    ):
        iv.append("glp1")
    if _has_any(fl, (r"\bmetformin", r"\bmetformina")):
        iv.append("metformin")
    if _has_any(fl, (r"\binsulin\b", r"\binsulina")):
        iv.append("insulin")
    if _has_any(
        fl,
        (
            r"\bdoac\b",
            r"\banticoagul",
            r"\bwarfarin",
            r"\bapixaban",
            r"\brivaroxaban",
        ),
    ):
        iv.append("anticoagulation")
    return _dedupe(iv)


def _comparator_from_text(fl: str) -> list[str]:
    comp: list[str] = []
    if _has_any(
        fl,
        (
            r"\bfrente\s+a\b",
            r"\bvs\.?\b",
            r"\bversus\b",
            r"\bcompar",
            r"\brespecto\s+a\b",
        ),
    ):
        if _has_any(fl, (r"\bmetformin", r"\bmetformina")):
            comp.append("metformin")
        if _has_any(fl, (r"\bplacebo", r"\bcontrol\b")):
            comp.append("placebo")
        if _has_any(fl, (r"\bwarfarin", r"\bdoac")):
            comp.append("warfarin")
    return _dedupe(comp)


def _outcomes_from_text(fl: str) -> list[str]:
    out: list[str] = []
    if _has_any(
        fl,
        (
            r"\bcardiovascular",
            r"\bcardiovasc",
            r"\briesgo\s+cardiovascular",
            r"\bheart\s+disease",
            r"\bcoronary",
        ),
    ):
        out.append("cardiovascular events")
    if _has_any(fl, (r"\bmace\b", r"\bmajor\s+adverse")):
        out.append("mace")
    if _has_any(fl, (r"\bmortal", r"\bdeath\b", r"\bmuerte")):
        out.append("mortality")
    if _has_any(fl, (r"\bheart\s+failure", r"\binsuficiencia\s+cardiaca", r"\bhf\b")):
        out.append("heart failure")
    if _has_any(fl, (r"\bstroke\b", r"\bictus", r"\bacv\b")):
        out.append("stroke")
    if _has_any(fl, (r"\bhba1c\b", r"\bglycemic", r"\bglucem")):
        out.append("glycemic control")
    if _has_any(
        fl,
        (
            r"\badverse\s+event",
            r"\bsafety\b",
            r"\bseguridad\b",
            r"\bhypoglyc",
            r"\bhipogluc",
            r"\btolerabil",
            r"\bside\s+effect",
        ),
    ):
        out.append("safety")
    if _has_any(
        fl,
        (
            r"\brenal\b",
            r"\bkidney\b",
            r"\bnefropat",
            r"\bckd\b",
            r"\begfr\b",
            r"\balbuminur",
            r"\briñon",
        ),
    ):
        out.append("renal outcomes")
    return _dedupe(out)


def _evidence_pref_from_text(fl: str) -> list[str]:
    pref: list[str] = []
    if _has_any(fl, (r"\brct\b", r"\brandomi", r"\bensa", r"\btrial\b")):
        pref.append("rct")
    if _has_any(fl, (r"\bmeta[-\s]?anal", r"\bsystematic\s+review", r"\brevision\s+sistem")):
        pref.append("meta-analysis")
    if _has_any(fl, (r"\bguia\b", r"\bguideline", r"\bconsensus")):
        pref.append("guideline")
    return _dedupe(pref)


_CV_OUTCOME_KEYS = frozenset(
    {
        "cardiovascular events",
        "mace",
        "mortality",
        "heart failure",
        "stroke",
    }
)


def intent_asks_cv_outcomes(intent: ClinicalIntent) -> bool:
    """True si la intención pide desenlaces cardiovasculares duros (MACE, mortalidad, etc.)."""
    return any(o.lower() in _CV_OUTCOME_KEYS for o in intent.outcomes)


_SAFETY_OUTCOME_KEYS = frozenset({"safety"})
_RENAL_OUTCOME_KEYS = frozenset({"renal outcomes"})
_GLYCEMIC_OUTCOME_KEYS = frozenset({"glycemic control"})

OutcomeTheme = Literal["cv", "safety", "renal", "glycemic", "general"]


def intent_asks_safety_outcomes(intent: ClinicalIntent) -> bool:
    return any(o.lower() in _SAFETY_OUTCOME_KEYS for o in intent.outcomes)


def intent_asks_renal_outcomes(intent: ClinicalIntent) -> bool:
    return any(o.lower() in _RENAL_OUTCOME_KEYS for o in intent.outcomes)


def primary_outcome_theme(intent: ClinicalIntent) -> OutcomeTheme:
    """
    Tema dominante de desenlace para planificación adaptativa de queries PubMed.

    Prioridad: CV > renal > safety > glycemic > general.
    """
    if intent_asks_cv_outcomes(intent):
        return "cv"
    if intent_asks_renal_outcomes(intent):
        return "renal"
    if intent_asks_safety_outcomes(intent):
        return "safety"
    if any(o.lower() in _GLYCEMIC_OUTCOME_KEYS for o in intent.outcomes):
        return "glycemic"
    return "general"


def _coerce_priority_axis(v: Any) -> IntentPriorityAxis:
    s = str(v or "").strip().lower()
    if s in ("population", "intervention", "outcome"):
        return s  # type: ignore[return-value]
    return "intervention"


def infer_priority_axis(query: str, intent: ClinicalIntent) -> IntentPriorityAxis:
    """
    Eje principal de la pregunta: población, terapia o desenlace.

    Guía si PubMed debe anclarse en enfermedad, fármaco o outcome (rerank refina el resto).
    """
    fl = _fold(query)
    scores: dict[IntentPriorityAxis, float] = {
        "population": 0.0,
        "intervention": 0.0,
        "outcome": 0.0,
    }
    scores["population"] += 1.2 * len(intent.population)
    scores["intervention"] += 1.5 * len(intent.interventions)
    scores["outcome"] += 1.5 * len(intent.outcomes)

    if re.search(
        r"\b(reduc|lower|prevent|mortality|mortalidad|mace|cardiovascular|heart failure|ictus|stroke)\b",
        fl,
    ):
        scores["outcome"] += 2.5
    if re.search(r"\b(cuantos|cohort|pacientes|poblacion|population|mayores|ancian)\b", fl):
        scores["population"] += 1.8
    if re.search(
        r"\b(sglt2|glp|metformin|metformina|inhibitor|agonist|tratam|therap|frente a|versus|vs)\b",
        fl,
    ):
        scores["intervention"] += 2.0
    if "obesity" in intent.population and re.search(r"\b(weight|peso|obes)\b", fl):
        scores["population"] += 2.0
        if intent.interventions:
            scores["intervention"] += 1.0

    if scores["outcome"] >= scores["intervention"] and scores["outcome"] >= scores["population"]:
        return "outcome"
    if scores["intervention"] >= scores["population"]:
        return "intervention"
    return "population"


def _noise_targets_for_intent(intent: ClinicalIntent) -> list[str]:
    """Poblaciones frecuentemente ruidosas si la pregunta no las invoca."""
    invited = {p.lower() for p in intent.population}
    noise: list[str] = []
    if "pcos" not in invited:
        noise.append("pcos")
    if "type 1 diabetes" not in invited and "type 2 diabetes" in invited:
        noise.append("type 1 diabetes")
    if "obesity" not in invited and "type 2 diabetes" in invited:
        noise.append("obesity")
    if "hiv" not in invited:
        noise.append("hiv")
    if "pregnancy" not in invited:
        noise.append("pregnancy")
    return noise


def extract_clinical_intent(
    query: str,
    clinical_context: Optional[Union[ClinicalContext, dict, Mapping[str, Any]]] = None,
) -> ClinicalIntent:
    """
    Extrae intención clínica desde la pregunta y, si existe, refuerza con cohorte estructurada.
    """
    q = (query or "").strip()
    fl = _fold(q)
    age_min, age_max = _age_from_query(q)

    population = _population_from_text(fl)
    interventions = _interventions_from_text(fl)
    comparator = _comparator_from_text(fl)
    outcomes = _outcomes_from_text(fl)
    evidence_preference = _evidence_pref_from_text(fl)

    intent = ClinicalIntent(
        population=population,
        interventions=interventions,
        comparator=comparator,
        outcomes=outcomes,
        evidence_preference=evidence_preference,
        age_min=age_min,
        age_max=age_max,
    )
    merged = merge_intent_with_cohort(intent, clinical_context)
    merged.priority_axis = infer_priority_axis(q, merged)
    return merged


def merge_intent_with_cohort(
    intent: ClinicalIntent,
    clinical_context: Optional[Union[ClinicalContext, dict, Mapping[str, Any]]],
) -> ClinicalIntent:
    """Fusiona señales de ``ClinicalContext`` (cohorte SQL) en la intención."""
    ctx = _as_clinical_context(clinical_context)
    if ctx is None:
        intent.population_noise = _noise_targets_for_intent(intent)
        return intent

    pop = list(intent.population)
    for c in ctx.population_conditions or []:
        cf = _fold(str(c))
        if re.search(r"diabet", cf) and "type 2 diabetes" not in pop and "type 1 diabetes" not in pop:
            pop.append("type 2 diabetes")
        if re.search(r"hipertens|hta", cf) and "hypertension" not in pop:
            pop.append("hypertension")
        if re.search(r"obes", cf) and "obesity" not in pop:
            pop.append("obesity")
        if re.search(r"pcos|polycystic", cf) and "pcos" not in pop:
            pop.append("pcos")

    # Medicación de cohorte → rerank / SQL; no se promueve a intervención PubMed automáticamente.
    iv = list(intent.interventions)

    age_min = intent.age_min
    age_max = intent.age_max
    try:
        if ctx.population_age_min is not None:
            v = int(ctx.population_age_min)
            age_min = v if age_min is None else max(age_min, v)
        if ctx.population_age_max is not None:
            age_max = int(ctx.population_age_max)
    except (TypeError, ValueError):
        pass

    pop = _dedupe(pop)
    if age_min is not None and age_min >= 65 and "older adults" not in pop:
        pop.append("older adults")

    merged = ClinicalIntent(
        population=pop,
        interventions=_dedupe(iv),
        comparator=intent.comparator,
        outcomes=intent.outcomes,
        evidence_preference=intent.evidence_preference,
        age_min=age_min,
        age_max=age_max,
    )
    merged.population_noise = _noise_targets_for_intent(merged)
    return merged
