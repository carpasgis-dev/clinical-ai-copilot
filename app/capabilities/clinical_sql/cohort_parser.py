"""
Fase 1 — cohort retrieval determinista (sin LLM, sin temporalidad avanzada).

texto libre → ``CohortQuery`` → SQL vía ``sql_builder.build_sql_from_cohort``.

Vocabulario preferente: términos desde SQLite (``load_known_*``). Bootstrap mínimo solo
si no se pasa vocabulario (tests / BD sin tablas de terminología).
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from app.capabilities.clinical_sql.terminology import fold_ascii

# ---------------------------------------------------------------------------
# Bootstrap: cubre términos frecuentes cuando la BD no tiene ese vocabulario
# (dataset pequeño que no generó esa condición) o cuando la carga falla.
# like_tok es la cadena que va en LIKE '%tok%' → debe coincidir con
# las descripciones en inglés del ETL Synthea.
# ---------------------------------------------------------------------------
_BOOTSTRAP_CONDITION_TRIGGERS: tuple[tuple[tuple[str, ...], str], ...] = (
    # Metabólicas / endocrinas
    (("diabet", "dm2", "t2dm", "glucemia alta"), "diabet"),
    (("hipotiroid", "hypothyroid", "tiroides", "tiroideo"), "hypothyroid"),
    (("hipertiroid", "hyperthyroid"), "hyperthyroid"),
    (("obesidad", "obeso", "obesa", "obesity"), "obes"),
    (("hiperlipid", "dislipem", "colesterol alto", "trigliceridos", "dyslipid"), "hyperlipid"),
    (("gota", "acido urico", "ácido úrico", "gout"), "gout"),
    # Cardiovascular (like_tok en inglés/SNOMED típico de Synthea: "Hypertension", no "hipertensión")
    (("hipertens", "hta", "tension alta", "presion alta", "hypertension"), "hypertens"),
    (("angina", "angina de pecho", "angina pectoris"), "angina"),
    (("infarto", "iam", "infarto agudo", "infarto de miocardio", "heart attack"), "infarct"),
    (("insuficiencia cardiaca", "heart failure", "insuf cardiaca", "fallo cardiaco"), "cardiac"),
    (("fibrilacion", "fibrilación", "auricular", "fibrilacion auricular"), "fibrillat"),
    (("arritmia", "arrhythmia"), "arrhythm"),
    (("ictus", "acv", "accidente cerebrovascular", "stroke", "apoplejia"), "stroke"),
    (("trombosis", "trombo", "thrombosis"), "thrombos"),
    (("embolia", "embolism", "tromboembolia"), "embol"),
    (("aterosclerosis", "arterioscler", "atheroscler"), "atheroscler"),
    (("cardiopatia", "coronary", "enfermedad coronaria", "coronaria"), "coronary"),
    # Respiratorio
    (("asma", "asmatico", "asmatica"), "asma"),
    (("epoc", "copd", "enfisema", "emphysema"), "obstruct"),
    (("neumonia", "pneumonia", "bronquitis", "bronchitis"), "pneumon"),
    (("apnea del sueno", "apnea de sueño", "sleep apnea"), "sleep apnea"),
    # Renal
    (
        ("insuficiencia renal", "renal cronica", "enfermedad renal cronica", "ckd", "chronic kidney"),
        "kidney",
    ),
    (("nefropat", "glomerulo"), "nephr"),
    # Digestivo
    (("reflujo", "erge", "gerd", "gastroesofagico", "gastroesofágico"), "gastroesophageal"),
    (("ulcera gastrica", "ulcera peptica", "peptic ulcer"), "peptic"),
    (("colitis", "crohn", "enfermedad inflamatoria"), "colitis"),
    # Musculoesquelético / autoinmune
    (("artritis", "arthritis"), "arthr"),
    (("osteoporosis",), "osteoporos"),
    (("lupus", "lep"), "lupus"),
    (("esclerosis multiple", "multiple sclerosis"), "multiple sclerosis"),
    # Mental
    (("depresion", "depresión", "depression", "deprimido"), "depress"),
    (("ansiedad", "anxiety"), "anxiet"),
    (("alzheimer", "demencia", "dementia"), "dement"),
    # Oncológico
    (("cancer", "cáncer", "neoplasia", "tumor maligno", "carcinoma", "malignant"), "malignant"),
    # Infeccioso / VIH
    (("vih", "hiv", "sida", "aids"), "hiv"),
)

_BOOTSTRAP_MED_TRIGGERS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("metform",), "metform"),
    (("aspirin", "aspirina", "acido acetilsalicilico", "ácido acetilsalicílico"), "aspir"),
    (("insulin", "insulina"), "insulin"),
    (("losartan",), "losartan"),
    (("enalapril",), "enalapril"),
    (("estatina", "statin", "atorvastat", "rosuvastat", "simvastat"), "statin"),
    (("omeprazol", "omeprazole", "pantoprazol", "lansoprazol"), "prazol"),
    (("antidepresivo", "antidepressant", "sertralina", "fluoxetina", "paroxetina"), "antidepress"),
    (("betabloqueo", "betabloqueante", "atenolol", "bisoprolol", "metoprolol"), "olol"),
    (("calcioantagonista", "amlodipino", "amlodipine", "nifedipino"), "dipine"),
    (("diuretico", "diurético", "furosemida", "hidroclorotiazida"), "furos"),
)

_COUNT_HINTS: frozenset[str] = frozenset(
    {
        "cuantos",
        "cuantas",
        "listar",
        "lista los",
        "lista las",
        "listado",
        "registros",
        "cohorte",
        "pacientes",
    }
)


def _fold(s: str) -> str:
    return fold_ascii(s)


def _sanitize_like_token(raw: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (raw or "").lower())[:40]


def _bootstrap_condition_tokens(q: str) -> list[str]:
    out: list[str] = []
    for triggers, like_tok in _BOOTSTRAP_CONDITION_TRIGGERS:
        if any(t in q for t in triggers):
            s = _sanitize_like_token(like_tok)
            if s and s not in out:
                out.append(s)
    return out


def _bootstrap_medication_tokens(q: str) -> list[str]:
    out: list[str] = []
    for triggers, like_tok in _BOOTSTRAP_MED_TRIGGERS:
        if any(t in q for t in triggers):
            s = _sanitize_like_token(like_tok)
            if s and s not in out:
                out.append(s)
    return out


def _stroke_token_likely_prevention_only(folded_q: str) -> bool:
    """
    True si «ictus/stroke» en el texto refieren a prevención (p. ej. FA + anticoagulación),
    no a antecedente patológico de evento ya padecido (que sí debe acotar la cohorte).
    """
    ql = (folded_q or "").lower()
    if re.search(r"antecedente\s+de\s+(el\s+)?(ictus|stroke|acv)\b", ql):
        return False
    if re.search(r"\b(historia|antecedentes)\s+de\s+.{0,40}\b(ictus|stroke|acv)\b", ql):
        return False
    if re.search(r"\b(ictus|stroke|acv)\s+previ", ql):
        return False
    if re.search(r"prevenci[oó]n\s+de(la?|l)?\s*(el\s+)?(ictus|stroke|acv)\b", ql):
        return True
    if re.search(r"\b(ictus|stroke|acv)\b.{0,80}prevenci[oó]n\b", ql):
        return True
    if re.search(r"prevenci[oó]n\s+.{0,80}\b(ictus|stroke|acv)\b", ql):
        return True
    return False


def _renal_token_likely_function_only(folded_q: str) -> bool:
    """True si solo se menciona función renal (p. ej. monitorización) sin ERC explícita."""
    ql = (folded_q or "").lower()
    if re.search(
        r"insuficiencia\s+renal|enfermedad\s+renal|nefropat|ckd\b|erc\b|irc\b|"
        r"cronica\s+renal|cr[oó]nica\s+renal|dialisis|d[ií]alisis|estadio\s+[1-5]",
        ql,
    ):
        return False
    return bool(re.search(r"funci[oó]n\s+renal", ql))


def _refine_condition_like_tokens_for_nl_intent(folded_q: str, tokens: Sequence[str]) -> list[str]:
    """
    Elimina tokens engañados por subcadenas del lenguaje natural o por vocabulario corto.

    - ``mellitus`` como subcadena de «diabetes mellitus» junto a ``diabet``.
    - ``stroke`` por la palabra «ictus» en «prevención de ictus» sin antecedente de evento.
    - ``kidney``/``nephr`` cuando solo aparece «función renal» sin nefropatía explícita.
    """
    out = [str(t).strip() for t in tokens if str(t).strip()]
    if not out:
        return out
    sset = {_sanitize_like_token(t) for t in out}
    if "mellitus" in sset and ("diabet" in sset or "diabetes" in sset):
        out = [t for t in out if _sanitize_like_token(t) != "mellitus"]
    if "stroke" in sset and _stroke_token_likely_prevention_only(folded_q):
        out = [t for t in out if _sanitize_like_token(t) != "stroke"]
    if ("kidney" in sset or "nephr" in sset or "renal" in sset) and _renal_token_likely_function_only(
        folded_q
    ):
        out = [t for t in out if _sanitize_like_token(t) not in ("kidney", "nephr", "renal")]
    return out


def _match_vocab_substrings(
    q: str,
    vocab: frozenset[str],
    *,
    min_term_len: int = 5,
) -> list[str]:
    picked: list[str] = []
    seen: set[str] = set()
    for term in sorted((t for t in vocab if len(t) >= min_term_len), key=len, reverse=True):
        if term in q:
            s = _sanitize_like_token(term)
            if not s or s in seen:
                continue
            seen.add(s)
            picked.append(s)
    return picked


def _query_words(q: str) -> list[str]:
    return [m.group() for m in re.finditer(r"[a-z0-9]{4,}", q)]


def _match_vocab_prefixes(q: str, vocab: frozenset[str], *, min_word: int = 5) -> list[str]:
    words = _query_words(q)
    picked: list[str] = []
    seen: set[str] = set()
    for w in words:
        if len(w) < min_word:
            continue
        for term in sorted((t for t in vocab if len(t) > len(w)), key=len, reverse=True):
            if term.startswith(w):
                s = _sanitize_like_token(term)
                if not s or s in seen:
                    continue
                seen.add(s)
                picked.append(s)
                break
    return picked


@dataclass
class CohortQuery:
    """Cohort estructurado: solo tokens LIKE seguros y límites demográficos."""

    condition_like_tokens: list[str] = field(default_factory=list)
    medication_like_tokens: list[str] = field(default_factory=list)
    age_min_years: int | None = None
    age_max_years: int | None = None
    sex: str | None = None
    count_only: bool = True
    alive_only: bool = True


def cohort_query_has_filters(q: CohortQuery) -> bool:
    return bool(
        q.condition_like_tokens
        or q.medication_like_tokens
        or q.age_min_years is not None
        or q.age_max_years is not None
        or (q.sex is not None and str(q.sex).strip() != "")
    )


def merge_cohort_queries(old: CohortQuery | None, new: CohortQuery) -> CohortQuery:
    """
    Fusión para memoria de sesión: el turno actual (``new``) añade o refina sobre ``old``.

    - Listas de tokens: unión deduplicada (condiciones y medicaciones).
    - Edad / sexo: el turno actual pisa solo si aporta valor explícito; si no, se conserva ``old``.
    - ``count_only`` / ``alive_only``: siempre los del turno actual (el NL del mensaje vigente).
    """
    if old is None:
        return CohortQuery(
            condition_like_tokens=list(new.condition_like_tokens),
            medication_like_tokens=list(new.medication_like_tokens),
            age_min_years=new.age_min_years,
            age_max_years=new.age_max_years,
            sex=new.sex,
            count_only=new.count_only,
            alive_only=new.alive_only,
        )

    merged_conds = _dedupe_like_tokens(
        list(old.condition_like_tokens) + list(new.condition_like_tokens)
    )
    merged_meds = _dedupe_like_tokens(
        list(old.medication_like_tokens) + list(new.medication_like_tokens)
    )
    age_min = new.age_min_years if new.age_min_years is not None else old.age_min_years
    age_max = new.age_max_years if new.age_max_years is not None else old.age_max_years
    sex_new = new.sex if new.sex is not None and str(new.sex).strip() != "" else None
    sex = sex_new if sex_new is not None else old.sex

    return CohortQuery(
        condition_like_tokens=merged_conds,
        medication_like_tokens=merged_meds,
        age_min_years=age_min,
        age_max_years=age_max,
        sex=sex,
        count_only=new.count_only,
        alive_only=new.alive_only,
    )


def _dedupe_like_tokens(seq: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in seq:
        sx = _sanitize_like_token(x)
        if not sx or sx in seen:
            continue
        seen.add(sx)
        out.append(sx)
    return out


def like_tokens_for_display(tokens: Sequence[str]) -> list[str]:
    """
    Tokens LIKE listos para textos (cohorte, UI, prompts): quita redundancia por prefijo.

    Si un token es prefijo estricto de otro (p. ej. ``diabet`` frente a ``diabetes``),
    se conserva solo el más largo. No altera ``CohortQuery`` ni el SQL generado.
    """
    items = [str(t).strip() for t in tokens if t and str(t).strip()]
    if len(items) <= 1:
        return items
    first_spelling: dict[str, str] = {}
    order: list[str] = []
    for t in items:
        k = t.lower()
        if k not in first_spelling:
            first_spelling[k] = t
            order.append(k)
    if len(order) <= 1:
        return [first_spelling[order[0]]]
    kept: list[str] = []
    for lo in order:
        redundant = any(
            other != lo and len(other) > len(lo) and other.startswith(lo) for other in order
        )
        if not redundant:
            kept.append(first_spelling[lo])
    return kept


# Etiquetas en español para tokens LIKE mostrados al usuario (UI / cohort_summary).
# Claves: resultado de ``_sanitize_like_token`` (minúsculas, sin espacios).
_COHORT_LIKE_TOKEN_LABEL_ES: dict[str, str] = {
    # Condiciones (alineado con bootstrap + vocabulario frecuente)
    "diabet": "diabetes",
    "diabetes": "diabetes",
    "hypothyroid": "hipotiroidismo",
    "hyperthyroid": "hipertiroidismo",
    "obes": "obesidad",
    "hyperlipid": "dislipemia",
    "gout": "gota",
    "hypertens": "hipertensión",
    "hypertension": "hipertensión",
    "angina": "angina de pecho",
    "infarct": "infarto de miocardio",
    "cardiac": "insuficiencia cardiaca",
    "fibrillat": "fibrilación auricular",
    "arrhythm": "arritmia",
    "stroke": "ictus",
    "thrombos": "trombosis",
    "embol": "embolia",
    "atheroscler": "ateroesclerosis",
    "coronary": "enfermedad coronaria",
    "asma": "asma",
    "obstruct": "EPOC",
    "pneumon": "neumonía",
    "sleepapnea": "apnea del sueño",
    "kidney": "enfermedad renal crónica",
    "nephr": "nefropatía",
    "gastroesophageal": "reflujo gastroesofágico",
    "peptic": "úlcera péptica",
    "colitis": "enfermedad inflamatoria intestinal",
    "arthr": "artritis",
    "osteoporos": "osteoporosis",
    "lupus": "lupus",
    "multiplesclerosis": "esclerosis múltiple",
    "depress": "depresión",
    "anxiet": "ansiedad",
    "dement": "demencia",
    "malignant": "neoplasia maligna",
    "hiv": "VIH",
    # Medicación
    "metform": "metformina",
    "metformin": "metformina",
    "aspir": "aspirina",
    "insulin": "insulina",
    "losartan": "losartán",
    "enalapril": "enalapril",
    "statin": "estatinas",
    "prazol": "inhibidor de la bomba de protones",
    "antidepress": "antidepresivos",
    "olol": "betabloqueantes",
    "dipine": "calcioantagonistas",
    "furos": "diuréticos",
    "semaglutide": "semaglutida",
}


def humanize_like_tokens_es(tokens: Sequence[str]) -> list[str]:
    """
    Convierte tokens LIKE internos en etiquetas legibles en español para textos de cohorte.

    No afecta al SQL ni a ``CohortQuery``; solo a ``population_*`` en contexto clínico.
    """
    out: list[str] = []
    for raw in tokens:
        t = str(raw).strip()
        if not t:
            continue
        key = _sanitize_like_token(t)
        label = _COHORT_LIKE_TOKEN_LABEL_ES.get(key)
        if label:
            out.append(label)
        elif len(t) <= 1:
            out.append(t.upper())
        else:
            out.append(t[0].upper() + t[1:].lower())
    return out


def parse_cohort_query(
    text: str,
    *,
    known_condition_terms: frozenset[str] | None = None,
    known_medication_terms: frozenset[str] | None = None,
) -> CohortQuery:
    """
    ``known_*``: vocabulario desde SQLite; si es ``None``, se usa solo bootstrap mínimo.
    """
    q = _fold(text or "")

    conds: list[str] = []
    if known_condition_terms:
        conds.extend(_match_vocab_substrings(q, known_condition_terms))
        conds.extend(_match_vocab_prefixes(q, known_condition_terms))
    # Siempre fusionar triggers bootstrap (ES↔EN): si el vocabulario ya encontró p. ej. diabetes,
    # sin esto no se añaden otras señales del texto (p. ej. hipertensión → ``hypertens`` en SQL) que no matchean el lexicon.
    conds.extend(_bootstrap_condition_tokens(q))

    meds: list[str] = []
    if known_medication_terms:
        meds.extend(_match_vocab_substrings(q, known_medication_terms))
        meds.extend(_match_vocab_prefixes(q, known_medication_terms))
    meds.extend(_bootstrap_medication_tokens(q))

    min_age: int | None = None
    max_age: int | None = None
    for pat in (
        r"mayores?\s+de\s+(\d{1,3})\b",
        r"mas\s+de\s+(\d{1,3})\b",
        r">\s*(\d{1,3})\b",
        r"desde\s+los?\s+(\d{1,3})\b",
    ):
        m = re.search(pat, q)
        if m:
            v = int(m.group(1))
            if 0 <= v <= 120:
                min_age = v
            break
    for pat in (
        r"menores?\s+de\s+(\d{1,3})\b",
        r"menos\s+de\s+(\d{1,3})\b",
        r"<\s*(\d{1,3})\b",
    ):
        m = re.search(pat, q)
        if m:
            v = int(m.group(1))
            if 0 <= v <= 120:
                max_age = v
            break
    m = re.search(r"\b(\d{1,3})\s*a(?:n|ñ)os?\b", q)
    if m and min_age is None and max_age is None:
        v = int(m.group(1))
        if 0 <= v <= 120:
            min_age = v

    sex: str | None = None
    if any(
        x in q
        for x in (
            "mujeres",
            "mujer",
            "femenino",
            "femenina",
            "sexo femenino",
            "pacientes mujeres",
        )
    ):
        sex = "F"
    elif any(
        x in q
        for x in (
            "hombres",
            "hombre",
            "masculino",
            "masculina",
            "sexo masculino",
            "pacientes hombres",
        )
    ):
        sex = "M"

    count_only = any(h in q for h in _COUNT_HINTS) or "cuant" in q

    conds = _refine_condition_like_tokens_for_nl_intent(q, conds)

    return CohortQuery(
        condition_like_tokens=_dedupe_like_tokens(conds),
        medication_like_tokens=_dedupe_like_tokens(meds),
        age_min_years=min_age,
        age_max_years=max_age,
        sex=sex,
        count_only=count_only,
        alive_only=True,
    )
