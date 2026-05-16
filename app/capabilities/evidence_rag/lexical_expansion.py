"""
Puente determinista cohorte/token → frase PubMed (MeSH + tiab).

Solo expansión léxica: enfermedades, fármacos y tokens de cohorte.
Sin conceptos terapéuticos compuestos (DOAC, MACE) ni política de evidencia.
"""
from __future__ import annotations

from app.capabilities.clinical_sql.terminology import fold_ascii

# Token de cohorte / etiqueta corta → cláusula PubMed curada
LEXICAL_TOKEN_TO_PUBMED: dict[str, str] = {
    "diabet": (
        "(\"Diabetes Mellitus, Type 2\"[Mesh] OR \"Diabetes Mellitus\"[Mesh] "
        'OR "type 2 diabetes"[tiab] OR T2DM[tiab] OR diabetes[tiab] OR diabetic[tiab])'
    ),
    "diabetes": (
        "(\"Diabetes Mellitus, Type 2\"[Mesh] OR \"Diabetes Mellitus\"[Mesh] "
        'OR "type 2 diabetes"[tiab] OR T2DM[tiab] OR diabetes[tiab] OR diabetic[tiab])'
    ),
    "mellitus": (
        "(\"Diabetes Mellitus, Type 2\"[Mesh] OR \"Diabetes Mellitus\"[Mesh] "
        'OR "type 2 diabetes"[tiab] OR T2DM[tiab] OR diabetes[tiab] OR diabetic[tiab])'
    ),
    "hipertens": (
        "(\"Hypertension\"[Mesh] OR hypertension[tiab] OR "
        '"blood pressure"[tiab] OR hypertensive[tiab])'
    ),
    "hypertension": (
        "(\"Hypertension\"[Mesh] OR hypertension[tiab] OR "
        '"blood pressure"[tiab] OR hypertensive[tiab])'
    ),
    "hipertension": (
        "(\"Hypertension\"[Mesh] OR hypertension[tiab] OR "
        '"blood pressure"[tiab] OR hypertensive[tiab])'
    ),
    "asma": "(\"Asthma\"[Mesh] OR asthma[tiab] OR asthmatic[tiab])",
    "obstruct": (
        "(\"Pulmonary Disease, Chronic Obstructive\"[Mesh] OR COPD[tiab] OR epoc[tiab])"
    ),
    "cardiac": (
        "(\"Heart Failure\"[Mesh] OR \"cardiac insufficiency\"[tiab] OR \"heart failure\"[tiab])"
    ),
    "metform": "(\"Metformin\"[Mesh] OR metformin[tiab] OR metform*[tiab])",
    "metformin": "(\"Metformin\"[Mesh] OR metformin[tiab] OR metform*[tiab])",
    "aspir": "(\"Aspirin\"[Mesh] OR aspirin[tiab] OR \"acetylsalicylic acid\"[tiab])",
    "insulin": "(\"Insulin\"[Mesh] OR insulin[tiab] OR insul*[tiab])",
    "losartan": "(\"Losartan\"[Mesh] OR losartan[tiab])",
    "enalapril": "(\"Enalapril\"[Mesh] OR enalapril[tiab])",
    "fibrillat": (
        "(\"Atrial Fibrillation\"[Mesh] OR \"atrial fibrillation\"[tiab] OR \"atrial flutter\"[tiab])"
    ),
    "warfarin": "(\"Warfarin\"[Mesh] OR warfarin[tiab] OR \"vitamin K antagonist\"[tiab])",
    "apixaban": "(\"Apixaban\"[Mesh] OR apixaban[tiab])",
    "rivaroxaban": "(\"Rivaroxaban\"[Mesh] OR rivaroxaban[tiab])",
    "dabigatran": "(\"Dabigatran\"[Mesh] OR dabigatran[tiab])",
    "edoxaban": "(\"Edoxaban\"[Mesh] OR edoxaban[tiab])",
    "stroke": "(\"stroke\"[Mesh] OR stroke[tiab] OR \"cerebrovascular accident\"[tiab])",
    "hyperlipid": (
        "(\"Hyperlipidemias\"[Mesh] OR hyperlipidemia[tiab] OR dyslipidemia[tiab] OR "
        '"blood lipids"[tiab])'
    ),
    "sglt2": (
        "(\"Sodium-Glucose Transporter 2 Inhibitors\"[Mesh] OR \"SGLT2 inhibitors\"[tiab] "
        "OR SGLT2[tiab] OR gliflozin*[tiab] OR empagliflozin[tiab] OR dapagliflozin[tiab] "
        "OR canagliflozin[tiab])"
    ),
    "glp1": (
        "(\"Glucagon-Like Peptide-1 Receptor\"[Mesh] OR \"GLP-1 receptor agonists\"[tiab] "
        "OR GLP-1[tiab] OR semaglutide[tiab] OR liraglutide[tiab] OR dulaglutide[tiab])"
    ),
    "renal": (
        "(\"Renal Insufficiency, Chronic\"[Mesh] OR \"chronic kidney disease\"[tiab] "
        'OR CKD[tiab] OR "renal failure"[tiab])'
    ),
    "ckd": (
        "(\"Renal Insufficiency, Chronic\"[Mesh] OR \"chronic kidney disease\"[tiab] "
        'OR CKD[tiab] OR "renal failure"[tiab])'
    ),
}

_LEXICAL_BY_FOLD: dict[str, str] = {
    fold_ascii(k.lower()): v for k, v in LEXICAL_TOKEN_TO_PUBMED.items()
}

_QUERY_DENYLIST = frozenset(
    {
        "vs",
        "alone",
        "therapy",
        "treatment",
        "placebo",
        "control",
        "drug",
        "medicine",
        "pill",
        "intervention",
    }
)


def _like_key_for_humanized_label(label: str) -> str | None:
    from app.capabilities.clinical_sql.cohort_parser import _COHORT_LIKE_TOKEN_LABEL_ES

    fl = fold_ascii((label or "").strip().lower())
    if not fl:
        return None
    for key, es_label in _COHORT_LIKE_TOKEN_LABEL_ES.items():
        if fold_ascii(es_label) == fl:
            return key
    return None


def expand_lexical_token_for_pubmed(token: str, *, is_drug: bool = False) -> str:
    """Expansión léxica única: token → MeSH/tiab. Sin conceptos clínicos compuestos."""
    raw = (token or "").strip().lower()
    if not raw:
        return ""

    t = fold_ascii(raw)

    if "type 2" in raw and "diabet" in raw:
        return LEXICAL_TOKEN_TO_PUBMED["diabet"]
    if "type 1" in raw and "diabet" in raw:
        return '(\"Diabetes Mellitus, Type 1\"[Mesh] OR \"type 1 diabetes\"[tiab] OR T1DM[tiab])'

    if t.startswith("type2diabet"):
        return LEXICAL_TOKEN_TO_PUBMED["diabet"]

    if t in _LEXICAL_BY_FOLD:
        return _LEXICAL_BY_FOLD[t]

    if raw in LEXICAL_TOKEN_TO_PUBMED:
        return LEXICAL_TOKEN_TO_PUBMED[raw]

    like_key = _like_key_for_humanized_label(raw)
    if like_key and like_key != t:
        return expand_lexical_token_for_pubmed(like_key, is_drug=is_drug)

    safe = "".join(c for c in t if c.isalnum() or c == "-")
    if len(safe) < 3:
        return ""

    if safe in _QUERY_DENYLIST:
        return ""

    return f"{safe}[tiab]"
