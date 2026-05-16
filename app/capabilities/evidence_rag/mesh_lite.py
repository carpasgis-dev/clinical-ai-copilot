"""
Expansiones deterministas tipo MeSH / términos PubMed (sin LLM, sin UMLS).

Mapea tokens de cohorte (p. ej. ``diabet``) a fragmentos de query más recuperables.
"""
from __future__ import annotations

from app.capabilities.clinical_sql.terminology import fold_ascii

# Mapa cohorte → PubMed. Para una segunda línea MeSH (p. ej. Humans[MeSH]) vía env, ver docs en README / .env.example.
TOKEN_TO_EVIDENCE_PHRASE: dict[str, str] = {
    "diabet": "(\"diabetes mellitus\"[tiab] OR diabet*[tiab])",
    "diabetes": "(\"diabetes mellitus\"[tiab] OR diabet*[tiab])",
    # Subcadena de «diabetes mellitus» a veces matcheada como término suelto en vocabulario/NL.
    "mellitus": "(\"diabetes mellitus\"[tiab] OR diabet*[tiab])",
    "hipertens": "(hypertension[tiab] OR \"blood pressure\"[tiab])",
    "hypertension": "(hypertension[tiab] OR \"blood pressure\"[tiab])",
    "asma": "(asthma[tiab] OR asthm*[tiab])",
    "obstruct": "(\"pulmonary disease, chronic obstructive\"[tiab] OR COPD[tiab] OR epoc[tiab])",
    "cardiac": "(\"heart failure\"[tiab] OR \"cardiac insufficiency\"[tiab])",
    "metform": "(metformin[tiab] OR metform*[tiab])",
    "metformin": "(metformin[tiab] OR metform*[tiab])",
    "aspir": "(aspirin[tiab] OR \"acetylsalicylic acid\"[tiab])",
    "insulin": "(insulin[tiab] OR insul*[tiab])",
    "losartan": "(losartan[tiab])",
    "enalapril": "(enalapril[tiab])",
    "fibrillat": "(\"atrial fibrillation\"[tiab] OR \"atrial flutter\"[tiab])",
    "warfarin": "(warfarin[tiab] OR \"vitamin K antagonist\"[tiab])",
    "hyperlipid": "(hyperlipidemia[tiab] OR dyslipidemia[tiab] OR \"blood lipids\"[tiab])",
    # Etiquetas ES humanizadas (fold → clave ASCII)
    "hipertension": "(hypertension[tiab] OR \"blood pressure\"[tiab])",
}

_TOKEN_BY_FOLD: dict[str, str] = {
    fold_ascii(k.lower()): v for k, v in TOKEN_TO_EVIDENCE_PHRASE.items()
}


def _like_key_for_humanized_label(label: str) -> str | None:
    """Si ``label`` es etiqueta ES de cohorte (p. ej. «fibrilación auricular»), devuelve clave LIKE."""
    from app.capabilities.clinical_sql.cohort_parser import _COHORT_LIKE_TOKEN_LABEL_ES

    fl = fold_ascii((label or "").strip().lower())
    if not fl:
        return None
    for key, es_label in _COHORT_LIKE_TOKEN_LABEL_ES.items():
        if fold_ascii(es_label) == fl:
            return key
    return None


def expand_cohort_token_for_pubmed(token: str) -> str:
    """Devuelve frase PubMed para un token de cohorte; si no hay mapa, escapa mínimo tiab."""
    raw = (token or "").strip().lower()
    if not raw:
        return ""
    t = fold_ascii(raw)
    if t in _TOKEN_BY_FOLD:
        return _TOKEN_BY_FOLD[t]
    if raw in TOKEN_TO_EVIDENCE_PHRASE:
        return TOKEN_TO_EVIDENCE_PHRASE[raw]
    like_key = _like_key_for_humanized_label(raw)
    if like_key and like_key != t:
        return expand_cohort_token_for_pubmed(like_key)
    # fallback conservador: solo wildcard tiab sobre token saneado (ASCII)
    safe = "".join(c for c in t if c.isalnum())[:30]
    if len(safe) < 3:
        return ""
    return f"({safe}[tiab] OR {safe}*[tiab])"
