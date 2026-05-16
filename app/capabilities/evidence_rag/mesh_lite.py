"""
Expansiones deterministas tipo MeSH / términos PubMed (sin LLM, sin UMLS).
Mapea tokens de cohorte (p. ej. ``diabet``) a fragmentos de query más recuperables.
Mejorado para usar [MeSH Terms] y evitar el sobreajuste con [tiab].
"""
from __future__ import annotations

from app.capabilities.clinical_sql.terminology import fold_ascii

# Mapa cohorte → PubMed enriquecido con [MeSH]
TOKEN_TO_EVIDENCE_PHRASE: dict[str, str] = {
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
    
    "hipertens": "(\"Hypertension\"[Mesh] OR hypertension[tiab] OR \"blood pressure\"[tiab] OR hypertensive[tiab])",
    "hypertension": "(\"Hypertension\"[Mesh] OR hypertension[tiab] OR \"blood pressure\"[tiab] OR hypertensive[tiab])",
    "hipertension": "(\"Hypertension\"[Mesh] OR hypertension[tiab] OR \"blood pressure\"[tiab] OR hypertensive[tiab])",

    "asma": "(\"Asthma\"[Mesh] OR asthma[tiab] OR asthmatic[tiab])",
    "obstruct": "(\"Pulmonary Disease, Chronic Obstructive\"[Mesh] OR COPD[tiab] OR epoc[tiab])",
    "cardiac": "(\"Heart Failure\"[Mesh] OR \"cardiac insufficiency\"[tiab] OR \"heart failure\"[tiab])",

    "metform": "(\"Metformin\"[Mesh] OR metformin[tiab] OR metform*[tiab])",
    "metformin": "(\"Metformin\"[Mesh] OR metformin[tiab] OR metform*[tiab])",
    "aspir": "(\"Aspirin\"[Mesh] OR aspirin[tiab] OR \"acetylsalicylic acid\"[tiab])",
    "insulin": "(\"Insulin\"[Mesh] OR insulin[tiab] OR insul*[tiab])",
    "losartan": "(\"Losartan\"[Mesh] OR losartan[tiab])",
    "enalapril": "(\"Enalapril\"[Mesh] OR enalapril[tiab])",
    
    "fibrillat": "(\"Atrial Fibrillation\"[Mesh] OR \"atrial fibrillation\"[tiab] OR \"atrial flutter\"[tiab])",
    "warfarin": "(\"Warfarin\"[Mesh] OR warfarin[tiab] OR \"vitamin K antagonist\"[tiab])",
    "hyperlipid": "(\"Hyperlipidemias\"[Mesh] OR hyperlipidemia[tiab] OR dyslipidemia[tiab] OR \"blood lipids\"[tiab])",
    
    # Adiciones para mejorar drogas modernas
    "sglt2": "(\"Sodium-Glucose Transporter 2 Inhibitors\"[Mesh] OR \"SGLT2 inhibitors\"[tiab] OR SGLT2[tiab] OR gliflozin*[tiab] OR empagliflozin[tiab] OR dapagliflozin[tiab] OR canagliflozin[tiab])",
    "glp1": "(\"Glucagon-Like Peptide-1 Receptor\"[Mesh] OR \"GLP-1 receptor agonists\"[tiab] OR GLP-1[tiab] OR semaglutide[tiab] OR liraglutide[tiab] OR dulaglutide[tiab])",
    
    # Renal / CKD
    "renal": "(\"Renal Insufficiency, Chronic\"[Mesh] OR \"chronic kidney disease\"[tiab] OR CKD[tiab] OR \"renal failure\"[tiab])",
    "ckd": "(\"Renal Insufficiency, Chronic\"[Mesh] OR \"chronic kidney disease\"[tiab] OR CKD[tiab] OR \"renal failure\"[tiab])",
    
    # Población
    "older adult": "(\"Aged\"[Mesh] OR \"older adult\"[tiab] OR \"older adults\"[tiab] OR elderly[tiab])",
    "elderly": "(\"Aged\"[Mesh] OR elderly[tiab] OR \"older adults\"[tiab])",
}

_TOKEN_BY_FOLD: dict[str, str] = {
    fold_ascii(k.lower()): v for k, v in TOKEN_TO_EVIDENCE_PHRASE.items()
}


def _like_key_for_humanized_label(label: str) -> str | None:
    from app.capabilities.clinical_sql.cohort_parser import _COHORT_LIKE_TOKEN_LABEL_ES
    fl = fold_ascii((label or "").strip().lower())
    if not fl:
        return None
    for key, es_label in _COHORT_LIKE_TOKEN_LABEL_ES.items():
        if fold_ascii(es_label) == fl:
            return key
    return None


def expand_cohort_token_for_pubmed(token: str, is_drug: bool = False) -> str:
    """Devuelve frase PubMed expandida (MeSH + tiab)."""
    raw = (token or "").strip().lower()
    if not raw:
        return ""
        
    t = fold_ascii(raw)

    # Frases PICO completas (evita type2diabetes[tiab] al quitar espacios)
    if "type 2" in raw and "diabet" in raw:
        return TOKEN_TO_EVIDENCE_PHRASE["diabet"]
    if "type 1" in raw and "diabet" in raw:
        return "(\"Diabetes Mellitus, Type 1\"[Mesh] OR \"type 1 diabetes\"[tiab] OR T1DM[tiab])"
    
    # Intercept common bad mappings from NLP
    if t.startswith("type2diabet"):
        return TOKEN_TO_EVIDENCE_PHRASE["diabet"]
    
    if t in _TOKEN_BY_FOLD:
        return _TOKEN_BY_FOLD[t]
        
    if raw in TOKEN_TO_EVIDENCE_PHRASE:
        return TOKEN_TO_EVIDENCE_PHRASE[raw]
        
    like_key = _like_key_for_humanized_label(raw)
    if like_key and like_key != t:
        return expand_cohort_token_for_pubmed(like_key, is_drug)
        
    safe = "".join(c for c in t if c.isalnum() or c == '-')
    if len(safe) < 3:
        return ""
        
    # Denylist explícita para evitar que stopwords NLP formen queries basura
    _DENYLIST = frozenset({"vs", "alone", "therapy", "treatment", "placebo", "control", "drug", "medicine", "pill", "intervention"})
    if safe in _DENYLIST:
        return ""
        
    # Fallback relaxado: NO inventar [Mesh] para que PubMed no lance de-mappings extraños
    # Si no conocemos el string, buscamos text word literal para que gane recall sin ruido categórico
    if is_drug:
        return f"{safe}[tiab]"
    return f"{safe}[tiab]"

