"""
Post-proceso de la salida del LLM para líneas PubMed.

Lógica alineada con ``prsn30.pubmed.query_planner`` (PRSN 3.0), sin dependencia del paquete.
"""
from __future__ import annotations

import os
import re

_SYSTEM = """You are a PubMed search expert. Your task: translate a SPANISH clinical question into ONE English PubMed search string.

⚠️ OUTPUT LANGUAGE: ENGLISH ONLY. Every word in your output must be in English. Never output Spanish words.

FORMAT: ONE line, PubMed syntax, nothing else.
- Combine 2–3 clinical concepts with AND.
- Use OR inside parentheses for English synonyms.
- Correct example: ("chronic musculoskeletal pain" OR "chronic pain") AND ("psychological intervention" OR "cognitive behavioral therapy" OR "stress management")
- Wrong example (Spanish): ("intervenciones psicológicas") AND ("dolor crónico")  ← FORBIDDEN

RULES:
- All terms must be in ENGLISH. Translate from Spanish before writing.
- Maximum 3 AND groups. Each group: 1–3 English phrases in parentheses.
- Extract keywords ONLY from the clinical question, not from the patient demographics.
- No Spanish words. No JSON. No labels. No explanation. No period at end. ONE LINE ONLY."""


def env_int(name: str, default: int) -> int:
    try:
        return max(0, int(os.getenv(name, str(default)).strip()))
    except ValueError:
        return default


def clip_text(s: str, max_chars: int) -> str:
    s = (s or "").strip()
    if max_chars <= 0 or len(s) <= max_chars:
        return s
    return s[: max_chars - 3].rstrip() + "..."


def strip_code_fence(raw: str) -> str:
    t = raw.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t).strip()
    return t


def pubmed_line_from_llm_text(raw: str) -> str:
    """Toma la primera línea útil del modelo (sin JSON)."""
    t = strip_code_fence(raw)
    if not t:
        return ""
    for line in t.splitlines():
        s = line.strip()
        if not s:
            continue
        s = re.sub(r"^[-*•]\s*", "", s)
        s = re.sub(
            r"^(Query|Search|PubMed|Keywords|Terms)\s*:\s*",
            "",
            s,
            flags=re.I,
        )
        s = s.strip('"\'`')
        if s:
            return s
    return ""


def refine_llm_pubmed_keywords(line: str, *, max_words: int = 6) -> str:
    """
    Limpia la línea del LLM para evitar 0 hits en PubMed:
    – elimina contexto social/laboral que el modelo copia del expediente
    – si la query ya tiene paréntesis/AND/OR explícitos (modelo estructurado), la respeta
    – si son palabras sueltas, recorta a máx. `max_words` palabras
    """
    s = (line or "").strip()
    if not s:
        return ""
    s = re.split(r"[.;]", s, maxsplit=1)[0].strip()

    _noise = [
        r"(?i)\bfull[- ]?time\s+employment\b",
        r"(?i)\bpart[- ]?time\s+employment\b",
        r"(?i)\bwith\s+full[- ]?time\b",
        r"(?i)\bwith\s+part[- ]?time\b",
        r"(?i)\b(young|working|employed)\s+adults?\b",
        r"(?i)\band\s+full[- ]?time\b",
        r"(?i)\bemployment\b",
        r"(?i)\bjob\s+stress\b",
        r"(?i)\boccupational\s+stress\b(?!.*management)",
        r"(?i)\bhousing\b",
        r"(?i)\beducation\s+level\b",
        r"(?i)\bhigher\s+education\b",
        r"(?i)\bsocial\s+isolation\b",
    ]
    for pat in _noise:
        s = re.sub(pat, " ", s)
    s = re.sub(r"(?i)\bunemployed\b", " ", s)
    s = re.sub(r"(?i)\bunemployment\b", " ", s)

    s = " ".join(s.split())

    if "(" in s or " AND " in s.upper() or " OR " in s.upper():
        return s

    s = re.sub(r"(?i)^\s*(in|with|and|or|for|of)\b", "", s)
    s = re.sub(r"(?i)\b(in|with|and|or|for|of)\s*$", "", s)
    s = " ".join(s.split())

    words = s.split()
    if len(words) > max_words:
        words = words[:max_words]
    return " ".join(words)


def coerce_pubmed_line_for_esearch(line: str, *, max_and_words: int = 5) -> str:
    """
    Si el modelo ya devolvió una query con paréntesis/AND/OR, la respeta.
    Si devolvió palabras sueltas sin operadores:
    - ≤ max_and_words: AND implícito (PubMed).
    - > max_and_words: dos grupos AND con OR interno.
    """
    s = " ".join((line or "").split())
    if not s:
        return ""
    if "(" in s or " AND " in s.upper() or " OR " in s.upper():
        return s
    words = s.split()
    if len(words) <= max_and_words:
        return s
    mid = len(words) // 2
    g1 = " OR ".join(words[:mid])
    g2 = " OR ".join(words[mid:])
    return f"({g1}) AND ({g2})"


def sanitize_pubmed_term(s: str, *, max_len: int = 400) -> str:
    s = " ".join(s.split())
    s = s.strip(' \t\n\r"\'`\u201c\u201d')
    if len(s) > max_len:
        s = s[:max_len].rstrip()
    return s


_SPANISH_MARKERS = re.compile(
    r"\b(intervenciones|psicológicas|psicologicas|manejo|estrés|estres|"
    r"dolor|crónico|cronico|musculoesquelético|musculoesqueletico|"
    r"terapia|cognitivo|conductual|aceptación|aceptacion|"
    r"ansiedad|bienestar|ejercicio|rehabilitación|rehabilitacion)\b",
    re.I,
)


def is_spanish_pubmed_line(text: str) -> bool:
    """Detecta si la query contiene palabras españolas típicas (PubMed en inglés)."""
    return bool(_SPANISH_MARKERS.search(text or ""))


def finalize_llm_pubmed_line(raw_line: str) -> str:
    """Cadena lista para esearch a partir de la primera línea extraída del modelo."""
    refined = refine_llm_pubmed_keywords(pubmed_line_from_llm_text(raw_line))
    return sanitize_pubmed_term(coerce_pubmed_line_for_esearch(refined))
