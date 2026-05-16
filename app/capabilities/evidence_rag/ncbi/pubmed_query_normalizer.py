"""
Normalización defensiva de términos PubMed antes de ESearch.

Objetivo: reducir HTTP 400 por mezclas fielded/unfielded, comillas Unicode,
paréntesis desbalanceados u operadores booleanos degenerados — sin LLM.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Tuple

# Campo por defecto para comillas sin etiqueta (coherencia con OR/AND fielded).
_DEFAULT_FIELD = "tiab"

_SMART_QUOTES = str.maketrans(
    {
        "\u201c": '"',
        "\u201d": '"',
        "\u2018": "'",
        "\u2019": "'",
        "\u00ab": '"',
        "\u00bb": '"',
    }
)


def sanitize_unicode(s: str) -> str:
    t = unicodedata.normalize("NFKC", s).translate(_SMART_QUOTES)
    t = re.sub(r"[\u200b-\u200d\ufeff]", "", t)
    return t


def collapse_boolean_ops(s: str) -> str:
    t = s
    t = re.sub(r"\bAND\s+AND\b", "AND", t, flags=re.IGNORECASE)
    t = re.sub(r"\bOR\s+OR\b", "OR", t, flags=re.IGNORECASE)
    t = re.sub(r"\bAND\s*\)", ")", t, flags=re.IGNORECASE)
    t = re.sub(r"\(\s*AND\b", "(", t, flags=re.IGNORECASE)
    t = re.sub(r"\bOR\s*\)", ")", t, flags=re.IGNORECASE)
    t = re.sub(r"\(\s*OR\b", "(", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def balance_parentheses(s: str) -> Tuple[str, List[str]]:
    notes: List[str] = []
    o = s.count("(")
    c = s.count(")")
    if o == c:
        return s, notes
    if o > c:
        need = o - c
        notes.append(f"equilibrio: añadidos {need} ')' finales")
        return s + (")" * need), notes
    excess = c - o
    fix = s.rstrip()
    removed = 0
    while excess > 0 and fix.endswith(")"):
        fix = fix[:-1].rstrip()
        excess -= 1
        removed += 1
    if excess > 0:
        notes.append(f"equilibrio: añadidos {excess} '(' iniciales")
        fix = ("(" * excess) + fix
    else:
        notes.append(f"equilibrio: eliminados {removed} ')' sobrantes")
    return fix.strip(), notes


def _tag_unfielded_quoted_phrases(s: str, field: str = _DEFAULT_FIELD) -> str:
    """
    Añade [field] a cada \"...\" que no vaya seguido ya de [tag].

    Evita mezclas tipo (\"a\"[tiab] OR \"b\") que a veces provocan 400 en ESearch.
    No toca cadenas que parecen solo año (4 dígitos).

    El cierre de una frase entre comillas va seguido de ``[`` si ya tiene campo;
    sin lookbehind, el ``\"`` de cierre se confundía con apertura (``\"...[tiab] OR \"``).
    """
    field_l = field.lower()
    # Apertura de comillas: no inmediatamente tras letra/dígito (evita el " de cierre de "mellitus").
    pat = re.compile(r'(?<![A-Za-z0-9])"([^"]+)"(\s*)(?!\[)')

    def repl(m: re.Match[str]) -> str:
        inner = m.group(1)
        sp = m.group(2) or ""
        if not inner.strip():
            return m.group(0)
        if re.fullmatch(r"\d{4}", inner.strip()):
            return m.group(0)
        # Ya viene con sintaxis de campo PubMed (p. ej. re-normalizar una query compuesta).
        if "[" in inner:
            return m.group(0)
        return f'"{inner}"[{field_l}]{sp}'

    return pat.sub(repl, s)


def normalize_pubmed_query(raw: str) -> Tuple[str, Dict[str, Any]]:
    """
    Devuelve (query_normalizada, meta).

    meta incluye: warnings (list[str]), steps_applied (list[str]).
    """
    warnings: List[str] = []
    steps: List[str] = []
    t = (raw or "").strip()
    if not t:
        return "", {"warnings": warnings, "steps_applied": steps}

    t0 = sanitize_unicode(t)
    if t0 != t:
        steps.append("sanitize_unicode")

    t1 = collapse_boolean_ops(t0)
    if t1 != t0:
        steps.append("collapse_boolean_ops")

    t2, pnotes = balance_parentheses(t1)
    warnings.extend(pnotes)
    if t2 != t1:
        steps.append("balance_parentheses")

    t3 = _tag_unfielded_quoted_phrases(t2)
    if t3 != t2:
        steps.append("tag_unfielded_quoted_phrases")

    t4 = collapse_boolean_ops(t3)
    t4 = re.sub(r"\s+", " ", t4).strip()
    if t4 != t3:
        steps.append("final_collapse_whitespace")

    return t4, {"warnings": warnings, "steps_applied": steps}


def boolean_max_depth(s: str) -> int:
    d = 0
    best = 0
    for ch in s:
        if ch == "(":
            d += 1
            best = max(best, d)
        elif ch == ")":
            d = max(0, d - 1)
    return best


def retrieval_metrics_for_query(normalized: str) -> Dict[str, Any]:
    """Métricas heurísticas solo a partir del texto (sin llamar a NCBI)."""
    q = (normalized or "").strip()
    if not q:
        return {
            "query_complexity": 0.0,
            "boolean_depth": 0,
            "estimated_specificity": "unknown",
        }
    n_and = len(re.findall(r"\bAND\b", q, flags=re.IGNORECASE))
    n_or = len(re.findall(r"\bOR\b", q, flags=re.IGNORECASE))
    n_paren = q.count("(") + q.count(")")
    depth = boolean_max_depth(q)
    score = (n_and * 2 + n_or + n_paren * 0.5) / max(len(q) / 40.0, 1.0)
    complexity = max(0.0, min(1.0, score / 8.0))

    if n_and >= 4 or (n_and >= 2 and depth >= 3):
        spec = "high"
    elif n_and >= 2 or len(q) > 120:
        spec = "medium"
    else:
        spec = "low"

    return {
        "query_complexity": round(complexity, 3),
        "boolean_depth": depth,
        "estimated_specificity": spec,
        "operator_counts": {"and": n_and, "or": n_or},
    }
