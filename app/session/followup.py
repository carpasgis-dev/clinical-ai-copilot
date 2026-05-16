"""
Fase 3.5 — detección heurística de follow-up conversacional (sin LLM).

Sirve para decidir si se fusiona la cohorte previa de sesión con el parse del turno actual.
"""
from __future__ import annotations

import re


# Consultas que suelen ser autocontenidas (nuevo tema): no forzar merge por brevedad.
_NEW_TOPIC_HINT = re.compile(
    r"cuantos|cuántos|cuantas|cuántas|listar|lista\s+los|lista\s+las|"
    r"en nuestra base|en nuestra cohorte|cuantos tenemos|cuántos tenemos|"
    r"distribuci|prevalencia|comorbilidad",
    re.IGNORECASE,
)


def is_followup_query(text: str) -> bool:
    """
    True si el texto parece ampliar o refinar la consulta anterior (referencia implícita).

    Heurística por prefijos y longitud; evita fusionar cohortes en preguntas nuevas largas.
    """
    raw = (text or "").strip()
    if not raw:
        return False
    t = raw.lower()
    # Prefijos típicos de continuación (ES).
    if re.match(
        r"^[\s¿?]*("
        r"y\s|¿y\s|y\s+con\b|y\s+las\b|y\s+los\b|"
        r"también\b|tambien\b|además\b|ademas\b|"
        r"solo\b|únicamente\b|unicamente\b|"
        r"igual\b|lo mismo\b|"
        r"más\b|mas\b"
        r")",
        t,
    ):
        return True
    # Mensaje corto sin señales de “nueva consulta de cohorte completa”.
    if len(raw) < 36 and not _NEW_TOPIC_HINT.search(t):
        return True
    return False
