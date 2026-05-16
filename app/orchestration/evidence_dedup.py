"""
Deduplicación de evidencia PubMed antes de síntesis (determinista y LLM).

Evita PMIDs, títulos y ``evidence_statements`` repetidos en ``MedicalAnswer`` y en el JSON
que recibe el LLM de síntesis.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence

_TITLE_FOLD_RE = re.compile(r"[^\w\s]+", re.UNICODE)


def _fold_title(title: str) -> str:
    s = re.sub(r"\s+", " ", (title or "").strip().lower())
    if not s:
        return ""
    return _TITLE_FOLD_RE.sub(" ", s).strip()


def _fold_statement(statement: str) -> str:
    s = re.sub(r"\s+", " ", (statement or "").strip().lower())
    return s[:400] if len(s) > 400 else s


def deduplicate_pmids(pmids: Iterable[str]) -> List[str]:
    """Conserva el primer PMID de cada valor; ignora vacíos."""
    seen: set[str] = set()
    out: list[str] = []
    for raw in pmids:
        pm = str(raw or "").strip()
        if not pm or pm in seen:
            continue
        seen.add(pm)
        out.append(pm)
    return out


def deduplicate_titles(
    items: Sequence[Mapping[str, Any]],
    *,
    title_key: str = "title",
    pmid_key: str = "pmid",
) -> List[Dict[str, Any]]:
    """
    Elimina filas duplicadas por PMID (prioridad) o por título normalizado idéntico.

    Si dos entradas comparten PMID, se conserva la primera. Si no hay PMID pero el
    título normalizado coincide, se descarta la repetición.
    """
    return deduplicate_articles(items, title_key=title_key, pmid_key=pmid_key)


def deduplicate_articles(
    articles: Sequence[Mapping[str, Any]],
    *,
    title_key: str = "title",
    pmid_key: str = "pmid",
) -> List[Dict[str, Any]]:
    """Alias explícito para artículos del ``evidence_bundle``."""
    out: list[Dict[str, Any]] = []
    seen_pmid: set[str] = set()
    seen_title: set[str] = set()
    for raw in articles:
        if not isinstance(raw, Mapping):
            continue
        row = dict(raw)
        pm = str(row.get(pmid_key) or "").strip()
        tit_fold = _fold_title(str(row.get(title_key) or ""))
        if pm:
            if pm in seen_pmid:
                continue
            seen_pmid.add(pm)
            if tit_fold:
                seen_title.add(tit_fold)
            out.append(row)
            continue
        if tit_fold:
            if tit_fold in seen_title:
                continue
            seen_title.add(tit_fold)
            out.append(row)
    return out


def deduplicate_evidence_statements(
    statements: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Un statement por PMID principal (primer ``citation_pmids``); evita texto idéntico sin PMID."""
    out: list[Dict[str, Any]] = []
    seen_pmid: set[str] = set()
    seen_text: set[str] = set()
    for raw in statements:
        if not isinstance(raw, Mapping):
            continue
        row = dict(raw)
        pmids = deduplicate_pmids(str(p) for p in (row.get("citation_pmids") or []))
        body_fold = _fold_statement(str(row.get("statement") or ""))
        if pmids:
            primary = pmids[0]
            if primary in seen_pmid:
                continue
            seen_pmid.add(primary)
            if body_fold:
                seen_text.add(body_fold)
            row["citation_pmids"] = pmids
            out.append(row)
            continue
        if body_fold and body_fold in seen_text:
            continue
        if body_fold:
            seen_text.add(body_fold)
        out.append(row)
    return out


def deduplicate_citations(
    citations: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Citas serializables (``MedicalAnswer.citations``) sin PMID ni título repetido."""
    return deduplicate_titles(citations, title_key="title", pmid_key="pmid")


def deduplicate_evidence_bundle_dict(eb: Mapping[str, Any]) -> Dict[str, Any]:
    """Normaliza ``articles`` y ``pmids`` de un ``evidence_bundle`` en dict."""
    out: Dict[str, Any] = dict(eb)
    raw_arts = [a for a in (eb.get("articles") or []) if isinstance(a, Mapping)]
    arts = deduplicate_articles(raw_arts)
    out["articles"] = arts
    from_arts = [str(a.get("pmid") or "").strip() for a in arts if str(a.get("pmid") or "").strip()]
    top = [str(p) for p in (eb.get("pmids") or []) if str(p).strip()]
    if from_arts:
        out["pmids"] = deduplicate_pmids(from_arts)
    else:
        out["pmids"] = deduplicate_pmids(top)
    return out


def deduplicate_medical_answer_evidence(ma: MutableMapping[str, Any]) -> Dict[str, Any]:
    """Limpia citas y ``evidence_statements`` del payload ``MedicalAnswer``."""
    out = dict(ma)
    cites = out.get("citations")
    if isinstance(cites, list):
        raw = [c for c in cites if isinstance(c, Mapping)]
        out["citations"] = deduplicate_citations(raw)
    est = out.get("evidence_statements")
    if isinstance(est, list):
        raw_st = [s for s in est if isinstance(s, Mapping)]
        out["evidence_statements"] = deduplicate_evidence_statements(raw_st)
    return out


def state_with_deduped_evidence(state: Mapping[str, Any]) -> Dict[str, Any]:
    """Copia superficial del estado con ``evidence_bundle`` deduplicado (pre-síntesis)."""
    out = dict(state)
    eb = state.get("evidence_bundle")
    if isinstance(eb, Mapping):
        out["evidence_bundle"] = deduplicate_evidence_bundle_dict(eb)
    return out
