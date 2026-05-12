"""
NCBI E-utilities (PubMed): esearch + efetch (XML).

Origen: ``sina_mcp/prsn3.0/src/prsn30/pubmed/eutils.py`` (misma lógica, imports locales).

Política de uso: https://www.ncbi.nlm.nih.gov/books/NBK25497/
"""
from __future__ import annotations

import os
import time
import xml.etree.ElementTree as ET
from typing import Any, List, Optional

import httpx

from app.capabilities.evidence_rag.ncbi.date_window import pdat_range_for_years_back
from app.capabilities.evidence_rag.ncbi.pubmed_record import PubMedArticleRecord

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def _local(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag


def _text(el: Optional[ET.Element]) -> str:
    if el is None:
        return ""
    return "".join(el.itertext()).strip()


def _ncbi_params() -> dict[str, str]:
    out: dict[str, str] = {}
    email = os.getenv("NCBI_EMAIL", "").strip()
    if email:
        out["email"] = email
    key = os.getenv("NCBI_API_KEY", "").strip()
    if key:
        out["api_key"] = key
    return out


def esearch_pubmed(
    term: str,
    *,
    retmax: int = 10,
    datetype: Optional[str] = None,
    mindate: Optional[str] = None,
    maxdate: Optional[str] = None,
    client: Optional[httpx.Client] = None,
) -> List[str]:
    """Devuelve lista de PMIDs (strings)."""
    params: dict[str, Any] = {
        "db": "pubmed",
        "term": term,
        "retmax": str(retmax),
        "retmode": "json",
        **_ncbi_params(),
    }
    if datetype and mindate and maxdate:
        params["datetype"] = datetype
        params["mindate"] = mindate
        params["maxdate"] = maxdate
    url = f"{EUTILS}/esearch.fcgi"
    own = client is None
    if own:
        client = httpx.Client(timeout=60.0)
    try:
        assert client is not None
        r = client.get(url, params=params)
        r.raise_for_status()
        raw = (r.text or "").strip()
        if not raw:
            raise ValueError(
                "La API de NCBI (esearch) devolvió una respuesta vacía. "
                "Compruebe red, firewall o proxy; más tarde reintente."
            )
        data = r.json()
        res = data.get("esearchresult") or {}
        err = res.get("ERROR") or res.get("ErrorMsg")
        if err:
            raise ValueError(f"PubMed rechazó la consulta: {err}")
        ids = res.get("idlist") or []
        return [str(x) for x in ids]
    finally:
        if own and client is not None:
            client.close()


def fetch_pubmed_xml(
    pmids: List[str],
    *,
    client: Optional[httpx.Client] = None,
) -> str:
    if not pmids:
        return ""
    params: dict[str, Any] = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        **_ncbi_params(),
    }
    url = f"{EUTILS}/efetch.fcgi"
    own = client is None
    if own:
        client = httpx.Client(timeout=120.0)
    try:
        assert client is not None
        r = client.get(url, params=params)
        r.raise_for_status()
        body = r.text or ""
        if not body.strip():
            raise ValueError(
                "PubMed (efetch) devolvió XML vacío. Suele deberse a límites de red, "
                "bloqueo temporal de NCBI o lista de PMIDs inválida."
            )
        return body
    finally:
        if own and client is not None:
            client.close()


def _first_year_from_pubdate(pubdate: ET.Element) -> Optional[str]:
    for y in pubdate.iter():
        if _local(y.tag) == "Year" and y.text:
            return y.text.strip()
    for m in pubdate.iter():
        if _local(m.tag) == "MedlineDate" and m.text:
            parts = m.text.split()
            for p in parts:
                if len(p) == 4 and p.isdigit():
                    return p
    return None


def parse_pubmed_fetch_xml(xml_text: str) -> List[PubMedArticleRecord]:
    if not (xml_text or "").strip():
        return []
    root = ET.fromstring(xml_text)
    out: List[PubMedArticleRecord] = []
    for art in root.iter():
        if _local(art.tag) != "PubmedArticle":
            continue
        mc = next((c for c in art if _local(c.tag) == "MedlineCitation"), None)
        if mc is None:
            continue

        pmid: Optional[str] = None
        for el in mc.iter():
            if _local(el.tag) == "PMID" and el.text:
                pmid = el.text.strip()
                break
        if not pmid:
            continue

        title = ""
        abstract_parts: List[str] = []
        year: Optional[str] = None
        doi: Optional[str] = None

        article = next((c for c in mc if _local(c.tag) == "Article"), None)
        if article is not None:
            for el in article.iter():
                ln = _local(el.tag)
                if ln == "ArticleTitle":
                    title = _text(el)
                elif ln == "AbstractText":
                    label = el.get("Label", "")
                    chunk = _text(el)
                    if label:
                        abstract_parts.append(f"{label}: {chunk}")
                    else:
                        abstract_parts.append(chunk)
                elif ln == "ELocationID" and el.get("EIdType") == "doi":
                    doi = _text(el) or doi
                elif ln == "Journal":
                    for ji in el.iter():
                        if _local(ji.tag) == "JournalIssue":
                            for pd in ji.iter():
                                if _local(pd.tag) == "PubDate":
                                    year = year or _first_year_from_pubdate(pd)

        for el in art.iter():
            if _local(el.tag) == "ArticleId" and el.get("IdType") == "doi":
                doi = _text(el) or doi

        abstract = "\n\n".join(abstract_parts) if abstract_parts else ""

        out.append(
            PubMedArticleRecord(
                pmid=pmid,
                title=title or "(sin título)",
                abstract=abstract or "(sin abstract disponible)",
                year=year,
                doi=doi,
            )
        )
    return out


def search_and_fetch_abstracts(
    term: str,
    *,
    retmax: int = 8,
    sleep_s: float = 0.34,
    pubmed_years_back: Optional[int] = None,
    mindate: Optional[str] = None,
    maxdate: Optional[str] = None,
) -> List[PubMedArticleRecord]:
    """
    esearch → pausa opcional (sin API key ~3 req/s) → efetch → parse.
    """
    md, xd = mindate, maxdate
    if md is None and xd is None and pubmed_years_back is not None:
        md, xd = pdat_range_for_years_back(pubmed_years_back)

    dt = "pdat" if md and xd else None

    with httpx.Client(timeout=120.0) as client:
        ids = esearch_pubmed(
            term,
            retmax=retmax,
            datetype=dt,
            mindate=md,
            maxdate=xd,
            client=client,
        )
        if not ids:
            return []
        if sleep_s > 0:
            time.sleep(sleep_s)
        xml_text = fetch_pubmed_xml(ids, client=client)
    return parse_pubmed_fetch_xml(xml_text)
