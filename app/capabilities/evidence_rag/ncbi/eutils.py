"""
NCBI E-utilities (PubMed): esearch + efetch (XML).

Origen: ``sina_mcp/prsn3.0/src/prsn30/pubmed/eutils.py`` (misma lógica, imports locales).

Política de uso: https://www.ncbi.nlm.nih.gov/books/NBK25497/
"""
from __future__ import annotations

import os
import time
import xml.etree.ElementTree as ET
from typing import Any, List, Optional, Tuple

import httpx

from app.capabilities.evidence_rag.copilot_errors import CopilotError
from app.capabilities.evidence_rag.evidence_rerank import weak_design_share_from_titles
from app.capabilities.evidence_rag.ncbi.date_window import pdat_range_for_years_back
from app.capabilities.evidence_rag.ncbi.pubmed_query_normalizer import (
    normalize_pubmed_query,
    retrieval_metrics_for_query,
)
from app.capabilities.evidence_rag.ncbi.pubmed_record import PubMedArticleRecord

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# Segunda búsqueda condicional: revisiones, meta-análisis y ECA (Publication Type en PubMed).
_PUBMED_SYNTHESIS_PUB_TYPES_CLAUSE = (
    '(Review[pt] OR Meta-Analysis[pt] OR "Randomized Controlled Trial"[pt])'
)

# Demasiados hits en ESearch → conjunto ruidoso; primera página con muchos diseños débiles → refinar.
_DEFAULT_REFINE_MIN_RESULT_COUNT = 2500
_DEFAULT_REFINE_WEAK_TITLE_SHARE = 0.5
_DEFAULT_REFINE_MIN_ARTICLES_FOR_WEAK_SHARE = 3


def append_synthesis_pub_types_to_pubmed_query(normalized_term: str) -> str:
    """Añade filtro OR de tipos de publicación útiles para síntesis clínica (PubMed)."""
    t = (normalized_term or "").strip()
    if not t:
        return ""
    return f"({t}) AND ({_PUBMED_SYNTHESIS_PUB_TYPES_CLAUSE})"


def _local(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag


def _text(el: Optional[ET.Element]) -> str:
    if el is None:
        return ""
    return "".join(el.itertext()).strip()


def _sanitize_ncbi_api_key(raw: str) -> str:
    """
    Evita enviar a NCBI basura típica de .env mal copiado (comentario en la misma línea
    interpretado como valor, o texto que empieza por '#').
    Las claves de NCBI son alfanuméricas; nos quedamos con el primer token sin '#'.
    """
    s = (raw or "").strip()
    if not s or s.startswith("#"):
        return ""
    if "#" in s:
        s = s.split("#", 1)[0].strip()
    return s.split()[0] if s else ""


def _ncbi_params() -> dict[str, str]:
    out: dict[str, str] = {}
    email = os.getenv("NCBI_EMAIL", "").strip()
    if email:
        out["email"] = email
    key = _sanitize_ncbi_api_key(os.getenv("NCBI_API_KEY", ""))
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

    Implementación delegada en ``search_and_fetch_with_debug`` (retrocompatibilidad).
    """
    recs, _dbg = search_and_fetch_with_debug(
        term,
        retmax=retmax,
        sleep_s=sleep_s,
        pubmed_years_back=pubmed_years_back,
        mindate=mindate,
        maxdate=maxdate,
    )
    return recs


def _simplify_pubmed_boolean(term: str) -> Optional[str]:
    """Si la query es una conjunción larga, prueba sin el último ') AND ('."""
    t = (term or "").strip()
    if len(t) < 40 or ") AND (" not in t:
        return None
    cut = t.rfind(") AND (")
    if cut <= 0:
        return None
    shorter = t[: cut + 1].strip()
    if len(shorter) < 25 or shorter == t:
        return None
    return shorter


def esearch_pubmed_detailed(
    term: str,
    *,
    retmax: int = 10,
    datetype: Optional[str] = None,
    mindate: Optional[str] = None,
    maxdate: Optional[str] = None,
    client: httpx.Client,
) -> dict[str, Any]:
    """
    esearch con metadatos para observabilidad.

    Returns:
        dict con keys: idlist (list[str]), result_count (int|None), http_status, error (str|None), stage.
    """
    out: dict[str, Any] = {
        "idlist": [],
        "result_count": None,
        "http_status": None,
        "error": None,
        "stage": "esearch",
    }
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
    try:
        r = client.get(url, params=params)
        out["http_status"] = r.status_code
        if r.status_code >= 400:
            body = (r.text or "")[:500].replace("\n", " ").strip()
            out["error"] = f"http esearch status={r.status_code}"
            if body:
                out["error"] += f": {body}"
            return out
        r.raise_for_status()
        raw = (r.text or "").strip()
        if not raw:
            out["error"] = "esearch: cuerpo HTTP vacío"
            return out
        data = r.json()
        res = data.get("esearchresult") or {}
        err = res.get("ERROR") or res.get("ErrorMsg")
        if err:
            out["error"] = f"PubMed esearch: {err}"
            return out
        cnt = res.get("count")
        if cnt is not None and str(cnt).strip().isdigit():
            out["result_count"] = int(str(cnt))
        ids = res.get("idlist") or []
        out["idlist"] = [str(x) for x in ids]
        
        # INTRA-STAGE FALLBACK: Si pusimos filtro de fecha y dio 0 hits, re-ejecutamos de inmediato sin límite de fecha
        if not out["idlist"] and datetype == "pdat":
            fallback_params = params.copy()
            for k in ("datetype", "mindate", "maxdate"):
                fallback_params.pop(k, None)
            
            try:
                r_fall = client.get(url, params=fallback_params)
                if r_fall.status_code < 400:
                    data_fall = r_fall.json()
                    res_fall = data_fall.get("esearchresult") or {}
                    if not (res_fall.get("ERROR") or res_fall.get("ErrorMsg")):
                        fall_ids = res_fall.get("idlist") or []
                        out["idlist"] = [str(x) for x in fall_ids]
                        if "count" in res_fall and str(res_fall["count"]).isdigit():
                            out["result_count"] = int(str(res_fall["count"]))
            except Exception:
                pass # Si el fallback falla por red, mantenemos el resultado original (0 hits)

    except httpx.TimeoutException as exc:
        out["error"] = f"timeout esearch: {exc}"
    except httpx.HTTPStatusError as exc:
        out["error"] = f"http esearch status={exc.response.status_code}"
        out["http_status"] = getattr(exc.response, "status_code", None)
    except (ValueError, KeyError, TypeError) as exc:
        out["error"] = f"esearch parse/json: {exc}"
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"esearch: {type(exc).__name__}: {exc}"
    return out


def fetch_pubmed_xml_safe(
    pmids: List[str],
    *,
    client: httpx.Client,
) -> tuple[str, Optional[str]]:
    """efetch; devuelve (xml_text, error)."""
    if not pmids:
        return "", None
    params: dict[str, Any] = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        **_ncbi_params(),
    }
    url = f"{EUTILS}/efetch.fcgi"
    try:
        r = client.get(url, params=params)
        if r.status_code != 200:
            return "", f"efetch http_status={r.status_code}"
        r.raise_for_status()
        body = r.text or ""
        if not body.strip():
            return "", "efetch: XML vacío"
        return body, None
    except httpx.TimeoutException as exc:
        return "", f"timeout efetch: {exc}"
    except httpx.HTTPStatusError as exc:
        return "", f"http efetch status={exc.response.status_code}"
    except Exception as exc:  # noqa: BLE001
        return "", f"efetch: {type(exc).__name__}: {exc}"


def parse_pubmed_fetch_xml_safe(xml_text: str) -> tuple[List[PubMedArticleRecord], Optional[str]]:
    if not (xml_text or "").strip():
        return [], "parse: XML vacío"
    try:
        return parse_pubmed_fetch_xml(xml_text), None
    except ET.ParseError as exc:
        return [], f"parse: XML inválido: {exc}"
    except Exception as exc:  # noqa: BLE001
        return [], f"parse: {type(exc).__name__}: {exc}"


def _pubmed_transport_or_esearch_code(msg: str) -> str:
    m = (msg or "").lower()
    if "timeout" in m:
        return "PUBMED_TIMEOUT"
    if "efetch" in m:
        return "PUBMED_EFETCH_ERROR"
    if "http" in m or "status=" in m:
        return "PUBMED_HTTP_ERROR"
    return "PUBMED_ESEARCH_FAILED"


def search_and_fetch_with_debug(
    term: str,
    *,
    retmax: int = 8,
    sleep_s: float = 0.34,
    pubmed_years_back: Optional[int] = None,
    mindate: Optional[str] = None,
    maxdate: Optional[str] = None,
    synthesis_pubtype_refine: bool = True,
    refine_min_result_count: int = _DEFAULT_REFINE_MIN_RESULT_COUNT,
    refine_weak_title_share: float = _DEFAULT_REFINE_WEAK_TITLE_SHARE,
    refine_min_articles_for_weak_share: int = _DEFAULT_REFINE_MIN_ARTICLES_FOR_WEAK_SHARE,
) -> Tuple[List[PubMedArticleRecord], dict[str, Any]]:
    """
    ``esearch`` + ``efetch`` (y opcionalmente un segundo ciclo con filtro de tipo de publicación).

    Tras un primer resultado válido, si el conteo global de ESearch es muy alto o la primera
    página es mayoritariamente «débil» por título (casos clínicos, editoriales, preclínico),
    se intenta otra búsqueda añadiendo
    ``Review[pt] OR Meta-Analysis[pt] OR Randomized Controlled Trial[pt]``; si falla o no hay
    artículos parseables, se conserva el resultado primario.

    Cero PMIDs con HTTP OK en el primer intento es válido (``outcome=zero_hits_esearch``).
    Errores de red, ESearch o parseo: ``CopilotError`` con ``code``.
    """
    cap = max(1, min(retmax, 250))
    planned = (term or "").strip()
    if not planned:
        return [], {
            "outcome": "no_query",
            "attempts": [],
            "errors": ["término PubMed vacío"],
            "final_idlist_length": 0,
            "articles_parsed": 0,
            "pubmed_query_planned": "",
            "normalized_query": "",
            "final_query_sent": "",
            "normalization": {"warnings": [], "steps_applied": []},
            "retrieval_metrics": retrieval_metrics_for_query(""),
        }

    q0, norm_meta = normalize_pubmed_query(planned)
    if not q0:
        return [], {
            "outcome": "no_query",
            "attempts": [],
            "errors": ["query PubMed vacía tras normalización"],
            "final_idlist_length": 0,
            "articles_parsed": 0,
            "pubmed_query_planned": planned,
            "normalized_query": "",
            "final_query_sent": "",
            "normalization": norm_meta,
            "retrieval_metrics": retrieval_metrics_for_query(""),
        }

    md0, xd0 = mindate, maxdate
    if md0 is None and xd0 is None and pubmed_years_back is not None:
        md0, xd0 = pdat_range_for_years_back(pubmed_years_back)

    if md0 and xd0:
        label, q, use_pdat, md, xd = "primary_pdat", q0, True, md0, xd0
    else:
        label, q, use_pdat, md, xd = "primary_no_pdat", q0, False, None, None

    with httpx.Client(timeout=120.0) as client:
        dt = "pdat" if use_pdat and md and xd else None
        att: dict[str, Any] = {
            "label": label,
            "query": q,
            "used_pdat": bool(dt),
            "mindate": md,
            "maxdate": xd,
            "result_count": None,
            "idlist_length": 0,
            "http_status": None,
            "stage_reached": "esearch",
            "error": None,
            "articles_parsed": 0,
        }
        es = esearch_pubmed_detailed(
            q,
            retmax=cap,
            datetype=dt,
            mindate=md,
            maxdate=xd,
            client=client,
        )
        att["result_count"] = es.get("result_count")
        att["http_status"] = es.get("http_status")
        ids = list(es.get("idlist") or [])
        att["idlist_length"] = len(ids)
        attempts = [att]
        errors: list[str] = []

        if es.get("error"):
            att["error"] = es["error"]
            msg = f"{label}: {es['error']}"
            errors.append(msg)
            code = _pubmed_transport_or_esearch_code(msg)
            raise CopilotError(code, msg)

        if not ids:
            debug = {
                "outcome": "zero_hits_esearch",
                "attempts": attempts,
                "errors": [],
                "final_idlist_length": 0,
                "articles_parsed": 0,
                "pubmed_query_planned": planned,
                "normalized_query": q0,
                "final_query_sent": q0,
                "normalization": norm_meta,
                "retrieval_metrics": {
                    **retrieval_metrics_for_query(q0),
                    "last_attempt_result_count": att.get("result_count"),
                },
            }
            return [], debug

        if sleep_s > 0:
            time.sleep(sleep_s)

        att["stage_reached"] = "efetch"
        xml_text, ferr = fetch_pubmed_xml_safe(ids, client=client)
        if ferr:
            att["error"] = ferr
            msg = f"{label}: {ferr}"
            errors.append(msg)
            raise CopilotError(_pubmed_transport_or_esearch_code(msg), msg)

        att["stage_reached"] = "parse"
        records, perr = parse_pubmed_fetch_xml_safe(xml_text)
        att["articles_parsed"] = len(records)
        if perr:
            att["error"] = perr
            msg = f"{label}: {perr}"
            errors.append(msg)
            raise CopilotError("PUBMED_PARSE_ERROR", msg)

        if not records:
            att["error"] = "parse: 0 artículos con PMIDs devueltos por esearch"
            msg = f"{label}: {att['error']}"
            raise CopilotError("PUBMED_NO_ARTICLES_PARSED", msg)

        best_records = records
        final_ids = ids
        final_query_for_debug = q0
        refine_meta: dict[str, Any] = {
            "applied": False,
            "reason": None,
            "refined_query": None,
            "kept_primary_after_refine": False,
        }

        rc = att.get("result_count")
        weak_share = weak_design_share_from_titles(r.title for r in records)
        need_refine = False
        refine_reason: Optional[str] = None
        if synthesis_pubtype_refine:
            if rc is not None and rc >= refine_min_result_count:
                need_refine = True
                refine_reason = "high_esearch_hit_count"
            elif (
                len(records) >= refine_min_articles_for_weak_share
                and weak_share >= refine_weak_title_share
            ):
                need_refine = True
                refine_reason = "weak_design_majority_in_page"

        if need_refine:
            if sleep_s > 0:
                time.sleep(sleep_s)

            q2 = (append_synthesis_pub_types_to_pubmed_query(q) or "").strip()

            lbl2 = "synthesis_pub_types_pdat" if use_pdat else "synthesis_pub_types_no_pdat"
            att2: dict[str, Any] = {
                "label": lbl2,
                "query": q2,
                "used_pdat": bool(dt),
                "mindate": md,
                "maxdate": xd,
                "result_count": None,
                "idlist_length": 0,
                "http_status": None,
                "stage_reached": "esearch",
                "error": None,
                "articles_parsed": 0,
            }
            es2 = esearch_pubmed_detailed(
                q2,
                retmax=cap,
                datetype=dt,
                mindate=md,
                maxdate=xd,
                client=client,
            )
            att2["result_count"] = es2.get("result_count")
            att2["http_status"] = es2.get("http_status")
            ids2 = list(es2.get("idlist") or [])
            att2["idlist_length"] = len(ids2)
            attempts.append(att2)

            if es2.get("error"):
                att2["error"] = es2["error"]
                refine_meta["reason"] = refine_reason
                refine_meta["kept_primary_after_refine"] = True
                refine_meta["refined_query"] = q2
            elif not ids2:
                att2["error"] = "zero PMIDs after synthesis pub-type filter"
                refine_meta["reason"] = refine_reason
                refine_meta["kept_primary_after_refine"] = True
                refine_meta["refined_query"] = q2
            else:
                if sleep_s > 0:
                    time.sleep(sleep_s)
                att2["stage_reached"] = "efetch"
                xml2, ferr2 = fetch_pubmed_xml_safe(ids2, client=client)
                if ferr2:
                    att2["error"] = ferr2
                    refine_meta["reason"] = refine_reason
                    refine_meta["kept_primary_after_refine"] = True
                    refine_meta["refined_query"] = q2
                else:
                    att2["stage_reached"] = "parse"
                    rec2, perr2 = parse_pubmed_fetch_xml_safe(xml2)
                    att2["articles_parsed"] = len(rec2)
                    if perr2:
                        att2["error"] = perr2
                        refine_meta["reason"] = refine_reason
                        refine_meta["kept_primary_after_refine"] = True
                        refine_meta["refined_query"] = q2
                    elif not rec2:
                        att2["error"] = "parse: 0 articles after refine"
                        refine_meta["reason"] = refine_reason
                        refine_meta["kept_primary_after_refine"] = True
                        refine_meta["refined_query"] = q2
                    else:
                        best_records = rec2
                        final_ids = ids2
                        final_query_for_debug = q2
                        refine_meta["applied"] = True
                        refine_meta["reason"] = refine_reason
                        refine_meta["refined_query"] = q2

        metrics = {
            **retrieval_metrics_for_query(q0),
            "last_attempt_result_count": att.get("result_count"),
            "synthesis_pubtype_refine_applied": bool(refine_meta.get("applied")),
            "synthesis_pubtype_refine_reason": refine_meta.get("reason"),
            "weak_design_share_primary_page": weak_share,
        }
        debug = {
            "outcome": "success",
            "attempts": attempts,
            "errors": errors,
            "final_idlist_length": len(final_ids),
            "articles_parsed": len(best_records),
            "pubmed_query_planned": planned,
            "normalized_query": q0,
            "final_query_sent": final_query_for_debug,
            "normalization": norm_meta,
            "retrieval_metrics": metrics,
            "synthesis_pubtype_refine": refine_meta,
        }
        return best_records, debug
