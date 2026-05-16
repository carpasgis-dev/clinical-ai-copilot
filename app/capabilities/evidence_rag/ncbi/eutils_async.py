"""
Variantes async de E-utilities (``httpx.AsyncClient``) para recuperación paralela.
"""
from __future__ import annotations

import asyncio
import os
from typing import Any, List, Optional, Tuple

import httpx

from app.capabilities.evidence_rag.copilot_errors import CopilotError
from app.capabilities.evidence_rag.evidence_rerank import weak_design_share_from_titles
from app.capabilities.evidence_rag.ncbi.date_window import pdat_range_for_years_back
from app.capabilities.evidence_rag.ncbi.eutils import (
    EUTILS,
    _DEFAULT_REFINE_MIN_ARTICLES_FOR_WEAK_SHARE,
    _DEFAULT_REFINE_MIN_RESULT_COUNT,
    _DEFAULT_REFINE_WEAK_TITLE_SHARE,
    _ncbi_params,
    _pubmed_transport_or_esearch_code,
    append_synthesis_pub_types_to_pubmed_query,
)
from app.capabilities.evidence_rag.ncbi.pubmed_query_normalizer import (
    normalize_pubmed_query,
    retrieval_metrics_for_query,
)
from app.capabilities.evidence_rag.ncbi.pubmed_record import PubMedArticleRecord
from app.capabilities.evidence_rag.ncbi.eutils import (
    parse_pubmed_fetch_xml_safe,
    search_and_fetch_with_debug,
)


def _ncbi_sleep_s() -> float:
    if _sanitize_ncbi_key_present():
        return 0.11
    return 0.34


def _sanitize_ncbi_key_present() -> bool:
    raw = os.getenv("NCBI_API_KEY", "").strip()
    if not raw or raw.startswith("#"):
        return False
    if "#" in raw:
        raw = raw.split("#", 1)[0].strip()
    return bool(raw.split()[0] if raw else "")


async def esearch_pubmed_detailed_async(
    term: str,
    *,
    retmax: int = 10,
    datetype: Optional[str] = None,
    mindate: Optional[str] = None,
    maxdate: Optional[str] = None,
    client: httpx.AsyncClient,
) -> dict[str, Any]:
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
        r = await client.get(url, params=params)
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


async def fetch_pubmed_xml_safe_async(
    pmids: List[str],
    *,
    client: httpx.AsyncClient,
) -> tuple[str, Optional[str]]:
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
        r = await client.get(url, params=params)
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


async def search_and_fetch_with_debug_async(
    term: str,
    *,
    retmax: int = 8,
    sleep_s: Optional[float] = None,
    pubmed_years_back: Optional[int] = None,
    mindate: Optional[str] = None,
    maxdate: Optional[str] = None,
    synthesis_pubtype_refine: bool = True,
    refine_min_result_count: int = _DEFAULT_REFINE_MIN_RESULT_COUNT,
    refine_weak_title_share: float = _DEFAULT_REFINE_WEAK_TITLE_SHARE,
    refine_min_articles_for_weak_share: int = _DEFAULT_REFINE_MIN_ARTICLES_FOR_WEAK_SHARE,
) -> Tuple[List[PubMedArticleRecord], dict[str, Any]]:
    """Misma semántica que ``search_and_fetch_with_debug`` con cliente HTTP async."""
    if sleep_s is None:
        sleep_s = _ncbi_sleep_s()
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

    async with httpx.AsyncClient(timeout=120.0) as client:
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
        es = await esearch_pubmed_detailed_async(
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
            raise CopilotError(_pubmed_transport_or_esearch_code(msg), msg)

        if not ids:
            return [], {
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

        if sleep_s > 0:
            await asyncio.sleep(sleep_s)

        att["stage_reached"] = "efetch"
        xml_text, ferr = await fetch_pubmed_xml_safe_async(ids, client=client)
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
                await asyncio.sleep(sleep_s)

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
            es2 = await esearch_pubmed_detailed_async(
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
                    await asyncio.sleep(sleep_s)
                att2["stage_reached"] = "efetch"
                xml2, ferr2 = await fetch_pubmed_xml_safe_async(ids2, client=client)
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
            "http_async": True,
        }
        return best_records, debug


def search_and_fetch_parallel_aware(
    term: str,
    *,
    retmax: int = 8,
    pubmed_years_back: Optional[int] = None,
) -> Tuple[List[PubMedArticleRecord], dict[str, Any]]:
    """Usa async HTTP si el modo paralelo está activo; si no, delega en sync."""
    from app.capabilities.evidence_rag.retrieval_parallel import (
        parallel_retrieval_enabled,
        run_coroutine_sync,
    )

    if parallel_retrieval_enabled():
        return run_coroutine_sync(
            search_and_fetch_with_debug_async(
                term,
                retmax=retmax,
                pubmed_years_back=pubmed_years_back,
            )
        )
    return search_and_fetch_with_debug(
        term,
        retmax=retmax,
        pubmed_years_back=pubmed_years_back,
    )
