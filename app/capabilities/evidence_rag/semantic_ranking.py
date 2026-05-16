"""
Rerank semántico clínico: bi-encoder (recall) + cross-encoder (precisión clínica).

Pipeline (modo ``full``):
  pool PubMed (≤200)
    → heurístico PICO+diseño (score base)
    → bi-encoder embed → top-K (≤40)
    → cross-encoder query↔doc (score dominante)
    → fusión ponderada 20/10/70
    → top-6 para síntesis

El cross-encoder recibe la pregunta en forma PICO estructurada en inglés clínico
(``build_intent_semantic_query``) en vez del texto crudo en ES, para que el modelo
evalúe intención clínica real, no solapamiento léxico.

Degrada silenciosamente a heurística si sentence-transformers falla o modo=off.
"""
from __future__ import annotations

import logging
import threading
from typing import Any

from app.capabilities.evidence_rag.clinical_intent import ClinicalIntent
from app.capabilities.evidence_rag.intent_semantic_query import build_intent_semantic_query
from app.capabilities.evidence_rag.domain_alignment import get_domain_aligner
from app.capabilities.evidence_rag.epistemic_ranking import finalize_rank_score
from app.capabilities.evidence_rag.semantic_config import (
    cross_encoder_model_name,
    embedding_model_name,
    semantic_device,
    semantic_embed_top_k,
    semantic_encode_batch_size,
    semantic_pre_pool_max,
    semantic_rerank_mode,
    semantic_score_weights,
)

_log = logging.getLogger(__name__)

# Estado de carga — tracking separado por modelo para permitir operación parcial
_lock = threading.Lock()
_embedder: Any = None
_cross_encoder: Any = None
_embed_error: str | None = None
_ce_error: str | None = None
_st_missing: bool = False   # sentence-transformers no instalado
_models_loaded: bool = False


# ---------------------------------------------------------------------------
# Carga y preload
# ---------------------------------------------------------------------------

def _load_models(*, force_reload: bool = False) -> bool:
    """
    Carga bi-encoder y cross-encoder según modo activo. Thread-safe.
    Soporta operación parcial: si CE falla pero embed cargó, usa modo embeddings.
    Devuelve True si al menos un modelo necesario está disponible.
    """
    global _embedder, _cross_encoder, _embed_error, _ce_error, _st_missing, _models_loaded

    mode = semantic_rerank_mode()
    if mode == "off":
        return False

    with _lock:
        if _models_loaded and not force_reload:
            return _embedder is not None or _cross_encoder is not None

        try:
            from sentence_transformers import CrossEncoder, SentenceTransformer
        except ImportError as exc:
            _st_missing = True
            _embed_error = _ce_error = f"sentence-transformers no instalado: {exc}"
            _log.warning("semantic_ranking: %s  →  pip install -r requirements-semantic.txt", exc)
            _models_loaded = True
            return False

        device = semantic_device()

        if mode in ("embeddings", "full") and (_embedder is None or force_reload):
            model_name = embedding_model_name()
            _log.info("semantic_ranking: cargando bi-encoder %s en %s …", model_name, device)
            try:
                _embedder = SentenceTransformer(model_name, device=device)
                _embed_error = None
                _log.info("semantic_ranking: bi-encoder listo.")
                get_domain_aligner(_embedder)  # Inicializar singleton de dominios
            except Exception as exc:
                _embed_error = f"bi-encoder ({model_name}): {exc}"
                _log.error("semantic_ranking: error bi-encoder — %s", exc)
                _embedder = None

        if mode in ("cross_encoder", "full") and (_cross_encoder is None or force_reload):
            ce_name = cross_encoder_model_name()
            _log.info("semantic_ranking: cargando cross-encoder %s en %s …", ce_name, device)
            try:
                _cross_encoder = CrossEncoder(ce_name, device=device, max_length=512)
                _ce_error = None
                _log.info("semantic_ranking: cross-encoder listo.")
            except Exception as exc:
                _ce_error = f"cross-encoder ({ce_name}): {exc}"
                _log.error("semantic_ranking: error cross-encoder — %s", exc)
                _cross_encoder = None

        _models_loaded = True
        if _embed_error or _ce_error:
            _log.warning(
                "semantic_ranking: errores parciales — embed=%s  CE=%s",
                _embed_error or "ok", _ce_error or "ok",
            )
        return _embedder is not None or _cross_encoder is not None


def _effective_mode() -> str:
    """Modo efectivo según qué modelos están realmente cargados."""
    requested = semantic_rerank_mode()
    has_embed = _embedder is not None
    has_ce = _cross_encoder is not None
    if requested == "full":
        if has_embed and has_ce:
            return "full"
        if has_ce:
            return "cross_encoder"
        if has_embed:
            return "embeddings"
    if requested == "cross_encoder" and has_ce:
        return "cross_encoder"
    if requested == "embeddings" and has_embed:
        return "embeddings"
    return "off"


def preload_models() -> None:
    """
    Llamar en ``@app.on_event('startup')`` para evitar latencia en la primera petición.
    No hace nada si ``COPILOT_SEMANTIC_RERANK=off``.
    """
    if semantic_rerank_mode() == "off":
        return
    _log.info("semantic_ranking: precargando modelos …")
    _load_models()
    eff = _effective_mode()
    _log.info(
        "semantic_ranking: preload completo — modo_efectivo=%s  embed=%s  CE=%s  device=%s",
        eff,
        embedding_model_name() if _embedder else "—",
        cross_encoder_model_name() if _cross_encoder else "—",
        semantic_device(),
    )


# ---------------------------------------------------------------------------
# Utilidades internas
# ---------------------------------------------------------------------------

def _normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    rng = hi - lo
    if rng < 1e-9:
        return [0.5] * len(values)
    return [(v - lo) / rng for v in values]


def _doc_text(article: dict[str, Any]) -> str:
    title = str(article.get("title") or "").strip()
    snip = str(article.get("abstract_snippet") or "").strip()
    if title and snip:
        return f"{title}. {snip}"
    return title or snip


def _embed_scores(query: str, articles: list[dict[str, Any]]) -> tuple[list[float], list[float], list[list[float]]]:
    if _embedder is None or not articles:
        return [0.0] * len(articles), [], []
    try:
        from sentence_transformers.util import cos_sim

        batch = semantic_encode_batch_size()
        texts = [_doc_text(a) for a in articles]
        q_emb = _embedder.encode(
            [query], normalize_embeddings=True,
            batch_size=batch, show_progress_bar=False,
        )
        d_emb = _embedder.encode(
            texts, normalize_embeddings=True,
            batch_size=batch, show_progress_bar=False,
        )
        scores = [float(s) for s in cos_sim(q_emb, d_emb)[0].tolist()]
        # q_emb tiene shape (1, dim), d_emb tiene (n, dim)
        return scores, q_emb[0].tolist(), d_emb.tolist()
    except Exception as exc:
        _log.warning("semantic_ranking: embed error — %s", exc)
        return [0.0] * len(articles), [], []


def _ce_scores(query: str, articles: list[dict[str, Any]]) -> list[float]:
    if _cross_encoder is None or not articles:
        return [0.0] * len(articles)
    try:
        batch = semantic_encode_batch_size()
        pairs = [[query, _doc_text(a)] for a in articles]
        raw = _cross_encoder.predict(pairs, batch_size=batch, show_progress_bar=False)
        return [float(x) for x in raw]
    except Exception as exc:
        _log.warning("semantic_ranking: cross-encoder error — %s", exc)
        return [0.0] * len(articles)


# ---------------------------------------------------------------------------
# Punto de entrada principal
# ---------------------------------------------------------------------------

def semantic_rank_articles(
    articles: list[dict[str, Any]],
    *,
    user_query: str,
    clinical_intent: ClinicalIntent | None,
    heuristic_scores: list[float],
    cap: int = 6,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Reordena ``articles`` con pipeline semántico clínico.

    ``heuristic_scores`` debe estar alineado por índice con ``articles``.
    Devuelve (lista ordenada[:cap], dict de debug).
    """
    mode = semantic_rerank_mode()
    semantic_q = build_intent_semantic_query(clinical_intent, user_query)

    debug: dict[str, Any] = {
        "mode": mode,
        "semantic_query": semantic_q,
        "applied": False,
        "fallback_reason": None,
        "embedding_model": None,
        "cross_encoder_model": None,
        "device": semantic_device(),
        "pool_size_in": len(articles),
    }

    def _heuristic_fallback(reason: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        debug["fallback_reason"] = reason
        # heuristic_scores ya incluyen finalize_rank_score desde rerank_article_dicts
        pairs = sorted(zip(heuristic_scores, articles), key=lambda x: x[0], reverse=True)
        out_fb = []
        for sc, art in pairs[:cap]:
            row = dict(art)
            if "semantic_scores" not in row:
                row["semantic_scores"] = {"fused": round(sc, 4)}
            out_fb.append(row)
        return out_fb, debug

    if mode == "off" or not articles:
        return _heuristic_fallback("mode_off")

    if not _load_models():
        reason = _embed_error or _ce_error or "models_unavailable"
        return _heuristic_fallback(reason)

    eff = _effective_mode()
    if eff == "off":
        return _heuristic_fallback(_embed_error or _ce_error or "no_model_loaded")

    debug["applied"] = True
    debug["effective_mode"] = eff
    debug["embedding_model"] = embedding_model_name() if _embedder else None
    debug["cross_encoder_model"] = cross_encoder_model_name() if _cross_encoder else None

    pool_max = semantic_pre_pool_max()
    pool = articles[:pool_max]
    h = heuristic_scores[:len(pool)]

    # --- Bi-encoder: acotar candidatos para CE ---
    if eff in ("embeddings", "full"):
        emb, q_vec, d_vecs = _embed_scores(semantic_q, pool)
    else:
        emb, q_vec, d_vecs = [0.0] * len(pool), [], []

    if eff == "full":
        k = min(semantic_embed_top_k(), len(pool))
        top_idx = sorted(range(len(pool)), key=lambda i: emb[i], reverse=True)[:k]
        ce_pool = [pool[i] for i in top_idx]
        ce_h    = [h[i]    for i in top_idx]
        ce_emb  = [emb[i]  for i in top_idx]
        ce_d_vecs = [d_vecs[i] for i in top_idx]
        ce      = _ce_scores(semantic_q, ce_pool)
        final_pool, final_h, final_emb, final_ce, final_d_vecs = ce_pool, ce_h, ce_emb, ce, ce_d_vecs

    elif eff == "cross_encoder":
        ce = _ce_scores(semantic_q, pool)
        final_pool, final_h, final_emb, final_ce, final_d_vecs = pool, h, emb, ce, d_vecs

    else:  # embeddings only
        final_pool, final_h, final_emb, final_ce, final_d_vecs = pool, h, emb, [0.0] * len(pool), d_vecs

    # --- Fusión ponderada ---
    w_h, w_e, w_c = semantic_score_weights()
    h_n = _normalize(final_h)
    e_n = _normalize(final_emb)
    c_n = _normalize(final_ce)

    aligner = get_domain_aligner()
    fused: list[tuple[float, int, dict[str, Any]]] = []
    for i, art in enumerate(final_pool):
        domain_alignment_score = 0.0
        explanation = ""
        intent_domain = ""
        paper_domain = ""
        
        if aligner and q_vec and final_d_vecs:
            analysis = aligner.analyze_domain_alignment(q_vec, final_d_vecs[i])
            domain_alignment_score = analysis["domain_alignment_score"]
            explanation = analysis["explanation"]
            intent_domain = analysis["intent_top_domain"]
            paper_domain = analysis["paper_top_domain"]
        
        # Penalización ligera o tie-breaker usando el abstract mismatch (domain prior)
        domain_prior = 0.05 * domain_alignment_score
        
        if eff == "embeddings":
            score = w_h * h_n[i] + (1.0 - w_h) * e_n[i] + domain_prior
        elif eff == "cross_encoder":
            score = w_h * h_n[i] + (1.0 - w_h) * c_n[i] + domain_prior
        else:  # full
            # PASO 2 & 3: Clinical Relevance Fusion Multimodal & Evidence Hierarchy
            # Evaluamos jerarquía + recencia + cross_encoder + embed + domain base
            evidence_hierarchy_weight = 0.15 * h_n[i]   # Asumimos que heurística ya premiaba metaanális/ECA
            bi_encoder_overlap        = 0.05 * e_n[i]
            cross_encoder_clinical    = 0.40 * c_n[i] 
            recency_or_base           = 0.10 * h_n[i]   # Approximation: heurística incluía PICO+recency
            domain_alignment_weight   = 0.10 * domain_alignment_score
            
            score = cross_encoder_clinical + evidence_hierarchy_weight + bi_encoder_overlap + recency_or_base + domain_alignment_weight

        title = str(art.get("title") or "")
        snip = str(art.get("abstract_snippet") or "")
        final_score, rank_meta = finalize_rank_score(
            score,
            title=title,
            abstract=snip,
            clinical_intent=clinical_intent,
        )

        row = dict(art)
        row["clinical_domain"] = paper_domain
        sem_scores: dict[str, Any] = {
            "heuristic_norm": round(h_n[i], 4),
            "embedding_norm": round(e_n[i], 4),
            "cross_encoder_norm": round(c_n[i], 4),
            "cross_encoder_raw": round(final_ce[i], 4),
            "domain_alignment": round(domain_alignment_score, 4),
            "fused_pre_epistemic": rank_meta["fused_pre_epistemic"],
            "epistemic_multiplier": rank_meta["epistemic_multiplier"],
            "epistemic_boost": rank_meta["epistemic_boost"],
            "evidence_type": rank_meta["evidence_type"],
            "noise_multiplier": rank_meta["noise_multiplier"],
            "fused": rank_meta["fused"],
            "domain_explainability": explanation,
        }
        row["semantic_scores"] = sem_scores
        fused.append((final_score, -i, row))

    fused.sort(key=lambda x: (x[0], x[1]), reverse=True)
    out = [t[2] for t in fused[:max(1, cap)]]

    debug["pool_size_after_embed"] = len(final_pool)
    debug["top_pmids"] = [str(a.get("pmid") or "") for a in out]
    debug["ce_score_range"] = (
        round(min(final_ce), 4),
        round(max(final_ce), 4),
    ) if final_ce and any(x != 0.0 for x in final_ce) else None

    return out, debug
