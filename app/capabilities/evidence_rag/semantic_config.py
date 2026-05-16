"""
Configuración del rerank semántico clínico (embeddings + cross-encoder).

Modos (``COPILOT_SEMANTIC_RERANK``):
  - ``off`` (default): solo heurística PICO + noise suppression.
  - ``embeddings``: bi-encoder sobre el pool → fusión con heurística.
  - ``cross_encoder``: cross-encoder (clinical intent query ↔ título+abstract) → fusión.
  - ``full``: embed top-K → cross-encoder → fusión ponderada (recomendado).

Modelos recomendados (configurables vía env):
  Bi-encoder  COPILOT_EMBEDDING_MODEL:
    - BAAI/bge-large-en-v1.5          general, alto rendimiento (~1.3 GB)
    - ncats/MedCPT-Query-Encoder       biomédico PubMed (asimétrico, par con Article-Encoder)
  Cross-encoder COPILOT_RERANKER_MODEL:
    - ncats/MedCPT-Cross-Encoder       biomédico PubMed (recomendado para preguntas clínicas)
    - BAAI/bge-reranker-large          general alta calidad (~1.3 GB), buen fallback
    - BAAI/bge-reranker-v2-m3          multilingual, acepta texto en ES sin traducción

Requiere: pip install -r requirements-semantic.txt
"""
from __future__ import annotations

import os
from typing import Literal

SemanticRerankMode = Literal["off", "embeddings", "cross_encoder", "full"]

# Modelos por defecto — todos disponibles en HuggingFace sin token.
# Bi-encoder: BAAI/bge-large-en-v1.5 (~1.3 GB, MTEB top-tier, dominio general/biomédico)
# Cross-encoder: BAAI/bge-reranker-large (~1.3 GB, alta calidad; evalúa query+doc juntos)
# Alternativas biomédicas CE (descomentar en .env):
#   BAAI/bge-reranker-v2-m3          multilingual, acepta texto en ES directamente
#   mixedbread-ai/mxbai-rerank-large-v1  excelente calidad general
#   cross-encoder/ms-marco-MiniLM-L-6-v2  ligero y rápido
_DEFAULT_EMBED_MODEL = "BAAI/bge-large-en-v1.5"
_DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-large"


def semantic_rerank_mode() -> SemanticRerankMode:
    v = (os.getenv("COPILOT_SEMANTIC_RERANK") or "off").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return "full"
    if v in ("embeddings", "embedding", "biencoder", "bi-encoder"):
        return "embeddings"
    if v in ("cross_encoder", "cross-encoder", "crossencoder", "reranker", "ce"):
        return "cross_encoder"
    if v in ("full", "both"):
        return "full"
    return "off"


def embedding_model_name() -> str:
    return (os.getenv("COPILOT_EMBEDDING_MODEL") or _DEFAULT_EMBED_MODEL).strip()


def cross_encoder_model_name() -> str:
    return (os.getenv("COPILOT_RERANKER_MODEL") or _DEFAULT_RERANKER_MODEL).strip()


def semantic_device() -> str:
    """Dispositivo para sentence-transformers: cuda si disponible, cpu si no."""
    forced = (os.getenv("COPILOT_SEMANTIC_DEVICE") or "").strip().lower()
    if forced in ("cpu", "cuda", "mps"):
        return forced
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def semantic_embed_top_k() -> int:
    """Candidatos que pasan del bi-encoder al cross-encoder (modo ``full``)."""
    try:
        return max(5, min(100, int(os.getenv("COPILOT_SEMANTIC_EMBED_TOP_K", "40"))))
    except ValueError:
        return 40


def semantic_pre_pool_max() -> int:
    """Máximo de candidatos que entran al pipeline semántico (tras merge PubMed)."""
    from app.config.settings import settings

    default = str(settings.evidence_retrieval_pool_max)
    try:
        raw = os.getenv("COPILOT_SEMANTIC_PRE_POOL_MAX", default)
        return max(10, min(500, int(raw)))
    except ValueError:
        return settings.evidence_retrieval_pool_max


def semantic_encode_batch_size() -> int:
    """Batch para SentenceTransformer.encode (más grande = más rápido en GPU)."""
    try:
        return max(8, min(256, int(os.getenv("COPILOT_SEMANTIC_BATCH_SIZE", "64"))))
    except ValueError:
        return 64


def semantic_score_weights() -> tuple[float, float, float]:
    """
    Pesos (heurística, embedding, cross_encoder).

    El cross-encoder domina: evalúa QUERY+DOCUMENTO juntos (intención clínica real).
    Heurística aporta señal de diseño de evidencia y PICO que el CE puede no capturar.
    """
    return (0.20, 0.10, 0.70)
