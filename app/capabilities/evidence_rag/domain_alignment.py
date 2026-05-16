from __future__ import annotations

import logging
import math
from typing import Any, Dict, List

_log = logging.getLogger(__name__)

# Fase 1: SOLO Prototypes Semánticos (Anchors). 
# Cero palabras clave, cero listas negras, cero diccionarios fijos.
# Esto orienta el espacio semántico de embeddings (BGE-large) basándose
# en su representación latente profunda.
DOMAIN_PROTOTYPES = {
    "clinical_human_therapeutics": "Clinical human therapeutics, randomized controlled trials, pharmacotherapy, medical interventions, patient management, chronic disease treatment.",
    "preclinical_basic_science": "Preclinical science, basic science, in vitro studies, cell lines, murine models, animal research, molecular pathways, disease mechanisms.",
    "infectious_disease": "Infectious disease, virology, viral pathogens, antibacterial, antiviral therapy, immunology, outbreaks, sepsis.",
    "oncology": "Oncology, cancer, solid tumors, hematologic malignancies, chemotherapy, radiation therapy, tumor progression, metastatic disease.",
    "population_epidemiology": "Population health, epidemiology, public health, demographic studies, incidence, prevalence, screening interventions, real-world evidence cohorts."
}

def dot_product(v1: List[float], v2: List[float]) -> float:
    return sum(x * y for x, y in zip(v1, v2))

def magnitude(v: List[float]) -> float:
    return math.sqrt(sum(x * x for x in v))

def compute_cosine_similarity(v1: List[float], v2: List[float]) -> float:
    mag1, mag2 = magnitude(v1), magnitude(v2)
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot_product(v1, v2) / (mag1 * mag2)

class SemanticDomainAligner:
    """
    Fase 2 y 3: Distribución de Afinidad y Alignment.
    """
    def __init__(self, embedder: Any):
        """
        Se ejecuta UNA VEZ (singleton) con el bi-encoder cargado.
        """
        _log.info("Inicializando SemanticDomainAligner (caché de embeddings de dominios)...")
        self.domains = list(DOMAIN_PROTOTYPES.keys())
        texts = [DOMAIN_PROTOTYPES[d] for d in self.domains]
        
        # Generar tensores y convertirlos a listas planas float
        embs = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        self.domain_embeddings = {
            domain: [float(x) for x in emb]
            for domain, emb in zip(self.domains, embs)
        }
        
    def get_domain_distribution(self, text_embedding: List[float]) -> Dict[str, float]:
        """
        Computa la afinidad de un vector (intent o paper) contra todos los macro-dominios.
        Devuelve algo como: {"cardiometabolic": 0.81, "nephrology": 0.62, "oncology": 0.11}
        """
        distribution = {}
        for domain_name, prototype_emb in self.domain_embeddings.items():
            score = max(0.0, compute_cosine_similarity(text_embedding, prototype_emb))
            distribution[domain_name] = round(score, 4)
        return distribution

    def get_top_domain(self, distribution: Dict[str, float]) -> str:
        if not distribution:
            return "unknown"
        return max(distribution.items(), key=lambda x: x[1])[0]

    def analyze_domain_alignment(self, intent_embedding: List[float], paper_embedding: List[float]) -> Dict[str, Any]:
        """
        Calcula las distribuciones semánticas para explicabilidad y un score ligero.
        Usa los dominios principalmente como 'semantic prior' y 'mismatch detection'.
        """
        intent_dist = self.get_domain_distribution(intent_embedding)
        paper_dist = self.get_domain_distribution(paper_embedding)
        
        # Double projection score - solo como señal débil adicional
        vec_intent = [intent_dist.get(d, 0.0) for d in self.domains]
        vec_paper = [paper_dist.get(d, 0.0) for d in self.domains]
        domain_alignment_score = compute_cosine_similarity(vec_intent, vec_paper)
        
        intent_top = self.get_top_domain(intent_dist)
        paper_top = self.get_top_domain(paper_dist)
        
        explanation = f"Paper domain: {paper_top} (Intent: {intent_top})"
        
        return {
            "intent_top_domain": intent_top,
            "paper_top_domain": paper_top,
            "domain_alignment_score": round(domain_alignment_score, 4),
            "explanation": explanation
        }

_aligner_instance: SemanticDomainAligner | None = None

def get_domain_aligner(embedder: Any = None) -> SemanticDomainAligner | None:
    """
    Devuelve la instancia singleton del alineador. Inyecta el embedder en el primer uso.
    """
    global _aligner_instance
    if _aligner_instance is None and embedder is not None:
        _aligner_instance = SemanticDomainAligner(embedder)
    return _aligner_instance
