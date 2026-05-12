"""
Interfaces (Protocols) de las capabilities del copiloto.

El orquestador y el grafo LangGraph SOLO interactúan con estos Protocols.
Las implementaciones concretas viven en:
    capabilities/clinical_sql/   ← derivado de sqlite-analyzer-mcp
    capabilities/evidence_rag/   ← NCBI, Europe PMC, multi-fuente (mismo ``EvidenceCapability``)

Regla de arquitectura:
    Si un nodo LangGraph importa algo que NO sea de este módulo
    o de app.schemas.copilot_state, la arquitectura está rota.

Por qué Protocol y no ABC:
    - Permite duck typing: las implementaciones no necesitan heredar nada.
    - runtime_checkable habilita isinstance() para tests y validación.
    - Facilita mocks en tests sin frameworks extra.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.schemas.copilot_state import (
    ClinicalContext,
    EvidenceBundle,
    SqlResult,
)


# ---------------------------------------------------------------------------
# Capability A — Datos clínicos estructurados
# ---------------------------------------------------------------------------

@runtime_checkable
class ClinicalCapability(Protocol):
    """
    Contrato para la capability de datos clínicos estructurados.
    Derivada de sqlite-analyzer-mcp.

    Responsabilidades:
        - Introspección segura de esquema
        - Ejecución SQL de solo lectura con validación
        - Extracción de contexto clínico resumido (nunca dumps)
        - Perfiles de pacientes y cohortes

    Lo que esta capability NO debe hacer:
        - Devolver dumps completos de BD al orquestador
        - Ejecutar SQL de escritura
        - Conocer nada de PubMed o RAG
    """

    def list_tables(self) -> list[str]:
        """
        Devuelve la lista de tablas disponibles en la BD clínica.
        Usado por el orquestador para validar disponibilidad.
        """
        ...

    def run_safe_query(self, sql: str) -> SqlResult:
        """
        Ejecuta SQL de solo lectura contra la BD clínica.

        La implementación es responsable de:
            - Validar y sanitizar la query (whitelist, no DDL/DML)
            - Limitar filas al máximo definido en SqlResult
            - Registrar la query ejecutada para trazabilidad
            - Devolver error estructurado (SqlResult.error) si falla,
              nunca lanzar excepción no capturada

        Args:
            sql: Query SQL de lectura. El caller debe asegurarse
                 de que es una query de solo lectura.

        Returns:
            SqlResult con filas, row_count, tables_used y la query ejecutada.
        """
        ...

    def extract_clinical_summary(self, free_text_query: str) -> ClinicalContext:
        """
        Dada una consulta en lenguaje natural, devuelve un resumen
        clínico estructurado mínimo para enriquecer una query PubMed.

        Contrato de salida:
            - Solo devuelve ClinicalContext (resumen estructurado)
            - NUNCA devuelve filas crudas ni referencias a columnas de BD
            - Respeta los límites de lista de ClinicalContext

        Ejemplos de input → output:
            "pacientes con diabetes e hipertensión > 65 años"
            → ClinicalContext(
                age_range=">65",
                conditions=["diabetes", "hipertensión"],
              )

        Args:
            free_text_query: Texto libre del usuario o del orquestador.

        Returns:
            ClinicalContext con el perfil extraído.
        """
        ...

    def health_check(self) -> bool:
        """
        True si la BD está accesible y al menos una tabla es consultable.
        Usado por el nodo Safety para informar disponibilidad en el trace.
        """
        ...


# ---------------------------------------------------------------------------
# Capability B — Recuperación de evidencia biomédica
# ---------------------------------------------------------------------------

@runtime_checkable
class EvidenceCapability(Protocol):
    """
    Contrato para la capability de evidencia biomédica.
    Derivada de PRSN 3.0 (PubMed E-utilities + RAG).

    Responsabilidades:
        - Construcción de queries PubMed (con o sin contexto clínico)
        - Búsqueda y recuperación de abstracts via E-utilities
        - Chunking y retrieval de texto completo Open Access (opcional v1)
        - Devolución de bundles con PMIDs verificables

    Lo que esta capability NO debe hacer:
        - Conocer nada de la BD clínica o SQL
        - Inventar PMIDs o citas no recuperadas
        - Devolver artículos sin PMID verificable
    """

    def build_pubmed_query(
        self,
        free_text: str,
        clinical_context: ClinicalContext | None = None,
    ) -> str:
        """
        Construye una query PubMed estructurada lista para eutils/esearch.

        Cuando se proporciona clinical_context (modo híbrido), enriquece
        la query con condiciones, rango de edad y medicamentos del perfil.

        Ejemplos:
            free_text="riesgo cardiovascular metformina"
            clinical_context=ClinicalContext(age_range=">65",
                                             conditions=["diabetes"])
            → "(metformin[tiab] OR metformina[tiab]) AND
               (cardiovascular risk[tiab]) AND
               (diabetes[MeSH]) AND (aged[MeSH])"

        Args:
            free_text: Pregunta o términos clave del usuario.
            clinical_context: Perfil clínico para enriquecer la query.
                              None si la ruta es EVIDENCE pura.

        Returns:
            String de query PubMed listo para pasar a esearch.
        """
        ...

    def retrieve_evidence(
        self,
        pubmed_query: str,
        retmax: int = 6,
        years_back: int = 5,
    ) -> EvidenceBundle:
        """
        Busca en PubMed con la query dada y devuelve un bundle acotado.

        Garantías:
            - Nunca devuelve más de _EVIDENCE_MAX_ART artículos en contexto.
            - Cada artículo tiene PMID verificable (sin inventados).
            - abstract_snippet respeta el límite de _ARTICLE_MAX_SNIPPET chars.
            - Si PubMed no responde, devuelve bundle vacío con search_term,
              nunca lanza excepción no capturada.

        Args:
            pubmed_query: Query construida por build_pubmed_query.
            retmax: Máximo artículos a recuperar (no más de 10 en v1).
            years_back: Ventana temporal en años (0 = sin filtro).

        Returns:
            EvidenceBundle con PMIDs, artículos y metadatos de recuperación.
        """
        ...

    def health_check(self) -> bool:
        """
        True si la API PubMed (eutils.ncbi.nlm.nih.gov) es alcanzable.
        Usado por el nodo Safety para informar disponibilidad en el trace.
        """
        ...
