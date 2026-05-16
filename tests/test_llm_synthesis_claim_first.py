"""Síntesis LLM en modo claim-first (facts sin abstracts)."""
from __future__ import annotations

import json

from app.orchestration.llm_synthesis import _build_facts_dict, facts_use_claim_first


def test_build_facts_claim_first_omits_abstracts() -> None:
    state = {
        "user_query": "SGLT2 heart failure diabetes",
        "route": "evidence",
        "evidence_bundle": {
            "articles": [
                {
                    "pmid": "1",
                    "title": "EMPA-REG OUTCOME trial",
                    "abstract_snippet": "Reduced heart failure hospitalization.",
                    "final_rank_score": 0.9,
                }
            ],
        },
        "clinical_evidence_frame": {
            "question_type": "treatment_efficacy",
            "intervention_concept_ids": ["sglt2_class"],
            "outcome_concept_ids": ["outcome_hf_hosp"],
            "therapeutic_objective": True,
        },
    }
    medical_answer = {
        "citations": [
            {
                "pmid": "1",
                "title": "EMPA-REG OUTCOME trial",
                "url": "https://pubmed.ncbi.nlm.nih.gov/1/",
            }
        ],
        "clinical_claim_bundle": {
            "question_type": "treatment_efficacy",
            "claims": [
                {
                    "axis_id": "sglt2_hf_hospitalization",
                    "axis_label": "SGLT2 ↓ hospitalización por insuficiencia cardíaca",
                    "statement": "Beneficio en IC.",
                    "direction": "benefit",
                    "evidence_strength": "moderate",
                    "consistency": "consistent",
                    "confidence": 0.7,
                    "landmark_support": ["EMPA-REG OUTCOME"],
                    "support": [{"pmid": "1"}],
                    "contradicting": [],
                }
            ],
            "unresolved_conflicts": [],
        },
        "clinical_claims_summary": "### Claim: SGLT2 ↓ HF",
        "sintesis_modo": "claim_first",
    }
    facts = _build_facts_dict(state, medical_answer)
    assert facts_use_claim_first(facts)
    assert facts["sintesis_modo"] == "claim_first"
    assert facts["citas_pubmed"] == []
    assert facts["claims_clinicos"] is not None
    assert len(facts["indice_pmids"]) == 1
    blob = json.dumps(facts, ensure_ascii=False).lower()
    assert facts["citas_pubmed"] == []
    assert "extracto_del_resumen" not in blob
