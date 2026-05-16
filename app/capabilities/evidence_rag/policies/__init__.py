from dataclasses import dataclass
from typing import Callable, Optional

@dataclass(frozen=True, slots=True)
class RetrievalPolicy:
    theme: str
    requires_cvot: bool
    epistemic_boost_allowed: bool
    pubmed_broad_clause: Optional[str] = None
    landmark_retrieval_allowed: bool = True

RETRIEVAL_POLICIES = {
    "cv": RetrievalPolicy(
        theme="cv",
        requires_cvot=True,
        epistemic_boost_allowed=True,
        pubmed_broad_clause='("cardiovascular disease"[tiab] OR "major adverse cardiovascular events"[tiab] OR "MACE"[tiab] OR "heart failure"[tiab])',
    ),
    "renal": RetrievalPolicy(
        theme="renal",
        requires_cvot=False,
        epistemic_boost_allowed=True,
        pubmed_broad_clause='("chronic kidney disease"[tiab] OR "renal outcomes"[tiab] OR "eGFR"[tiab])',
    ),
    "safety": RetrievalPolicy(
        theme="safety",
        requires_cvot=False,
        epistemic_boost_allowed=False,
        pubmed_broad_clause='("adverse events"[tiab] OR "safety"[tiab] OR "tolerability"[tiab])',
    ),
    "general": RetrievalPolicy(
        theme="general",
        requires_cvot=False,
        epistemic_boost_allowed=True,
        pubmed_broad_clause=None
    )
}
