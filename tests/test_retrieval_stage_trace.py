import pytest
from app.orchestration.executor import _aggregate_retrieval_debug

def test_aggregate_retrieval_debug():
    rd = {"original": "val"}
    stages = [
        {"stage_id": "stage1", "pmids": [], "articles_count": 0, "raw_debug": {"attempts": [{"query": "bad"}]}},
        {"stage_id": "stage2", "pmids": ["123"], "articles_count": 1, "raw_debug": {"attempts": [{"query": "good"}]}}
    ]
    merged = [{"pmid": "123"}]
    res = _aggregate_retrieval_debug(rd, stages, merged)
    
    assert res["outcome"] == "partial_primary_miss"
    assert len(res["attempts"]) == 2
    assert res["attempts"][0]["stage_id"] == "stage1"
    assert res["attempts"][1]["stage_id"] == "stage2"
    assert res["attempts"][1]["pmids_from_stage"] == ["123"]

