"""
Unit tests for Counterfactual Retrieval Engine (Phase 6).
Verifies intervention application, counterfactual retrieval execution, path delta tracking, and explanation generation.
"""

import pytest
from graphrag_audit import (
    CounterfactualEngine,
    CounterfactualResult,
    AuditableRetriever,
)
from tests.test_audit_retrieval import MockNeo4jDriver


def test_intervention_application():
    engine = CounterfactualEngine()
    query = "Patient has severe bacterial pneumonia with fever and elevated WBC"

    # REMOVE_EVIDENCE
    cf_rem = engine.apply_intervention(query, {"type": "REMOVE_EVIDENCE", "target": "elevated WBC"})
    assert "elevated WBC" not in cf_rem
    assert "pneumonia" in cf_rem

    # MASK_EVIDENCE
    cf_mask = engine.apply_intervention(query, {"type": "MASK_EVIDENCE", "target": "pneumonia"})
    assert "[MASKED]" in cf_mask

    # MODIFY_FINDING
    cf_mod = engine.apply_intervention(query, {"type": "MODIFY_FINDING", "target": "bacterial", "replacement": "viral"})
    assert "viral pneumonia" in cf_mod


def test_counterfactual_retrieval_execution():
    mock_db = MockNeo4jDriver()
    retriever = AuditableRetriever(driver=mock_db, config={"retrieval": {"min_confidence_threshold": 0.0}})
    engine = CounterfactualEngine()

    query = "What evidence supports bacterial pneumonia treatment?"
    intervention = {
        "type": "REMOVE_EVIDENCE",
        "target": "pneumonia",
        "clinical_rationale": "Evaluate if treatment path persists without explicit pneumonia indication."
    }

    cf_res = engine.counterfactual_retrieve(
        query=query,
        intervention=intervention,
        retriever=retriever
    )

    assert isinstance(cf_res, CounterfactualResult)
    assert cf_res.original_query == query
    assert "pneumonia" not in cf_res.counterfactual_query.lower()
    assert cf_res.intervention == intervention
    assert isinstance(cf_res.retained_paths, list)
    assert isinstance(cf_res.removed_paths, list)
    assert isinstance(cf_res.added_paths, list)
    assert len(cf_res.counterfactual_explanation) > 20
    assert "Counterfactual test applied" in cf_res.counterfactual_explanation


def test_counterfactual_suite():
    mock_db = MockNeo4jDriver()
    retriever = AuditableRetriever(driver=mock_db, config={"retrieval": {"min_confidence_threshold": 0.0}})
    engine = CounterfactualEngine()

    query = "What treatments and lab tests are indicated for pneumonia?"
    suite = [
        {"type": "REMOVE_EVIDENCE", "target": "pneumonia"},
        {"type": "MASK_EVIDENCE", "target": "lab tests"},
    ]

    results = engine.run_suite(query=query, retriever=retriever, interventions=suite)
    assert len(results) == 2
    for r in results:
        assert isinstance(r, CounterfactualResult)
        assert r.original_query == query
