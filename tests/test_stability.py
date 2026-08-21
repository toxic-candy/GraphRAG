"""
Unit tests for Retrieval Stability Testing (Phase 5).
Verifies keyword removal, masking, numeric perturbations, Jaccard similarities, and stability report generation.
"""

import pytest
from graphrag_audit import (
    RetrievalStabilityTester,
    StabilityReport,
    AuditableRetriever,
    CandidatePath,
    CandidateEdge,
)
from tests.test_audit_retrieval import MockNeo4jDriver


def test_perturbation_operators():
    tester = RetrievalStabilityTester()
    query = "Patient has severe bacterial pneumonia with WBC 15.4 K/uL and fever"

    # 1. Removal
    removed = tester.perturb_remove_keyword(query, "fever")
    assert "fever" not in removed
    assert "pneumonia" in removed

    # 2. Masking
    masked = tester.perturb_mask_keyword(query, "pneumonia")
    assert "[MASKED]" in masked
    assert "pneumonia" not in masked.lower()

    # 3. Numeric perturbation
    num_perturbed = tester.perturb_numeric_values(query, factor=0.20)
    assert "18.5" in num_perturbed or "15.4" not in num_perturbed


def test_jaccard_and_mae_metrics():
    tester = RetrievalStabilityTester()

    e1 = CandidateEdge("Pneumonia", "treats", "Ceftriaxone")
    e2 = CandidateEdge("WBC", "monitors", "Pneumonia")

    p1 = CandidatePath(path_id="p1", nodes=[], edges=[e1.to_dict()], path_score=0.85)
    p2 = CandidatePath(path_id="p2", nodes=[], edges=[e2.to_dict()], path_score=0.75)
    p3 = CandidatePath(path_id="p3", nodes=[], edges=[e1.to_dict()], path_score=0.80)

    # Identical sets -> Jaccard 1.0
    assert tester.compute_path_jaccard([p1], [p1]) == 1.0
    assert tester.compute_edge_jaccard([p1], [p1]) == 1.0

    # Disjoint sets -> Jaccard 0.0
    assert tester.compute_path_jaccard([p1], [p2]) == 0.0

    # Confidence MAE for shared path p1 vs p3 (|0.85 - 0.80| = 0.05)
    mae = tester.compute_confidence_mae([p1], [p3])
    assert mae == pytest.approx(0.05, rel=1e-3)


def test_evaluate_stability_execution():
    mock_db = MockNeo4jDriver()
    retriever = AuditableRetriever(driver=mock_db, config={"retrieval": {"min_confidence_threshold": 0.0}})
    tester = RetrievalStabilityTester()

    query = "What evidence supports bacterial pneumonia diagnosis?"
    report = tester.evaluate_stability(query=query, retriever=retriever, keywords_to_test=["pneumonia", "bacterial"])

    assert isinstance(report, StabilityReport)
    assert report.query == query
    assert report.num_perturbations >= 2
    assert 0.0 <= report.mean_path_jaccard <= 1.0
    assert 0.0 <= report.mean_edge_jaccard <= 1.0
    assert 0.0 <= report.top_path_retention_rate <= 1.0
    assert report.stability_grade in ["HIGH", "MODERATE", "LOW"]
    assert len(report.perturbed_evaluations) >= 2
