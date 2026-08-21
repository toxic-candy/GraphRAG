"""
Unit tests for Edge-Level and Path-Level Confidence Scoring (Phase 4).
Verifies component score calculations, provenance mappings, aggregation methods, and path scoring.
"""

import pytest
from graphrag_audit import EdgeConfidenceScorer
from utils import get_embedding


def test_confidence_semantic_score():
    scorer = EdgeConfidenceScorer()
    query_emb = get_embedding("bacterial pneumonia infection and antibiotic therapy")

    # Relevant triple vs unrelated triple
    score_rel = scorer.semantic_score("Pneumonia", "treats", "Ceftriaxone", query_embedding=query_emb)
    score_unrel = scorer.semantic_score("Fracture", "causes", "Bone pain", query_embedding=query_emb)

    assert 0.0 <= score_rel <= 1.0
    assert 0.0 <= score_unrel <= 1.0


def test_confidence_graph_support_score():
    scorer = EdgeConfidenceScorer()

    # REFERENCE edge should have high topological support
    ref_score = scorer.graph_support_score("Pneumonia", "REFERENCE", "Bacterial Pneumonia", properties={})
    assert ref_score >= 0.80

    # ICD chapter grouped edge
    icd_score = scorer.graph_support_score("Pneumonia", "icd_related", "Bronchopneumonia", properties={"icd_chapter": "480"})
    assert icd_score >= 0.70

    # Structural round-robin fallback edge
    rr_score = scorer.graph_support_score("Lisinopril", "treats", "Pneumonia", properties={"extraction_method": "round_robin_assignment"})
    assert rr_score <= 0.50


def test_confidence_source_reliability():
    scorer = EdgeConfidenceScorer()

    # MIMIC evidence-backed
    mimic_rel = scorer.source_reliability_score(source_type="MIMIC", provenance="evidence_backed")
    assert mimic_rel >= 0.90

    # Guideline evidence-backed
    guide_rel = scorer.source_reliability_score(source_type="clinical_guideline", provenance="evidence_backed")
    assert guide_rel >= 0.85

    # Dictionary derived
    dict_rel = scorer.source_reliability_score(source_type="dictionary", provenance="dictionary_derived")
    assert dict_rel >= 0.70

    # Structurally generated (round-robin fallback)
    struct_rel = scorer.source_reliability_score(source_type="structural", provenance="structurally_generated")
    assert struct_rel <= 0.50


def test_confidence_aggregations():
    # 1. Weighted Sum (default)
    scorer_ws = EdgeConfidenceScorer(config={"aggregation_method": "weighted_sum"})
    res_ws = scorer_ws.calculate_edge_confidence(
        source_id="Pneumonia",
        relation_type="treats",
        target_id="Vancomycin",
        query_embedding=get_embedding("pneumonia treatment"),
        properties={"source_type": "MIMIC", "provenance": "evidence_backed", "extraction_method": "mimic_structured_admission"}
    )
    assert "confidence" in res_ws
    assert 0.0 <= res_ws["confidence"] <= 1.0

    # 2. Multiplicative (Geometric Mean)
    scorer_mul = EdgeConfidenceScorer(config={"aggregation_method": "multiplicative"})
    res_mul = scorer_mul.calculate_edge_confidence(
        source_id="Pneumonia",
        relation_type="treats",
        target_id="Vancomycin",
        query_embedding=get_embedding("pneumonia treatment"),
        properties={"source_type": "MIMIC", "provenance": "evidence_backed"}
    )
    assert res_mul["aggregation"] == "multiplicative"
    assert 0.0 <= res_mul["confidence"] <= 1.0

    # 3. Harmonic Mean
    scorer_hm = EdgeConfidenceScorer(config={"aggregation_method": "harmonic_mean"})
    res_hm = scorer_hm.calculate_edge_confidence(
        source_id="Pneumonia",
        relation_type="treats",
        target_id="Vancomycin",
        query_embedding=get_embedding("pneumonia treatment"),
        properties={"source_type": "MIMIC", "provenance": "evidence_backed"}
    )
    assert res_hm["aggregation"] == "harmonic_mean"
    assert 0.0 <= res_hm["confidence"] <= 1.0


def test_path_confidence_scoring():
    scorer = EdgeConfidenceScorer(config={"hop_penalty_factor": 0.90})

    # Single hop
    path_1hop = scorer.calculate_path_confidence(edge_scores=[0.80], hop_count=1)
    assert path_1hop == pytest.approx(0.80, rel=1e-3)

    # 2 hops with penalty (0.80 * 0.90 = 0.72)
    path_2hop = scorer.calculate_path_confidence(edge_scores=[0.80, 0.80], hop_count=2)
    assert path_2hop == pytest.approx(0.72, rel=1e-3)
    assert path_2hop < path_1hop
