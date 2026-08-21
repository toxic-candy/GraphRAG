"""
Unit tests for Auditable Retrieval (Phase 3).
Verifies candidate path construction, confidence scoring, alternative path rejection logging, and audit trail generation.
"""

import pytest
import json
from graphrag_audit import (
    CandidateNode,
    CandidateEdge,
    CandidatePath,
    AuditRetrievalResult,
    AuditableRetriever,
)


class MockNeo4jDriver:
    """Mock database driver simulating Neo4j Cypher queries for testing."""

    def __init__(self, summary_rows=None, intra_rows=None, ref_rows=None):
        self.summary_rows = summary_rows if summary_rows is not None else [
            {
                "gid": "gid_pneumonia_01",
                "content": ["Patient diagnosed with acute bacterial pneumonia showing fever and elevated WBC."],
            },
            {
                "gid": "gid_copd_02",
                "content": ["Patient with chronic obstructive pulmonary disease without acute exacerbation."],
            },
            {
                "gid": "gid_chf_03",
                "content": ["Patient admitted for congestive heart failure evaluation."],
            },
        ]
        self.intra_rows = intra_rows if intra_rows is not None else [
            {
                "source_id": "Pneumonia",
                "source_type": "Diagnosis",
                "source_props": {"source": "structured_fallback"},
                "target_id": "Ceftriaxone",
                "target_type": "Medication",
                "target_props": {"source": "structured_fallback"},
                "rel_type": "treats",
                "rel_props": {
                    "source_type": "structural",
                    "provenance": "structurally_generated",
                    "extraction_method": "round_robin_assignment"
                },
                "gid": "gid_pneumonia_01"
            },
            {
                "source_id": "WBC 14.8",
                "source_type": "LabTest",
                "source_props": {"source": "structured_fallback"},
                "target_id": "Pneumonia",
                "target_type": "Diagnosis",
                "target_props": {"source": "structured_fallback"},
                "rel_type": "monitors",
                "rel_props": {
                    "source_type": "rule_based",
                    "provenance": "evidence_backed",
                    "extraction_method": "lab_finding"
                },
                "gid": "gid_pneumonia_01"
            },
        ]
        self.ref_rows = ref_rows if ref_rows is not None else [
            {
                "n_id": "Pneumonia",
                "n_type": "Diagnosis",
                "n_gid": "gid_pneumonia_01",
                "m_id": "Bacterial Pneumonia",
                "m_type": "Diagnosis",
                "m_gid": "gid_guideline_01",
                "o_id": "Vancomycin",
                "o_type": "Procedure",
                "o_gid": "gid_guideline_01",
                "ref_type": "REFERENCE",
                "conn_type": "indicated_for",
                "ref_props": {"source": "trinity_linker"},
                "conn_props": {"provenance": "dictionary_derived"}
            }
        ]

    def query(self, cypher: str, params: dict = None):
        cypher_clean = cypher.strip()
        if "MATCH (s:Summary)" in cypher_clean:
            return self.summary_rows
        elif "MATCH (n)-[r:REFERENCE]->(m)" in cypher_clean:
            return self.ref_rows
        elif "MATCH (n)-[r]->(m)" in cypher_clean:
            return self.intra_rows
        return []


def test_audit_data_structures():
    """Test candidate models and serialization."""
    node = CandidateNode(id="Pneumonia", gid="gid_01", type="Diagnosis")
    assert node.id == "Pneumonia"

    edge = CandidateEdge(
        source_id="Pneumonia",
        target_id="Antibiotic",
        relation_type="treats",
        confidence_score=0.85
    )
    assert edge.display_str == "(Pneumonia) -[treats]-> (Antibiotic)"

    path = CandidatePath(
        path_id="path_001",
        nodes=[{"id": "Pneumonia"}, {"id": "Antibiotic"}],
        edges=[edge.to_dict()],
        path_score=0.85,
        selection_status="selected"
    )
    assert "Pneumonia --[treats]--> Antibiotic" in path.readable_path

    audit_res = AuditRetrievalResult(
        query="What antibiotic treats pneumonia?",
        candidate_paths=[path],
        selected_paths=[path],
        rejected_paths=[]
    )
    data = audit_res.to_dict()
    assert data["query"] == "What antibiotic treats pneumonia?"
    assert len(data["selected_paths"]) == 1

    json_str = audit_res.to_json()
    assert json.loads(json_str)["query"] == "What antibiotic treats pneumonia?"


def test_auditable_retriever_execution():
    """Test end-to-end AuditableRetriever trace generation with mock database."""
    mock_db = MockNeo4jDriver()
    retriever = AuditableRetriever(
        driver=mock_db,
        config={
            "retrieval": {
                "top_k": 2,
                "max_hops": 2,
                "max_evidence": 50,
                "min_confidence_threshold": 0.0
            }
        }
    )

    query = "What evidence supports bacterial pneumonia diagnosis and treatment?"
    result = retriever.retrieve(query=query)

    # 1. Result must contain basic metadata
    assert isinstance(result, AuditRetrievalResult)
    assert result.query == query
    assert len(result.seed_nodes) == 3

    # 2. Verify selected vs. rejected seeds
    selected_seeds = [s for s in result.seed_nodes if s["status"] == "selected"]
    rejected_seeds = [s for s in result.seed_nodes if s["status"] == "rejected"]
    assert len(selected_seeds) <= 2
    assert len(rejected_seeds) >= 1
    assert rejected_seeds[0]["rejection_reason"] is not None

    # 3. Verify candidate paths and path scoring
    assert len(result.candidate_paths) > 0
    for p in result.candidate_paths:
        assert p.path_score >= 0.0
        assert p.selection_status in ["selected", "rejected"]

    # 4. Verify rejected alternative paths log explicit reasons
    assert len(result.selected_paths) > 0
    for rej_path in result.rejected_paths:
        assert rej_path.rejection_reason is not None

    # 5. Verify edge confidence decomposition
    assert len(result.edge_confidence) > 0
    for ec in result.edge_confidence:
        assert "semantic_score" in ec
        assert "graph_support" in ec
        assert "source_reliability" in ec
        assert "stability" in ec
        assert "confidence" in ec

    # 6. Verify evidence and decision explanation
    assert len(result.evidence) > 0
    assert len(result.decision_rationale) > 20
    assert "Retriever evaluated" in result.decision_rationale


def test_auditable_retriever_threshold_rejection():
    """Test that candidate paths falling below min_confidence_threshold are properly rejected."""
    mock_db = MockNeo4jDriver()
    retriever = AuditableRetriever(
        driver=mock_db,
        config={
            "retrieval": {
                "top_k": 3,
                "max_hops": 2,
                "min_confidence_threshold": 0.999  # Intentionally high to trigger rejection
            }
        }
    )
    result = retriever.retrieve(query="Pneumonia treatment")
    # All candidates should be rejected with appropriate reasons
    for path in result.candidate_paths:
        assert path.selection_status == "rejected"
        assert "below minimum threshold" in path.rejection_reason


def test_auditable_retriever_empty_graph():
    """Test graceful handling when no nodes or relationships exist in graph."""
    empty_db = MockNeo4jDriver(summary_rows=[], intra_rows=[], ref_rows=[])
    retriever = AuditableRetriever(driver=empty_db)
    result = retriever.retrieve(query="Non-existent condition")

    assert result.seed_nodes == []
    assert result.candidate_paths == []
    assert result.selected_paths == []
    assert result.rejected_paths == []
    assert result.evidence == []
    assert "Retriever evaluated 0" in result.decision_rationale


def test_auditable_retriever_custom_confidence_weights():
    """Test that custom confidence weights correctly impact edge and path confidence."""
    mock_db = MockNeo4jDriver()
    custom_weights = {
        "semantic": 0.10,
        "graph_support": 0.50,
        "source_reliability": 0.30,
        "stability": 0.10,
    }
    retriever = AuditableRetriever(
        driver=mock_db,
        config={
            "retrieval": {"top_k": 2, "min_confidence_threshold": 0.0},
            "confidence_weights": custom_weights
        }
    )
    result = retriever.retrieve(query="Pneumonia treatment")
    assert result.retrieval_parameters["confidence_weights"] == custom_weights
    assert len(result.edge_confidence) > 0

