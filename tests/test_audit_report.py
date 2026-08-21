"""
Unit tests for Structured Audit Report Generation (Phase 8).
Verifies complete audit report compilation, serialization to JSON, and Markdown generation.
"""

import pytest
import json
from graphrag_audit import (
    AuditReport,
    AuditReportCompiler,
    AuditableRetriever,
)
from tests.test_audit_retrieval import MockNeo4jDriver


def test_audit_report_serialization():
    report = AuditReport(
        query="What treatments are indicated for pneumonia?",
        patient_id="patient_10000690",
        selected_paths=[
            {"path_id": "path_001", "readable_path": "Pneumonia --[treats]--> Ceftriaxone", "path_score": 0.82}
        ],
        edge_confidences=[
            {
                "edge": "(Pneumonia)-[treats]->(Ceftriaxone)",
                "confidence": 0.82,
                "semantic_score": 0.85,
                "graph_support": 0.70,
                "source_reliability": 0.90,
                "provenance": {"provenance": "evidence_backed"}
            }
        ],
        evidence=["Pneumonia treats Ceftriaxone"],
        answer="Ceftriaxone is indicated based on retrieved evidence.",
        decision_rationale="Evaluated candidate subgraphs and selected top matching paths."
    )

    # 1. JSON serialization
    json_str = report.to_json()
    parsed = json.loads(json_str)
    assert parsed["query"] == "What treatments are indicated for pneumonia?"
    assert parsed["patient_id"] == "patient_10000690"
    assert len(parsed["selected_paths"]) == 1

    # 2. Markdown generation
    md = report.to_markdown()
    assert "# Clinical Graph Audit Report" in md
    assert "Pneumonia --[treats]--> Ceftriaxone" in md
    assert "Ceftriaxone is indicated" in md


def test_audit_report_compiler_end_to_end():
    mock_db = MockNeo4jDriver()
    retriever = AuditableRetriever(driver=mock_db, config={"retrieval": {"min_confidence_threshold": 0.0}})
    compiler = AuditReportCompiler(retriever=retriever)

    report = compiler.generate_full_report(
        query="What evidence supports bacterial pneumonia diagnosis and treatment?",
        patient_id="patient_10000690",
        run_stability=True,
        run_counterfactuals=True,
        run_compliance=True
    )

    assert isinstance(report, AuditReport)
    assert report.query == "What evidence supports bacterial pneumonia diagnosis and treatment?"
    assert report.patient_id == "patient_10000690"
    assert len(report.selected_paths) > 0
    assert report.stability_metrics is not None
    assert "mean_path_jaccard" in report.stability_metrics
    assert len(report.counterfactual_tests) > 0
    assert report.compliance_verification is not None
    assert report.compliance_verification["overall_status"] in ["COMPLIANT", "WARNING", "NON_COMPLIANT"]

    # Verify report is valid JSON
    assert json.loads(report.to_json()) is not None
