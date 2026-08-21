"""
Unit tests for Clinical Compliance Verification (Phase 7).
Verifies rule-based evaluation (PNE-001 through PNE-005) against retrieved clinical reasoning paths.
"""

import pytest
from graphrag_audit import (
    PneumoniaComplianceVerifier,
    ComplianceResult,
    ComplianceCheck,
    AuditRetrievalResult,
    CandidatePath,
    CandidateEdge,
)


def test_compliance_verifier_full_pass():
    verifier = PneumoniaComplianceVerifier()

    # Create mock retrieval result with complete pneumonia evidence and antibiotic indication
    e1 = CandidateEdge(source_id="Pneumonia", relation_type="treats", target_id="Ceftriaxone")
    e2 = CandidateEdge(source_id="WBC 14.8", relation_type="monitors", target_id="Pneumonia")

    p1 = CandidatePath(path_id="p1", nodes=[], edges=[e1.to_dict()], evidence_text="Pneumonia treats Ceftriaxone")
    p2 = CandidatePath(path_id="p2", nodes=[], edges=[e2.to_dict()], evidence_text="WBC 14.8 monitors Pneumonia")

    retrieval_res = AuditRetrievalResult(
        query="What is the recommended antibiotic for acute bacterial pneumonia with elevated WBC?",
        selected_paths=[p1, p2],
        evidence=[
            "Pneumonia treats Ceftriaxone",
            "WBC 14.8 monitors Pneumonia",
            "Chest X-Ray shows right lower lobe infiltrate"
        ]
    )

    result = verifier.verify(retrieval_res)
    assert isinstance(result, ComplianceResult)
    assert result.overall_status == "COMPLIANT"
    assert result.compliance_score >= 0.80
    assert result.total_rules_evaluated == 5

    # Verify individual checks
    rule_ids = {c.rule_id: c.status for c in result.checks}
    assert rule_ids["PNE-001"] == "PASS"  # Diagnostic confirmation
    assert rule_ids["PNE-002"] == "PASS"  # Antibiotic pathway justified
    assert rule_ids["PNE-003"] == "PASS"  # Inflammatory / severity monitoring


def test_compliance_verifier_warning_sparse_evidence():
    verifier = PneumoniaComplianceVerifier()

    # Insufficient evidence (only 1 item)
    retrieval_res = AuditRetrievalResult(
        query="What is the status?",
        selected_paths=[],
        evidence=["Pneumonia diagnosed"]
    )

    result = verifier.verify(retrieval_res)
    assert result.overall_status in ["WARNING", "NON_COMPLIANT"]
    rule_ids = {c.rule_id: c.status for c in result.checks}
    assert rule_ids["PNE-005"] == "WARNING"
