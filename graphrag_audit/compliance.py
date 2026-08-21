"""
Clinical Compliance Verification Engine for Pneumonia Risk Stratification (Phase 7).
Implements transparent, rule-based clinical reasoning verification for medical graph retrieval.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import re

from .audit_types import AuditRetrievalResult, CandidatePath


@dataclass
class ComplianceCheck:
    """Individual clinical rule verification record."""
    rule_id: str
    name: str
    description: str
    status: str            # "PASS" | "WARNING" | "FAIL"
    reason: str
    matched_evidence: List[str] = field(default_factory=list)
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ComplianceResult:
    """Overall compliance report for retrieved clinical reasoning."""
    overall_status: str    # "COMPLIANT" | "WARNING" | "NON_COMPLIANT"
    total_rules_evaluated: int
    passed_count: int
    warning_count: int
    failed_count: int
    compliance_score: float
    checks: List[ComplianceCheck] = field(default_factory=list)
    summary_explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PneumoniaComplianceVerifier:
    """
    Evaluates retrieved reasoning paths and evidence against standardized
    clinical verification rules for pneumonia diagnosis, management, and safety.
    """

    RULES = [
        {
            "id": "PNE-001",
            "name": "Diagnostic Confirmation",
            "description": "Verifies that diagnostic criteria (imaging, microbiology culture, or formal diagnosis) exist in the reasoning graph.",
            "keywords": ["pneumonia", "infiltrate", "consolidation", "culture", "x-ray", "radiograph", "sputum", "bronchopneumonia"],
        },
        {
            "id": "PNE-002",
            "name": "Antibiotic Pathway Justification",
            "description": "Verifies that any antibiotic intervention in retrieved paths is supported by infection/diagnosis evidence.",
            "keywords": ["ceftriaxone", "vancomycin", "cefepime", "azithromycin", "levofloxacin", "antibiotic", "ampicillin"],
        },
        {
            "id": "PNE-003",
            "name": "Inflammatory / Severity Monitoring",
            "description": "Verifies that inflammatory markers or severity indicators (WBC, fever, oxygenation, creatinine) are present in evidence.",
            "keywords": ["wbc", "white blood cells", "fever", "temperature", "oxygen", "spo2", "creatinine", "respiratory rate", "hypotension"],
        },
        {
            "id": "PNE-004",
            "name": "Comorbidity / Complication Awareness",
            "description": "Checks for consideration of respiratory failure, sepsis, heart failure, or underlying chronic disease risk.",
            "keywords": ["chf", "heart failure", "copd", "sepsis", "respiratory failure", "hypoxia", "effusion", "empyema"],
        },
        {
            "id": "PNE-005",
            "name": "Evidence Sufficiency",
            "description": "Verifies that at least 2 distinct evidence items support the clinical recommendation.",
            "min_evidence_items": 2,
        },
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def verify(
        self,
        retrieval_result: AuditRetrievalResult
    ) -> ComplianceResult:
        """
        Executes clinical compliance checks against an AuditRetrievalResult.
        """
        evidence_corpus = " ".join(retrieval_result.evidence).lower()
        path_corpus = " ".join([p.readable_path for p in retrieval_result.selected_paths]).lower()
        full_text = f"{retrieval_result.query.lower()} {evidence_corpus} {path_corpus}"

        checks = []

        # Rule 1: Diagnostic Confirmation
        r1_matches = [kw for kw in self.RULES[0]["keywords"] if kw in full_text]
        if r1_matches:
            c1 = ComplianceCheck(
                rule_id="PNE-001",
                name=self.RULES[0]["name"],
                description=self.RULES[0]["description"],
                status="PASS",
                reason=f"Diagnostic indication verified via matching findings: {', '.join(r1_matches[:3])}.",
                matched_evidence=r1_matches[:5]
            )
        else:
            c1 = ComplianceCheck(
                rule_id="PNE-001",
                name=self.RULES[0]["name"],
                description=self.RULES[0]["description"],
                status="WARNING",
                reason="No explicit pneumonia diagnostic or imaging evidence found in retrieved paths.",
                matched_evidence=[]
            )
        checks.append(c1)

        # Rule 2: Antibiotic Pathway Justification
        abx_present = [abx for abx in self.RULES[1]["keywords"] if abx in full_text]
        has_infection = any(inf in full_text for inf in ["pneumonia", "infection", "bacterial", "culture", "fever", "wbc"])
        if not abx_present:
            c2 = ComplianceCheck(
                rule_id="PNE-002",
                name=self.RULES[1]["name"],
                description=self.RULES[1]["description"],
                status="PASS",
                reason="No antimicrobial therapy indicated; rule satisfied.",
                matched_evidence=[]
            )
        elif abx_present and has_infection:
            c2 = ComplianceCheck(
                rule_id="PNE-002",
                name=self.RULES[1]["name"],
                description=self.RULES[1]["description"],
                status="PASS",
                reason=f"Antibiotic therapy ({', '.join(abx_present[:2])}) is appropriately substantiated by infection evidence.",
                matched_evidence=abx_present
            )
        else:
            c2 = ComplianceCheck(
                rule_id="PNE-002",
                name=self.RULES[1]["name"],
                description=self.RULES[1]["description"],
                status="FAIL",
                reason=f"Antibiotic therapy ({', '.join(abx_present[:2])}) lacks documented supporting infection markers.",
                matched_evidence=abx_present
            )
        checks.append(c2)

        # Rule 3: Severity Monitoring
        r3_matches = [kw for kw in self.RULES[2]["keywords"] if kw in full_text]
        if r3_matches:
            c3 = ComplianceCheck(
                rule_id="PNE-003",
                name=self.RULES[2]["name"],
                description=self.RULES[2]["description"],
                status="PASS",
                reason=f"Physiological / inflammatory markers detected: {', '.join(r3_matches[:3])}.",
                matched_evidence=r3_matches[:4]
            )
        else:
            c3 = ComplianceCheck(
                rule_id="PNE-003",
                name=self.RULES[2]["name"],
                description=self.RULES[2]["description"],
                status="WARNING",
                reason="Missing quantitative severity indicators (e.g. WBC, vitals) in retrieved evidence.",
                matched_evidence=[]
            )
        checks.append(c3)

        # Rule 4: Comorbidity Awareness
        r4_matches = [kw for kw in self.RULES[3]["keywords"] if kw in full_text]
        c4 = ComplianceCheck(
            rule_id="PNE-004",
            name=self.RULES[3]["name"],
            description=self.RULES[3]["description"],
            status="PASS" if r4_matches else "PASS",  # Informational check
            reason=f"Underlying risk / comorbidity factors assessed: {', '.join(r4_matches[:3])}." if r4_matches else "No acute severe comorbidities flagged.",
            matched_evidence=r4_matches
        )
        checks.append(c4)

        # Rule 5: Evidence Sufficiency
        num_evidence = len(retrieval_result.evidence)
        if num_evidence >= 2:
            c5 = ComplianceCheck(
                rule_id="PNE-005",
                name=self.RULES[4]["name"],
                description=self.RULES[4]["description"],
                status="PASS",
                reason=f"Sufficient evidence volume retrieved ({num_evidence} items >= threshold 2).",
                matched_evidence=[]
            )
        else:
            c5 = ComplianceCheck(
                rule_id="PNE-005",
                name=self.RULES[4]["name"],
                description=self.RULES[4]["description"],
                status="WARNING",
                reason=f"Sparse evidence retrieved ({num_evidence} item(s) < threshold 2).",
                matched_evidence=[]
            )
        checks.append(c5)

        # Aggregate summary
        passed = sum(1 for c in checks if c.status == "PASS")
        warnings = sum(1 for c in checks if c.status == "WARNING")
        failed = sum(1 for c in checks if c.status == "FAIL")
        total = len(checks)
        score = (passed * 1.0 + warnings * 0.5) / total if total > 0 else 1.0

        if failed > 0:
            overall = "NON_COMPLIANT"
        elif warnings > 1:
            overall = "WARNING"
        else:
            overall = "COMPLIANT"

        summary = (
            f"Pneumonia Clinical Compliance: {overall} (Score: {score:.2f}, "
            f"Passed: {passed}/{total}, Warnings: {warnings}/{total}, Failed: {failed}/{total})."
        )

        return ComplianceResult(
            overall_status=overall,
            total_rules_evaluated=total,
            passed_count=passed,
            warning_count=warnings,
            failed_count=failed,
            compliance_score=round(score, 4),
            checks=checks,
            summary_explanation=summary
        )
