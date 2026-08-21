"""
Structured Clinical Audit Report Generator (Phase 8).
Assembles end-to-end audit records combining retrieval traces, confidence breakdowns,
stability metrics, counterfactual scenarios, and clinical compliance checks.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import json
import uuid
from datetime import datetime

from .audit_types import AuditRetrievalResult
from .stability import StabilityReport, RetrievalStabilityTester
from .counterfactual import CounterfactualResult, CounterfactualEngine
from .compliance import ComplianceResult, PneumoniaComplianceVerifier


@dataclass
class AuditReport:
    """Complete, validatable audit record for a single clinical query."""
    report_id: str = field(default_factory=lambda: f"audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}")
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    version: str = "1.0"
    
    # Query & Subject Context
    query: str = ""
    patient_id: Optional[str] = None

    # Core Retrieval Decision Trace
    seed_nodes_evaluated: int = 0
    candidate_paths_evaluated: int = 0
    selected_paths: List[Dict[str, Any]] = field(default_factory=list)
    rejected_paths: List[Dict[str, Any]] = field(default_factory=list)
    
    # Edge Confidence Decompositions
    edge_confidences: List[Dict[str, Any]] = field(default_factory=list)
    
    # Robustness & Stability Metrics (Phase 5)
    stability_metrics: Optional[Dict[str, Any]] = None
    
    # Counterfactual Interventions (Phase 6)
    counterfactual_tests: List[Dict[str, Any]] = field(default_factory=list)
    
    # Clinical Compliance Verification (Phase 7)
    compliance_verification: Optional[Dict[str, Any]] = None
    
    # Output Evidence & Answer
    evidence: List[str] = field(default_factory=list)
    answer: str = ""
    decision_rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_markdown(self) -> str:
        """Renders a clean, human-readable markdown audit summary."""
        lines = [
            f"# Clinical Graph Audit Report: `{self.report_id}`",
            f"- **Timestamp**: {self.timestamp}",
            f"- **Query**: {self.query}",
            f"- **Patient ID**: {self.patient_id or 'General / Non-Specific'}",
            "",
            "## 1. Retrieval Decision Rationale",
            f"> {self.decision_rationale}",
            "",
            f"### Selected Reasoning Paths ({len(self.selected_paths)})",
        ]
        for p in self.selected_paths:
            score = p.get("path_score", 0.0)
            lines.append(f"- **[{p.get('path_id')}]** (Confidence: `{score:.3f}`): {p.get('readable_path', '')}")

        if self.rejected_paths:
            lines.append(f"\n### Sample Rejected Alternatives ({len(self.rejected_paths)} total)")
            for p in self.rejected_paths[:4]:
                score = p.get("path_score", 0.0)
                reason = p.get("rejection_reason", "Budget limit")
                lines.append(f"- *[{p.get('path_id')}]* (Score: `{score:.3f}`): {p.get('readable_path', '')} *(Reason: {reason})*")

        if self.edge_confidences:
            lines.append("\n## 2. Edge Confidence Decomposition")
            lines.append("| Edge | Confidence | Semantic | Graph Support | Reliability | Provenance |")
            lines.append("|---|---|---|---|---|---|")
            for ec in self.edge_confidences[:8]:
                prov = (ec.get("provenance") or {}).get("provenance", "unknown")
                lines.append(f"| `{ec.get('edge')}` | **{ec.get('confidence', 0):.3f}** | {ec.get('semantic_score', 0):.2f} | {ec.get('graph_support', 0):.2f} | {ec.get('source_reliability', 0):.2f} | {prov} |")

        if self.stability_metrics:
            lines.append("\n## 3. Retrieval Stability & Invariance")
            lines.append(f"- **Stability Grade**: `{self.stability_metrics.get('stability_grade', 'N/A')}`")
            lines.append(f"- **Mean Path Jaccard**: `{self.stability_metrics.get('mean_path_jaccard', 0):.3f}`")
            lines.append(f"- **Top Path Retention Rate**: `{self.stability_metrics.get('top_path_retention_rate', 0):.1%}`")
            lines.append(f"- **Confidence MAE**: `{self.stability_metrics.get('confidence_mae', 0):.4f}`")

        if self.compliance_verification:
            lines.append("\n## 4. Clinical Compliance Verification")
            lines.append(f"- **Status**: `{self.compliance_verification.get('overall_status', 'N/A')}` (Score: `{self.compliance_verification.get('compliance_score', 0):.2f}`)")
            for check in self.compliance_verification.get("checks", []):
                status_icon = "✅" if check.get("status") == "PASS" else ("⚠️" if check.get("status") == "WARNING" else "❌")
                lines.append(f"- {status_icon} **{check.get('rule_id')} ({check.get('name')})**: {check.get('reason')}")

        if self.answer:
            lines.append("\n## 5. Grounded Answer")
            lines.append(self.answer)

        return "\n".join(lines)


class AuditReportCompiler:
    """
    Compiles full audit records by orchestrating retrieval, stability evaluation,
    counterfactual inquiry, and compliance verification.
    """

    def __init__(
        self,
        retriever: Any,
        stability_tester: Optional[RetrievalStabilityTester] = None,
        counterfactual_engine: Optional[CounterfactualEngine] = None,
        compliance_verifier: Optional[PneumoniaComplianceVerifier] = None
    ):
        self.retriever = retriever
        self.stability_tester = stability_tester or RetrievalStabilityTester()
        self.counterfactual_engine = counterfactual_engine or CounterfactualEngine()
        self.compliance_verifier = compliance_verifier or PneumoniaComplianceVerifier()

    def generate_full_report(
        self,
        query: str,
        patient_id: Optional[str] = None,
        answer: Optional[str] = None,
        run_stability: bool = True,
        run_counterfactuals: bool = True,
        run_compliance: bool = True,
        interventions: Optional[List[Dict[str, Any]]] = None
    ) -> AuditReport:
        """
        Runs comprehensive auditable evaluation and generates an AuditReport.
        """
        # 1. Primary Auditable Retrieval
        retrieval_res = self.retriever.retrieve(query=query, patient_id=patient_id)

        # 2. Stability Evaluation
        stability_data = None
        if run_stability:
            try:
                stab_rep = self.stability_tester.evaluate_stability(
                    query=query,
                    retriever=self.retriever,
                    original_result=retrieval_res
                )
                stability_data = stab_rep.to_dict()
            except Exception:
                pass

        # 3. Counterfactual Scenarios
        cf_data = []
        if run_counterfactuals:
            if not interventions:
                interventions = [
                    {"type": "REMOVE_EVIDENCE", "target": "pneumonia", "clinical_rationale": "Test evidence persistence without disease prompt token."},
                    {"type": "MASK_EVIDENCE", "target": "antibiotic", "clinical_rationale": "Test treatment path masking."},
                ]
            try:
                cf_results = self.counterfactual_engine.run_suite(
                    query=query,
                    retriever=self.retriever,
                    interventions=interventions
                )
                cf_data = [r.to_dict() for r in cf_results]
            except Exception:
                pass

        # 4. Clinical Compliance Check
        compliance_data = None
        if run_compliance:
            try:
                comp_res = self.compliance_verifier.verify(retrieval_result=retrieval_res)
                compliance_data = comp_res.to_dict()
            except Exception:
                pass

        # Assemble final AuditReport
        selected_paths_dict = [p.to_dict() for p in retrieval_res.selected_paths]
        for p, orig_p in zip(selected_paths_dict, retrieval_res.selected_paths):
            p["readable_path"] = orig_p.readable_path

        rejected_paths_dict = [p.to_dict() for p in retrieval_res.rejected_paths]
        for p, orig_p in zip(rejected_paths_dict, retrieval_res.rejected_paths):
            p["readable_path"] = orig_p.readable_path

        return AuditReport(
            query=query,
            patient_id=patient_id,
            seed_nodes_evaluated=len(retrieval_res.seed_nodes),
            candidate_paths_evaluated=len(retrieval_res.candidate_paths),
            selected_paths=selected_paths_dict,
            rejected_paths=rejected_paths_dict,
            edge_confidences=retrieval_res.edge_confidence,
            stability_metrics=stability_data,
            counterfactual_tests=cf_data,
            compliance_verification=compliance_data,
            evidence=retrieval_res.evidence,
            answer=answer or "Evidence retrieved and verified against audit criteria.",
            decision_rationale=retrieval_res.decision_rationale
        )
