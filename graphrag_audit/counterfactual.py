"""
Counterfactual Retrieval Engine for Medical Knowledge Graphs (Phase 6).
Enables systematic 'what-if' clinical reasoning by perturbing evidence and
measuring changes in graph retrieval pathways, confidence scores, and explanations.
"""

import re
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

from .audit_types import CandidatePath, AuditRetrievalResult


@dataclass
class CounterfactualResult:
    """Complete record of a counterfactual retrieval inquiry."""
    original_query: str
    counterfactual_query: str
    intervention: Dict[str, Any]
    
    # Path deltas
    retained_paths: List[str] = field(default_factory=list)
    removed_paths: List[str] = field(default_factory=list)
    added_paths: List[str] = field(default_factory=list)

    # Quantitative metrics
    top_path_shifted: bool = False
    original_top_path: Optional[str] = None
    counterfactual_top_path: Optional[str] = None
    original_mean_confidence: float = 0.0
    counterfactual_mean_confidence: float = 0.0
    confidence_delta: float = 0.0

    # Audit Trail Explanation
    counterfactual_explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CounterfactualEngine:
    """
    Executes controlled counterfactual interventions on clinical evidence/queries
    and analyzes topological and confidence shifts in the retrieved reasoning graph.
    """

    SUPPORTED_INTERVENTIONS = {
        "REMOVE_EVIDENCE",    # Remove a symptom, lab, or clinical observation
        "MASK_EVIDENCE",      # Replace finding with generic [MASKED] token
        "MODIFY_FINDING",     # Replace one condition/drug with an alternative
        "VALUE_PERTURBATION", # Alter quantitative test result (e.g. WBC count)
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def apply_intervention(self, query: str, intervention: Dict[str, Any]) -> str:
        """
        Applies a clinical counterfactual intervention to a query string.
        """
        i_type = intervention.get("type", "REMOVE_EVIDENCE").upper()
        target = intervention.get("target", "")

        if not target:
            return query

        if i_type == "REMOVE_EVIDENCE":
            pattern = re.compile(re.escape(target), re.IGNORECASE)
            result = pattern.sub("", query)
            return re.sub(r"\s+", " ", result).strip()

        elif i_type == "MASK_EVIDENCE":
            pattern = re.compile(re.escape(target), re.IGNORECASE)
            return pattern.sub("[MASKED]", query)

        elif i_type == "MODIFY_FINDING":
            replacement = intervention.get("replacement", "")
            pattern = re.compile(re.escape(target), re.IGNORECASE)
            return pattern.sub(replacement, query)

        elif i_type == "VALUE_PERTURBATION":
            factor = float(intervention.get("factor", 0.20))
            def _adjust(match):
                try:
                    val = float(match.group(0))
                    new_val = val * (1.0 + factor)
                    return f"{new_val:.1f}" if "." in match.group(0) else str(int(round(new_val)))
                except ValueError:
                    return match.group(0)
            return re.sub(r"\b\d+(\.\d+)?\b", _adjust, query)

        return query

    def counterfactual_retrieve(
        self,
        query: str,
        intervention: Dict[str, Any],
        retriever: Any,
        original_result: Optional[AuditRetrievalResult] = None
    ) -> CounterfactualResult:
        """
        Executes counterfactual retrieval for a given intervention and calculates
        path and confidence deltas relative to original baseline retrieval.
        """
        # 1. Obtain original retrieval baseline
        if original_result is None:
            original_result = retriever.retrieve(query=query)

        orig_paths = [p.readable_path for p in original_result.selected_paths]
        orig_confs = [p.path_score for p in original_result.selected_paths]
        orig_top = orig_paths[0] if orig_paths else None
        orig_mean_conf = float(np.mean(orig_confs)) if orig_confs else 0.0

        # 2. Construct and run counterfactual query
        cf_query = self.apply_intervention(query, intervention)
        cf_result = retriever.retrieve(query=cf_query)

        cf_paths = [p.readable_path for p in cf_result.selected_paths]
        cf_confs = [p.path_score for p in cf_result.selected_paths]
        cf_top = cf_paths[0] if cf_paths else None
        cf_mean_conf = float(np.mean(cf_confs)) if cf_confs else 0.0

        # 3. Analyze path differences
        set_orig = set(orig_paths)
        set_cf = set(cf_paths)

        retained = list(set_orig & set_cf)
        removed = list(set_orig - set_cf)
        added = list(set_cf - set_orig)

        top_shifted = (orig_top != cf_top) if (orig_top and cf_top) else False
        conf_delta = round(cf_mean_conf - orig_mean_conf, 4)

        # 4. Synthesize counterfactual explanation
        i_type = intervention.get("type", "INTERVENTION")
        target = intervention.get("target", "target finding")
        rationale = intervention.get("clinical_rationale", "")

        explanation_parts = [
            f"Counterfactual test applied intervention [{i_type} on '{target}'] (Rationale: {rationale or 'N/A'}).",
            f"Resulted in {len(retained)} retained paths, {len(removed)} removed paths, and {len(added)} newly surfaced paths.",
            f"Top path shifted: {top_shifted} (Original: '{orig_top}' -> Counterfactual: '{cf_top}').",
            f"Confidence shift: {conf_delta:+.3f} (Original mean: {orig_mean_conf:.3f}, Counterfactual mean: {cf_mean_conf:.3f})."
        ]
        if removed:
            explanation_parts.append(f"Paths removed due to intervention: {'; '.join(removed[:2])}.")

        explanation = " ".join(explanation_parts)

        return CounterfactualResult(
            original_query=query,
            counterfactual_query=cf_query,
            intervention=intervention,
            retained_paths=retained,
            removed_paths=removed,
            added_paths=added,
            top_path_shifted=top_shifted,
            original_top_path=orig_top,
            counterfactual_top_path=cf_top,
            original_mean_confidence=round(orig_mean_conf, 4),
            counterfactual_mean_confidence=round(cf_mean_conf, 4),
            confidence_delta=conf_delta,
            counterfactual_explanation=explanation
        )

    def run_suite(
        self,
        query: str,
        retriever: Any,
        interventions: List[Dict[str, Any]]
    ) -> List[CounterfactualResult]:
        """
        Executes a batch suite of counterfactual scenarios for a query.
        """
        orig_res = retriever.retrieve(query=query)
        results = []
        for interv in interventions:
            res = self.counterfactual_retrieve(
                query=query,
                intervention=interv,
                retriever=retriever,
                original_result=orig_res
            )
            results.append(res)
        return results
