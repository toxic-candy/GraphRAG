"""
Retrieval Stability and Robustness Testing Engine (Phase 5).
Measures graph retrieval sensitivity and invariance under clinical evidence perturbations.
"""

import re
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple, Set

from utils import cosine_similarity, get_embedding
from .audit_types import AuditRetrievalResult, CandidatePath


@dataclass
class StabilityReport:
    """Quantitative measurement of retrieval stability across perturbations."""
    query: str
    num_perturbations: int
    mean_path_jaccard: float
    mean_edge_jaccard: float
    confidence_mae: float
    top_path_retention_rate: float
    perturbed_evaluations: List[Dict[str, Any]] = field(default_factory=list)
    stability_grade: str = "HIGH"  # "HIGH" | "MODERATE" | "LOW"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RetrievalStabilityTester:
    """
    Applies systematic perturbations to clinical queries/evidence and measures
    graph retrieval invariance.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def perturb_remove_keyword(self, text: str, keyword: str) -> str:
        """Removes all mentions of a specific clinical keyword or finding."""
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        perturbed = pattern.sub("", text)
        return re.sub(r"\s+", " ", perturbed).strip()

    def perturb_mask_keyword(self, text: str, keyword: str, mask: str = "[MASKED]") -> str:
        """Masks a specific clinical concept with a placeholder."""
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        return pattern.sub(mask, text)

    def perturb_numeric_values(self, text: str, factor: float = 0.20) -> str:
        """
        Perturbs numeric measurements in text by a specified percentage (±factor).
        Example: 'WBC 14.8' -> 'WBC 17.7' or 'WBC 11.8'.
        """
        def _replace_num(match):
            val_str = match.group(0)
            try:
                val = float(val_str)
                delta = val * factor
                # Deterministic slight increase for test repeatability
                new_val = val + delta
                return f"{new_val:.1f}" if "." in val_str else str(int(round(new_val)))
            except ValueError:
                return val_str

        return re.sub(r"\b\d+(\.\d+)?\b", _replace_num, text)

    def compute_path_jaccard(
        self,
        paths_a: List[CandidatePath],
        paths_b: List[CandidatePath]
    ) -> float:
        """Computes Jaccard similarity between two sets of retrieved reasoning paths."""
        set_a = {p.readable_path for p in paths_a}
        set_b = {p.readable_path for p in paths_b}
        if not set_a and not set_b:
            return 1.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return float(intersection / union) if union > 0 else 0.0

    def compute_edge_jaccard(
        self,
        paths_a: List[CandidatePath],
        paths_b: List[CandidatePath]
    ) -> float:
        """Computes Jaccard similarity over constituent edge triples."""
        def _get_edges(paths):
            edges = set()
            for p in paths:
                for e in p.edges:
                    s = e.get("source_id") or e.get("source")
                    t = e.get("target_id") or e.get("target")
                    r = e.get("relation_type") or e.get("type")
                    if s and t and r:
                        edges.add((s, r, t))
            return edges

        edges_a = _get_edges(paths_a)
        edges_b = _get_edges(paths_b)
        if not edges_a and not edges_b:
            return 1.0
        intersection = len(edges_a & edges_b)
        union = len(edges_a | edges_b)
        return float(intersection / union) if union > 0 else 0.0

    def compute_confidence_mae(
        self,
        paths_a: List[CandidatePath],
        paths_b: List[CandidatePath]
    ) -> float:
        """Computes Mean Absolute Error (MAE) of path scores across shared paths."""
        scores_a = {p.readable_path: p.path_score for p in paths_a}
        scores_b = {p.readable_path: p.path_score for p in paths_b}

        common = set(scores_a.keys()) & set(scores_b.keys())
        if not common:
            return 0.0

        errors = [abs(scores_a[k] - scores_b[k]) for k in common]
        return float(np.mean(errors))

    def evaluate_stability(
        self,
        query: str,
        retriever: Any,
        original_result: Optional[AuditRetrievalResult] = None,
        keywords_to_test: Optional[List[str]] = None
    ) -> StabilityReport:
        """
        Executes a suite of perturbation tests on the query and produces a StabilityReport.
        """
        if original_result is None:
            original_result = retriever.retrieve(query=query)

        orig_paths = original_result.selected_paths
        orig_top_path = orig_paths[0].readable_path if orig_paths else None

        # Determine target keywords from query if not specified
        if not keywords_to_test:
            words = [w for w in re.findall(r"\b[A-Za-z]{4,}\b", query) if w.lower() not in {"what", "which", "where", "clinical", "patient", "about"}]
            keywords_to_test = words[:3] or ["pneumonia"]

        evaluations = []
        path_jaccards = []
        edge_jaccards = []
        confidence_maes = []
        top_retained_count = 0

        # Generate perturbation variations
        perturbations = []
        for kw in keywords_to_test:
            # 1. Removal
            p_remove = self.perturb_remove_keyword(query, kw)
            if p_remove != query:
                perturbations.append(("REMOVE", kw, p_remove))
            # 2. Masking
            p_mask = self.perturb_mask_keyword(query, kw)
            if p_mask != query:
                perturbations.append(("MASK", kw, p_mask))

        # 3. Numeric perturbation
        p_num = self.perturb_numeric_values(query, factor=0.20)
        if p_num != query:
            perturbations.append(("NUMERIC_DELTA", "numbers", p_num))

        if not perturbations:
            perturbations.append(("NOOP", "none", query))

        for p_type, target, p_query in perturbations:
            p_result = retriever.retrieve(query=p_query)
            p_paths = p_result.selected_paths

            pj = self.compute_path_jaccard(orig_paths, p_paths)
            ej = self.compute_edge_jaccard(orig_paths, p_paths)
            c_mae = self.compute_confidence_mae(orig_paths, p_paths)

            p_top = p_paths[0].readable_path if p_paths else None
            top_retained = (orig_top_path == p_top) if orig_top_path and p_top else False
            if top_retained:
                top_retained_count += 1

            path_jaccards.append(pj)
            edge_jaccards.append(ej)
            confidence_maes.append(c_mae)

            evaluations.append({
                "perturbation_type": p_type,
                "target": target,
                "perturbed_query": p_query,
                "path_jaccard": round(pj, 4),
                "edge_jaccard": round(ej, 4),
                "confidence_mae": round(c_mae, 4),
                "top_path_retained": top_retained,
            })

        mean_pj = float(np.mean(path_jaccards)) if path_jaccards else 1.0
        mean_ej = float(np.mean(edge_jaccards)) if edge_jaccards else 1.0
        mean_c_mae = float(np.mean(confidence_maes)) if confidence_maes else 0.0
        retention_rate = float(top_retained_count / len(perturbations)) if perturbations else 1.0

        grade = "HIGH" if mean_pj >= 0.70 and retention_rate >= 0.60 else ("MODERATE" if mean_pj >= 0.40 else "LOW")

        return StabilityReport(
            query=query,
            num_perturbations=len(perturbations),
            mean_path_jaccard=round(mean_pj, 4),
            mean_edge_jaccard=round(mean_ej, 4),
            confidence_mae=round(mean_c_mae, 4),
            top_path_retention_rate=round(retention_rate, 4),
            perturbed_evaluations=evaluations,
            stability_grade=grade,
        )
