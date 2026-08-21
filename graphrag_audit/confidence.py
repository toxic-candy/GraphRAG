"""
Edge-Level and Path-Level Confidence Scoring Engine for Medical GraphRAG.
Implements modular, transparent, and configurable confidence calculation:
  Confidence = f(semantic similarity, graph support, source reliability, retrieval stability)
"""

from typing import Dict, Any, List, Optional, Union
import numpy as np
from utils import get_embedding, cosine_similarity


class EdgeConfidenceScorer:
    """
    Computes inspectable confidence scores for individual graph edges and composite paths.
    Note: These are heuristic confidence scores for research auditability,
    not calibrated probabilistic posteriors.
    """

    DEFAULT_WEIGHTS = {
        "semantic": 0.35,
        "graph_support": 0.25,
        "source_reliability": 0.25,
        "stability": 0.15,
    }

    SOURCE_RELIABILITY_MAP = {
        # (source_type, provenance): score
        ("MIMIC", "evidence_backed"): 0.92,
        ("clinical_guideline", "evidence_backed"): 0.88,
        ("dictionary", "dictionary_derived"): 0.75,
        ("clinical_guideline", "dictionary_derived"): 0.72,
        ("LLM_extracted", "evidence_backed"): 0.70,
        ("LLM_extracted", "LLM_extracted"): 0.65,
        ("rule_based", "evidence_backed"): 0.65,
        ("clinical_guideline", "structurally_generated"): 0.50,
        ("structural", "structurally_generated"): 0.40,
        ("unknown", "unknown"): 0.50,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        custom_weights = self.config.get("confidence_weights", {})
        self.weights = {**self.DEFAULT_WEIGHTS, **custom_weights}
        self.aggregation_method = self.config.get("aggregation_method", "weighted_sum")
        self.hop_penalty_factor = float(self.config.get("hop_penalty_factor", 0.95))

    def semantic_score(
        self,
        source_id: str,
        relation_type: str,
        target_id: str,
        query_embedding: Optional[List[float]] = None
    ) -> float:
        """
        Computes cosine similarity between query embedding and the triple text representation.
        Normalized to range [0.0, 1.0].
        """
        if query_embedding is None:
            return 0.50

        triple_text = f"{source_id} {relation_type} {target_id}"
        triple_emb = get_embedding(triple_text)
        sim = cosine_similarity(query_embedding, triple_emb)

        # Normalize cosine [-1.0, 1.0] to [0.0, 1.0]
        norm_score = (sim + 1.0) / 2.0 if sim < 0 else sim
        return float(np.clip(norm_score, 0.0, 1.0))

    def graph_support_score(
        self,
        source_id: str,
        relation_type: str,
        target_id: str,
        properties: Optional[Dict[str, Any]] = None,
        driver: Any = None
    ) -> float:
        """
        Estimates the structural topological support for an edge.
        Factors in cross-layer Trinity REFERENCE linkage, ICD ontology chapter groupings,
        and database connectivity if driver is provided.
        """
        props = properties or {}
        score = 0.50

        # Cross-layer REFERENCE edges represent curated ontology/evidence alignments
        if relation_type == "REFERENCE":
            score = 0.85
        elif props.get("icd_chapter"):
            score = 0.78
        elif props.get("extraction_method") == "lab_finding":
            score = 0.82
        elif props.get("extraction_method") == "mimic_structured_admission":
            score = 0.90
        elif props.get("extraction_method") == "round_robin_assignment":
            score = 0.45

        # Query driver for path redundancy if available
        if driver is not None and hasattr(driver, "query"):
            try:
                redundancy_query = """
                    MATCH (s {id: $source})-[r]->(t {id: $target})
                    RETURN count(r) AS edge_count
                """
                res = driver.query(redundancy_query, {"source": source_id, "target": target_id})
                if res and res[0].get("edge_count", 0) > 1:
                    score = min(1.0, score + 0.10)
            except Exception:
                pass

        return float(np.clip(score, 0.0, 1.0))

    def source_reliability_score(
        self,
        source_type: Optional[str] = None,
        provenance: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Returns inspectable reliability rating based on data provenance.
        """
        props = properties or {}
        s_type = source_type or props.get("source_type", "unknown")
        prov = provenance or props.get("provenance", "unknown")

        lookup_key = (s_type, prov)
        if lookup_key in self.SOURCE_RELIABILITY_MAP:
            return self.SOURCE_RELIABILITY_MAP[lookup_key]

        # Partial matching fallback
        if prov == "evidence_backed":
            return 0.85
        elif prov == "dictionary_derived":
            return 0.75
        elif prov == "structurally_generated":
            return 0.40
        elif s_type == "MIMIC":
            return 0.90
        elif s_type == "LLM_extracted":
            return 0.65

        return 0.50

    def stability_score(
        self,
        edge_id: Optional[str] = None,
        stability_history: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Returns edge retention stability rate under perturbation.
        Defaults to 1.0 when no perturbation history is registered.
        """
        if stability_history and edge_id and edge_id in stability_history:
            return float(np.clip(stability_history[edge_id], 0.0, 1.0))
        return 1.0

    def calculate_edge_confidence(
        self,
        source_id: str,
        relation_type: str,
        target_id: str,
        query_embedding: Optional[List[float]] = None,
        properties: Optional[Dict[str, Any]] = None,
        driver: Any = None,
        stability_history: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Calculates all component scores and computes composite confidence score.
        """
        props = properties or {}
        edge_key = f"({source_id})-[{relation_type}]->({target_id})"

        sem = self.semantic_score(source_id, relation_type, target_id, query_embedding)
        sup = self.graph_support_score(source_id, relation_type, target_id, props, driver)
        rel = self.source_reliability_score(props.get("source_type"), props.get("provenance"), props)
        stab = self.stability_score(edge_key, stability_history)

        if self.aggregation_method == "multiplicative":
            # Geometric mean
            composite = (sem * sup * rel * stab) ** 0.25
        elif self.aggregation_method == "harmonic_mean":
            scores = [max(0.001, s) for s in (sem, sup, rel, stab)]
            composite = 4.0 / sum(1.0 / s for s in scores)
        else:  # default: weighted_sum
            w = self.weights
            total_weight = sum(w.values()) or 1.0
            composite = (
                w["semantic"] * sem +
                w["graph_support"] * sup +
                w["source_reliability"] * rel +
                w["stability"] * stab
            ) / total_weight

        return {
            "edge": edge_key,
            "source_id": source_id,
            "target_id": target_id,
            "relation_type": relation_type,
            "semantic_score": round(sem, 4),
            "graph_support": round(sup, 4),
            "source_reliability": round(rel, 4),
            "stability": round(stab, 4),
            "confidence": round(float(composite), 4),
            "aggregation": self.aggregation_method,
            "provenance": {
                "source_type": props.get("source_type", "unknown"),
                "provenance": props.get("provenance", "unknown"),
                "extraction_method": props.get("extraction_method", "unknown"),
            }
        }

    def calculate_path_confidence(
        self,
        edge_scores: List[float],
        hop_count: Optional[int] = None
    ) -> float:
        """
        Aggregates edge confidence scores into a path confidence score.
        Applies a configurable hop penalty for longer multi-hop paths to favor concise reasoning.
        """
        if not edge_scores:
            return 0.0

        mean_edge_conf = float(np.mean(edge_scores))
        hops = hop_count if hop_count is not None else len(edge_scores)
        penalty = self.hop_penalty_factor ** max(0, hops - 1)

        return float(np.clip(mean_edge_conf * penalty, 0.0, 1.0))
