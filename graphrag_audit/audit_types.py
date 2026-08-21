"""
Data models and type definitions for Auditable Graph Retrieval.
Provides structured representation for candidates, decisions, paths, and audit results.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import json
from datetime import datetime


@dataclass
class CandidateNode:
    """A node considered during graph retrieval."""
    id: str
    gid: str
    type: str = "Entity"
    score: float = 0.0
    properties: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateEdge:
    """A directed edge traversed or scored during path discovery."""
    source_id: str
    target_id: str
    relation_type: str
    gid: str = ""
    semantic_score: float = 0.0
    graph_support_score: float = 0.0
    source_reliability_score: float = 0.0
    stability_score: float = 1.0
    confidence_score: float = 0.0
    properties: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def display_str(self) -> str:
        return f"({self.source_id}) -[{self.relation_type}]-> ({self.target_id})"


@dataclass
class CandidatePath:
    """An explicit candidate reasoning pathway through the medical graph."""
    path_id: str
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    path_score: float = 0.0
    edge_scores: List[float] = field(default_factory=list)
    selection_status: str = "pending"  # "selected" | "rejected" | "pending"
    rejection_reason: Optional[str] = None
    evidence_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def readable_path(self) -> str:
        """Returns human-readable representation, e.g. Fever -> symptom_of -> Infection -> associated_with -> Pneumonia"""
        if not self.edges:
            return " -> ".join([str(n.get("id", "")) for n in self.nodes])
        
        parts = []
        for i, edge in enumerate(self.edges):
            if i == 0:
                parts.append(edge.get("source_id", ""))
            rel = edge.get("relation_type", "RELATED_TO")
            parts.append(f"--[{rel}]-->")
            parts.append(edge.get("target_id", ""))
        return " ".join(parts)


@dataclass
class AuditRetrievalResult:
    """
    Comprehensive record of the graph retrieval operation.
    Exposes:
      1. What did the system consider? (seed_nodes, candidate_paths)
      2. Why was this path selected? (selected_paths vs rejected_paths with scores & reasons)
      3. Edge & Path confidence breakdown.
      4. Extracted evidence citations.
    """
    query: str
    query_embedding: Optional[List[float]] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    patient_id: Optional[str] = None

    # What the system considered
    seed_nodes: List[Dict[str, Any]] = field(default_factory=list)
    candidate_paths: List[CandidatePath] = field(default_factory=list)

    # Decisions: Selected vs. Rejected
    selected_paths: List[CandidatePath] = field(default_factory=list)
    rejected_paths: List[CandidatePath] = field(default_factory=list)

    # Detailed scores
    retrieval_scores: Dict[str, float] = field(default_factory=dict)
    edge_confidence: List[Dict[str, Any]] = field(default_factory=list)
    path_confidence: List[Dict[str, Any]] = field(default_factory=list)

    # Retrieval parameters utilized
    retrieval_parameters: Dict[str, Any] = field(default_factory=dict)

    # Extracted evidence strings & contributing GIDs
    evidence: List[str] = field(default_factory=list)
    evidence_gids: List[str] = field(default_factory=list)
    
    # Audit Trail Explanation
    decision_rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # Omit large embedding vector from direct dict export if desired, or keep
        if self.query_embedding is not None and len(self.query_embedding) > 10:
            data["query_embedding_dim"] = len(self.query_embedding)
            data["query_embedding"] = f"[{len(self.query_embedding)} dims]"
        return data

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)
