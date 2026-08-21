"""
Auditable Knowledge Retriever for Medical GraphRAG.
Implements observable, transparent, and quantifiable graph retrieval.
Exposes candidate consideration, edge/path confidence scoring, and explicit selection/rejection rationale.
"""

import os
import uuid
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from utils import get_embedding, cosine_similarity
from .confidence import EdgeConfidenceScorer
from .audit_types import (
    CandidateNode,
    CandidateEdge,
    CandidatePath,
    AuditRetrievalResult,
)


class AuditableRetriever:
    """
    Retrieves evidence from a Medical Knowledge Graph while producing a
    complete, verifiable audit trail of the decision process.
    """

    def __init__(self, driver, config: Optional[Dict[str, Any]] = None):
        self.driver = driver
        self.config = config or {}

        retrieval_cfg = self.config.get("retrieval", {})
        self.top_k = int(retrieval_cfg.get("top_k", self.config.get("top_k", 3)))
        self.max_hops = int(retrieval_cfg.get("max_hops", self.config.get("max_hops", 2)))
        self.max_evidence = int(retrieval_cfg.get("max_evidence", self.config.get("max_evidence", 120)))
        self.max_candidate_paths = int(retrieval_cfg.get("max_candidate_paths", 50))
        self.min_confidence_threshold = float(retrieval_cfg.get("min_confidence_threshold", 0.0))

        # Initialize confidence scorer (Phase 4 integration)
        self.scorer = EdgeConfidenceScorer(config=self.config)
        self.weights = self.scorer.weights

    def retrieve(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None,
        patient_id: Optional[str] = None
    ) -> AuditRetrievalResult:
        """
        Executes auditable retrieval for a clinical query.
        Returns full AuditRetrievalResult containing decisions, paths, and scores.
        """
        start_time = datetime.now()

        # 1. Embed query
        if query_embedding is None:
            query_embedding = get_embedding(query)

        # 2. Discover and score seed candidates
        all_seed_candidates = self._identify_seed_candidates(query_embedding, patient_id=patient_id)
        
        # Determine top seed GIDs
        selected_seed_gids = [
            s["gid"] for s in all_seed_candidates if s.get("status") == "selected"
        ]
        retrieval_scores = {s["gid"]: float(s.get("score", 0.0)) for s in all_seed_candidates}

        # 3. Explore candidate graph paths starting from seeds
        candidate_paths, candidate_edges = self._explore_candidate_paths(
            seed_gids=selected_seed_gids,
            query_embedding=query_embedding
        )

        # 4. Score all candidate paths
        scored_paths = self._score_candidate_paths(
            candidate_paths=candidate_paths,
            query_embedding=query_embedding
        )

        # 5. Select top paths and log rejected alternatives with explicit rationale
        selected_paths, rejected_paths = self._select_and_partition_paths(scored_paths)

        # 6. Extract evidence and contributing GIDs
        evidence_lines, evidence_gids = self._extract_evidence_from_paths(
            selected_paths=selected_paths,
            seed_gids=selected_seed_gids
        )

        # 7. Collect edge and path confidence breakdowns
        edge_confidences = [
            {
                "edge": e.display_str,
                "source": e.source_id,
                "target": e.target_id,
                "relation_type": e.relation_type,
                "gid": e.gid,
                "semantic_score": round(e.semantic_score, 4),
                "graph_support": round(e.graph_support_score, 4),
                "source_reliability": round(e.source_reliability_score, 4),
                "stability": round(e.stability_score, 4),
                "confidence": round(e.confidence_score, 4),
                "provenance": e.provenance,
            }
            for e in candidate_edges
        ]

        path_confidences = [
            {
                "path_id": p.path_id,
                "readable_path": p.readable_path,
                "path_score": round(p.path_score, 4),
                "status": p.selection_status,
                "rejection_reason": p.rejection_reason,
            }
            for p in scored_paths
        ]

        # 8. Synthesize decision explanation
        rationale = self._synthesize_decision_rationale(
            query=query,
            seed_candidates=all_seed_candidates,
            selected_paths=selected_paths,
            rejected_paths=rejected_paths,
            evidence_count=len(evidence_lines)
        )

        return AuditRetrievalResult(
            query=query,
            query_embedding=query_embedding,
            timestamp=start_time.isoformat(),
            patient_id=patient_id,
            seed_nodes=all_seed_candidates,
            candidate_paths=scored_paths,
            selected_paths=selected_paths,
            rejected_paths=rejected_paths,
            retrieval_scores=retrieval_scores,
            edge_confidence=edge_confidences,
            path_confidence=path_confidences,
            retrieval_parameters={
                "top_k": self.top_k,
                "max_hops": self.max_hops,
                "max_evidence": self.max_evidence,
                "min_confidence_threshold": self.min_confidence_threshold,
                "confidence_weights": self.weights,
            },
            evidence=evidence_lines[:self.max_evidence],
            evidence_gids=list(dict.fromkeys(evidence_gids)),
            decision_rationale=rationale
        )

    def _identify_seed_candidates(
        self,
        query_embedding: List[float],
        patient_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Discovers all potential seed documents/subgraphs, scores them against
        the query, and logs both selected and rejected candidates.
        """
        candidates = []

        # Check for Summary nodes in Neo4j
        summary_query = """
            MATCH (s:Summary)
            RETURN s.content AS content, s.gid AS gid
        """
        try:
            summary_rows = self.driver.query(summary_query)
        except Exception:
            summary_rows = []

        if summary_rows:
            for row in summary_rows:
                content = row.get("content", "")
                gid = row.get("gid", "")
                if not gid:
                    continue

                # Compute similarity
                text = " ".join(content) if isinstance(content, list) else str(content)
                text_emb = get_embedding(text) if text else None
                sim = cosine_similarity(query_embedding, text_emb) if text_emb else 0.0

                candidates.append({
                    "gid": gid,
                    "type": "Summary",
                    "preview": text[:120] + "..." if len(text) > 120 else text,
                    "score": float(sim),
                })
        else:
            # Fallback to Entity nodes with embeddings
            entity_query = """
                MATCH (n)
                WHERE n.embedding IS NOT NULL AND n.gid IS NOT NULL AND NOT n:Summary
                RETURN DISTINCT n.gid AS gid, labels(n)[0] AS label, n.id AS id, n.embedding AS embedding
                LIMIT 300
            """
            try:
                entity_rows = self.driver.query(entity_query)
            except Exception:
                entity_rows = []

            seen_gids = set()
            for row in entity_rows:
                gid = row.get("gid")
                emb = row.get("embedding")
                if not gid or gid in seen_gids:
                    continue
                seen_gids.add(gid)
                sim = cosine_similarity(query_embedding, emb) if emb else 0.0
                candidates.append({
                    "gid": gid,
                    "type": row.get("label", "Entity"),
                    "preview": f"Entity {row.get('id', '')}",
                    "score": float(sim),
                })

        # Rank all candidates by score descending
        candidates.sort(key=lambda x: x["score"], reverse=True)

        # Mark selection status
        for rank, cand in enumerate(candidates, 1):
            if rank <= self.top_k and cand["score"] >= self.min_confidence_threshold:
                cand["status"] = "selected"
                cand["rejection_reason"] = None
            else:
                cand["status"] = "rejected"
                if cand["score"] < self.min_confidence_threshold:
                    cand["rejection_reason"] = f"Score {cand['score']:.3f} below threshold {self.min_confidence_threshold}"
                else:
                    cand["rejection_reason"] = f"Rank {rank} exceeds top_k limit of {self.top_k}"

        return candidates

    def _explore_candidate_paths(
        self,
        seed_gids: List[str],
        query_embedding: List[float]
    ) -> Tuple[List[CandidatePath], List[CandidateEdge]]:
        """
        Explores candidate paths up to max_hops from the seed GIDs,
        traversing intra-GID relationships and inter-layer REFERENCE relationships.
        """
        paths = []
        all_edges_dict = {}

        if not seed_gids:
            return paths, []

        # 1. Fetch 1-hop intra-GID relationships
        intra_query = """
            MATCH (n)
            WHERE n.gid IN $gids AND NOT n:Summary
            MATCH (n)-[r]->(m)
            WHERE n.gid = m.gid AND NOT m:Summary AND TYPE(r) <> 'REFERENCE'
            RETURN n.id AS source_id, labels(n)[0] AS source_type, properties(n) AS source_props,
                   m.id AS target_id, labels(m)[0] AS target_type, properties(m) AS target_props,
                   TYPE(r) AS rel_type, properties(r) AS rel_props, n.gid AS gid
            LIMIT 200
        """
        try:
            intra_rows = self.driver.query(intra_query, {"gids": seed_gids})
        except Exception:
            intra_rows = []

        # 2. Fetch cross-GID REFERENCE relationships (Trinity links)
        ref_query = """
            MATCH (n)
            WHERE n.gid IN $gids AND NOT n:Summary
            MATCH (n)-[r:REFERENCE]->(m)
            WHERE NOT m:Summary
            MATCH (m)-[s]-(o)
            WHERE NOT o:Summary AND TYPE(s) <> 'REFERENCE'
            RETURN n.id AS n_id, labels(n)[0] AS n_type, n.gid AS n_gid,
                   m.id AS m_id, labels(m)[0] AS m_type, m.gid AS m_gid,
                   o.id AS o_id, labels(o)[0] AS o_type, o.gid AS o_gid,
                   TYPE(r) AS ref_type, TYPE(s) AS conn_type,
                   properties(r) AS ref_props, properties(s) AS conn_props
            LIMIT 200
        """
        try:
            ref_rows = self.driver.query(ref_query, {"gids": seed_gids})
        except Exception:
            ref_rows = []

        path_idx = 1

        # Build 1-hop intra-GID paths
        for row in intra_rows:
            s_id = str(row.get("source_id", ""))
            t_id = str(row.get("target_id", ""))
            r_type = str(row.get("rel_type", "RELATED_TO"))
            gid = str(row.get("gid", ""))
            rel_props = row.get("rel_props", {}) or {}

            edge_key = (s_id, r_type, t_id)
            if edge_key not in all_edges_dict:
                edge = self._create_scored_edge(
                    source_id=s_id,
                    target_id=t_id,
                    relation_type=r_type,
                    gid=gid,
                    properties=rel_props,
                    query_embedding=query_embedding
                )
                all_edges_dict[edge_key] = edge
            else:
                edge = all_edges_dict[edge_key]

            node_s = {"id": s_id, "type": row.get("source_type", "Entity"), "properties": row.get("source_props", {})}
            node_t = {"id": t_id, "type": row.get("target_type", "Entity"), "properties": row.get("target_props", {})}

            path = CandidatePath(
                path_id=f"path_{path_idx:03d}",
                nodes=[node_s, node_t],
                edges=[edge.to_dict()],
                evidence_text=f"{s_id} {r_type} {t_id}"
            )
            paths.append(path)
            path_idx += 1

        # Build 2-hop cross-layer REFERENCE paths
        for row in ref_rows:
            n_id = str(row.get("n_id", ""))
            m_id = str(row.get("m_id", ""))
            o_id = str(row.get("o_id", ""))
            ref_type = str(row.get("ref_type", "REFERENCE"))
            conn_type = str(row.get("conn_type", "RELATED_TO"))

            edge1_key = (n_id, ref_type, m_id)
            if edge1_key not in all_edges_dict:
                edge1 = self._create_scored_edge(
                    source_id=n_id,
                    target_id=m_id,
                    relation_type=ref_type,
                    gid=str(row.get("n_gid", "")),
                    properties=row.get("ref_props", {}) or {},
                    query_embedding=query_embedding
                )
                all_edges_dict[edge1_key] = edge1
            else:
                edge1 = all_edges_dict[edge1_key]

            edge2_key = (m_id, conn_type, o_id)
            if edge2_key not in all_edges_dict:
                edge2 = self._create_scored_edge(
                    source_id=m_id,
                    target_id=o_id,
                    relation_type=conn_type,
                    gid=str(row.get("m_gid", "")),
                    properties=row.get("conn_props", {}) or {},
                    query_embedding=query_embedding
                )
                all_edges_dict[edge2_key] = edge2
            else:
                edge2 = all_edges_dict[edge2_key]

            path = CandidatePath(
                path_id=f"path_{path_idx:03d}",
                nodes=[
                    {"id": n_id, "type": row.get("n_type", "Entity")},
                    {"id": m_id, "type": row.get("m_type", "Entity")},
                    {"id": o_id, "type": row.get("o_type", "Entity")},
                ],
                edges=[edge1.to_dict(), edge2.to_dict()],
                evidence_text=f"{n_id} has reference to {m_id} {conn_type} {o_id}"
            )
            paths.append(path)
            path_idx += 1

        return paths[:self.max_candidate_paths], list(all_edges_dict.values())

    def _create_scored_edge(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        gid: str,
        properties: Dict[str, Any],
        query_embedding: List[float]
    ) -> CandidateEdge:
        """
        Creates a CandidateEdge and delegates scoring to EdgeConfidenceScorer.
        """
        scored_dict = self.scorer.calculate_edge_confidence(
            source_id=source_id,
            relation_type=relation_type,
            target_id=target_id,
            query_embedding=query_embedding,
            properties=properties,
            driver=self.driver,
        )

        return CandidateEdge(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            gid=gid,
            semantic_score=scored_dict["semantic_score"],
            graph_support_score=scored_dict["graph_support"],
            source_reliability_score=scored_dict["source_reliability"],
            stability_score=scored_dict["stability"],
            confidence_score=scored_dict["confidence"],
            properties=properties,
            provenance=scored_dict["provenance"],
        )

    def _score_candidate_paths(
        self,
        candidate_paths: List[CandidatePath],
        query_embedding: List[float]
    ) -> List[CandidatePath]:
        """
        Computes composite confidence scores for all candidate reasoning paths using EdgeConfidenceScorer.
        """
        for path in candidate_paths:
            if not path.edges:
                path.path_score = 0.0
                path.edge_scores = []
                continue

            edge_confs = [float(e.get("confidence_score", 0.5)) for e in path.edges]
            path.edge_scores = edge_confs
            path.path_score = self.scorer.calculate_path_confidence(
                edge_scores=edge_confs,
                hop_count=len(path.edges)
            )

        # Sort candidate paths by score descending
        candidate_paths.sort(key=lambda p: p.path_score, reverse=True)
        return candidate_paths

    def _select_and_partition_paths(
        self,
        scored_paths: List[CandidatePath]
    ) -> Tuple[List[CandidatePath], List[CandidatePath]]:
        """
        Partitions paths into selected vs. rejected, attaching explicit rejection rationale.
        """
        selected = []
        rejected = []
        path_budget = min(self.top_k * 4, 15)

        for rank, path in enumerate(scored_paths, 1):
            if rank <= path_budget and path.path_score >= self.min_confidence_threshold:
                path.selection_status = "selected"
                path.rejection_reason = None
                selected.append(path)
            else:
                path.selection_status = "rejected"
                if path.path_score < self.min_confidence_threshold:
                    path.rejection_reason = f"Confidence {path.path_score:.3f} below minimum threshold {self.min_confidence_threshold}"
                else:
                    path.rejection_reason = f"Rank {rank} exceeds path selection budget of {path_budget}"
                rejected.append(path)

        return selected, rejected

    def _extract_evidence_from_paths(
        self,
        selected_paths: List[CandidatePath],
        seed_gids: List[str]
    ) -> Tuple[List[str], List[str]]:
        """
        Generates readable evidence strings from selected paths and lists contributing GIDs.
        """
        evidence_lines = []
        contributing_gids = list(seed_gids)

        for path in selected_paths:
            if path.evidence_text and path.evidence_text not in evidence_lines:
                evidence_lines.append(path.evidence_text)
            for edge in path.edges:
                g = edge.get("gid")
                if g and g not in contributing_gids:
                    contributing_gids.append(g)

        return evidence_lines, contributing_gids

    def _synthesize_decision_rationale(
        self,
        query: str,
        seed_candidates: List[Dict[str, Any]],
        selected_paths: List[CandidatePath],
        rejected_paths: List[CandidatePath],
        evidence_count: int
    ) -> str:
        """
        Synthesizes human-readable audit trail explanation for the retrieval decision.
        """
        num_seeds = len(seed_candidates)
        num_selected_seeds = len([s for s in seed_candidates if s.get("status") == "selected"])
        total_paths = len(selected_paths) + len(rejected_paths)
        mean_selected_conf = np.mean([p.path_score for p in selected_paths]) if selected_paths else 0.0

        top_rejections = []
        for r in rejected_paths[:3]:
            top_rejections.append(f"'{r.readable_path}' ({r.rejection_reason})")

        rejection_summary = "; ".join(top_rejections) if top_rejections else "None"

        rationale = (
            f"Retriever evaluated {num_seeds} subgraph seed candidates for query: '{query}'. "
            f"Selected top {num_selected_seeds} seed subgraphs based on semantic similarity. "
            f"Discovered {total_paths} candidate reasoning paths. "
            f"Selected {len(selected_paths)} paths (mean confidence {mean_selected_conf:.3f}) yielding {evidence_count} evidence items. "
            f"Rejected {len(rejected_paths)} alternative paths. Notable rejections: {rejection_summary}."
        )
        return rationale
