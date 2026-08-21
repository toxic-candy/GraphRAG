"""
Baseline Experiment Runner for Medical GraphRAG.
Executes the frozen baseline retrieval and answering pipeline, recording reproducible outputs.
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils import load_yaml_config
from post_graph_inference import (
    _connect_neo4j,
    _normalize_neo4j_url,
    _question_summary,
    _neighbor_gids,
    _collect_evidence,
    _answer_with_citations
)
from retrieve import select_top_gids


def run_baseline_experiment(config_path: str, neo4j_password: str = None, output_path: str = None):
    # Load config
    config = load_yaml_config(config_path, override_env=True)
    
    neo4j_cfg = config.get("neo4j", {})
    url = _normalize_neo4j_url(neo4j_cfg.get("uri") or os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    username = neo4j_cfg.get("username") or os.getenv("NEO4J_USERNAME", "neo4j")
    password = neo4j_password or neo4j_cfg.get("password") or os.getenv("NEO4J_PASSWORD")
    
    if not password:
        raise ValueError("NEO4J_PASSWORD must be provided via CLI, config, or environment.")
        
    retrieval_cfg = config.get("retrieval", {})
    top_k = int(retrieval_cfg.get("top_k", 3))
    max_hops = int(retrieval_cfg.get("max_hops", 2))
    max_evidence = int(retrieval_cfg.get("max_evidence", 120))
    
    queries = config.get("queries", [
        "What are the risk factors and treatment options for pneumonia?"
    ])
    
    n4j = _connect_neo4j(url=url, username=username, password=password)
    
    results = []
    timestamp = datetime.now().isoformat()
    
    print(f"\n==========================================")
    print(f"Running Baseline Experiment: {len(queries)} queries")
    print(f"==========================================\n")
    
    for idx, query in enumerate(queries, 1):
        print(f"[{idx}/{len(queries)}] Query: {query}")
        
        q_summary = _question_summary(query)
        seed_gids = select_top_gids(n4j, q_summary, top_k=top_k)
        
        expanded_gids = list(seed_gids)
        for gid in seed_gids:
            expanded_gids.extend(_neighbor_gids(n4j, gid, max_hops=max_hops, max_items=20))
            
        ordered_gids = list(dict.fromkeys([g for g in expanded_gids if g]))
        evidence = _collect_evidence(n4j, ordered_gids, max_evidence=max_evidence)
        
        answer = _answer_with_citations(query, evidence) if evidence else "No evidence retrieved."
        
        record = {
            "query_index": idx,
            "query": query,
            "seed_gids": seed_gids,
            "retrieved_gids": ordered_gids,
            "evidence_count": len(evidence),
            "evidence": evidence,
            "answer": answer
        }
        results.append(record)
        print(f"   Retrieved GIDs: {len(ordered_gids)}, Evidence Items: {len(evidence)}")
        
    experiment_record = {
        "experiment_name": config.get("experiment", {}).get("name", "baseline_standard_graphrag"),
        "timestamp": timestamp,
        "dataset_version": config.get("experiment", {}).get("dataset_version", "pneumonia-100"),
        "random_seed": config.get("experiment", {}).get("random_seed", 42),
        "parameters": {
            "top_k": top_k,
            "max_hops": max_hops,
            "max_evidence": max_evidence,
            "embedding_model": config.get("embedding", {}).get("model", "text-embedding-3-small"),
            "llm_model": config.get("llm", {}).get("model", "meta-llama/llama-3-8b-instruct"),
        },
        "results": results
    }
    
    # Save output
    out_dir = Path(config.get("experiment", {}).get("output_dir", "./experiments/outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if output_path is None:
        out_file = out_dir / f"baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    else:
        out_file = Path(output_path)
        
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(experiment_record, f, indent=2)
        
    print(f"\nSaved baseline experiment results to: {out_file}")
    return experiment_record


def main():
    parser = argparse.ArgumentParser(description="Run baseline GraphRAG experiment.")
    parser.add_argument("--config", type=str, default="experiments/configs/baseline_config.yaml")
    parser.add_argument("--neo4j-password", type=str, default=os.getenv("NEO4J_PASSWORD"))
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    
    run_baseline_experiment(args.config, args.neo4j_password, args.output)


if __name__ == "__main__":
    main()
