import argparse
import os
import re
from urllib.parse import urlparse

from camel.storages import Neo4jGraph

from simple_neo4j_graph import SimpleNeo4jGraph
from retrieve import select_top_gids
from summarize import process_chunks
from utils import call_llm, link_context, ret_context


POST_SYS_PROMPT = """
You are a medical QA assistant grounded on graph-retrieved evidence.
Answer the user's question with concise clinical reasoning and cite supporting evidence IDs like [E1], [E2].
If evidence is insufficient, state uncertainty clearly and avoid unsupported claims.
"""


def _use_llm_answering():
    # Default local/offline mode for reproducibility.
    return os.getenv("USE_LLM_ANSWER", "0") == "1"


def _normalize_neo4j_url(raw_url: str | None) -> str:
    if not raw_url:
        return "bolt://localhost:7687"

    url = raw_url.strip()
    parsed = urlparse(url)

    if parsed.scheme in {"http", "https"}:
        host = parsed.hostname or "localhost"
        return f"bolt://{host}:7687"

    if parsed.scheme in {"bolt", "neo4j", "bolt+s", "bolt+ssc", "neo4j+s", "neo4j+ssc"}:
        host = parsed.hostname or "localhost"
        port = parsed.port
        if port == 7474:
            return f"{parsed.scheme}://{host}:7687"
        return url

    if "://" not in url:
        if url.endswith(":7474"):
            url = url[:-5] + ":7687"
        return f"bolt://{url}"

    return url


def _connect_neo4j(url: str, username: str, password: str):
    try:
        return Neo4jGraph(url=url, username=username, password=password)
    except Exception as e:
        print(f"Neo4jGraph init failed ({e}). Falling back to SimpleNeo4jGraph without APOC.")
        return SimpleNeo4jGraph(url=url, username=username, password=password)


def _question_summary(question: str):
    if os.getenv("USE_LLM_SUMMARY", "0") == "1":
        return process_chunks(question)
    return [question[:1500]]


def _extract_patient_id(question: str):
    """Normalize a patient ID appearing in natural-language questions.

    Examples matched:
      - what is the diagnosis for id: "patient_10457410"
      - diagnosis for patient_10457410
      - patient 10457410
    """
    if not question:
        return None

    patterns = [
        r"(?i)\b(?:patient_id|id)\s*[:=]\s*[\"']?(?:patient[_\-\s]*)?(\d{5,})[\"']?",
        r"(?i)\bpatient[_\-\s]*(\d{5,})\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, question)
        if match:
            return f"patient_{match.group(1)}"
    return None


def _patient_seed_gids(n4j, patient_id: str | None):
    if not patient_id:
        return []

    try:
        rows = n4j.query(
            """
            MATCH (p)
            WHERE p.id = $patient_id OR p.gid = $patient_id
            RETURN DISTINCT p.gid AS gid, labels(p)[0] AS label, p.id AS id
            LIMIT 20
            """,
            {"patient_id": patient_id},
        )
    except TypeError:
        rows = n4j.query(
            """
            MATCH (p)
            WHERE p.id = $patient_id OR p.gid = $patient_id
            RETURN DISTINCT p.gid AS gid, labels(p)[0] AS label, p.id AS id
            LIMIT 20
            """
        )
    return [row["gid"] for row in rows if row.get("gid")]


def _neighbor_gids(n4j, seed_gid: str, max_hops: int = 2, max_items: int = 20):
    hop = max(1, min(4, int(max_hops)))
    query = f"""
        MATCH (n)
        WHERE n.gid = $gid AND NOT n:Summary
        MATCH p=(n)-[:REFERENCE*1..{hop}]-(m)
        WHERE NOT m:Summary
        RETURN DISTINCT m.gid AS gid
        LIMIT $limit
    """
    rows = n4j.query(query, {"gid": seed_gid, "limit": max_items})
    return [r["gid"] for r in rows if r.get("gid")]


def _filter_patient_evidence(n4j, evidence, patient_id: str | None, max_evidence: int = 120):
    """Keep only evidence that is actually connected to the patient graph, not just text snippets."""
    if not patient_id:
        return evidence[:max_evidence]

    try:
        connected = n4j.query(
            """
            MATCH (p {id: $patient_id})
            MATCH (p)-[*1..5]-(n)
            RETURN DISTINCT n.id AS node_id, coalesce(n.description, '') AS description
            """,
            {"patient_id": patient_id},
        )
    except Exception:
        connected = []

    valid_node_ids = set()
    valid_descriptions = set()
    for row in connected:
        node_id = str(row.get("node_id") or "").strip()
        desc = str(row.get("description") or "").strip()
        if node_id:
            valid_node_ids.add(node_id.lower())
        if desc:
            valid_descriptions.add(desc.lower())

    filtered = []
    seen = set()
    for item in evidence:
        content = item.get("content") if isinstance(item, dict) else str(item)
        if not content:
            continue

        lower = content.lower()
        related = any(
            token.lower() in lower for token in sorted(valid_node_ids | valid_descriptions)
        ) or any(
            term in lower for term in (patient_id.lower(), "diagnosed_with", "has_condition", "treats", "monitors", "performed_for")
        )

        if not related:
            continue

        key = content.strip()
        if key in seen:
            continue
        seen.add(key)
        filtered.append(item)
        if len(filtered) >= max_evidence:
            break

    return filtered


def _collect_evidence(n4j, gids, max_evidence: int = 120, patient_id: str | None = None):
    evidence = []
    seen = set()

    for gid in gids:
        local_ctx = ret_context(n4j, gid)
        ref_ctx = link_context(n4j, gid)
        for line in local_ctx + ref_ctx:
            clean = str(line).strip()
            if not clean or clean in seen:
                continue
            seen.add(clean)
            evidence.append({"id": f"E{len(evidence)+1}", "content": clean})
            if len(evidence) >= max_evidence:
                return _filter_patient_evidence(n4j, evidence, patient_id, max_evidence=max_evidence)

    return _filter_patient_evidence(n4j, evidence, patient_id, max_evidence=max_evidence)


def _is_medication_question(question: str) -> bool:
    q = (question or "").lower()
    return any(term in q for term in [
        "drug", "drugs", "medication", "medications", "medicine", "prescribed",
        "given", "administered", "treatment", "therapy", "meds", "antibiotic",
        "what was given", "what drugs"
    ])


def _build_grounded_response(question: str, evidence, fallback_response: str | None = None) -> str:
    """Construct a grounded response from evidence when the question asks about medications or treatments."""
    if not evidence or not _is_medication_question(question):
        return fallback_response or ""

    entries = []
    seen = set()
    for item in evidence:
        content = item.get("content") if isinstance(item, dict) else str(item)
        if not content:
            continue
        lower = content.lower()
        if not any(token in lower for token in [
            "mg", "ml", "iv", "po", "oral", "tablet", "capsule",
            "flush", "solution", "medication", "drug", "prescribed",
            "administered", "given", "dose", "antibiotic", "infusion"
        ]):
            continue

        entry = content.strip()
        if entry in seen:
            continue
        seen.add(entry)
        entries.append(entry)

    if not entries:
        return fallback_response or ""

    citations = []
    for item in evidence:
        content = item.get("content") if isinstance(item, dict) else str(item)
        if content in entries:
            citations.append(f"[{item['id']}]" if isinstance(item, dict) and 'id' in item else "[E]")

    meds_text = "; ".join(entries[:8])
    if citations:
        return f"The medications explicitly recorded in the connected patient evidence are: {meds_text}. { ' '.join(citations[:len(entries)]) }"
    return f"The medications explicitly recorded in the connected patient evidence are: {meds_text}."


def _answer_with_citations(question: str, evidence):
    numbered = []
    if evidence and isinstance(evidence[0], dict):
        for item in evidence:
            numbered.append(f"[{item['id']}] {item['content']}")
    else:
        for i, item in enumerate(evidence, start=1):
            numbered.append(f"[E{i}] {item}")

    if _use_llm_answering():
        user_prompt = (
            f"Question: {question}\n\n"
            "Evidence:\n"
            + "\n".join(numbered)
            + "\n\nPlease answer using only the evidence above and include [E#] citations. For medication/treatment questions, list every medication or treatment from the evidence explicitly and do not omit supported items."
        )
        answer = call_llm(POST_SYS_PROMPT, user_prompt)
        grounded = _build_grounded_response(question, evidence, fallback_response=answer)
        if grounded:
            return grounded
        return answer

    def _fmt(raw: str) -> str:
        # Insert spaces around ALL-CAPS relationship tokens, e.g. CAUSES -> --CAUSES-->
        return re.sub(r"([A-Z][a-z].+?)([A-Z_]{3,})([A-Z][a-z])", r"\1 --\2--> \3", raw)

    selected = numbered[: min(8, len(numbered))]
    lines = [
        "-" * 60,
        "  Local Answer Mode  (set USE_LLM_ANSWER=1 for LLM response)",
        "-" * 60,
        "",
        "  Likely clinical problems / interventions from graph evidence:",
        "",
    ]
    for item in selected:
        lines.append(f"    {_fmt(item)}")
    lines += [
        "",
        "-" * 60,
        "  Guidance: prioritize findings connected by repeated",
        "  relation / context patterns in the evidence above.",
        "-" * 60,
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Post-graph inference for three-layer Medical-Graph-RAG")
    parser.add_argument("--question", type=str, help="Question text")
    parser.add_argument("--question-file", type=str, help="Path to a question file")
    parser.add_argument("--top-k", type=int, default=4, help="Top summary-matched gids")
    parser.add_argument("--max-hops", type=int, default=2, help="REFERENCE traversal hops")
    parser.add_argument("--max-evidence", type=int, default=120, help="Max evidence lines for answer")

    parser.add_argument("--neo4j-url", type=str, default=os.getenv("NEO4J_URL") or os.getenv("NEO4J_URI"))
    parser.add_argument("--neo4j-username", type=str, default=os.getenv("NEO4J_USERNAME", "neo4j"))
    parser.add_argument("--neo4j-password", type=str, default=os.getenv("NEO4J_PASSWORD"))

    parser.add_argument("--audit", action="store_true", help="Enable auditable retrieval with full decision logging")
    parser.add_argument("--audit-output", type=str, default=None, help="Save structured audit record to JSON file")

    args = parser.parse_args()

    if not args.neo4j_password:
        raise ValueError("NEO4J_PASSWORD is not set. Export NEO4J_PASSWORD or pass --neo4j-password.")

    question = args.question
    if not question and args.question_file:
        with open(args.question_file, "r", encoding="utf-8") as f:
            question = f.read().strip()

    if not question:
        raise ValueError("Provide --question or --question-file.")

    n4j = _connect_neo4j(
        url=_normalize_neo4j_url(args.neo4j_url),
        username=args.neo4j_username,
        password=args.neo4j_password,
    )

    patient_id = _extract_patient_id(question)

    if args.audit:
        from graphrag_audit import AuditableRetriever
        retriever = AuditableRetriever(
            driver=n4j,
            config={
                "top_k": max(1, args.top_k),
                "max_hops": args.max_hops,
                "max_evidence": args.max_evidence,
            }
        )
        audit_res = retriever.retrieve(query=question, patient_id=patient_id)
        evidence = audit_res.evidence
        answer = _answer_with_citations(question, evidence) if evidence else "No evidence retrieved."

        print("=" * 80)
        print("AUDITABLE RETRIEVAL TRACE")
        print("=" * 80)
        print(f"Query: {question}")
        print(f"Decision Rationale: {audit_res.decision_rationale}\n")
        
        print(f"=== Selected Paths ({len(audit_res.selected_paths)}) ===")
        for p in audit_res.selected_paths:
            print(f"  [{p.path_id}] (Score: {p.path_score:.3f}) {p.readable_path}")

        print(f"\n=== Rejected Alternative Paths ({len(audit_res.rejected_paths)}) ===")
        for p in audit_res.rejected_paths[:5]:
            print(f"  [{p.path_id}] (Score: {p.path_score:.3f}) {p.readable_path} --> REASON: {p.rejection_reason}")

        print(f"\n=== Edge Confidence Breakdown ({len(audit_res.edge_confidence)}) ===")
        for e in audit_res.edge_confidence[:8]:
            print(f"  {e['edge']} => Confidence: {e['confidence']:.3f} (Semantic: {e['semantic_score']:.2f}, Graph: {e['graph_support']:.2f}, Reliability: {e['source_reliability']:.2f})")

        print(f"\n=== Evidence Count ===")
        print(len(evidence))

        print(f"\n=== Answer ===")
        print(answer)

        if args.audit_output:
            import json
            out_dict = audit_res.to_dict()
            out_dict["answer"] = answer
            out_dir = os.path.dirname(args.audit_output)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(args.audit_output, "w", encoding="utf-8") as f:
                json.dump(out_dict, f, indent=2)
            print(f"\nAudit record saved to {args.audit_output}")
        return

    q_summary = _question_summary(question)
    seed_gids = select_top_gids(n4j, q_summary, top_k=max(1, args.top_k))
    patient_seed_gids = _patient_seed_gids(n4j, patient_id)

    expanded_gids = list(dict.fromkeys(patient_seed_gids + list(seed_gids)))
    for gid in seed_gids:
        expanded_gids.extend(_neighbor_gids(n4j, gid, max_hops=args.max_hops, max_items=20))
    for gid in patient_seed_gids:
        expanded_gids.extend(_neighbor_gids(n4j, gid, max_hops=args.max_hops, max_items=20))

    expanded_gids = list(dict.fromkeys(expanded_gids))

    # Preserve order while deduplicating.
    ordered_unique_gids = list(dict.fromkeys([g for g in expanded_gids if g]))

    patient_id = _extract_patient_id(question)
    evidence = _collect_evidence(n4j, ordered_unique_gids, max_evidence=max(10, args.max_evidence), patient_id=patient_id)
    if not evidence:
        print("No evidence retrieved from graph. Check graph construction and REFERENCE links.")
        return

    answer = _answer_with_citations(question, evidence)

    print("=== Selected GIDs ===")
    for gid in ordered_unique_gids:
        print(gid)

    print("\n=== Evidence Count ===")
    print(len(evidence))

    print("\n=== Answer ===")
    print(answer)


if __name__ == "__main__":
    main()

