"""
Pneumonia QA Validation Pipeline
Evaluates GraphRAG and baseline pipelines on pneumonia_qa_event.jsonl / pneumonia_qa_note.jsonl.

Usage:
    python experiments/validate_pneumonia.py \
        --data-dir pneumonia/ \
        --split both \
        --mode both \
        --limit 50 \
        --sample-mode random \
        --seed 42 \
        --run-audit \
        --output-dir experiments/outputs/validation/
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from post_graph_inference import (
    _answer_with_citations,
    _collect_evidence,
    _connect_neo4j,
    _neighbor_gids,
    _normalize_neo4j_url,
    _patient_seed_gids,
    _question_summary,
    _extract_patient_id,
)
from retrieve import select_top_gids
from utils import call_llm, get_embedding

try:
    from rouge_score import rouge_scorer as _rouge_scorer_mod
    _ROUGE_AVAILABLE = True
except ImportError:
    _ROUGE_AVAILABLE = False
    print("[validate] WARNING: rouge_score not installed; ROUGE-L will use built-in LCS.", file=sys.stderr)

try:
    import bert_score as _bert_score_mod
    _BERTSCORE_AVAILABLE = True
except ImportError:
    _BERTSCORE_AVAILABLE = False
    print("[validate] WARNING: bert_score not installed; BERTScore-F1 will be None.", file=sys.stderr)


# ---------------------------------------------------------------------------
# Section 1 — CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Pneumonia QA Validation Pipeline")
    p.add_argument("--data-dir", required=True, help="Path to the pneumonia/ folder")
    p.add_argument("--split", choices=["event", "note", "both"], default="both")
    p.add_argument("--mode", choices=["graphrag", "baseline", "both"], default="both")
    p.add_argument("--limit", type=int, default=None, help="Max questions per split")
    p.add_argument("--sample-mode", choices=["first", "random"], default="first")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--max-hops", type=int, default=2)
    p.add_argument("--run-audit", action="store_true", help="Enable graphrag_audit checks")
    p.add_argument(
        "--output-dir",
        default="experiments/outputs/validation/",
        help="Directory to save results",
    )
    p.add_argument("--neo4j-url", default=os.getenv("NEO4J_URL") or os.getenv("NEO4J_URI"))
    p.add_argument("--neo4j-username", default=os.getenv("NEO4J_USERNAME", "neo4j"))
    p.add_argument("--neo4j-password", default=os.getenv("NEO4J_PASSWORD"))
    return p


# ---------------------------------------------------------------------------
# Section 2 — Dataset Loading
# ---------------------------------------------------------------------------

def load_qa(
    path: str,
    limit: Optional[int],
    sample_mode: str,
    seed: int,
) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if limit is None or limit >= len(rows):
        return rows

    if sample_mode == "random":
        rng = random.Random(seed)
        return rng.sample(rows, limit)

    return rows[:limit]


def load_cohort(path: str) -> Dict[str, Dict[str, Any]]:
    cohort: Dict[str, Dict[str, Any]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = str(row.get("subject_id", "")).strip()
            if sid:
                cohort[sid] = dict(row)
    return cohort


# ---------------------------------------------------------------------------
# Section 3 — Inference Runners
# ---------------------------------------------------------------------------

def _token_count_estimate(text: str) -> int:
    return max(1, len(text.split()))


def _call_llm_tracked(
    sys_prompt: str, user_prompt: str
) -> Tuple[str, int, int]:
    """Call LLM and return (response_text, prompt_tokens, completion_tokens)."""
    try:
        from utils import _client, _chat_model_name, _run_with_hard_timeout

        client = _client()
        resp = _run_with_hard_timeout(
            45,
            client.chat.completions.create,
            model=_chat_model_name(),
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=500,
            n=1,
            stop=None,
            temperature=0.5,
            timeout=20,
        )
        content = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        if usage:
            pt = getattr(usage, "prompt_tokens", _token_count_estimate(sys_prompt + user_prompt))
            ct = getattr(usage, "completion_tokens", _token_count_estimate(content))
        else:
            pt = _token_count_estimate(sys_prompt + user_prompt)
            ct = _token_count_estimate(content)
        return content, pt, ct
    except Exception as exc:
        print(f"[validate] LLM error: {exc}", file=sys.stderr)
        return "", _token_count_estimate(sys_prompt + user_prompt), 0


def run_graphrag(
    question: str,
    subject_id: Optional[str],
    n4j: Any,
    top_k: int,
    max_hops: int,
) -> Tuple[str, List[Any], float, int, int]:
    """Run GraphRAG inference. Returns (answer, evidence, latency_s, prompt_tok, completion_tok)."""
    t0 = time.perf_counter()
    patient_id = _extract_patient_id(question) or (f"patient_{subject_id}" if subject_id else None)

    q_summary = _question_summary(question)
    seed_gids = select_top_gids(n4j, q_summary, top_k=max(1, top_k))
    patient_seed_gids = _patient_seed_gids(n4j, patient_id)

    expanded = list(dict.fromkeys(patient_seed_gids + list(seed_gids)))
    for gid in seed_gids:
        expanded.extend(_neighbor_gids(n4j, gid, max_hops=max_hops, max_items=20))
    for gid in patient_seed_gids:
        expanded.extend(_neighbor_gids(n4j, gid, max_hops=max_hops, max_items=20))

    ordered = list(dict.fromkeys(g for g in expanded if g))
    evidence = _collect_evidence(n4j, ordered, max_evidence=120, patient_id=patient_id)

    if not evidence:
        latency = time.perf_counter() - t0
        return "No evidence retrieved.", [], latency, 0, 0

    if os.getenv("USE_LLM_ANSWER", "0") == "1":
        numbered = [
            f"[{e['id']}] {e['content']}" if isinstance(e, dict) else f"[E{i+1}] {e}"
            for i, e in enumerate(evidence)
        ]
        from post_graph_inference import POST_SYS_PROMPT
        user_prompt = (
            f"Question: {question}\n\nEvidence:\n"
            + "\n".join(numbered[:8])
            + "\n\nAnswer using only the evidence above and include [E#] citations."
        )
        answer, pt, ct = _call_llm_tracked(POST_SYS_PROMPT, user_prompt)
    else:
        answer = _answer_with_citations(question, evidence)
        pt = _token_count_estimate(question)
        ct = _token_count_estimate(answer)

    latency = time.perf_counter() - t0
    return answer, evidence, latency, pt, ct


def run_baseline_q(
    question: str,
    n4j: Any,
    top_k: int,
    max_hops: int,
) -> Tuple[str, List[Any], float, int, int]:
    """Run baseline inference (mirrors run_baseline_experiment per question)."""
    t0 = time.perf_counter()

    q_summary = _question_summary(question)
    seed_gids = select_top_gids(n4j, q_summary, top_k=top_k)

    expanded = list(seed_gids)
    for gid in seed_gids:
        expanded.extend(_neighbor_gids(n4j, gid, max_hops=max_hops, max_items=20))

    ordered = list(dict.fromkeys(g for g in expanded if g))
    evidence = _collect_evidence(n4j, ordered, max_evidence=120)

    answer = _answer_with_citations(question, evidence) if evidence else "No evidence retrieved."
    pt = _token_count_estimate(question)
    ct = _token_count_estimate(answer)

    latency = time.perf_counter() - t0
    return answer, evidence, latency, pt, ct


# ---------------------------------------------------------------------------
# Section 4 — Metric Functions
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.strip()


def _tokenize(text: str) -> List[str]:
    return _normalize(text).split()


def exact_match(pred: str, gold: str) -> float:
    return 1.0 if _normalize(pred) == _normalize(gold) else 0.0


def token_f1(pred: str, gold: str) -> float:
    pred_toks = _tokenize(pred)
    gold_toks = _tokenize(gold)
    if not pred_toks or not gold_toks:
        return 0.0
    pred_counts: Dict[str, int] = defaultdict(int)
    gold_counts: Dict[str, int] = defaultdict(int)
    for t in pred_toks:
        pred_counts[t] += 1
    for t in gold_toks:
        gold_counts[t] += 1
    common = sum(min(pred_counts[t], gold_counts[t]) for t in pred_counts)
    if common == 0:
        return 0.0
    precision = common / len(pred_toks)
    recall = common / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def _lcs_len(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0
    m, n = len(a), len(b)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[n]


def rouge_l(pred: str, gold: str) -> float:
    if _ROUGE_AVAILABLE:
        scorer = _rouge_scorer_mod.RougeScorer(["rougeL"], use_stemmer=False)
        scores = scorer.score(gold, pred)
        return float(scores["rougeL"].fmeasure)
    pred_toks = _tokenize(pred)
    gold_toks = _tokenize(gold)
    if not pred_toks or not gold_toks:
        return 0.0
    lcs = _lcs_len(pred_toks, gold_toks)
    p = lcs / len(pred_toks)
    r = lcs / len(gold_toks)
    if p + r == 0:
        return 0.0
    return float(2 * p * r / (p + r))


def bertscore_f1(preds: List[str], golds: List[str]) -> List[Optional[float]]:
    if not _BERTSCORE_AVAILABLE or not preds:
        return [None] * len(preds)
    try:
        _, _, f1 = _bert_score_mod.score(
            preds, golds, lang="en", verbose=False, rescale_with_baseline=False
        )
        return [float(v) for v in f1.tolist()]
    except Exception as exc:
        print(f"[validate] BERTScore error: {exc}", file=sys.stderr)
        return [None] * len(preds)


_JUDGE_SYS = (
    "You are a clinical QA evaluator. Rate the PREDICTED answer vs the GOLD answer\n"
    "on a scale 1-5 for clinical accuracy, then flag any unsupported claims.\n\n"
    "Respond with EXACTLY this JSON (no markdown):\n"
    '{"score": <1-5>, "hallucination": <true|false>, "reason": "<one sentence>"}'
)


def llm_judge(question: str, gold: str, pred: str) -> Dict[str, Any]:
    user = f"Question: {question}\nGold: {gold}\nPredicted: {pred}"
    try:
        raw, pt, ct = _call_llm_tracked(_JUDGE_SYS, user)
        data = json.loads(raw)
        return {
            "judge_score": float(data.get("score", 3)),
            "hallucination": bool(data.get("hallucination", False)),
            "judge_reason": str(data.get("reason", "")),
            "judge_prompt_tokens": pt,
            "judge_completion_tokens": ct,
        }
    except Exception:
        return {
            "judge_score": None,
            "hallucination": None,
            "judge_reason": None,
            "judge_prompt_tokens": 0,
            "judge_completion_tokens": 0,
        }


def context_recall(gold: str, evidence: List[Any]) -> float:
    gold_toks = set(_tokenize(gold))
    if not gold_toks:
        return 1.0
    ev_text = " ".join(
        e["content"] if isinstance(e, dict) else str(e) for e in evidence
    )
    ev_toks = set(_tokenize(ev_text))
    return len(gold_toks & ev_toks) / len(gold_toks)


def context_precision(pred: str, evidence: List[Any]) -> float:
    pred_toks = set(_tokenize(pred))
    if not pred_toks:
        return 0.0
    ev_text = " ".join(
        e["content"] if isinstance(e, dict) else str(e) for e in evidence
    )
    ev_toks = set(_tokenize(ev_text))
    return len(pred_toks & ev_toks) / len(pred_toks)


_FAITH_SYS = (
    "You are a faithfulness checker. Given EVIDENCE and an ANSWER, determine if the\n"
    "answer only contains claims supported by the evidence.\n\n"
    "Respond with EXACTLY this JSON (no markdown):\n"
    '{"faithful": <true|false>, "reason": "<one sentence>"}'
)


def faithfulness(pred: str, evidence: List[Any]) -> Optional[bool]:
    if not evidence:
        return None
    ev_text = "\n".join(
        (e["content"] if isinstance(e, dict) else str(e)) for e in evidence[:12]
    )
    user = f"Evidence:\n{ev_text}\n\nAnswer:\n{pred}"
    try:
        raw, _, _ = _call_llm_tracked(_FAITH_SYS, user)
        data = json.loads(raw)
        return bool(data.get("faithful", True))
    except Exception:
        return None


def entity_coverage(evidence: List[Any], evidence_gids: List[str], n4j: Any) -> Optional[int]:
    if not evidence_gids or n4j is None:
        return None
    try:
        rows = n4j.query(
            """
            MATCH (n)
            WHERE n.gid IN $gids AND NOT n:Summary
            RETURN DISTINCT labels(n)[0] AS lbl
            """,
            {"gids": evidence_gids[:50]},
        )
        return len(rows)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Section 5 — Audit Runner
# ---------------------------------------------------------------------------

def run_audit(
    question: str,
    subject_id: Optional[str],
    n4j: Any,
    cohort_row: Optional[Dict[str, Any]],
    top_k: int,
    max_hops: int,
) -> Dict[str, Any]:
    from graphrag_audit import (
        AuditableRetriever,
        PneumoniaComplianceVerifier,
        RetrievalStabilityTester,
        CounterfactualEngine,
    )

    patient_id = f"patient_{subject_id}" if subject_id else None
    retriever = AuditableRetriever(
        driver=n4j,
        config={"top_k": top_k, "max_hops": max_hops, "max_evidence": 120},
    )

    audit_res = retriever.retrieve(query=question, patient_id=patient_id)

    verifier = PneumoniaComplianceVerifier()
    compliance_res = verifier.verify(audit_res)
    compliance_pass_rate = compliance_res.compliance_score

    edge_confs = [e.get("confidence", 0.0) for e in audit_res.edge_confidence]
    mean_edge_conf = float(sum(edge_confs) / len(edge_confs)) if edge_confs else None

    stability_tester = RetrievalStabilityTester()
    try:
        stab_report = stability_tester.evaluate_stability(
            query=question, retriever=retriever, original_result=audit_res
        )
        stability_score = stab_report.mean_path_jaccard
        top_path_retention = stab_report.top_path_retention_rate
    except Exception as exc:
        print(f"[validate] Stability error: {exc}", file=sys.stderr)
        stability_score = None
        top_path_retention = None

    cf_sensitivity = None
    if cohort_row:
        try:
            los_val = cohort_row.get("hadm_los", "")
            cf_engine = CounterfactualEngine()
            cf_res = cf_engine.counterfactual_retrieve(
                query=question,
                intervention={
                    "type": "VALUE_PERTURBATION",
                    "target": str(los_val)[:20],
                    "factor": 0.20,
                    "clinical_rationale": "Assess sensitivity to length-of-stay perturbation",
                },
                retriever=retriever,
                original_result=audit_res,
            )
            cf_sensitivity = cf_res.top_path_shifted
        except Exception as exc:
            print(f"[validate] Counterfactual error: {exc}", file=sys.stderr)

    return {
        "compliance_pass_rate": compliance_pass_rate,
        "compliance_status": compliance_res.overall_status,
        "mean_edge_confidence": mean_edge_conf,
        "stability_score": stability_score,
        "top_path_retention_rate": top_path_retention,
        "counterfactual_top_path_shifted": cf_sensitivity,
    }


# ---------------------------------------------------------------------------
# Section 6 — Resume + Streaming Output
# ---------------------------------------------------------------------------

def load_done_ids(jsonl_path: Path) -> Set[str]:
    done: Set[str] = set()
    if not jsonl_path.exists():
        return done
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                key = _result_key(rec["split"], rec["mode"], rec["q_id"])
                done.add(key)
            except Exception:
                pass
    return done


def _result_key(split: str, mode: str, q_id: str) -> str:
    return f"{split}|{mode}|{q_id}"


def stream_result(fh: Any, record: Dict[str, Any]) -> None:
    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    fh.flush()


# ---------------------------------------------------------------------------
# Section 7 — Aggregation & Reporting
# ---------------------------------------------------------------------------

def _safe_mean(values: List[Any]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return float(sum(vals) / len(vals)) if vals else None


def _percentile(values: List[float], pct: float) -> Optional[float]:
    if not values:
        return None
    sorted_vals = sorted(values)
    idx = min(int(len(sorted_vals) * pct / 100), len(sorted_vals) - 1)
    return sorted_vals[idx]


def aggregate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    latencies = [r["latency_s"] for r in records if r.get("latency_s") is not None]
    prompt_toks = [r.get("prompt_tokens", 0) for r in records]
    comp_toks = [r.get("completion_tokens", 0) for r in records]

    def _field(key: str) -> List[Any]:
        return [r.get(key) for r in records]

    result: Dict[str, Any] = {
        "n": len(records),
        "median_latency_s": _percentile(latencies, 50),
        "p95_latency_s": _percentile(latencies, 95),
        "avg_prompt_tokens": _safe_mean(prompt_toks),
        "avg_completion_tokens": _safe_mean(comp_toks),
        "exact_match": _safe_mean(_field("exact_match")),
        "token_f1": _safe_mean(_field("token_f1")),
        "rouge_l": _safe_mean(_field("rouge_l")),
        "bertscore_f1": _safe_mean(_field("bertscore_f1")),
        "judge_score": _safe_mean(_field("judge_score")),
        "hallucination_rate": _safe_mean(
            [1.0 if h else 0.0 for h in _field("hallucination") if h is not None]
        ),
        "context_recall": _safe_mean(_field("context_recall")),
        "context_precision": _safe_mean(_field("context_precision")),
        "faithfulness_rate": _safe_mean(
            [1.0 if f else 0.0 for f in _field("faithfulness") if f is not None]
        ),
        "entity_coverage": _safe_mean(_field("entity_coverage")),
    }

    for k in ("compliance_pass_rate", "mean_edge_confidence", "stability_score", "top_path_retention_rate"):
        result[k] = _safe_mean(_field(k))

    cf_flags = [r.get("counterfactual_top_path_shifted") for r in records]
    cf_valid = [1.0 if v else 0.0 for v in cf_flags if v is not None]
    result["counterfactual_sensitivity_rate"] = _safe_mean(cf_valid) if cf_valid else None

    return result


_METRICS_DISPLAY = [
    ("exact_match", "EM"),
    ("token_f1", "Token-F1"),
    ("rouge_l", "ROUGE-L"),
    ("bertscore_f1", "BERTScore-F1"),
    ("judge_score", "LLM-Judge (1-5)"),
    ("hallucination_rate", "Halluc. Rate"),
    ("context_recall", "Ctx Recall"),
    ("context_precision", "Ctx Precision"),
    ("faithfulness_rate", "Faithfulness"),
    ("entity_coverage", "Entity Cov."),
    ("median_latency_s", "Median Lat.(s)"),
    ("p95_latency_s", "P95 Lat.(s)"),
    ("avg_prompt_tokens", "Avg Prompt Tok"),
    ("avg_completion_tokens", "Avg Compl Tok"),
    ("compliance_pass_rate", "Compliance Rate"),
    ("mean_edge_confidence", "Mean Edge Conf"),
    ("stability_score", "Stability Score"),
    ("top_path_retention_rate", "Path Retention"),
    ("counterfactual_sensitivity_rate", "CF Sensitivity"),
]


def print_comparison_table(summaries: Dict[Tuple[str, str], Dict[str, Any]]) -> None:
    keys = sorted(summaries.keys())
    metric_w = 22
    col_w = max(18, max(len(f"{s}/{m}") for s, m in keys) + 4)

    sep = "+" + ("-" * metric_w) + "+" + (("-" * col_w) + "+") * len(keys)
    print(sep)
    header_row = "|" + "Metric".ljust(metric_w) + "|"
    for s, m in keys:
        header_row += f"{s}/{m}".center(col_w) + "|"
    print(header_row)
    print(sep)

    for field_key, label in _METRICS_DISPLAY:
        row = "|" + label.ljust(metric_w) + "|"
        for key in keys:
            val = summaries[key].get(field_key)
            if val is None:
                cell = "N/A"
            elif isinstance(val, float):
                cell = f"{val:.4f}"
            else:
                cell = str(val)
            row += cell.center(col_w) + "|"
        print(row)

    print(sep)


# ---------------------------------------------------------------------------
# Main Orchestration
# ---------------------------------------------------------------------------

def _llm_enabled() -> bool:
    return bool(
        os.getenv("GROQ_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("OPENROUTER_API_KEY")
    )


def _evaluate_split_mode(
    *,
    split: str,
    mode: str,
    qa_rows: List[Dict[str, Any]],
    cohort: Dict[str, Dict[str, Any]],
    n4j: Any,
    top_k: int,
    max_hops: int,
    run_audit_flag: bool,
    out_dir: Path,
    bertscore_batch_preds: List[str],
    bertscore_batch_golds: List[str],
    bertscore_batch_ids: List[str],
) -> List[Dict[str, Any]]:
    jsonl_path = out_dir / f"results_{split}_{mode}.jsonl"
    done_ids = load_done_ids(jsonl_path)
    records: List[Dict[str, Any]] = []
    llm_on = _llm_enabled()

    with open(jsonl_path, "a", encoding="utf-8") as fh:
        total = len(qa_rows)
        for idx, row in enumerate(qa_rows):
            hadm_id = str(row.get("hadm_id", ""))
            subject_id = str(row.get("subject_id", ""))
            q_id = f"{hadm_id}_{idx}"
            question = row.get("q", "")
            gold = row.get("a", "")

            if _result_key(split, mode, q_id) in done_ids:
                continue

            print(f"  [{idx+1}/{total}] {split}/{mode} hadm={hadm_id}", end=" ... ", flush=True)

            cohort_row = cohort.get(subject_id)

            try:
                if mode == "graphrag":
                    answer, evidence, latency, pt, ct = run_graphrag(
                        question, subject_id, n4j, top_k, max_hops
                    )
                else:
                    answer, evidence, latency, pt, ct = run_baseline_q(
                        question, n4j, top_k, max_hops
                    )
            except Exception as exc:
                print(f"\nINFERENCE ERROR: {exc}", file=sys.stderr)
                answer, evidence, latency, pt, ct = "", [], 0.0, 0, 0

            em = exact_match(answer, gold)
            tf1 = token_f1(answer, gold)
            rl = rouge_l(answer, gold)

            bertscore_batch_preds.append(answer)
            bertscore_batch_golds.append(gold)
            bertscore_batch_ids.append(q_id)

            if llm_on:
                judge_dict = llm_judge(question, gold, answer)
            else:
                judge_dict = {
                    "judge_score": None,
                    "hallucination": None,
                    "judge_reason": None,
                    "judge_prompt_tokens": 0,
                    "judge_completion_tokens": 0,
                }

            ctx_recall = context_recall(gold, evidence)
            ctx_prec = context_precision(answer, evidence)
            faith = faithfulness(answer, evidence) if llm_on else None

            ev_gids: List[str] = []
            if isinstance(evidence, list) and evidence and isinstance(evidence[0], dict):
                ev_gids = list({e.get("id", "") for e in evidence if isinstance(e, dict)})
            ent_cov = entity_coverage(evidence, ev_gids, n4j)

            audit_dict: Dict[str, Any] = {}
            if run_audit_flag and mode == "graphrag":
                try:
                    audit_dict = run_audit(
                        question, subject_id, n4j, cohort_row, top_k, max_hops
                    )
                except Exception as exc:
                    print(f"\n[validate] Audit error: {exc}", file=sys.stderr)

            pt_total = pt + judge_dict["judge_prompt_tokens"] + judge_dict["judge_completion_tokens"]

            record: Dict[str, Any] = {
                "split": split,
                "mode": mode,
                "q_id": q_id,
                "hadm_id": hadm_id,
                "subject_id": subject_id,
                "question": question,
                "gold": gold,
                "answer": answer,
                "event_type": row.get("event_type"),
                "official_split": row.get("official_split"),
                "latency_s": round(latency, 4),
                "prompt_tokens": pt_total,
                "completion_tokens": ct,
                "exact_match": em,
                "token_f1": round(tf1, 4),
                "rouge_l": round(rl, 4),
                "bertscore_f1": None,
                "judge_score": judge_dict["judge_score"],
                "hallucination": judge_dict["hallucination"],
                "judge_reason": judge_dict["judge_reason"],
                "context_recall": round(ctx_recall, 4),
                "context_precision": round(ctx_prec, 4),
                "faithfulness": faith,
                "entity_coverage": ent_cov,
                **audit_dict,
            }

            stream_result(fh, record)
            records.append(record)
            print(f"EM={em:.2f} TF1={tf1:.2f} ROUGE-L={rl:.2f} lat={latency:.2f}s")

    return records


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits_to_run = ["event", "note"] if args.split == "both" else [args.split]
    modes_to_run = ["graphrag", "baseline"] if args.mode == "both" else [args.mode]

    qa_files = {
        "event": data_dir / "pneumonia_qa_event.jsonl",
        "note": data_dir / "pneumonia_qa_note.jsonl",
    }
    cohort_file = data_dir / "pneumonia_cohort.csv"
    cohort = load_cohort(str(cohort_file)) if cohort_file.exists() else {}

    if not args.neo4j_password:
        print(
            "[validate] WARNING: NEO4J_PASSWORD not set. Graph inference will fail.",
            file=sys.stderr,
        )
        n4j = None
    else:
        n4j = _connect_neo4j(
            url=_normalize_neo4j_url(args.neo4j_url),
            username=args.neo4j_username,
            password=args.neo4j_password,
        )

    all_summaries: Dict[Tuple[str, str], Dict[str, Any]] = {}
    all_records_by_key: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    bertscore_registry: Dict[Tuple[str, str], Tuple[List[str], List[str], List[str]]] = {}

    for split in splits_to_run:
        qa_path = qa_files[split]
        if not qa_path.exists():
            print(f"[validate] SKIP: {qa_path} not found.", file=sys.stderr)
            continue

        qa_rows = load_qa(
            str(qa_path),
            limit=args.limit,
            sample_mode=args.sample_mode,
            seed=args.seed,
        )
        print(f"\n=== Split: {split} | {len(qa_rows)} questions ===")

        for mode in modes_to_run:
            print(f"--- Mode: {mode} ---")
            bp: List[str] = []
            bg: List[str] = []
            bi: List[str] = []

            records = _evaluate_split_mode(
                split=split,
                mode=mode,
                qa_rows=qa_rows,
                cohort=cohort,
                n4j=n4j,
                top_k=args.top_k,
                max_hops=args.max_hops,
                run_audit_flag=args.run_audit,
                out_dir=out_dir,
                bertscore_batch_preds=bp,
                bertscore_batch_golds=bg,
                bertscore_batch_ids=bi,
            )

            bertscore_registry[(split, mode)] = (bp, bg, bi)
            all_records_by_key[(split, mode)] = records

    if _BERTSCORE_AVAILABLE:
        print("\n[validate] Running BERTScore batch pass...")
        for (split, mode), (bp, bg, bi) in bertscore_registry.items():
            if not bp:
                continue
            bs_scores = bertscore_f1(bp, bg)
            jsonl_path = out_dir / f"results_{split}_{mode}.jsonl"
            all_lines: List[Dict[str, Any]] = []
            if jsonl_path.exists():
                with open(jsonl_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            all_lines.append(json.loads(line))

            id_to_bs = dict(zip(bi, bs_scores))
            for rec in all_lines:
                q_id = rec.get("q_id", "")
                if q_id in id_to_bs and id_to_bs[q_id] is not None:
                    rec["bertscore_f1"] = round(float(id_to_bs[q_id]), 4)

            with open(jsonl_path, "w", encoding="utf-8") as f:
                for rec in all_lines:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            for rec in all_records_by_key.get((split, mode), []):
                q_id = rec.get("q_id", "")
                if q_id in id_to_bs and id_to_bs[q_id] is not None:
                    rec["bertscore_f1"] = round(float(id_to_bs[q_id]), 4)

    print("\n[validate] Aggregating results...")
    for key, records in all_records_by_key.items():
        if records:
            all_summaries[key] = aggregate(records)
        else:
            split, mode = key
            jsonl_path = out_dir / f"results_{split}_{mode}.jsonl"
            loaded: List[Dict[str, Any]] = []
            if jsonl_path.exists():
                with open(jsonl_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            loaded.append(json.loads(line))
            if loaded:
                all_summaries[key] = aggregate(loaded)

    summary_data = {f"{s}_{m}": v for (s, m), v in all_summaries.items()}
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"[validate] Saved summary -> {summary_path}")

    if all_summaries:
        print("\n" + "=" * 80)
        print("VALIDATION RESULTS COMPARISON TABLE")
        print("=" * 80)
        print_comparison_table(all_summaries)

    print("\n[validate] Done.")


if __name__ == "__main__":
    main()
