export interface EvidenceItem {
  id: string;
  path_id: string;
  content: string;
  confidence: number;
}

export interface CandidateNode {
  gid: string;
  type: string;
  preview: string;
  score: number;
  status: "selected" | "rejected";
  rejection_reason: string | null;
}

export interface CandidatePath {
  path_id: string;
  readable_path: string;
  path_score: number;
  selection_status: "selected" | "rejected" | "pending";
  rejection_reason: string | null;
  evidence_text: string;
  nodes: any[];
  edges: any[];
}

export interface AuditTrace {
  query: string;
  retrieval: {
    candidate_count: number;
    selected: CandidateNode[];
    rejected: CandidateNode[];
    parameters: any;
  };
  graph_expansion: {
    candidate_paths: CandidatePath[];
    selected_paths: CandidatePath[];
    rejected_paths: CandidatePath[];
  };
  evidence: EvidenceItem[];
  answer: {
    text: string;
    citations: string[];
  };
}

export function parseAuditTrace(raw: any): AuditTrace {
  let normalizedEvidence: EvidenceItem[] = [];
  if (raw.evidence && Array.isArray(raw.evidence)) {
    if (typeof raw.evidence[0] === "string") {
      normalizedEvidence = raw.evidence.map((text: string, idx: number) => {
        const matchingPath = (raw.selected_paths || []).find((p: any) => p.evidence_text === text);
        return {
          id: `E${idx + 1}`,
          path_id: matchingPath ? matchingPath.path_id : `legacy_path_${idx}`,
          content: text,
          confidence: matchingPath ? matchingPath.path_score : 0,
        };
      });
    } else {
      normalizedEvidence = raw.evidence;
    }
  }

  const answerText = raw.answer || "";
  const citations = Array.from(answerText.matchAll(/\[E\d+\]/g)).map((m: any) => m[0].replace(/\[|\]/g, ""));

  const seedNodes: CandidateNode[] = raw.seed_nodes || [];
  const selectedSeeds = seedNodes.filter(s => s.status === "selected");
  const rejectedSeeds = seedNodes.filter(s => s.status === "rejected");

  const candidatePaths: CandidatePath[] = raw.candidate_paths || [];
  const selectedPaths = raw.selected_paths || candidatePaths.filter(p => p.selection_status === "selected");
  const rejectedPaths = raw.rejected_paths || candidatePaths.filter(p => p.selection_status === "rejected");

  return {
    query: raw.query || "Unknown Query",
    retrieval: {
      candidate_count: seedNodes.length,
      selected: selectedSeeds,
      rejected: rejectedSeeds,
      parameters: raw.retrieval_parameters || {}
    },
    graph_expansion: {
      candidate_paths: candidatePaths,
      selected_paths: selectedPaths,
      rejected_paths: rejectedPaths
    },
    evidence: normalizedEvidence,
    answer: {
      text: answerText,
      citations
    }
  };
}
