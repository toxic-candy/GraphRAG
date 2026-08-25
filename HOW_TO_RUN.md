# How to Run Auditable Medical GraphRAG

This guide provides step-by-step instructions to execute the entire implemented pipeline: from data preprocessing and Neo4j graph construction to auditable retrieval, baseline evaluation, and the unit test suite.

---

## 1. Environment Setup

### 1.1 Activate Virtual Environment
```bash
# If .venv exists
source .venv/bin/activate

# Or recreate and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install pyyaml neo4j requests numpy pydantic pytest tiktoken anthropic pillow unstructured openai python-dotenv pandas
```

### 1.2 Configure Groq API (Free Tier) & Environment Variables
Copy `.env.example` to `.env` (already done) and put your free Groq API key:
```bash
# In .env:
GROQ_API_KEY=gsk_your_key_here
GROQ_MODEL=llama-3.3-70b-versatile
USE_LLM_ANSWER=1
```

Get a free key instantly at [https://console.groq.com/keys](https://console.groq.com/keys).

### 1.3 Start Neo4j (Local Docker)
```bash
docker rm -f medpneu-neo4j >/dev/null 2>&1 || true
docker run -d \
  --name medpneu-neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/test1234 \
  -e NEO4J_PLUGINS='["apoc"]' \
  neo4j:5
```

*(Note: If no API key is provided, the system automatically falls back to deterministic local hash embeddings and structured rule-based reasoning).*

---

## 2. Data Preprocessing & Three-Layer Dataset

### Step 2.1: Preprocess MIMIC-IV Clinical Data into Patient Reports
```bash
python preprocess_mimic_demo.py \
  --mimic-root ./mimic-iv-3.1-pneumonia-100 \
  --output-dir ./dataset/mimic_demo_10_pneumonia \
  --n-patients 10
```

### Step 2.2: Build the Three-Layer Text Hierarchy (Bottom / Middle / Top)
```bash
python prepare_three_layer_data.py \
  --repo-root . \
  --top-path ./dataset/mimic_demo_10_pneumonia \
  --out-root ./dataset/three_layer
```
Expected output folders:
- `dataset/three_layer/bottom` (Medical ontology dictionaries)
- `dataset/three_layer/middle` (Clinical guideline / knowledge chunks)
- `dataset/three_layer/top` (Patient narrative reports)

---

## 3. Ingest Graph with Provenance into Neo4j

Run graph ingestion with Trinity cross-layer `REFERENCE` links and provenance metadata:
```bash
USE_LLM_EXTRACTION=0 USE_LLM_SUMMARY=0 USE_REMOTE_EMBEDDINGS=0 \
python three_layer_import.py \
  --neo4j-password test1234 \
  --clear \
  --bottom ./dataset/three_layer/bottom \
  --middle ./dataset/three_layer/middle \
  --top ./dataset/three_layer/top \
  --trinity
```

---

## 4. Run Auditable Retrieval (Phase 3 Core Feature)

Run auditable retrieval with candidate path discovery, edge confidence decomposition, and explicit selection/rejection logging:

```bash
python post_graph_inference.py \
  --neo4j-password test1234 \
  --question "What are the clinical findings, diagnoses, and recommended treatments for this pneumonia patient?" \
  --top-k 3 \
  --max-hops 2 \
  --audit \
  --audit-output experiments/outputs/audit_sample.json
```

### What you will see:
1. **Decision Rationale**: Summary of candidate subgraphs evaluated, selected seeds, candidate paths discovered, and rejected alternatives.
2. **Selected Paths**: Top-ranked multi-hop graph reasoning paths with composite confidence scores.
3. **Rejected Alternative Paths**: Ranked paths that were rejected along with the explicit human-interpretable reason (e.g. score below threshold, budget cutoff).
4. **Edge Confidence Breakdown**: Component decomposition for each edge (`semantic_score`, `graph_support`, `source_reliability`, `stability`).
5. **Cited Answer**: Final clinical answer grounded in graph evidence.
6. **JSON Audit File**: Structured record saved to `experiments/outputs/audit_sample.json`.

---

## 5. Run Baseline Experiments (Phase 1)

Execute the reproducible baseline benchmark runner:
```bash
python experiments/baseline/run_baseline.py \
  --config experiments/configs/baseline_config.yaml \
  --neo4j-password test1234
```
Results will be saved to `experiments/outputs/baseline_<timestamp>.json`.

---

## 6. Run Standard Inference (Legacy/Baseline Mode)

To run inference without the audit trail (standard baseline mode):
```bash
python post_graph_inference.py \
  --neo4j-password test1234 \
  --question "What are the risk factors and treatment options for pneumonia?" \
  --top-k 2 \
  --max-hops 2
```

---

## 7. Run Automated Test Suite

Execute the full pytest suite verifying data structures, provenance tagging, candidate path exploration, confidence scoring, and edge cases:
```bash
.venv/bin/pytest -v tests/
```

All 7 test cases will execute:
- `tests/test_audit_retrieval.py::test_audit_data_structures`
- `tests/test_audit_retrieval.py::test_auditable_retriever_execution`
- `tests/test_audit_retrieval.py::test_auditable_retriever_threshold_rejection`
- `tests/test_audit_retrieval.py::test_auditable_retriever_empty_graph`
- `tests/test_audit_retrieval.py::test_auditable_retriever_custom_confidence_weights`
- `tests/test_provenance.py::test_provenance_patient_clinical_data`
- `tests/test_provenance.py::test_provenance_dictionary_data`
