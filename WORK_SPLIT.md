# Medical-Graph-RAG — 4-Person Work Split Report

> **Project:** Medical Knowledge Graph Construction & Retrieval-Augmented Generation  
> **Source:** [2408.04187](2408.04187v1.pdf) — *MedGraphRAG: Enhancing Medical Reasoning with Graph-based RAG*  
> **Dataset:** MIMIC-IV Clinical Database Demo v2.2

---

## System Architecture Overview

```
[Raw MIMIC-IV CSVs]
        │
        ▼
 Person 1: Data Engineering & Preprocessing
        │   preprocess_mimic_demo.py  →  patient report .txt files
        │   prepare_three_layer_data.py  → bottom / middle / top .txt layers
        │
        ▼
 Person 2: Graph Construction & LLM Extraction
        │   create_graph.py            → entity/relation extraction (LLM or fallback)
        │   summarize.py              → per-chunk LLM summaries
        │   agentic_chunker.py        → adaptive text chunking
        │   dataloader.py / data_chunk.py → file reading & splitting
        │
        ▼
 Person 3: Graph Storage, Schema & Import Orchestration
        │   three_layer_import.py     → orchestrates bottom/middle/top import
        │   simple_neo4j_graph.py     → Neo4j adapter (APOC-free fallback)
        │   cleangraph.py             → graph maintenance / deduplication
        │   utils.py (graph ops)      → add_gid, ref_link, add_ge_emb
        │
        ▼
 Person 4: Retrieval, Inference & Evaluation
            retrieve.py               → GID selection by embedding similarity
            post_graph_inference.py   → end-to-end query → evidence → answer
            utils.py (context ops)    → ret_context, link_context, call_llm
            run.py                    → original single-file pipeline runner
```

---

## Person 1 — Data Engineering & Preprocessing

**Scope:** Raw clinical data ingestion, structuring, and three-layer dataset preparation.

### Responsibilities

| Task | File | Details |
|------|------|---------|
| MIMIC-IV table loading | `preprocess_mimic_demo.py` | Loads 9 CSV tables: `patients`, `admissions`, `diagnoses_icd`, `procedures_icd`, `labevents`, `prescriptions`, `microbiologyevents`, and their ICD/lab dictionaries |
| ICD code resolution | `preprocess_mimic_demo.py` | Joins `d_icd_diagnoses` and `d_icd_procedures` to resolve raw ICD codes to human-readable `long_title` names |
| Narrative report assembly | `preprocess_mimic_demo.py` | Transforms structured CSV rows into natural-language `.txt` patient reports (demographics, admissions, diagnoses, procedures, lab results, medications, microbiology) |
| Bottom-layer dictionary files | `prepare_three_layer_data.py` | Produces `DIAGNOSIS`, `PROCEDURE`, and `LAB_TEST` ontology dictionary files from ICD & lab master tables (≤ 2,500 / 1,800 / 1,800 entries each) |
| Middle-layer guideline files | `prepare_three_layer_data.py` | Extracts `MedicalBook.xlsx` sheets and/or synthesizes guideline-style chunked narratives from ICD titles |
| Top-layer patient files | `prepare_three_layer_data.py` | Copies preprocessed patient report `.txt` files into `dataset/three_layer/top/` |
| Output layout | Both scripts | Produces `dataset/three_layer/{bottom,middle,top}/` hierarchy consumed by Person 3's importer |

### Key Design Decisions
- Reports cover up to 3 admissions per patient, the 12 most-sequenced diagnoses, 8 procedures, and the 20 most recent lab results.
- ICD version (9 vs 10) is preserved through the join keys to avoid code collisions.
- The `MedicalBook.xlsx` sheet content provides a non-MIMIC knowledge source for the middle layer.

### Input / Output
- **Input:** `mimic-iv-clinical-database-demo-2.2/hosp/*.csv`, `MedicalBook.xlsx`
- **Output:** `dataset/mimic_demo_10/*.txt`, `dataset/three_layer/{bottom,middle,top}/*.txt`

---

## Person 2 — Graph Construction & LLM Extraction

**Scope:** Converting raw text chunks into typed graph elements (nodes + relationships) using LLMs or rule-based fallback.

### Responsibilities

| Task | File | Details |
|------|------|---------|
| Text chunking | `agentic_chunker.py`, `data_chunk.py` | Splits documents into semantically coherent chunks; `agentic_chunker.py` uses an LLM to propose chunk boundaries and titles |
| File loading | `dataloader.py` | `load_high()` reads raw `.txt` files |
| LLM-based entity/relation extraction | `create_graph.py` → `_extract_graph_elements_from_text()` | Calls the configured OpenAI-compatible model with a structured medical extraction prompt; enforces `timeout=45` on the HTTP client |
| Camel-AI graph parsing | `create_graph.py` → `UnstructuredIO` | Parses LLM JSON output into `GraphElement` objects (nodes + relationships) via `camel.loaders.UnstructuredIO` |
| Structured fallback extraction | `create_graph.py` → `_extract_fallback()` | Regex + keyword pattern extraction when `USE_LLM_EXTRACTION=0`; produces chain-style entity-relation triples across medical entity types |
| Per-chunk summarization | `summarize.py` → `process_chunks()` | Splits text into ≤500-token chunks; calls the LLM summary prompt in parallel via `ThreadPoolExecutor`; uses `timeout=45` natively (no signal) |
| Alternate graph builder | `create_graph_with_description.py` | Variant that includes natural-language relationship descriptions as node properties |

### Key Design Decisions
- `USE_LLM_EXTRACTION=1` enables full LLM extraction; `=0` falls back to deterministic regex patterns — allowing offline/dev runs.
- `USE_LLM_SUMMARY=1` enables LLM summarization; `=0` skips it, using the question text directly.
- All `signal.alarm`-based timeouts were removed; the OpenAI `timeout` parameter on the HTTP client is used exclusively to avoid `signal only works in main thread` crashes when called from `ThreadPoolExecutor` workers.

### Input / Output
- **Input:** Chunked text strings from document layer files
- **Output:** `GraphElement` objects (in-memory) handed to Person 3's storage layer

---

## Person 3 — Graph Storage, Schema & Import Orchestration

**Scope:** Storing extracted graph elements in Neo4j, managing the three-layer import flow, and maintaining graph hygiene.

### Responsibilities

| Task | File | Details |
|------|------|---------|
| Three-layer import orchestration | `three_layer_import.py` → `ThreeLayerImporter` | Iterates bottom → middle → top layer files; calls `create_metagraph()` per file; assigns a UUID `gid` per document |
| Node embedding | `utils.py` → `add_ge_emb()` | Batches all node texts into a single remote embedding API call (`USE_REMOTE_EMBEDDINGS=1`) or uses a local `CamelEmbeddingModel` fallback; stores embedding vectors on each node |
| GID tagging | `utils.py` → `add_gid()` | Tags every node in a graph element with the document's UUID `gid` for later retrieval |
| Cross-layer Trinity linking | `three_layer_import.py` → `create_trinity_links()`, `utils.py` → `ref_link()` | Creates `REFERENCE` relationships between nodes of adjacent layers (bottom↔middle, middle↔top) |
| Neo4j adapter — APOC path | `camel.storages.Neo4jGraph` | Full-featured adapter used when APOC plugin is available |
| Neo4j adapter — fallback | `simple_neo4j_graph.py` → `SimpleNeo4jGraph` | Custom APOC-free adapter using raw `neo4j` driver; supports `add_graph_elements()` and `query()` without `apoc.meta.data()` |
| Graph deduplication | `cleangraph.py` | Utility for removing orphan or duplicate nodes post-import |
| Summary node creation | `utils.py` → `add_sum()` | Creates `Summary` nodes in Neo4j linked to the originating document's `gid` when `USE_LLM_SUMMARY=1` |

### Key Design Decisions
- The dual Neo4j adapter pattern (`Neo4jGraph` → `SimpleNeo4jGraph` fallback) allows deployment both with and without APOC plugins.
- Batched embedding (`add_ge_emb`) replaced per-node sequential HTTP calls to eliminate import bottlenecks.
- `REFERENCE` relationships are the primary cross-document traversal primitive used in retrieval.

### Input / Output
- **Input:** `GraphElement` objects from Person 2; Neo4j connection credentials
- **Output:** Populated Neo4j graph with typed nodes, embeddings, `gid` properties, and `REFERENCE` cross-links

---

## Person 4 — Retrieval, Inference & Evaluation

**Scope:** Querying the populated graph, ranking evidence, and generating clinically grounded answers.

### Responsibilities

| Task | File | Details |
|------|------|---------|
| GID selection by embedding similarity | `retrieve.py` → `select_top_gids()` | Embeds the query, fetches all `Summary` node embeddings from Neo4j, ranks by cosine similarity, returns top-K GIDs. Falls back to entity node embeddings if no `Summary` nodes exist |
| REFERENCE hop expansion | `post_graph_inference.py` → `_neighbor_gids()` | Traverses `REFERENCE` edges up to `--max-hops` depth from each seed GID to expand the candidate evidence neighborhood |
| Local context retrieval | `utils.py` → `ret_context()` | Fetches all intra-GID node–node relationships (excluding `REFERENCE`) as evidence triples |
| Cross-document context retrieval | `utils.py` → `link_context()` | Fetches cross-GID `REFERENCE`-linked entity relationships as additional evidence |
| Evidence deduplication | `post_graph_inference.py` → `_collect_evidence()` | Deduplicated evidence lines collected across all expanded GIDs, capped at `--max-evidence` |
| LLM answer generation | `post_graph_inference.py` → `_answer_with_citations()` | When `USE_LLM_ANSWER=1`: passes numbered evidence to the LLM with citation instruction. When `=0`: returns the top-8 evidence lines deterministically |
| Question summarization at inference | `post_graph_inference.py` → `_question_summary()` | Optionally runs the question through `process_chunks()` before embedding for longer, complex questions |
| Original single-file pipeline | `run.py` | Legacy runner integrating document load → graph build → inference in a single script |

### Key Design Decisions
- Cosine similarity ranking in pure NumPy (no vector index required) keeps the retrieval layer database-agnostic.
- The `Summary`-node fallback in `select_top_gids()` ensures retrieval works even when `USE_LLM_SUMMARY=0` during import.
- The `USE_LLM_ANSWER=0` deterministic mode enables fully offline/reproducible evaluation without any LLM costs at query time.
- `--top-k` and `--max-hops` are the two primary levers for trading coverage vs. precision.

### Input / Output
- **Input:** Natural-language clinical question; populated Neo4j graph
- **Output:** Ranked GID list, evidence count, cited answer (stdout)

---

## Environment Variables Summary

| Variable | Default | Effect |
|----------|---------|--------|
| `USE_LLM_EXTRACTION` | `0` | `1` = LLM entity extraction; `0` = regex fallback |
| `USE_LLM_SUMMARY` | `0` | `1` = LLM chunk summaries + `Summary` nodes; `0` = skip |
| `USE_REMOTE_EMBEDDINGS` | `0` | `1` = batch remote embedding API; `0` = local CamelEmbeddingModel |
| `USE_LLM_ANSWER` | `0` | `1` = LLM-generated cited answer; `0` = deterministic top-8 |
| `OPENAI_API_KEY` | — | API key for OpenRouter / OpenAI |
| `OPENAI_API_BASE_URL` | `https://openrouter.ai/api/v1` | Custom LLM endpoint |
| `OPENAI_MODEL` | `meta-llama/llama-3-8b-instruct` | Model for extraction, summary, and answering |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection URL |
| `NEO4J_USERNAME` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | — | Neo4j password |

---

## End-to-End Command Reference

```bash
# Step 1 — Preprocess MIMIC-IV into patient reports (Person 1)
python preprocess_mimic_demo.py --n-patients 10

# Step 2 — Build the three-layer dataset (Person 1)
python prepare_three_layer_data.py

# Step 3 — Import all three layers into Neo4j (Person 2 + 3)
USE_LLM_EXTRACTION=1 USE_LLM_SUMMARY=1 USE_REMOTE_EMBEDDINGS=1 \
python three_layer_import.py \
  --neo4j-password test1234 \
  --clear \
  --bottom ./dataset/three_layer/bottom \
  --middle ./dataset/three_layer/middle \
  --top    ./dataset/three_layer/top \
  --trinity

# Step 4 — Query the graph (Person 4)
USE_REMOTE_EMBEDDINGS=1 USE_LLM_ANSWER=1 \
python post_graph_inference.py \
  --neo4j-password test1234 \
  --question "What are the major clinical problems and likely interventions for this patient?" \
  --top-k 5 \
  --max-hops 2
```
