# GraphRAG

GraphRAG builds a three-layer medical knowledge graph from MIMIC clinical data and guideline documents. Neo4j stores the graph, and retrieval can follow cross-layer `REFERENCE` links to produce evidence-grounded clinical answers.

## Windows prerequisites

- Python 3.10 or newer
- Docker Desktop with Linux containers enabled
- Neo4j 5 running locally
- MIMIC-IV data if you want to regenerate patient reports

From PowerShell:

```powershell
Set-Location -Path "d:\GraphRAG"
```

## Install dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or invoke `.venv\Scripts\python.exe` directly.

## Start Neo4j

Start Docker Desktop, then create the container once:

```powershell
docker run -d --name medpneu-neo4j `
  -p 7474:7474 -p 7687:7687 `
  -e NEO4J_AUTH=neo4j/test1234 `
  -e NEO4J_PLUGINS='["apoc"]' `
  neo4j:5
```

For later runs, use `docker start medpneu-neo4j`. Neo4j Browser is at <http://localhost:7474>; the local Bolt address is `bolt://localhost:7687`.

## Configure environment variables

```powershell
$env:NEO4J_URI = "bolt://localhost:7687"
$env:NEO4J_USERNAME = "neo4j"
$env:NEO4J_PASSWORD = "test1234"
$env:USE_LLM_EXTRACTION = "0"
$env:USE_LLM_SUMMARY = "0"
$env:USE_REMOTE_EMBEDDINGS = "0"
$env:USE_LLM_ANSWER = "1"
```

For an OpenAI-compatible provider, set the API key, endpoint, and model in the session. Keep credentials out of source control:

```powershell
$env:GROQ_API_KEY = "<your-api-key>"
$env:OPENAI_API_BASE_URL = "https://api.groq.com/openai/v1"
$env:OPENAI_MODEL = "groq/compound-mini"
```

## Build the three-layer dataset

Generate reports from the local MIMIC-IV demo database:

```powershell
python preprocess_mimic_demo.py `
  --mimic-root mimic-iv-clinical-database-demo-2.2 `
  --output-dir dataset/mimic_demo_100_real `
  --n-patients 100
```

Prepare the bottom, middle, and top layers:

```powershell
python prepare_three_layer_data.py `
  --repo-root . `
  --top-path dataset/mimic_demo_100_real `
  --out-root dataset/three_layer
```

This creates `dataset/three_layer/bottom`, `dataset/three_layer/middle`, and `dataset/three_layer/top`. For the smaller checked-in sample, use `dataset/mimic_demo_10_pneumonia` as `--top-path`.

## Import the graph

```powershell
python three_layer_import.py `
  --neo4j-url bolt://localhost:7687 `
  --neo4j-username neo4j `
  --neo4j-password test1234 `
  --clear `
  --bottom dataset/three_layer/bottom `
  --middle dataset/three_layer/middle `
  --top dataset/three_layer/top `
  --trinity
```

`--trinity` creates cross-layer `REFERENCE` links. The importer supports a non-APOC Neo4j fallback. Review the entity, summary, relationship, `REFERENCE`, and per-layer counts printed at the end.

## Query the graph

```powershell
python post_graph_inference.py `
  --neo4j-url bolt://localhost:7687 `
  --neo4j-username neo4j `
  --neo4j-password test1234 `
  --question "What are the major clinical problems and likely interventions for this patient?" `
  --top-k 2 `
  --max-hops 2
```

Use `--question-file` for a prompt file. Add `--audit --output-dir audit_output` to write retrieval and answer audit artifacts.

## Inspect Neo4j Browser

Open <http://localhost:7474>, sign in with `neo4j`, and run:

```cypher
MATCH p=(n)-[r]->(m)
RETURN p
LIMIT 200
```

```cypher
MATCH p=(a:Entity)-[:REFERENCE]->(b:Entity)
RETURN p
LIMIT 300
```

```cypher
MATCH ()-[r]->()
RETURN type(r) AS relationship_type, count(*) AS count
ORDER BY count DESC
```

## Other entry points

- `three_layer_import.py`: current three-layer graph importer
- `post_graph_inference.py`: graph retrieval and clinical QA
- `preprocess_mimic_demo.py`: convert MIMIC tables into report text
- `prepare_three_layer_data.py`: build bottom, middle, and top datasets
- `run.py`: legacy single-dataset construction and inference runner
- `tests/`: project tests
- `RUNNING_GUIDE.md`: expanded Windows operations notes

## Troubleshooting

**Neo4j connection refused:** confirm Docker Desktop is running, then run `docker ps` and check ports `7474` and `7687`.

**PowerShell cannot activate `.venv`:** use `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or invoke `.venv\Scripts\python.exe` directly.

**APOC warnings:** the importer can fall back to the non-APOC client. Confirm graph paths and Neo4j credentials before rerunning.
Or use a prompt file:
