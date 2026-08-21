# Baseline Reproduction Guide

This document describes how to execute and reproduce the frozen baseline of the Medical Knowledge Graph RAG System.

## Architecture
The frozen baseline implements the three-layer knowledge graph RAG pipeline:
1. **Top Layer**: MIMIC-IV Clinical Data (demographics, diagnoses, procedures, lab results, medications, microbiology).
2. **Middle Layer**: Clinical guideline / knowledge chunks derived from ICD ontology.
3. **Bottom Layer**: Medical ontology dictionaries (ICD-9/10 diagnoses, procedures, lab item dictionary).

## Reproduction Steps

### 1. Data Preparation
```bash
# 1. Preprocess MIMIC-IV demo data to patient narrative reports
python preprocess_mimic_demo.py \
  --mimic-root ./mimic-iv-3.1-pneumonia-100 \
  --output-dir ./dataset/mimic_demo_10_pneumonia \
  --n-patients 10

# 2. Build three-layer dataset (Bottom / Middle / Top)
python prepare_three_layer_data.py \
  --repo-root . \
  --top-path ./dataset/mimic_demo_10_pneumonia \
  --out-root ./dataset/three_layer
```

### 2. Neo4j Graph Ingestion
```bash
USE_LLM_EXTRACTION=0 USE_LLM_SUMMARY=0 USE_REMOTE_EMBEDDINGS=0 \
python three_layer_import.py \
  --neo4j-password <YOUR_PASSWORD> \
  --clear \
  --bottom ./dataset/three_layer/bottom \
  --middle ./dataset/three_layer/middle \
  --top ./dataset/three_layer/top \
  --trinity
```

### 3. Run Baseline Experiment Runner
```bash
python experiments/baseline/run_baseline.py \
  --config experiments/configs/baseline_config.yaml \
  --neo4j-password <YOUR_PASSWORD>
```

The output trace and metrics will be saved into `experiments/outputs/baseline_<timestamp>.json`.
