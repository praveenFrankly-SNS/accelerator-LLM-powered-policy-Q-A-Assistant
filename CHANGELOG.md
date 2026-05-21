# Changelog

All notable changes to the LLM-Powered Policy Q&A Assistant accelerator are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [1.0.0] — 2026-05-21

### Added
- **Notebook 01** — Bronze ingestion: Auto Loader from Unity Catalog Volumes, SHA-256 doc IDs, version tracking, pipeline audit log
- **Notebook 02** — Silver chunking: token-aware chunking (400 tokens / 50 overlap), sentence-boundary aware, quarantine table for rejected records, data quality assertions
- **Notebook 03** — Vector Search: Mosaic AI Vector Search endpoint + Delta Sync index (triggered mode), BGE-Large-EN-v1.5 embeddings via Foundation Models API, retrieval smoke test
- **Notebook 04** — RAG chain: LangChain + ChatDatabricks + DatabricksVectorSearch, strict grounding prompt, 20-question evaluation dataset, MLflow LangChain autolog, UC model registry
- **Notebook 05** — Serving + UI: Mosaic AI Model Serving (serverless, scale-to-zero), Gradio UI on Databricks Apps, Gold table logging (qa_interactions, qa_feedback)
- **Guardrails layer**: PII detection/redaction, prompt injection detection, input length limit, similarity threshold disclaimer, output PII scan, verbatim reproduction prevention
- **RUNME.py**: single entry point, runs full pipeline end-to-end with parameterized widgets
- **databricks.yml**: full DAB with dev/staging/prod targets, job clusters with cost tags, weekly sync job
- **config.py**: centralized configuration for all models, UC paths, chunking params, guardrail thresholds
- **5 sample policy documents**: leave, IT security, expenses, data privacy, remote work (synthetic, no real PII)
- **README.md**: full documentation including architecture, prerequisites, cost estimate, failure modes, guardrails
- **CONTRIBUTING.md**: contribution guidelines and issue reporting
- **LICENSE**: Apache 2.0
- **requirements.txt**: pinned dependencies
- **docs/model_card.md**: model card with intended use, limitations, evaluation results

### Architecture
- Medallion Architecture: Bronze → Silver → Gold with quarantine at every stage
- Unity Catalog: all tables, volumes, model registry, Vector Search index under UC governance
- MLflow: full experiment tracking, LangChain traces, model signatures, UC model registry
- Databricks Workflows: 5-task pipeline with retry, alerting, cost tags
