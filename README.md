# LLM-Powered Policy Q&A Assistant

**Accelerator Version:** 1.0.0 | **Author:** SNS Square | **Runtime:** DBR 14.3 LTS ML

> Ask natural-language questions about company policy documents and get cited, grounded answers — powered by Databricks Mosaic AI, Llama 3.1, and RAG.

---

## Problem Statement

Enterprise HR, legal, and compliance teams maintain dozens of policy documents (leave, IT security, expenses, data privacy, remote work). Employees waste time searching through PDFs for answers, and HR teams field repetitive questions. Incorrect or outdated answers create compliance risk.

## Solution

A production-grade Retrieval-Augmented Generation (RAG) system that:
- Ingests policy documents (PDF, DOCX, TXT) into a governed Medallion Architecture
- Chunks and embeds documents using BGE-Large via Mosaic AI Foundation Models API
- Retrieves the top-5 most relevant chunks per query using Mosaic AI Vector Search
- Generates cited, grounded answers using Llama 3.1 — never guesses, always cites sources
- Applies enterprise guardrails: PII detection, prompt injection blocking, hallucination disclaimers
- Serves a Gradio UI on Databricks Apps with full audit logging to Gold Delta tables

---

## Architecture

```
PDF / DOCX / TXT files
        ↓  (Auto Loader — incremental, append-only)
[Bronze] catalog.schema.raw_documents
        ↓  (token-aware chunking, 400 tokens / 50 overlap, quarantine for bad records)
[Silver] catalog.schema.document_chunks   ← Vector Search source table
        ↓  (BGE-Large-EN-v1.5 via Foundation Models API)
Mosaic AI Vector Search Index
        ↑
User Query
  → Input Guardrails (PII redaction, injection detection, length check)
  → Embed query → cosine similarity → top-5 chunks
  → Llama 3.1 + strict grounding prompt
  → Output Guardrails (PII scan, verbatim reproduction check)
  → Cited Answer [Source: filename, Page X, Section Y]
        ↓
Gradio UI (Databricks Apps — public URL, zero external hosting)
        ↓
[Gold] catalog.schema.qa_interactions + qa_feedback
```

See [`/docs/architecture_diagram.png`](docs/architecture_diagram.png) for the full visual.

---

## Databricks Features Used

| Feature | Role |
|---|---|
| Delta Lake (Unity Catalog) | Bronze / Silver / Gold tables — fully governed |
| Unity Catalog Volumes | Stores raw PDF / DOCX files |
| Mosaic AI Foundation Models API | BGE-Large-EN embeddings + Llama 3.1 LLM |
| Mosaic AI Vector Search | Semantic similarity search (1 endpoint = free tier) |
| LangChain on Databricks | RAG chain: retriever + prompt + LLM + output parser |
| MLflow (LangChain autolog) | Full trace: retrieval → prompt → generation → latency |
| Mosaic AI Model Serving (serverless) | REST endpoint, scales to zero = free when idle |
| Databricks Apps (Gradio) | Demo UI — hosted on Databricks, public URL |
| Unity AI Gateway | Guardrails: PII detection, prompt injection, hallucination guard |
| Databricks Workflows | Orchestration, retry, alerting, audit trail |
| Databricks Lakehouse Monitoring | Gold table drift, freshness, row count alerts |

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Databricks Runtime | 14.3 LTS ML or later |
| Unity Catalog | Enabled on workspace |
| Mosaic AI Foundation Models API | Enabled (free trial or pay-per-token) |
| Mosaic AI Vector Search | 1 endpoint available (free tier quota) |
| Databricks Apps | Enabled on workspace |
| Secrets scope | `policy_qa_scope` with key `db_token` (see Setup below) |

### Secrets Setup

```bash
# Create secrets scope
databricks secrets create-scope policy_qa_scope

# Add your Databricks PAT (max 90-day expiry recommended)
databricks secrets put-secret policy_qa_scope db_token
```

---

## How to Run

### Option A — One-command deploy (recommended)

```bash
# Clone the repo
git clone https://github.com/sns-square/llm-powered-policy-qa-assistant.git
cd llm-powered-policy-qa-assistant

# Deploy to dev (uses default catalog: dev_policy_qa)
databricks bundle deploy --target dev
databricks bundle run policy_qa_pipeline --target dev
```

### Option B — Run RUNME notebook manually

1. Import the repo into your Databricks workspace
2. Open `RUNME.py` and attach to a DBR 14.3 LTS ML cluster
3. Set widget values (catalog, schema, environment)
4. Run all cells — the full pipeline runs end-to-end

### Option C — Run notebooks individually

```
01_ingest_documents.py   → Bronze layer
02_chunk_and_process.py  → Silver layer
03_embed_and_index.py    → Vector Search
04_build_rag_chain.py    → RAG chain + evaluation
05_serve_and_demo.py     → Model Serving + Gradio UI
```

---

## Expected Outputs

After a successful run:

| Asset | Location |
|---|---|
| Bronze table | `dev_policy_qa.policy_assistant.raw_documents` |
| Silver table | `dev_policy_qa.policy_assistant.document_chunks` |
| Quarantine table | `dev_policy_qa.policy_assistant.rejected_chunks` |
| Vector Search index | `dev_policy_qa.policy_assistant.policy_chunks_index` |
| Registered model | `dev_policy_qa.policy_assistant.policy_qa_rag_chain` |
| Serving endpoint | `policy_qa_endpoint` (serverless) |
| Gradio UI | Databricks Apps — check Apps panel for public URL |
| Interaction log | `dev_policy_qa.policy_assistant.qa_interactions` |
| Feedback log | `dev_policy_qa.policy_assistant.qa_feedback` |
| Pipeline audit log | `dev_policy_qa.policy_assistant.pipeline_audit_log` |
| MLflow experiment | `/Shared/policy_qa_assistant` |

---

## Cost Estimate

| Item | Cost |
|---|---|
| BGE-Large embeddings (50 docs, ~500K tokens) | ~$0.05 one-time |
| Llama 3.1 8B — dev queries (200 queries) | ~$0.05 |
| Llama 3.1 70B — final demo recording (1 session) | ~$0.50 |
| Vector Search endpoint | Free (1 endpoint = free tier quota) |
| Serverless serving | Free when idle (scales to zero) |
| **Total to build + demo** | **< $2** |

**Monthly production estimate** (500 queries/day, Llama 3.1 8B):
- ~150K tokens/day × $0.20/1M tokens × 30 days = **~$0.90/month**

---

## Guardrails

This accelerator implements enterprise-grade guardrails at every layer:

**Input guardrails (before LLM):**
- PII detection and redaction (email, phone, SSN, credit card patterns)
- Prompt injection detection (blocks jailbreak attempts)
- Input length limit (2,000 tokens max)
- Control character sanitization

**Output guardrails (before user):**
- PII scan on LLM response
- Verbatim reproduction prevention (proprietary document protection)
- Hallucination disclaimer when similarity score < 0.70
- Graceful not-found response when answer is not in documents

**Unity AI Gateway** (production): configure PII guardrail, prompt injection guardrail, hallucination guard, and payload logging via the Databricks UI.

---

## Failure Modes

| Scenario | Behavior |
|---|---|
| Empty source Volume | Pipeline logs 0 records, exits cleanly with audit entry |
| Corrupt / unreadable PDF | Record written to `rejected_chunks` quarantine table |
| Schema change in source | Delta schema enforcement catches it; pipeline fails with clear error |
| LLM unavailable | 3 retries with exponential backoff; graceful error message to user |
| Vector Search cold start | Retry logic in chain; UI shows warm-up message |
| Query not in any document | LLM explicitly says "I could not find information..." — never guesses |
| Stale Vector Search index | Weekly sync job re-triggers index sync automatically |

---

## Known Limitations

- **1 Vector Search endpoint** on free tier — this accelerator uses the quota. Additional accelerators need a paid tier.
- **Serverless cold start** — first query after idle period takes ~5–10 seconds.
- **PDF extraction quality** — scanned PDFs (image-based) will produce poor text. Use OCR pre-processing for scanned documents.
- **Language** — optimized for English. Non-English policy documents will work but retrieval quality may be lower.
- **Evaluation set** — 20 Q&A pairs covers the sample policies. Expand with your own documents before production.

---

## Human-in-the-Loop Statement

> **Important:** This system produces AI-generated recommendations based on retrieved policy documents. All answers should be verified with HR or the relevant department before making decisions that affect employees or individuals. This tool does not constitute legal, HR, or compliance advice. Model outputs are recommendations, not automated decisions.

---

## Contact

**SNS Square** | Databricks Bronze Partner
- GitHub Issues: [github.com/sns-square/llm-powered-policy-qa-assistant/issues](https://github.com/sns-square/llm-powered-policy-qa-assistant/issues)
- Email: accelerators@snssquare.com
