# Databricks notebook source
# MAGIC %md
# MAGIC # 🚀 LLM-Powered Policy Q&A Assistant — RUNME
# MAGIC
# MAGIC **Accelerator Version:** 1.0.0 | **Author:** SNS Square | **Runtime:** DBR 14.3 LTS ML
# MAGIC
# MAGIC ## Single entry point — runs the complete pipeline end-to-end
# MAGIC
# MAGIC ```
# MAGIC PDF/DOCX/TXT files
# MAGIC       ↓  (Auto Loader)
# MAGIC [Bronze] raw_documents        ← Notebook 01
# MAGIC       ↓  (chunk + clean)
# MAGIC [Silver] document_chunks      ← Notebook 02
# MAGIC       ↓  (BGE-Large embeddings)
# MAGIC Vector Search Index           ← Notebook 03
# MAGIC       ↑
# MAGIC User Query → embed → search → top-5 chunks
# MAGIC       ↓
# MAGIC Llama 3.1 (Foundation Models API)
# MAGIC + LangChain RAG chain + MLflow traces
# MAGIC       ↓                       ← Notebook 04
# MAGIC Cited Answer [Source: doc, Page X]
# MAGIC       ↓
# MAGIC Gradio UI (Databricks Apps)   ← Notebook 05
# MAGIC       ↓
# MAGIC [Gold] qa_interactions + qa_feedback
# MAGIC ```
# MAGIC
# MAGIC ## Prerequisites
# MAGIC | Requirement | Notes |
# MAGIC |---|---|
# MAGIC | Databricks Runtime | 14.3 LTS ML or later |
# MAGIC | Unity Catalog | Enabled on workspace |
# MAGIC | Mosaic AI Foundation Models API | Enabled (free trial or pay-per-token) |
# MAGIC | Mosaic AI Vector Search | 1 endpoint available (free tier quota) |
# MAGIC | Databricks Apps | Enabled on workspace |
# MAGIC | Secrets scope | `policy_qa_scope` with key `db_token` |
# MAGIC
# MAGIC ## Estimated cost to run once
# MAGIC - Embeddings (BGE-Large, 50 docs): **< $0.10**
# MAGIC - LLM queries during dev (Llama 3.1 8B, 200 queries): **< $0.05**
# MAGIC - Final demo (Llama 3.1 70B, 1 session): **< $0.50**
# MAGIC - **Total: < $2** (covered by free trial credits)

# COMMAND ----------

# DBTITLE 1,Configure Pipeline Parameters
# ── Edit these values for your workspace ──────────────────────────────────────
dbutils.widgets.text("catalog_name",  "dev_policy_qa",      "Unity Catalog name")
dbutils.widgets.text("schema_name",   "policy_assistant",   "Schema name")
dbutils.widgets.text("environment",   "dev",                "Environment (dev/staging/prod)")
dbutils.widgets.text("llm_model",     "databricks-meta-llama-3-1-8b-instruct", "LLM model")
dbutils.widgets.text("use_fm_api",    "true",               "Use Foundation Models API (true/false)")

catalog_name = dbutils.widgets.get("catalog_name")
schema_name  = dbutils.widgets.get("schema_name")
environment  = dbutils.widgets.get("environment")
llm_model    = dbutils.widgets.get("llm_model")
use_fm_api   = dbutils.widgets.get("use_fm_api").lower() == "true"

print("=" * 60)
print("🚀 LLM-Powered Policy Q&A Assistant")
print("=" * 60)
print(f"   Catalog     : {catalog_name}")
print(f"   Schema      : {schema_name}")
print(f"   Environment : {environment}")
print(f"   LLM Model   : {llm_model}")
print(f"   FM API      : {use_fm_api}")
print("=" * 60)

# COMMAND ----------

# MAGIC %md ## Step 1 — Document Ingestion (Bronze)

# COMMAND ----------

# DBTITLE 1,Run Notebook 01 — Ingest Documents
result_01 = dbutils.notebook.run(
    "./notebooks/01_ingest_documents",
    timeout_seconds=1800,
    arguments={
        "catalog_name": catalog_name,
        "schema_name":  schema_name,
    }
)
print(f"✅ Notebook 01 complete: {result_01}")

# COMMAND ----------

# MAGIC %md ## Step 2 — Chunking & Processing (Silver)

# COMMAND ----------

# DBTITLE 1,Run Notebook 02 — Chunk and Process
result_02 = dbutils.notebook.run(
    "./notebooks/02_chunk_and_process",
    timeout_seconds=1800,
    arguments={
        "catalog_name": catalog_name,
        "schema_name":  schema_name,
    }
)
print(f"✅ Notebook 02 complete: {result_02}")

# COMMAND ----------

# MAGIC %md ## Step 3 — Embeddings & Vector Search Index

# COMMAND ----------

# DBTITLE 1,Run Notebook 03 — Embed and Index
result_03 = dbutils.notebook.run(
    "./notebooks/03_embed_and_index",
    timeout_seconds=3600,
    arguments={
        "catalog_name": catalog_name,
        "schema_name":  schema_name,
        "use_fm_api":   str(use_fm_api).lower(),
    }
)
print(f"✅ Notebook 03 complete: {result_03}")

# COMMAND ----------

# MAGIC %md ## Step 4 — RAG Chain + MLflow

# COMMAND ----------

# DBTITLE 1,Run Notebook 04 — Build RAG Chain
result_04 = dbutils.notebook.run(
    "./notebooks/04_build_rag_chain",
    timeout_seconds=3600,
    arguments={
        "catalog_name": catalog_name,
        "schema_name":  schema_name,
        "llm_model":    llm_model,
        "use_fm_api":   str(use_fm_api).lower(),
    }
)
print(f"✅ Notebook 04 complete: {result_04}")

# COMMAND ----------

# MAGIC %md ## Step 5 — Model Serving + Gradio UI

# COMMAND ----------

# DBTITLE 1,Run Notebook 05 — Serve and Demo
result_05 = dbutils.notebook.run(
    "./notebooks/05_serve_and_demo",
    timeout_seconds=1800,
    arguments={
        "catalog_name": catalog_name,
        "schema_name":  schema_name,
        "llm_model":    llm_model,
        "use_fm_api":   str(use_fm_api).lower(),
    }
)
print(f"✅ Notebook 05 complete: {result_05}")

# COMMAND ----------

# MAGIC %md ## ✅ Pipeline Complete

# COMMAND ----------

# DBTITLE 1,Pipeline Summary
print("=" * 60)
print("🎉 LLM-Powered Policy Q&A Assistant — PIPELINE COMPLETE")
print("=" * 60)
print(f"   01 Ingest Documents  : {result_01[:60]}")
print(f"   02 Chunk & Process   : {result_02[:60]}")
print(f"   03 Embed & Index     : {result_03[:60]}")
print(f"   04 Build RAG Chain   : {result_04[:60]}")
print(f"   05 Serve & Demo      : {result_05[:60]}")
print()
print(f"📊 Unity Catalog  : {catalog_name}.{schema_name}")
print(f"🔍 Vector Search  : policy_qa_vs_endpoint / policy_chunks_index")
print(f"🤖 Serving        : policy_qa_endpoint")
print(f"🌐 Gradio UI      : Check Databricks Apps for public URL")
print()
print("📈 MLflow Experiment: /Shared/policy_qa_assistant")
print("   → View traces, evaluation metrics, and model versions")
