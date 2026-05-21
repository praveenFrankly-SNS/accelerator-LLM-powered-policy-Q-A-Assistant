# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 03 — Embed & Index (Vector Search)
# MAGIC
# MAGIC **Accelerator:** LLM-Powered Policy Q&A Assistant | **Version:** 1.0.0
# MAGIC **Author:** SNS Square | **Runtime:** DBR 14.3 LTS ML | **Updated:** 2026-05-21
# MAGIC
# MAGIC ## 🔍 What This Notebook Does
# MAGIC
# MAGIC Generates embeddings for every Silver chunk and creates a Mosaic AI Vector Search index
# MAGIC for semantic similarity retrieval.
# MAGIC
# MAGIC ```
# MAGIC [Silver] document_chunks
# MAGIC       ↓  (BGE-Large-EN-v1.5 via Foundation Models API)
# MAGIC Vector Search Index  ←  Delta Sync (triggered mode)
# MAGIC       ↑
# MAGIC User Query → embed → cosine similarity → top-5 chunks
# MAGIC ```
# MAGIC
# MAGIC ## Prerequisites
# MAGIC - Notebook 02 completed successfully (Silver table populated)
# MAGIC - Mosaic AI Vector Search enabled on workspace (1 endpoint = free tier)
# MAGIC - Foundation Models API enabled (or `use_fm_api=false` for local fallback)

# COMMAND ----------

# MAGIC %md ## 📦 Environment Setup

# COMMAND ----------

# DBTITLE 1,Install Required Libraries
# MAGIC %pip install databricks-vectorsearch>=0.22 mlflow>=2.14.0 tiktoken>=0.5.1

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md ## ⚙️ Configuration

# COMMAND ----------

# DBTITLE 1,Import Libraries
import json, time
from datetime import datetime, timezone

import mlflow
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count
from pyspark.sql.types import StructType, StructField, StringType, LongType, TimestampType
import warnings
warnings.filterwarnings("ignore")

print("📚 Libraries imported")

# COMMAND ----------

# DBTITLE 1,Pipeline Parameters
dbutils.widgets.text("catalog_name",       "dev_policy_qa",    "Unity Catalog name")
dbutils.widgets.text("schema_name",        "policy_assistant", "Schema name")
dbutils.widgets.text("vs_endpoint_name",   "policy_qa_vs_endpoint", "Vector Search endpoint name")
dbutils.widgets.text("use_fm_api",         "true",             "Use Foundation Models API (true/false)")

catalog_name     = dbutils.widgets.get("catalog_name")
schema_name      = dbutils.widgets.get("schema_name")
VS_ENDPOINT_NAME = dbutils.widgets.get("vs_endpoint_name")
use_fm_api       = dbutils.widgets.get("use_fm_api").lower() == "true"

VS_INDEX_NAME    = f"{catalog_name}.{schema_name}.policy_chunks_index"
SOURCE_TABLE     = f"{catalog_name}.{schema_name}.document_chunks"
EMBEDDING_MODEL  = "databricks-bge-large-en"   # Foundation Models API endpoint name
PIPELINE_RUN_ID  = f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"

print(f"🔧 Vector Search Configuration")
print(f"   Catalog          : {catalog_name}")
print(f"   Schema           : {schema_name}")
print(f"   Source table     : {SOURCE_TABLE}")
print(f"   VS endpoint      : {VS_ENDPOINT_NAME}")
print(f"   VS index         : {VS_INDEX_NAME}")
print(f"   Embedding model  : {EMBEDDING_MODEL}")
print(f"   Use FM API       : {use_fm_api}")

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

# MAGIC %md ## ✅ Pre-flight: Validate Silver Table

# COMMAND ----------

# DBTITLE 1,Validate Silver Table Has Data
print("🔍 Validating Silver table...")

silver_df    = spark.table(SOURCE_TABLE)
total_chunks = silver_df.count()

assert total_chunks > 0, (
    f"❌ Silver table {SOURCE_TABLE} is empty. "
    "Run Notebook 02 first."
)

null_text = silver_df.filter(
    col("chunk_text").isNull() | (col("chunk_text") == "")
).count()
assert null_text == 0, f"❌ {null_text} chunks have null/empty text in Silver table"

print(f"✅ Silver table validated: {total_chunks} chunks, 0 null texts")
silver_df.groupBy("doc_id").count().show(truncate=False)

# COMMAND ----------

# MAGIC %md ## 🏗️ Vector Search Infrastructure

# COMMAND ----------

# DBTITLE 1,Enable Change Data Feed on Source Table (Required for Delta Sync)
print("⚙️  Enabling Change Data Feed on document_chunks (required for VS Delta Sync)...")

spark.sql(f"""
  ALTER TABLE {SOURCE_TABLE}
  SET TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')
""")
print("✅ Change Data Feed enabled")

# COMMAND ----------

# DBTITLE 1,Create or Reuse Vector Search Endpoint
from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient(disable_notice=True)

def wait_for_endpoint(client, endpoint_name: str, timeout_minutes: int = 20):
    """Poll until endpoint is ONLINE or timeout."""
    deadline = time.time() + timeout_minutes * 60
    while time.time() < deadline:
        try:
            ep = client.get_endpoint(endpoint_name)
            state = ep.get("endpoint_status", {}).get("state", "UNKNOWN")
            print(f"   Endpoint state: {state}")
            if state == "ONLINE":
                return True
            if state in ("OFFLINE", "FAILED"):
                raise RuntimeError(f"Endpoint entered state: {state}")
        except Exception as e:
            if "does not exist" not in str(e).lower():
                raise
        time.sleep(30)
    raise TimeoutError(f"Endpoint {endpoint_name} did not become ONLINE within {timeout_minutes} min")


print(f"🔍 Checking Vector Search endpoint: {VS_ENDPOINT_NAME}")
try:
    ep = vsc.get_endpoint(VS_ENDPOINT_NAME)
    state = ep.get("endpoint_status", {}).get("state", "UNKNOWN")
    print(f"✅ Endpoint already exists — state: {state}")
    if state != "ONLINE":
        print("   Waiting for endpoint to come online...")
        wait_for_endpoint(vsc, VS_ENDPOINT_NAME)
except Exception as e:
    if "does not exist" in str(e).lower():
        print(f"🏗️  Creating new Vector Search endpoint: {VS_ENDPOINT_NAME}")
        vsc.create_endpoint(
            name=VS_ENDPOINT_NAME,
            endpoint_type="STANDARD"
        )
        wait_for_endpoint(vsc, VS_ENDPOINT_NAME)
        print(f"✅ Endpoint {VS_ENDPOINT_NAME} is ONLINE")
    else:
        raise

# COMMAND ----------

# DBTITLE 1,Create or Sync Vector Search Index
def wait_for_index(client, endpoint_name: str, index_name: str, timeout_minutes: int = 30):
    """Poll until index is ONLINE or timeout."""
    deadline = time.time() + timeout_minutes * 60
    while time.time() < deadline:
        try:
            idx   = client.get_index(endpoint_name, index_name)
            state = idx.describe().get("status", {}).get("ready", False)
            detailed = idx.describe().get("status", {}).get("message", "")
            print(f"   Index ready: {state} | {detailed[:80]}")
            if state:
                return True
        except Exception as e:
            print(f"   Waiting... ({e})")
        time.sleep(30)
    raise TimeoutError(f"Index {index_name} did not become ready within {timeout_minutes} min")


print(f"🔍 Checking Vector Search index: {VS_INDEX_NAME}")
try:
    idx = vsc.get_index(VS_ENDPOINT_NAME, VS_INDEX_NAME)
    print(f"✅ Index already exists — triggering sync...")
    idx.sync()
    wait_for_index(vsc, VS_ENDPOINT_NAME, VS_INDEX_NAME)
    print("✅ Index sync complete")

except Exception as e:
    if "does not exist" in str(e).lower():
        print(f"🏗️  Creating Delta Sync index: {VS_INDEX_NAME}")
        vsc.create_delta_sync_index(
            endpoint_name      = VS_ENDPOINT_NAME,
            index_name         = VS_INDEX_NAME,
            source_table_name  = SOURCE_TABLE,
            pipeline_type      = "TRIGGERED",          # cost-efficient for demo
            primary_key        = "chunk_id",
            embedding_source_column = "chunk_text",
            embedding_model_endpoint_name = EMBEDDING_MODEL,
        )
        print("⏳ Waiting for index to become ready (first sync may take 5–10 min)...")
        wait_for_index(vsc, VS_ENDPOINT_NAME, VS_INDEX_NAME)
        print(f"✅ Index {VS_INDEX_NAME} is ready")
    else:
        raise

# COMMAND ----------

# MAGIC %md ## 🧪 Retrieval Smoke Test

# COMMAND ----------

# DBTITLE 1,Test Similarity Search
print("🧪 Running retrieval smoke test...")

test_queries = [
    "How many days of annual leave do employees get?",
    "What is the password expiry policy?",
    "Can I get reimbursed for alcohol during business travel?",
]

idx = vsc.get_index(VS_ENDPOINT_NAME, VS_INDEX_NAME)

for query in test_queries:
    results = idx.similarity_search(
        query_text   = query,
        columns      = ["chunk_id", "chunk_text", "metadata_json"],
        num_results  = 3,
    )
    hits = results.get("result", {}).get("data_array", [])
    print(f"\n❓ Query: {query}")
    for i, hit in enumerate(hits[:2]):
        chunk_text = hit[1][:120].replace("\n", " ")
        print(f"   [{i+1}] {chunk_text}...")

print("\n✅ Retrieval smoke test passed")

# COMMAND ----------

# MAGIC %md ## 📊 Log Index Metadata to MLflow

# COMMAND ----------

# DBTITLE 1,Log Vector Search Metadata to MLflow
mlflow.set_experiment("/Shared/policy_qa_assistant")

with mlflow.start_run(run_name="03_vector_search_index") as run:
    mlflow.log_params({
        "vs_endpoint_name":   VS_ENDPOINT_NAME,
        "vs_index_name":      VS_INDEX_NAME,
        "source_table":       SOURCE_TABLE,
        "embedding_model":    EMBEDDING_MODEL,
        "pipeline_type":      "TRIGGERED",
        "total_chunks_indexed": total_chunks,
        "catalog_name":       catalog_name,
        "schema_name":        schema_name,
    })
    mlflow.log_metrics({
        "chunks_indexed": total_chunks,
        "smoke_test_queries": len(test_queries),
    })
    mlflow.set_tags({
        "accelerator":        "policy_qa_assistant",
        "notebook":           "03_embed_and_index",
        "pipeline_run_id":    PIPELINE_RUN_ID,
        "source_system":      "unity_catalog_silver",
        "pipeline_version":   "1.0.0",
    })
    run_id = run.info.run_id

print(f"✅ MLflow run logged: {run_id}")

# COMMAND ----------

# DBTITLE 1,Write Pipeline Audit Log
end_time = datetime.now(timezone.utc)
audit_schema = StructType([
    StructField("run_id",         StringType(),    True),
    StructField("notebook",       StringType(),    True),
    StructField("start_time",     TimestampType(), True),
    StructField("end_time",       TimestampType(), True),
    StructField("rows_processed", LongType(),      True),
    StructField("status",         StringType(),    True),
    StructField("error_message",  StringType(),    True),
])

spark.createDataFrame([{
    "run_id":         PIPELINE_RUN_ID,
    "notebook":       "03_embed_and_index",
    "start_time":     datetime.now(timezone.utc),
    "end_time":       end_time,
    "rows_processed": total_chunks,
    "status":         "SUCCESS",
    "error_message":  None,
}], schema=audit_schema).write.mode("append").saveAsTable(
    f"{catalog_name}.{schema_name}.pipeline_audit_log"
)
print(f"✅ Audit log written — run_id: {PIPELINE_RUN_ID}")

# COMMAND ----------

# MAGIC %md ## 📋 Vector Search Layer Complete
# MAGIC
# MAGIC ### ✅ What Was Created:
# MAGIC | Asset | Location |
# MAGIC |---|---|
# MAGIC | Vector Search endpoint | `policy_qa_vs_endpoint` |
# MAGIC | Vector Search index | `{catalog_name}.{schema_name}.policy_chunks_index` |
# MAGIC | Embedding model | `databricks-bge-large-en` (Foundation Models API) |
# MAGIC | MLflow run | `/Shared/policy_qa_assistant` |
# MAGIC
# MAGIC ### ➡️ Next: Notebook 04 — Build RAG Chain + MLflow

# COMMAND ----------

dbutils.notebook.exit(
    f"SUCCESS: {total_chunks} chunks indexed in {VS_INDEX_NAME}, "
    f"endpoint={VS_ENDPOINT_NAME}, run_id={PIPELINE_RUN_ID}"
)
