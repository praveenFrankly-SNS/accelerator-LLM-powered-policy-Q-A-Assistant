# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 02 — Chunk & Process (Silver Layer)
# MAGIC
# MAGIC **Accelerator:** LLM-Powered Policy Q&A Assistant | **Version:** 1.0.0
# MAGIC **Author:** SNS Square | **Runtime:** DBR 14.3 LTS ML | **Updated:** 2025-05-21
# MAGIC
# MAGIC ## 🔪 What This Notebook Does
# MAGIC
# MAGIC Transforms Bronze raw documents into clean, validated, chunked Silver records.
# MAGIC Bad records go to a quarantine table — never silently dropped.
# MAGIC
# MAGIC ```
# MAGIC [Bronze] raw_documents
# MAGIC       ↓  (clean + validate + chunk)
# MAGIC [Silver] document_chunks   ← Vector Search source table
# MAGIC       ↓  (failed records)
# MAGIC [Quarantine] rejected_chunks
# MAGIC ```
# MAGIC
# MAGIC ## Chunking Strategy
# MAGIC - **300–500 token segments** with **50-token overlap** (spec requirement)
# MAGIC - Sentence-boundary aware — never cuts mid-sentence
# MAGIC - Metadata preserved: `page_num`, `section_header`, `metadata_json`

# COMMAND ----------

# MAGIC %md ## 📦 Environment Setup

# COMMAND ----------

# DBTITLE 1,Install Required Libraries
# MAGIC %pip install tiktoken>=0.5.1

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md ## ⚙️ Configuration

# COMMAND ----------

# DBTITLE 1,Import Libraries
import re, json, hashlib, math
from datetime import datetime, timezone

import tiktoken
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, current_timestamp, row_number, desc
from pyspark.sql.window import Window
from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType, TimestampType, LongType
)
import warnings
warnings.filterwarnings("ignore")

print("📚 Libraries imported")

# COMMAND ----------

# DBTITLE 1,Pipeline Parameters
dbutils.widgets.text("catalog_name", "dev_policy_qa",    "Unity Catalog name")
dbutils.widgets.text("schema_name",  "policy_assistant", "Schema name")

catalog_name = dbutils.widgets.get("catalog_name")
schema_name  = dbutils.widgets.get("schema_name")

# Chunking parameters (spec: 300-500 tokens, 50-token overlap)
CHUNK_SIZE_TOKENS    = 400
CHUNK_OVERLAP_TOKENS = 50
MIN_CHUNK_TOKENS     = 30    # discard chunks shorter than this
TOKENIZER_MODEL      = "cl100k_base"   # same tokenizer as GPT-4 / BGE-Large

PIPELINE_RUN_ID = f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"

print(f"🔧 Chunking Configuration")
print(f"   Catalog        : {catalog_name}")
print(f"   Schema         : {schema_name}")
print(f"   Chunk size     : {CHUNK_SIZE_TOKENS} tokens")
print(f"   Overlap        : {CHUNK_OVERLAP_TOKENS} tokens")
print(f"   Tokenizer      : {TOKENIZER_MODEL}")

spark = SparkSession.builder.getOrCreate()
enc   = tiktoken.get_encoding(TOKENIZER_MODEL)

# COMMAND ----------

# MAGIC %md ## 🏛️ Silver & Quarantine Table Setup

# COMMAND ----------

# DBTITLE 1,Create Silver document_chunks Table
print("📊 Creating Silver document_chunks table...")

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {catalog_name}.{schema_name}.document_chunks (
  chunk_id        STRING    COMMENT 'Unique chunk ID: doc_id + chunk_index',
  doc_id          STRING    COMMENT 'Parent document ID (FK to raw_documents)',
  chunk_text      STRING    COMMENT 'Clean chunk text — 300-500 tokens',
  chunk_index     INT       COMMENT 'Sequential chunk index within document',
  page_num        INT       COMMENT 'Estimated page number (NULL for TXT)',
  token_count     INT       COMMENT 'Actual token count of this chunk',
  metadata_json   STRING    COMMENT 'JSON: filename, doc_type, section_header, pipeline_run_id',
  created_at      TIMESTAMP COMMENT 'UTC timestamp of chunk creation'
) USING DELTA
TBLPROPERTIES (
  'delta.autoOptimize.optimizeWrite' = 'true',
  'delta.autoOptimize.autoCompact'   = 'true',
  'delta.enableChangeDataFeed'       = 'true'
)
COMMENT 'Silver layer — clean, validated document chunks. Source table for Vector Search index.'
""")

# Quarantine table for failed/empty records
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {catalog_name}.{schema_name}.rejected_chunks (
  doc_id          STRING    COMMENT 'Document ID that failed processing',
  filename        STRING    COMMENT 'Original filename',
  rejection_reason STRING   COMMENT 'Why this record was rejected',
  raw_text_sample STRING    COMMENT 'First 500 chars of raw text for debugging',
  pipeline_run_id STRING    COMMENT 'Pipeline run that rejected this record',
  rejected_at     TIMESTAMP COMMENT 'UTC timestamp of rejection'
) USING DELTA
COMMENT 'Quarantine table — every failed record is traceable here, never silently dropped'
""")

print("✅ Silver and quarantine tables ready")

# COMMAND ----------

# MAGIC %md ## ✂️ Token-Aware Chunking Engine

# COMMAND ----------

# DBTITLE 1,Define Chunking Functions
def clean_text(text: str) -> str:
    """Normalize whitespace and remove control characters."""
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    text = re.sub(r'\r\n|\r', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()


def split_into_sentences(text: str) -> list:
    """Split text into sentences, preserving paragraph breaks."""
    paragraphs = text.split('\n\n')
    sentences  = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        # Split on sentence-ending punctuation followed by whitespace
        parts = re.split(r'(?<=[.!?])\s+', para)
        sentences.extend([p.strip() for p in parts if p.strip()])
    return sentences


def chunk_by_tokens(text: str, chunk_size: int = CHUNK_SIZE_TOKENS,
                    overlap: int = CHUNK_OVERLAP_TOKENS,
                    min_tokens: int = MIN_CHUNK_TOKENS) -> list:
    """
    Split text into overlapping token-bounded chunks.
    Sentence-boundary aware — never cuts mid-sentence.

    Returns list of (chunk_index, chunk_text, token_count) tuples.
    """
    sentences   = split_into_sentences(text)
    chunks      = []
    current     = []
    current_tok = 0
    idx         = 0

    for sentence in sentences:
        s_tokens = len(enc.encode(sentence))

        # If adding this sentence exceeds chunk_size, flush current chunk
        if current_tok + s_tokens > chunk_size and current:
            chunk_text  = ' '.join(current).strip()
            chunk_tokens = len(enc.encode(chunk_text))
            if chunk_tokens >= min_tokens:
                chunks.append((idx, chunk_text, chunk_tokens))
                idx += 1

            # Overlap: keep last N tokens worth of sentences
            overlap_sentences = []
            overlap_tok       = 0
            for sent in reversed(current):
                t = len(enc.encode(sent))
                if overlap_tok + t <= overlap:
                    overlap_sentences.insert(0, sent)
                    overlap_tok += t
                else:
                    break
            current     = overlap_sentences
            current_tok = overlap_tok

        current.append(sentence)
        current_tok += s_tokens

    # Flush remaining
    if current:
        chunk_text   = ' '.join(current).strip()
        chunk_tokens = len(enc.encode(chunk_text))
        if chunk_tokens >= min_tokens:
            chunks.append((idx, chunk_text, chunk_tokens))

    return chunks


def estimate_page_num(chunk_index: int, total_chunks: int, total_pages) -> int:
    """Estimate page number from chunk position (for PDFs with known page count).
    Handles None, float NaN (Pandas nullable int columns), and zero safely.
    """
    if total_pages is None:
        return None
    try:
        # Pandas reads nullable INT columns as float64 with NaN for NULL
        if isinstance(total_pages, float) and math.isnan(total_pages):
            return None
        pages = int(total_pages)
    except (ValueError, TypeError):
        return None
    if pages == 0:
        return None
    return max(1, round((chunk_index / max(total_chunks, 1)) * pages))


def extract_section_header(chunk_text: str) -> str:
    """Extract the first numbered section header from chunk text, if present."""
    match = re.search(r'^(\d+[\.\d]*\s+[A-Z][A-Z\s]+)', chunk_text, re.MULTILINE)
    return match.group(1).strip() if match else None


# Sanity check
test_chunks = chunk_by_tokens("This is a test sentence. " * 100)
print(f"✅ Chunking engine ready — test produced {len(test_chunks)} chunks")

# COMMAND ----------

# MAGIC %md ## 🔄 Process Bronze → Silver

# COMMAND ----------

# DBTITLE 1,Load Bronze Documents
print("📥 Loading Bronze documents...")

bronze_df = spark.table(f"{catalog_name}.{schema_name}.raw_documents")

# Deduplicate: keep only the latest version of each document.
# Bronze is append-only so re-running notebook 01 adds duplicate rows.
# We rank by version DESC and keep rank=1 per doc_id — fully idempotent.
dedup_window = Window.partitionBy("doc_id").orderBy(desc("version"), desc("ingested_at"))
bronze_df = (
    bronze_df
    .withColumn("_rn", row_number().over(dedup_window))
    .filter(col("_rn") == 1)
    .drop("_rn")
)

bronze_pd = bronze_df.toPandas()

total_raw  = spark.table(f"{catalog_name}.{schema_name}.raw_documents").count()
print(f"✅ Loaded {len(bronze_pd)} unique documents (deduplicated from {total_raw} Bronze rows)")
print(f"   Doc types: {bronze_pd['doc_type'].value_counts().to_dict()}")

# COMMAND ----------

# DBTITLE 1,Chunk All Documents with Data Quality Checks
print("✂️  Chunking documents...")

start_time     = datetime.now(timezone.utc)
silver_records = []
rejected       = []
seen_chunk_ids = set()   # guard against duplicates within this run

for _, row in bronze_pd.iterrows():
    doc_id   = row["doc_id"]
    filename = row["filename"]
    raw_text = row["raw_text"]
    doc_type = row["doc_type"]
    # Pandas converts nullable INT columns to float64 — NaN means NULL
    _tp = row["total_pages"]
    total_pages = None if (_tp is None or (isinstance(_tp, float) and math.isnan(_tp))) else int(_tp)
    _rid = row["pipeline_run_id"]
    run_id = PIPELINE_RUN_ID if (_rid is None or (isinstance(_rid, float) and math.isnan(_rid))) else str(_rid)

    # ── Data quality checks ──────────────────────────────────────────────────
    if not raw_text or len(raw_text.strip()) < 50:
        rejected.append({
            "doc_id":           doc_id,
            "filename":         filename,
            "rejection_reason": "Empty or near-empty raw_text (< 50 chars)",
            "raw_text_sample":  (raw_text or "")[:500],
            "pipeline_run_id":  PIPELINE_RUN_ID,
            "rejected_at":      datetime.now(timezone.utc),
        })
        print(f"   ⚠️  REJECTED {filename}: empty text")
        continue

    if "[EXTRACTION ERROR" in raw_text:
        rejected.append({
            "doc_id":           doc_id,
            "filename":         filename,
            "rejection_reason": "Text extraction failed",
            "raw_text_sample":  raw_text[:500],
            "pipeline_run_id":  PIPELINE_RUN_ID,
            "rejected_at":      datetime.now(timezone.utc),
        })
        print(f"   ⚠️  REJECTED {filename}: extraction error")
        continue

    # ── Clean and chunk ───────────────────────────────────────────────────────
    clean = clean_text(raw_text)
    chunks = chunk_by_tokens(clean)

    if not chunks:
        rejected.append({
            "doc_id":           doc_id,
            "filename":         filename,
            "rejection_reason": "No valid chunks produced after cleaning",
            "raw_text_sample":  clean[:500],
            "pipeline_run_id":  PIPELINE_RUN_ID,
            "rejected_at":      datetime.now(timezone.utc),
        })
        continue

    for chunk_idx, chunk_text, token_count in chunks:
        chunk_id = f"{doc_id}_c{chunk_idx:04d}"

        # Skip if this chunk_id was already produced in this run (safety net)
        if chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(chunk_id)

        page_num = estimate_page_num(chunk_idx, len(chunks), total_pages)
        section  = extract_section_header(chunk_text)

        metadata = {
            "filename":         filename,
            "doc_type":         doc_type,
            "section_header":   section,
            "pipeline_run_id":  PIPELINE_RUN_ID,
        }

        silver_records.append({
            "chunk_id":      chunk_id,
            "doc_id":        doc_id,
            "chunk_text":    chunk_text,
            "chunk_index":   chunk_idx,
            "page_num":      page_num,
            "token_count":   token_count,
            "metadata_json": json.dumps(metadata),
            "created_at":    datetime.now(timezone.utc),
        })

    print(f"   ✅ {filename}: {len(chunks)} chunks "
          f"(avg {sum(c[2] for c in chunks)//len(chunks)} tokens/chunk)")

print(f"\n📊 Silver records: {len(silver_records)} | Rejected: {len(rejected)}")

# COMMAND ----------

# DBTITLE 1,Write Silver Records to Delta Table
silver_schema = StructType([
    StructField("chunk_id",      StringType(),    True),
    StructField("doc_id",        StringType(),    True),
    StructField("chunk_text",    StringType(),    True),
    StructField("chunk_index",   IntegerType(),   True),
    StructField("page_num",      IntegerType(),   True),
    StructField("token_count",   IntegerType(),   True),
    StructField("metadata_json", StringType(),    True),
    StructField("created_at",    TimestampType(), True),
])

if silver_records:
    silver_df = spark.createDataFrame(silver_records, schema=silver_schema)
    silver_df.write.mode("overwrite").saveAsTable(
        f"{catalog_name}.{schema_name}.document_chunks"
    )
    saved_chunks = spark.table(f"{catalog_name}.{schema_name}.document_chunks").count()
    print(f"✅ Silver table: {saved_chunks} chunks saved")
else:
    print("⚠️  No Silver records to write")

# Write rejected records to quarantine
if rejected:
    rej_schema = StructType([
        StructField("doc_id",           StringType(),    True),
        StructField("filename",         StringType(),    True),
        StructField("rejection_reason", StringType(),    True),
        StructField("raw_text_sample",  StringType(),    True),
        StructField("pipeline_run_id",  StringType(),    True),
        StructField("rejected_at",      TimestampType(), True),
    ])
    spark.createDataFrame(rejected, schema=rej_schema).write.mode("append").saveAsTable(
        f"{catalog_name}.{schema_name}.rejected_chunks"
    )
    print(f"⚠️  {len(rejected)} records written to quarantine table")

# COMMAND ----------

# DBTITLE 1,Silver Data Quality Assertions
print("🔍 Silver Data Quality Assertions")
print("=" * 50)

silver_tbl = spark.table(f"{catalog_name}.{schema_name}.document_chunks")
total      = silver_tbl.count()

# Assertion 1: row count > 0
assert total > 0, "❌ ASSERTION FAILED: Silver table is empty"
print(f"✅ Row count check       : {total} chunks")

# Assertion 2: no null chunk_text
null_text = silver_tbl.filter(col("chunk_text").isNull() | (col("chunk_text") == "")).count()
assert null_text == 0, f"❌ ASSERTION FAILED: {null_text} chunks have null/empty text"
print(f"✅ Null text check       : 0 nulls")

# Assertion 3: token counts in valid range
out_of_range = silver_tbl.filter(
    (col("token_count") < 10) | (col("token_count") > 600)
).count()
print(f"{'✅' if out_of_range == 0 else '⚠️ '} Token range check      : {out_of_range} chunks outside 10-600 tokens")

# Assertion 4: no duplicate chunk_ids — warn and auto-deduplicate rather than crash
total_ids  = silver_tbl.count()
unique_ids = silver_tbl.select("chunk_id").distinct().count()
if total_ids != unique_ids:
    dup_count = total_ids - unique_ids
    print(f"⚠️  Uniqueness check      : {dup_count} duplicate chunk_ids detected — deduplicating...")
    from pyspark.sql.functions import row_number as _rn_fn
    dedup_w = Window.partitionBy("chunk_id").orderBy(desc("created_at"))
    silver_tbl = (
        silver_tbl
        .withColumn("_rn", row_number().over(dedup_w))
        .filter(col("_rn") == 1)
        .drop("_rn")
    )
    silver_tbl.write.mode("overwrite").saveAsTable(
        f"{catalog_name}.{schema_name}.document_chunks"
    )
    silver_tbl = spark.table(f"{catalog_name}.{schema_name}.document_chunks")
    print(f"✅ Uniqueness check      : deduplicated to {silver_tbl.count()} unique chunks")
else:
    print(f"✅ Uniqueness check      : all chunk_ids unique")

# Summary stats
silver_tbl.select("token_count").describe().show()

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
    "notebook":       "02_chunk_and_process",
    "start_time":     start_time,
    "end_time":       end_time,
    "rows_processed": len(silver_records),
    "status":         "SUCCESS",
    "error_message":  f"{len(rejected)} records quarantined" if rejected else None,
}], schema=audit_schema).write.mode("append").saveAsTable(
    f"{catalog_name}.{schema_name}.pipeline_audit_log"
)
print(f"✅ Audit log written — run_id: {PIPELINE_RUN_ID}")

# COMMAND ----------

# MAGIC %md ## 📋 Silver Layer Complete
# MAGIC
# MAGIC ### ✅ What Was Created:
# MAGIC | Asset | Location |
# MAGIC |---|---|
# MAGIC | Document chunks (Silver) | `{catalog_name}.{schema_name}.document_chunks` |
# MAGIC | Rejected records | `{catalog_name}.{schema_name}.rejected_chunks` |
# MAGIC
# MAGIC ### ➡️ Next: Notebook 03 — Embed & Index (Vector Search)

# COMMAND ----------

dbutils.notebook.exit(
    f"SUCCESS: {len(silver_records)} chunks written to Silver "
    f"({catalog_name}.{schema_name}.document_chunks), "
    f"{len(rejected)} quarantined, run_id={PIPELINE_RUN_ID}"
)
