# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 01 — Policy Document Ingestion (Bronze Layer)
# MAGIC
# MAGIC **Accelerator:** LLM-Powered Policy Q&A Assistant | **Version:** 1.0.0
# MAGIC **Author:** SNS Square | **Runtime:** DBR 14.3 LTS ML | **Updated:** 2025-05-21
# MAGIC
# MAGIC ## 🏢 What This Notebook Does
# MAGIC
# MAGIC Ingests raw policy documents (PDF / DOCX / TXT) into the **Bronze layer** of the
# MAGIC Medallion Architecture. No transformations — raw text preserved exactly as-is.
# MAGIC
# MAGIC ```
# MAGIC /Volumes/catalog/schema/raw_policies/   ← Unity Catalog Volume
# MAGIC         ↓  (Auto Loader — incremental, schema-on-read)
# MAGIC [Bronze] catalog.schema.raw_documents   ← Delta Lake, append-only
# MAGIC ```
# MAGIC
# MAGIC ## 📋 Prerequisites
# MAGIC - Unity Catalog enabled
# MAGIC - DBR 14.3 LTS ML or later
# MAGIC - Sample policies in `/data/sample_policies/` (auto-uploaded by this notebook)

# COMMAND ----------

# MAGIC %md ## 📦 Environment Setup

# COMMAND ----------

# DBTITLE 1,Install Required Libraries
# MAGIC %pip install pypdf>=3.17.0 python-docx>=1.1.0

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md ## ⚙️ Configuration

# COMMAND ----------

# DBTITLE 1,Import Libraries
import os, hashlib, json
from datetime import datetime, timezone
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, current_timestamp, lit
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

# Unity Catalog Volume path for raw files
VOLUME_PATH  = f"/Volumes/{catalog_name}/{schema_name}/raw_policies"
PIPELINE_RUN_ID = f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"

print(f"🔧 Ingestion Configuration")
print(f"   Catalog        : {catalog_name}")
print(f"   Schema         : {schema_name}")
print(f"   Volume path    : {VOLUME_PATH}")
print(f"   Pipeline run   : {PIPELINE_RUN_ID}")

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

# MAGIC %md ## 🏛️ Unity Catalog Infrastructure Setup

# COMMAND ----------

# DBTITLE 1,Create Catalog, Schema, and Volume
print("🏗️ Setting up Unity Catalog infrastructure...")

spark.sql(f"CREATE CATALOG  IF NOT EXISTS {catalog_name}")
spark.sql(f"CREATE SCHEMA   IF NOT EXISTS {catalog_name}.{schema_name}")
spark.sql(f"CREATE VOLUME   IF NOT EXISTS {catalog_name}.{schema_name}.raw_policies")

print(f"✅ Catalog  : {catalog_name}")
print(f"✅ Schema   : {schema_name}")
print(f"✅ Volume   : {catalog_name}.{schema_name}.raw_policies")

# COMMAND ----------

# DBTITLE 1,Create Bronze raw_documents Table
print("📊 Creating Bronze raw_documents table...")

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {catalog_name}.{schema_name}.raw_documents (
  doc_id          STRING    COMMENT 'SHA-256 hash of filename — stable unique identifier',
  filename        STRING    COMMENT 'Original filename (e.g. leave_policy_v3.pdf)',
  doc_type        STRING    COMMENT 'File type: PDF, DOCX, or TXT',
  raw_text        STRING    COMMENT 'Full extracted text — no transformations applied',
  total_pages     INT       COMMENT 'Page count (PDFs); NULL for TXT/DOCX',
  source          STRING    COMMENT 'Source path in Unity Catalog Volume',
  pipeline_run_id STRING    COMMENT 'Pipeline run identifier for lineage tracing',
  ingested_at     TIMESTAMP COMMENT 'UTC timestamp of ingestion',
  version         INT       COMMENT 'Incremental version — increments on re-ingest of same file'
) USING DELTA
TBLPROPERTIES (
  'delta.autoOptimize.optimizeWrite' = 'true',
  'delta.autoOptimize.autoCompact'   = 'true',
  'delta.enableChangeDataFeed'       = 'true'
)
COMMENT 'Bronze layer — raw policy documents, append-only, full history preserved'
""")

# Also create the pipeline audit log table
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {catalog_name}.{schema_name}.pipeline_audit_log (
  run_id          STRING    COMMENT 'Pipeline run identifier',
  notebook        STRING    COMMENT 'Notebook name',
  start_time      TIMESTAMP COMMENT 'Run start time (UTC)',
  end_time        TIMESTAMP COMMENT 'Run end time (UTC)',
  rows_processed  LONG      COMMENT 'Number of records processed',
  status          STRING    COMMENT 'SUCCESS or FAILED',
  error_message   STRING    COMMENT 'Error details if status = FAILED'
) USING DELTA
COMMENT 'Operational audit log for all pipeline runs'
""")

print("✅ Bronze table and audit log created")

# COMMAND ----------

# MAGIC %md ## 📄 Sample Policy Documents
# MAGIC
# MAGIC > **For your own policies:** Upload PDF/DOCX/TXT files to the Unity Catalog Volume at
# MAGIC > `/Volumes/{catalog_name}/{schema_name}/raw_policies/` and re-run this notebook.
# MAGIC > The sample documents below are replaced automatically.

# COMMAND ----------

# DBTITLE 1,Write Sample Policy Files to Volume
SAMPLE_POLICIES = {
    "employee_leave_policy_v3.txt": """EMPLOYEE LEAVE POLICY — Version 3.2 | Effective: 2024-01-01

1. ANNUAL LEAVE
Employees accrue paid time off based on tenure:
- 0–2 years: 15 days/year | 2–5 years: 20 days/year | 5+ years: 25 days/year
Maximum carryover: 5 days. Unused days beyond 5 are forfeited on Dec 31.
Leave requests must be submitted 5 business days in advance and approved by manager.

2. SICK LEAVE
All employees receive 10 days paid sick leave per calendar year.
Sick leave does not carry over and is not paid out on termination.
Medical certificate required for absences exceeding 3 consecutive days.

3. MATERNITY / PATERNITY LEAVE
Maternity: 16 weeks paid. Extendable by 4 weeks unpaid on written request.
Paternity: 2 weeks paid, taken within 3 months of birth or adoption.

4. BEREAVEMENT LEAVE
Immediate family (spouse, child, parent, sibling): 5 days paid.
Extended family (grandparent, in-law): 3 days paid.

5. PUBLIC HOLIDAYS
All gazetted public holidays are paid. Working on a public holiday earns double pay
or a compensatory day off at the employee's election.

Contact: hr@company.com | Extension 1234""",

    "it_security_acceptable_use_v2.txt": """IT SECURITY AND ACCEPTABLE USE POLICY — Version 2.1 | Effective: 2024-03-15

1. PASSWORD REQUIREMENTS
Minimum 12 characters. Must include uppercase, lowercase, numbers, and special characters.
Passwords expire every 90 days. MFA is mandatory for all systems.
Never share passwords. Accounts lock after 5 failed attempts.

2. DATA CLASSIFICATION
PUBLIC: No restrictions. INTERNAL: Internal use only.
CONFIDENTIAL: Encrypt in transit; approved storage only.
RESTRICTED: Encrypt at rest and in transit; need-to-know access only.
HR records are classified CONFIDENTIAL.

3. ACCEPTABLE USE
Permitted: business communication, job research, approved development.
Prohibited: illegal content, unauthorized software, cryptocurrency mining,
sharing confidential data on personal devices, bypassing security controls.

4. REMOTE WORK SECURITY
VPN required for all remote access to company systems.
Public Wi-Fi prohibited without VPN. WPA2/WPA3 required on home routers.

5. INCIDENT REPORTING
Report security incidents within 1 hour: security@company.com | +1-800-SEC-HELP

Violations may result in termination and legal prosecution.""",

    "expense_reimbursement_policy_v4.txt": """EXPENSE REIMBURSEMENT POLICY — Version 4.0 | Effective: 2024-01-01

1. GENERAL RULES
Expenses must be business-related, reasonable, and supported by receipts.
Submit within 30 days of incurring the expense via Concur.
Pre-approval required: >$500 (manager), >$2,000 (department head).

2. TRAVEL
Air: Economy for flights <6 hours. Business class requires VP approval for >6 hours.
Book 14 days in advance via company travel portal.
Hotel: Max $200/night domestic, $300/night international.
Mileage: $0.67/mile (2024 IRS rate) for personal vehicle use.

3. MEALS (DURING TRAVEL)
Breakfast $20 | Lunch $30 | Dinner $60 | Daily cap $100.
Receipts required for meals over $25. Alcohol is NOT reimbursable.

4. CLIENT ENTERTAINMENT
Business meals: up to $150/person with manager approval and attendee list.
Events (sports, concerts): up to $300/person with VP approval.

5. NON-REIMBURSABLE
Personal entertainment, alcohol, traffic fines, personal grooming,
family member expenses, first-class upgrades without approval.

Contact: finance@company.com""",

    "data_privacy_protection_policy_v1.txt": """DATA PRIVACY AND PROTECTION POLICY — Version 1.5 | Effective: 2024-05-01

1. SCOPE
Applies to all employees handling personal data. Complies with GDPR, CCPA,
and India DPDP Act 2023.

2. DATA SUBJECT RIGHTS
Access: Respond within 30 days. Rectification: Correct inaccurate data.
Erasure: Delete on request ('right to be forgotten').
Portability: Provide data in machine-readable format.
Contact: privacy@company.com

3. DATA RETENTION
Employee records: 7 years post-employment.
Customer transactions: 5 years post last transaction.
CCTV footage: 30 days unless required for investigation.

4. DATA BREACH RESPONSE
Internal DPO notification: within 1 hour.
Regulatory notification: within 72 hours (if required).
Notify affected individuals without undue delay if high risk.
Steps: Contain → Assess → Notify DPO (dpo@company.com) → Document → Remediate.

5. THIRD-PARTY PROCESSORS
Must sign DPA before receiving data. Must demonstrate SOC 2 or ISO 27001.
Must notify company of breaches within 24 hours.""",

    "remote_work_flexible_working_v2.txt": """REMOTE WORK AND FLEXIBLE WORKING POLICY — Version 2.0 | Effective: 2024-02-01

1. ELIGIBILITY
Available after 3 months of employment with satisfactory performance rating.
Not available for roles requiring physical presence or in first 90 days.

2. ARRANGEMENTS
Hybrid: Up to 3 days remote per week; minimum 2 days in office.
Fully Remote: Requires VP approval; reviewed annually; quarterly in-person meetings.
Flexible Hours: Core hours 10 AM–3 PM local time. Flex outside core with manager approval.
Compressed Week: 4-day/10-hour week on 3-month trial with manager approval.

3. HOME OFFICE REQUIREMENTS
Dedicated quiet workspace. Internet: min 25 Mbps download / 10 Mbps upload.
Company provides: laptop, peripherals, software, VPN.
Employee provides: internet (stipend $50/month provided).

4. AVAILABILITY
Respond to messages within 2 hours during working hours.
Camera on for video calls is encouraged. Update calendar with working hours.
Approved tools only: Teams/Zoom for video, Teams for IM, SharePoint/OneDrive for files.

5. REVOCATION
Remote privileges revoked with 2 weeks notice for performance issues,
attendance problems, or business needs requiring on-site presence.

Contact: hr@company.com | IT remote setup: helpdesk@company.com"""
}

print(f"📁 Writing {len(SAMPLE_POLICIES)} sample policy files to Volume...")

for filename, content in SAMPLE_POLICIES.items():
    file_path = f"{VOLUME_PATH}/{filename}"
    try:
        dbutils.fs.put(file_path, content, overwrite=True)
        print(f"   ✅ {filename}")
    except Exception as e:
        print(f"   ❌ {filename}: {e}")

print(f"\n✅ Sample policies written to {VOLUME_PATH}")

# COMMAND ----------

# MAGIC %md ## 📥 Document Ingestion with Auto Loader

# COMMAND ----------

# DBTITLE 1,Define Text Extraction Functions
def extract_text_from_txt(content: str) -> tuple:
    """Extract text from plain text content."""
    return content.strip(), None

def extract_text_from_file(file_path: str, filename: str) -> tuple:
    """
    Extract raw text from a file in the Unity Catalog Volume.
    Returns (raw_text, total_pages).
    Supports: TXT (direct read), PDF (pypdf), DOCX (python-docx).
    """
    ext = Path(filename).suffix.lower()
    try:
        if ext == ".txt":
            content = dbutils.fs.head(file_path, 1_000_000)
            return content.strip(), None

        elif ext == ".pdf":
            from pypdf import PdfReader
            import io
            # Read bytes via dbutils
            raw_bytes = dbutils.fs.head(file_path, 10_000_000)
            reader    = PdfReader(io.BytesIO(raw_bytes.encode("latin-1")))
            pages     = [p.extract_text() or "" for p in reader.pages]
            return "\n\n".join(pages).strip(), len(pages)

        elif ext in (".docx", ".doc"):
            from docx import Document
            import io
            raw_bytes = dbutils.fs.head(file_path, 10_000_000)
            doc       = Document(io.BytesIO(raw_bytes.encode("latin-1")))
            text      = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
            return text.strip(), None

        else:
            return f"[Unsupported file type: {ext}]", None

    except Exception as e:
        return f"[EXTRACTION ERROR: {str(e)}]", None

print("✅ Text extraction functions defined")

# COMMAND ----------

# DBTITLE 1,Ingest All Files from Volume into Bronze Table
print(f"📥 Scanning {VOLUME_PATH} for policy documents...")

start_time = datetime.now(timezone.utc)
ingested_records = []
errors = []

try:
    files = dbutils.fs.ls(VOLUME_PATH)
except Exception as e:
    files = []
    print(f"⚠️  Could not list volume: {e}")

supported_extensions = {".txt", ".pdf", ".docx", ".doc"}

for file_info in files:
    filename = file_info.name
    ext      = Path(filename).suffix.lower()

    if ext not in supported_extensions:
        continue

    file_path = file_info.path
    doc_id    = hashlib.sha256(filename.encode()).hexdigest()[:16]
    doc_type  = ext.lstrip(".").upper()

    try:
        raw_text, total_pages = extract_text_from_file(file_path, filename)

        # Determine version: count existing records for this doc_id
        existing = spark.sql(f"""
            SELECT COALESCE(MAX(version), 0) AS max_ver
            FROM {catalog_name}.{schema_name}.raw_documents
            WHERE doc_id = '{doc_id}'
        """).collect()[0]["max_ver"]

        ingested_records.append({
            "doc_id":          doc_id,
            "filename":        filename,
            "doc_type":        doc_type,
            "raw_text":        raw_text,
            "total_pages":     total_pages,
            "source":          file_path,
            "pipeline_run_id": PIPELINE_RUN_ID,
            "ingested_at":     datetime.now(timezone.utc),
            "version":         existing + 1,
        })
        print(f"   ✅ {filename} ({doc_type}, {len(raw_text):,} chars)")

    except Exception as e:
        errors.append({"file": filename, "error": str(e)})
        print(f"   ❌ {filename}: {e}")

print(f"\n📊 Ingested: {len(ingested_records)} files | Errors: {len(errors)}")

# COMMAND ----------

# DBTITLE 1,Write Bronze Records to Delta Table
if ingested_records:
    bronze_schema = StructType([
        StructField("doc_id",          StringType(),    True),
        StructField("filename",        StringType(),    True),
        StructField("doc_type",        StringType(),    True),
        StructField("raw_text",        StringType(),    True),
        StructField("total_pages",     IntegerType(),   True),
        StructField("source",          StringType(),    True),
        StructField("pipeline_run_id", StringType(),    True),
        StructField("ingested_at",     TimestampType(), True),
        StructField("version",         IntegerType(),   True),
    ])

    bronze_df = spark.createDataFrame(ingested_records, schema=bronze_schema)
    bronze_df.write.mode("append").saveAsTable(
        f"{catalog_name}.{schema_name}.raw_documents"
    )

    saved = spark.table(f"{catalog_name}.{schema_name}.raw_documents").count()
    print(f"✅ Bronze table now has {saved} total records")
else:
    print("⚠️  No records to write — check Volume path and file formats")

# COMMAND ----------

# DBTITLE 1,Data Quality Validation
print("🔍 Bronze Data Quality Report")
print("=" * 50)

bronze_tbl = spark.table(f"{catalog_name}.{schema_name}.raw_documents")
total_docs = bronze_tbl.count()

print(f"📄 Total documents in Bronze : {total_docs}")
print(f"📁 This run ingested          : {len(ingested_records)}")

if total_docs > 0:
    bronze_tbl.groupBy("doc_type").count().show(truncate=False)
    null_text = bronze_tbl.filter(col("raw_text").isNull() | (col("raw_text") == "")).count()
    print(f"⚠️  Documents with empty text : {null_text}")
    if null_text > 0:
        print("   → These will be quarantined in Notebook 02")

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

audit_record = [{
    "run_id":         PIPELINE_RUN_ID,
    "notebook":       "01_ingest_documents",
    "start_time":     start_time,
    "end_time":       end_time,
    "rows_processed": len(ingested_records),
    "status":         "SUCCESS" if not errors else "PARTIAL",
    "error_message":  json.dumps(errors) if errors else None,
}]

spark.createDataFrame(audit_record, schema=audit_schema).write.mode("append").saveAsTable(
    f"{catalog_name}.{schema_name}.pipeline_audit_log"
)
print(f"✅ Audit log written — run_id: {PIPELINE_RUN_ID}")

# COMMAND ----------

# MAGIC %md ## 📋 Bronze Layer Complete
# MAGIC
# MAGIC ### ✅ What Was Created:
# MAGIC | Asset | Location |
# MAGIC |---|---|
# MAGIC | Raw documents (Bronze) | `{catalog_name}.{schema_name}.raw_documents` |
# MAGIC | Pipeline audit log | `{catalog_name}.{schema_name}.pipeline_audit_log` |
# MAGIC | Raw files (Volume) | `/Volumes/{catalog_name}/{schema_name}/raw_policies/` |
# MAGIC
# MAGIC ### ➡️ Next: Notebook 02 — Chunk & Process (Silver)

# COMMAND ----------

dbutils.notebook.exit(
    f"SUCCESS: {len(ingested_records)} documents ingested into Bronze "
    f"({catalog_name}.{schema_name}.raw_documents), run_id={PIPELINE_RUN_ID}"
)
