# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 05 — Model Serving + Gradio UI (Databricks Apps)
# MAGIC
# MAGIC **Accelerator:** LLM-Powered Policy Q&A Assistant | **Version:** 1.0.0
# MAGIC **Author:** SNS Square | **Runtime:** DBR 14.3 LTS ML | **Updated:** 2026-05-21
# MAGIC
# MAGIC ## 🌐 What This Notebook Does
# MAGIC
# MAGIC 1. Deploys the registered RAG chain to a **Mosaic AI Model Serving** serverless endpoint
# MAGIC 2. Launches a **Gradio UI** on **Databricks Apps** — public URL, zero external hosting
# MAGIC 3. Logs every interaction to `Gold: qa_interactions` and feedback to `qa_feedback`
# MAGIC
# MAGIC ```
# MAGIC UC Model Registry (policy_qa_rag_chain)
# MAGIC       ↓  (serverless endpoint — scales to zero)
# MAGIC Model Serving: policy_qa_endpoint
# MAGIC       ↑
# MAGIC Gradio UI (Databricks Apps — public URL)
# MAGIC       ↓
# MAGIC [Gold] qa_interactions + qa_feedback
# MAGIC ```
# MAGIC
# MAGIC ## ⚠️ Cold Start Note
# MAGIC Serverless endpoints have a ~5–10 second cold start on the first query.
# MAGIC The UI displays a warm-up message to set user expectations.

# COMMAND ----------

# MAGIC %md ## 📦 Environment Setup

# COMMAND ----------

# DBTITLE 1,Install Required Libraries
# MAGIC %pip install gradio>=4.26.0 databricks-sdk>=0.24.0 mlflow>=2.14.0 requests>=2.31.0

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md ## ⚙️ Configuration

# COMMAND ----------

# DBTITLE 1,Import Libraries
import json, time, uuid, requests
from datetime import datetime, timezone

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedModelInput,
    ServedModelInputWorkloadSize,
)
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, StringType, BooleanType,
    LongType, DoubleType, TimestampType
)
import warnings
warnings.filterwarnings("ignore")

print("📚 Libraries imported")

# COMMAND ----------

# DBTITLE 1,Pipeline Parameters
dbutils.widgets.text("catalog_name",     "dev_policy_qa",    "Unity Catalog name")
dbutils.widgets.text("schema_name",      "policy_assistant", "Schema name")
dbutils.widgets.text("llm_model",        "databricks-meta-llama-3-1-8b-instruct", "LLM model")
dbutils.widgets.text("use_fm_api",       "true",             "Use Foundation Models API")

catalog_name     = dbutils.widgets.get("catalog_name")
schema_name      = dbutils.widgets.get("schema_name")
LLM_MODEL        = dbutils.widgets.get("llm_model")
use_fm_api       = dbutils.widgets.get("use_fm_api").lower() == "true"

MODEL_NAME           = f"{catalog_name}.{schema_name}.policy_qa_rag_chain"
SERVING_ENDPOINT     = "policy_qa_endpoint"
PIPELINE_RUN_ID      = f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"

# Databricks workspace URL (for REST API calls from Gradio)
WORKSPACE_URL = spark.conf.get("spark.databricks.workspaceUrl", "")
DB_TOKEN      = dbutils.secrets.get(scope="policy_qa_scope", key="db_token")

print(f"🔧 Serving Configuration")
print(f"   Catalog          : {catalog_name}")
print(f"   Schema           : {schema_name}")
print(f"   Model            : {MODEL_NAME}")
print(f"   Serving endpoint : {SERVING_ENDPOINT}")
print(f"   Workspace URL    : {WORKSPACE_URL}")

spark = SparkSession.builder.getOrCreate()
w     = WorkspaceClient()

# COMMAND ----------

# MAGIC %md ## 🚀 Deploy Model Serving Endpoint

# COMMAND ----------

# DBTITLE 1,Get Latest Model Version from UC Registry
mlflow.set_registry_uri("databricks-uc")
client = mlflow.tracking.MlflowClient()

versions = client.search_model_versions(f"name='{MODEL_NAME}'")
if not versions:
    raise ValueError(
        f"No versions found for model {MODEL_NAME}. "
        "Run Notebook 04 first."
    )

latest_version = max(versions, key=lambda v: int(v.version))
model_version  = latest_version.version
print(f"✅ Latest model version: {model_version}")

# COMMAND ----------

# DBTITLE 1,Create or Update Serving Endpoint
def wait_for_endpoint_ready(client: WorkspaceClient, endpoint_name: str, timeout_min: int = 20):
    """Poll until serving endpoint is ready."""
    deadline = time.time() + timeout_min * 60
    while time.time() < deadline:
        ep    = client.serving_endpoints.get(endpoint_name)
        state = ep.state.config_update.value if ep.state else "UNKNOWN"
        ready = ep.state.ready.value if ep.state else "NOT_READY"
        print(f"   State: {state} | Ready: {ready}")
        if ready == "READY":
            return True
        if state in ("UPDATE_FAILED",):
            raise RuntimeError(f"Endpoint update failed: {ep}")
        time.sleep(30)
    raise TimeoutError(f"Endpoint {endpoint_name} not ready within {timeout_min} min")


served_model = ServedModelInput(
    model_name          = MODEL_NAME,
    model_version       = model_version,
    workload_size       = ServedModelInputWorkloadSize.SMALL,
    scale_to_zero_enabled = True,   # serverless — free when idle
)

config = EndpointCoreConfigInput(served_models=[served_model])

try:
    existing = w.serving_endpoints.get(SERVING_ENDPOINT)
    print(f"🔄 Updating existing endpoint: {SERVING_ENDPOINT}")
    w.serving_endpoints.update_config(SERVING_ENDPOINT, served_models=[served_model])
except Exception as e:
    if "does not exist" in str(e).lower() or "RESOURCE_DOES_NOT_EXIST" in str(e):
        print(f"🏗️  Creating new serving endpoint: {SERVING_ENDPOINT}")
        w.serving_endpoints.create(
            name   = SERVING_ENDPOINT,
            config = config,
        )
    else:
        raise

print("⏳ Waiting for endpoint to be ready...")
wait_for_endpoint_ready(w, SERVING_ENDPOINT)
print(f"✅ Serving endpoint {SERVING_ENDPOINT} is READY")

# COMMAND ----------

# DBTITLE 1,Smoke Test the Serving Endpoint
print("🧪 Testing serving endpoint...")

test_payload = {"inputs": [{"query": "How many days of annual leave do I get after 3 years?"}]}
headers      = {
    "Authorization": f"Bearer {DB_TOKEN}",
    "Content-Type":  "application/json",
}
endpoint_url = f"https://{WORKSPACE_URL}/serving-endpoints/{SERVING_ENDPOINT}/invocations"

response = requests.post(endpoint_url, headers=headers, json=test_payload, timeout=60)
if response.status_code == 200:
    result = response.json()
    print(f"✅ Endpoint smoke test passed")
    print(f"   Response: {str(result)[:200]}")
else:
    print(f"⚠️  Endpoint returned {response.status_code}: {response.text[:200]}")

# COMMAND ----------

# MAGIC %md ## 🎨 Gradio UI — Databricks Apps

# COMMAND ----------

# DBTITLE 1,Define Gradio Application
import gradio as gr

# ── Helper: call serving endpoint ─────────────────────────────────────────────
def call_rag_endpoint(query: str) -> tuple:
    """
    Call the Model Serving endpoint and return (response, latency_ms).
    Handles cold start with retry.
    """
    headers = {
        "Authorization": f"Bearer {DB_TOKEN}",
        "Content-Type":  "application/json",
    }
    payload = {"inputs": [{"query": query}]}
    url     = f"https://{WORKSPACE_URL}/serving-endpoints/{SERVING_ENDPOINT}/invocations"

    for attempt in range(3):
        t0 = time.time()
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            latency_ms = int((time.time() - t0) * 1000)
            if resp.status_code == 200:
                data     = resp.json()
                # Handle both list and dict response formats
                if isinstance(data, dict) and "predictions" in data:
                    answer = data["predictions"][0]
                elif isinstance(data, list):
                    answer = data[0]
                else:
                    answer = str(data)
                return answer, latency_ms
            elif resp.status_code == 503 and attempt < 2:
                print(f"   Cold start — retrying ({attempt+1}/3)...")
                time.sleep(10)
            else:
                return f"⚠️ Service error ({resp.status_code}). Please try again.", latency_ms
        except requests.Timeout:
            latency_ms = int((time.time() - t0) * 1000)
            if attempt < 2:
                time.sleep(5)
            else:
                return "⚠️ Request timed out. The endpoint may be warming up — please try again in 10 seconds.", latency_ms
        except Exception as e:
            return f"⚠️ Unexpected error: {str(e)}", 0

    return "⚠️ Service unavailable after 3 attempts.", 0


# ── Helper: log interaction to Gold table ─────────────────────────────────────
def log_interaction(
    session_id: str, query: str, response: str,
    latency_ms: int, disclaimer_added: bool = False
) -> str:
    """Write interaction record to qa_interactions Gold table."""
    interaction_id = str(uuid.uuid4())
    try:
        schema = StructType([
            StructField("interaction_id",       StringType(),    True),
            StructField("session_id",           StringType(),    True),
            StructField("user_query",           StringType(),    True),
            StructField("llm_response",         StringType(),    True),
            StructField("retrieved_chunk_ids",  StringType(),    True),
            StructField("top_similarity_score", DoubleType(),    True),
            StructField("disclaimer_added",     BooleanType(),   True),
            StructField("latency_ms",           LongType(),      True),
            StructField("model_version",        StringType(),    True),
            StructField("pipeline_run_id",      StringType(),    True),
            StructField("created_at",           TimestampType(), True),
        ])
        spark.createDataFrame([{
            "interaction_id":       interaction_id,
            "session_id":           session_id,
            "user_query":           query,
            "llm_response":         response,
            "retrieved_chunk_ids":  "[]",
            "top_similarity_score": 0.0,
            "disclaimer_added":     disclaimer_added,
            "latency_ms":           latency_ms,
            "model_version":        LLM_MODEL,
            "pipeline_run_id":      PIPELINE_RUN_ID,
            "created_at":           datetime.now(timezone.utc),
        }], schema=schema).write.mode("append").saveAsTable(
            f"{catalog_name}.{schema_name}.qa_interactions"
        )
    except Exception as e:
        print(f"⚠️  Could not log interaction: {e}")
    return interaction_id


def log_feedback(interaction_id: str, rating: str, feedback_text: str = ""):
    """Write feedback record to qa_feedback Gold table."""
    try:
        schema = StructType([
            StructField("feedback_id",     StringType(),    True),
            StructField("interaction_id",  StringType(),    True),
            StructField("rating",          StringType(),    True),
            StructField("feedback_text",   StringType(),    True),
            StructField("created_at",      TimestampType(), True),
        ])
        spark.createDataFrame([{
            "feedback_id":    str(uuid.uuid4()),
            "interaction_id": interaction_id,
            "rating":         rating,
            "feedback_text":  feedback_text,
            "created_at":     datetime.now(timezone.utc),
        }], schema=schema).write.mode("append").saveAsTable(
            f"{catalog_name}.{schema_name}.qa_feedback"
        )
    except Exception as e:
        print(f"⚠️  Could not log feedback: {e}")


# ── Gradio Chat Handler ────────────────────────────────────────────────────────
SESSION_ID = str(uuid.uuid4())
last_interaction_id = {"value": None}

def chat(message: str, history: list) -> tuple:
    """Main chat handler — calls RAG endpoint and logs interaction."""
    if not message.strip():
        return "", history

    # Warm-up message for first query
    if not history:
        history = history + [
            [None, "👋 Welcome to the Policy Q&A Assistant. "
                   "Ask me anything about company policies. "
                   "⏳ First query may take ~10 seconds to warm up the endpoint."]
        ]

    response, latency_ms = call_rag_endpoint(message)
    interaction_id = log_interaction(SESSION_ID, message, response, latency_ms)
    last_interaction_id["value"] = interaction_id

    history = history + [[message, response]]
    return "", history


def submit_feedback(rating: str, feedback_text: str) -> str:
    """Handle thumbs up/down feedback."""
    iid = last_interaction_id.get("value")
    if not iid:
        return "⚠️ No recent interaction to rate."
    log_feedback(iid, rating, feedback_text)
    return f"✅ Thank you for your feedback ({rating})!"


# ── Gradio UI Layout ───────────────────────────────────────────────────────────
EXAMPLE_QUESTIONS = [
    "How many days of annual leave do I get after 5 years of service?",
    "What is the password expiry policy?",
    "Is alcohol reimbursable during business travel?",
    "How long are employee records retained?",
    "What is the minimum internet speed for remote work?",
    "What happens if I don't submit expenses within 30 days?",
]

with gr.Blocks(
    title="Policy Q&A Assistant — SNS Square",
    theme=gr.themes.Soft(primary_hue="blue"),
    css="""
        .header-box { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                      padding: 20px; border-radius: 10px; margin-bottom: 20px; }
        .header-title { color: #e94560; font-size: 1.8em; font-weight: bold; }
        .header-sub { color: #a8b2d8; font-size: 0.95em; margin-top: 5px; }
        .disclaimer { background: #fff3cd; border-left: 4px solid #ffc107;
                      padding: 10px; border-radius: 4px; font-size: 0.85em; }
    """
) as demo:

    # Header
    with gr.Row():
        gr.HTML("""
            <div class="header-box">
                <div class="header-title">📋 Policy Q&A Assistant</div>
                <div class="header-sub">
                    Powered by Databricks · Mosaic AI · Llama 3.1 · RAG
                    &nbsp;|&nbsp; Built by <strong>SNS Square</strong>
                </div>
            </div>
        """)

    with gr.Row():
        # Left: Chat
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Policy Assistant",
                height=500,
                show_copy_button=True,
                bubble_full_width=False,
            )
            with gr.Row():
                msg_box = gr.Textbox(
                    placeholder="Ask a question about company policies...",
                    label="Your Question",
                    scale=4,
                    lines=2,
                )
                send_btn = gr.Button("Ask →", variant="primary", scale=1)

            gr.HTML("""
                <div class="disclaimer">
                    ⚠️ <strong>Human-in-the-loop:</strong> Answers are AI-generated recommendations
                    based on policy documents. Always verify critical decisions with HR or the
                    relevant department. This tool does not constitute legal or HR advice.
                </div>
            """)

        # Right: Examples + Feedback
        with gr.Column(scale=1):
            gr.Markdown("### 💡 Example Questions")
            for q in EXAMPLE_QUESTIONS:
                gr.Button(q, size="sm").click(
                    fn=lambda x=q: x,
                    outputs=msg_box
                )

            gr.Markdown("---")
            gr.Markdown("### 📊 Rate This Answer")
            with gr.Row():
                thumbs_up   = gr.Button("👍 Helpful",   variant="secondary")
                thumbs_down = gr.Button("👎 Not Helpful", variant="secondary")
            feedback_text   = gr.Textbox(
                placeholder="Optional: What was wrong or missing?",
                label="Feedback (optional)",
                lines=2,
            )
            feedback_status = gr.Textbox(label="Feedback Status", interactive=False)

    # Footer
    gr.HTML("""
        <div style="text-align:center; color:#888; font-size:0.8em; margin-top:20px;">
            LLM-Powered Policy Q&A Assistant v1.0.0 · SNS Square · Databricks Accelerator<br>
            All interactions are logged for quality improvement. No personal data is stored.
        </div>
    """)

    # Event handlers
    send_btn.click(chat, [msg_box, chatbot], [msg_box, chatbot])
    msg_box.submit(chat, [msg_box, chatbot], [msg_box, chatbot])
    thumbs_up.click(
        fn=lambda t: submit_feedback("thumbs_up", t),
        inputs=[feedback_text], outputs=[feedback_status]
    )
    thumbs_down.click(
        fn=lambda t: submit_feedback("thumbs_down", t),
        inputs=[feedback_text], outputs=[feedback_status]
    )

print("✅ Gradio app defined")

# COMMAND ----------

# DBTITLE 1,Launch on Databricks Apps
print("🚀 Launching Gradio UI on Databricks Apps...")
print("   ⏳ First query may take ~10 seconds (serverless cold start)")
print()

demo.launch(
    server_name = "0.0.0.0",
    server_port = 8080,
    share       = False,   # Databricks Apps provides the public URL
)

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
    "notebook":       "05_serve_and_demo",
    "start_time":     datetime.now(timezone.utc),
    "end_time":       end_time,
    "rows_processed": 0,
    "status":         "SUCCESS",
    "error_message":  None,
}], schema=audit_schema).write.mode("append").saveAsTable(
    f"{catalog_name}.{schema_name}.pipeline_audit_log"
)
print(f"✅ Audit log written — run_id: {PIPELINE_RUN_ID}")

# COMMAND ----------

# MAGIC %md ## 📋 Serving & Demo Complete
# MAGIC
# MAGIC ### ✅ What Was Created:
# MAGIC | Asset | Location |
# MAGIC |---|---|
# MAGIC | Model Serving endpoint | `policy_qa_endpoint` (serverless, scales to zero) |
# MAGIC | Gradio UI | Databricks Apps — check Apps panel for public URL |
# MAGIC | Interaction log | `{catalog_name}.{schema_name}.qa_interactions` |
# MAGIC | Feedback log | `{catalog_name}.{schema_name}.qa_feedback` |
# MAGIC
# MAGIC ### 🎉 Pipeline Complete!
# MAGIC View the full pipeline in **Databricks Workflows** and traces in **MLflow Experiments**.

# COMMAND ----------

dbutils.notebook.exit(
    f"SUCCESS: Serving endpoint={SERVING_ENDPOINT}, "
    f"Gradio UI launched on Databricks Apps, run_id={PIPELINE_RUN_ID}"
)
