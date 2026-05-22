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
# MAGIC %pip install "gradio>=4.0.0" "databricks-sdk==0.40.0" "mlflow==2.22.5" "requests==2.32.3"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md ## ⚙️ Configuration

# COMMAND ----------

# DBTITLE 1,Import Libraries
import json, time, uuid, requests, os
from datetime import datetime, timezone

# Set proxy exclusion for localhost to prevent Gradio launch connection errors on Databricks
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"
os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"

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

spark = SparkSession.builder.getOrCreate()
w     = WorkspaceClient()

# Databricks workspace URL (for REST API calls from Gradio)
WORKSPACE_URL = spark.conf.get("spark.databricks.workspaceUrl", "")
DB_TOKEN      = dbutils.secrets.get(scope="policy_qa_scope", key="db_token")

print(f"🔧 Serving Configuration")
print(f"   Catalog          : {catalog_name}")
print(f"   Schema           : {schema_name}")
print(f"   Model            : {MODEL_NAME}")
print(f"   Serving endpoint : {SERVING_ENDPOINT}")
print(f"   Workspace URL    : {WORKSPACE_URL}")

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
def wait_for_endpoint_ready(client: WorkspaceClient, endpoint_name: str, timeout_min: int = 25):
    """
    Poll until serving endpoint is READY or fails definitively.
    Surfaces the actual deployment_state_message on failure so you can
    diagnose model load errors without digging through the SDK object dump.
    """
    deadline = time.time() + timeout_min * 60
    while time.time() < deadline:
        ep     = client.serving_endpoints.get(endpoint_name)
        state  = ep.state.config_update.value  if ep.state else "UNKNOWN"
        ready  = ep.state.ready.value          if ep.state else "NOT_READY"
        print(f"   State: {state} | Ready: {ready}")

        if ready == "READY":
            return True

        # Dig out the human-readable deployment message from served entities
        if state in ("UPDATE_FAILED", "UPDATE_CANCELED"):
            msg = "Model server failed to load — check serving logs."
            try:
                pending = ep.pending_config or ep.config
                if pending and pending.served_entities:
                    for entity in pending.served_entities:
                        s = entity.state
                        if s and s.deployment_state_message:
                            msg = s.deployment_state_message
                            break
                elif pending and pending.served_models:
                    for model in pending.served_models:
                        s = model.state
                        if s and s.deployment_state_message:
                            msg = s.deployment_state_message
                            break
            except Exception:
                pass
            raise RuntimeError(
                f"Endpoint '{endpoint_name}' deployment failed.\n"
                f"Reason: {msg}\n\n"
                f"To diagnose: Databricks UI → Serving → {endpoint_name} → Logs tab.\n"
                f"Common causes:\n"
                f"  1. pip install failed — check requirements in notebook 04\n"
                f"  2. load_context() raised an exception — check chain_config.json artifact\n"
                f"  3. Vector Search endpoint not reachable from serving container"
            )

        time.sleep(30)

    raise TimeoutError(f"Endpoint {endpoint_name} not ready within {timeout_min} min")


served_model = ServedModelInput(
    model_name            = MODEL_NAME,
    model_version         = model_version,
    workload_size         = ServedModelInputWorkloadSize.SMALL,
    scale_to_zero_enabled = True,   # serverless — free when idle
)

config = EndpointCoreConfigInput(served_models=[served_model])

def _endpoint_is_failed(client: WorkspaceClient, endpoint_name: str) -> bool:
    """Return True if the endpoint exists but is in a FAILED/NOT_READY state."""
    try:
        ep    = client.serving_endpoints.get(endpoint_name)
        state = ep.state.config_update.value if ep.state else ""
        ready = ep.state.ready.value         if ep.state else ""
        return state in ("UPDATE_FAILED", "UPDATE_CANCELED") or ready == "NOT_READY"
    except Exception:
        return False

try:
    existing = w.serving_endpoints.get(SERVING_ENDPOINT)
    existing_state = existing.state.config_update.value if existing.state else ""

    if existing_state in ("UPDATE_FAILED", "UPDATE_CANCELED"):
        # Previous deployment failed — delete and recreate for a clean slate
        print(f"⚠️  Endpoint {SERVING_ENDPOINT} is in {existing_state} state.")
        print(f"   Deleting and recreating for a clean deployment...")
        w.serving_endpoints.delete(SERVING_ENDPOINT)
        time.sleep(10)   # brief pause for deletion to propagate
        print(f"🏗️  Creating fresh endpoint: {SERVING_ENDPOINT}")
        w.serving_endpoints.create(name=SERVING_ENDPOINT, config=config)
    else:
        print(f"🔄 Updating existing endpoint: {SERVING_ENDPOINT}")
        w.serving_endpoints.update_config(SERVING_ENDPOINT, served_models=[served_model])

except Exception as e:
    if "does not exist" in str(e).lower() or "RESOURCE_DOES_NOT_EXIST" in str(e):
        print(f"🏗️  Creating new serving endpoint: {SERVING_ENDPOINT}")
        w.serving_endpoints.create(name=SERVING_ENDPOINT, config=config)
    else:
        raise

print("⏳ Waiting for endpoint to be ready (may take 10–15 min on first deploy)...")
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

# Ensure proxy exclusion for localhost/127.0.0.1 is active in this execution context
import os
import requests

# Save current proxy settings
proxies = ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]
saved_proxies = {p: os.environ.get(p) for p in proxies}

# Temporarily clear proxy settings during launch to prevent local health check interception
for p in proxies:
    if p in os.environ:
        del os.environ[p]

os.environ["no_proxy"] = "*"
os.environ["NO_PROXY"] = "*"

try:
    gr.close_all()
except Exception as e:
    print(f"⚠️ Warning: Could not close previous Gradio instances: {e}")

# Determine server port: use DATABRICKS_APP_PORT if running inside Databricks Apps,
# otherwise find a free port starting from 8080 for interactive/notebook execution.
if "DATABRICKS_APP_PORT" in os.environ:
    server_port = int(os.environ["DATABRICKS_APP_PORT"])
    server_name = "0.0.0.0"
    share_gradio = False
    print(f"👉 Using Databricks Apps port: {server_port}")
else:
    server_name = "0.0.0.0"
    server_port = None
    share_gradio = True
    print("👉 Launching interactive Gradio with share=True (Port assigned dynamically)")

try:
    demo.launch(
        server_name = server_name,
        server_port = server_port,
        share       = share_gradio,
        inline      = True,
        show_error  = True
    )
finally:
    # Restore proxy settings for subsequent network requests (e.g. Model Serving calls)
    for p, val in saved_proxies.items():
        if val is not None:
            os.environ[p] = val
        elif p in os.environ:
            del os.environ[p]

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
