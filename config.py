# =============================================================================
# LLM-Powered Policy Q&A Assistant — Shared Configuration
# Accelerator Version : 1.0.0
# Author              : SNS Square
# Last Updated        : 2025-05-21
# Databricks Runtime  : 14.3 LTS ML (tested)
# =============================================================================

# ── Deployment switch ─────────────────────────────────────────────────────────
# True  → Databricks-native stack (Foundation Models API + Vector Search)
#         Default for all demos, Brickbuilder submission, and production.
# False → sentence-transformers + NumPy fallback (zero-cost, dev/offline only)
#         Use ONLY when you have no Databricks workspace access at all.
USE_FOUNDATION_MODELS_API = True

# ── Model selection ───────────────────────────────────────────────────────────
# Embedding model served via Mosaic AI Foundation Models API
EMBEDDING_MODEL = "bge-large-en-v1.5"          # 1024-dim, best retrieval quality

# LLM — switch to 70B only for final demo recording (higher cost)
LLM_MODEL = "databricks-meta-llama-3-1-8b-instruct"   # dev / CI
# LLM_MODEL = "databricks-meta-llama-3-1-70b-instruct" # final demo recording

# Fallback models (used when USE_FOUNDATION_MODELS_API = False)
FALLBACK_EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # ~90 MB, CPU, HuggingFace
FALLBACK_LLM_MODEL       = "google/flan-t5-base" # ~250 MB, CPU, HuggingFace

# ── Unity Catalog ─────────────────────────────────────────────────────────────
# Override via databricks.yml widget or dbutils.widgets in each notebook
DEFAULT_CATALOG = "dev_policy_qa"
DEFAULT_SCHEMA  = "policy_assistant"

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE_TOKENS = 400    # target tokens per chunk (300-500 per spec)
CHUNK_OVERLAP_TOKENS = 50  # overlap tokens between consecutive chunks
MIN_CHUNK_CHARS   = 100    # discard chunks shorter than this

# ── Vector Search ─────────────────────────────────────────────────────────────
VS_ENDPOINT_NAME  = "policy_qa_vs_endpoint"
VS_INDEX_NAME     = "policy_chunks_index"
VS_EMBEDDING_DIM  = 1024   # BGE-Large output dimension
TOP_K_RETRIEVAL   = 5      # chunks retrieved per query (spec: top-5)

# ── RAG chain ─────────────────────────────────────────────────────────────────
MAX_ANSWER_TOKENS         = 512
SIMILARITY_THRESHOLD      = 0.70   # below this → add disclaimer (spec guardrail)
INPUT_MAX_TOKENS          = 2000   # reject queries longer than this

# ── MLflow ────────────────────────────────────────────────────────────────────
MLFLOW_EXPERIMENT_NAME = "/Shared/policy_qa_assistant"
MLFLOW_MODEL_NAME      = "policy_qa_rag_chain"   # registered in UC model registry

# ── Model Serving ─────────────────────────────────────────────────────────────
SERVING_ENDPOINT_NAME = "policy_qa_endpoint"

# ── Cost reference (README transparency) ─────────────────────────────────────
# BGE-Large embeddings : ~$0.0001 / 1K tokens  → 50 docs ≈ $0.10 one-time
# Llama 3.1 8B         : ~$0.20   / 1M tokens  → 200 test queries ≈ $0.05
# Llama 3.1 70B        : ~$0.90   / 1M tokens  → 1 demo session ≈ $0.50
# Vector Search        : 1 endpoint = free tier quota (no per-query cost)
# Total realistic spend to build + demo : < $2
