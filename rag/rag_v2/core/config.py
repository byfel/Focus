
import os
from typing import List

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = "/opt/ai/rag_v2"
DATA_ROOT = "/opt/ai/rag"  # Usa dados da versão atual

# ============================================================
# OLLAMA
# ============================================================

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_URL = f"{OLLAMA_BASE_URL}/api/embed"
OLLAMA_CHAT_URL = f"{OLLAMA_BASE_URL}/api/chat"

# ============================================================
# MODELOS
# ============================================================

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

SUPPORTED_MODELS = [
    "gemma4:26b-a4b-it-q4_K_M",
    "qwen3.6:27b",
    "qwen3.6:35b",
    "gemma4:12b",
    "gemma4:e4b",
    "gemma3:12b",
    "gemma3:4b",
]

DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "gemma4:12b")
QUERY_EXPANSION_MODEL = os.getenv("QUERY_EXPANSION_MODEL", "gemma3:4b")

# ============================================================
# QDRANT
# ============================================================

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "rag_documents")
QDRANT_VECTOR_SIZE = int(os.getenv("QDRANT_VECTOR_SIZE", "768"))

# ============================================================
# RAG PARAMETERS
# ============================================================

TOP_K = int(os.getenv("RAG_TOP_K", "12"))
FINAL_CONTEXTS = int(os.getenv("RAG_FINAL_CONTEXTS", "10"))
THRESHOLD = float(os.getenv("RAG_THRESHOLD", "0.30"))

QUERY_EXPANSION_ENABLED = os.getenv("QUERY_EXPANSION_ENABLED", "true").lower() == "true"
QUERY_EXPANSION_COUNT = int(os.getenv("QUERY_EXPANSION_COUNT", "3"))
EXPANSION_TOP_K = int(os.getenv("EXPANSION_TOP_K", str(TOP_K)))

# ============================================================
# PERFORMANCE
# ============================================================

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "10"))
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"

# ============================================================
# AUTENTICAÇÃO
# ============================================================

AUTH_MODE = os.getenv("RAG_AUTH_MODE", "local")
LOCAL_USERNAME = os.getenv("RAG_LOCAL_USERNAME", "admin")
LOCAL_PASSWORD = os.getenv("RAG_LOCAL_PASSWORD", "admin123")
SESSION_COOKIE = "rag_session"

