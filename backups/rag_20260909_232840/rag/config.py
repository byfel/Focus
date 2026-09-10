import os
import logging

# ============================================================
# LOGGING
#
# Substitui os `print()` espalhados pelo pipeline. Configure o
# nível via RAG_LOG_LEVEL=DEBUG para ver o detalhamento de
# retrieval (RRF, scores, etc.) que antes ia pro stdout.
# ============================================================
LOG_LEVEL = os.getenv("RAG_LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ============================================================
# EMBEDDING
# ============================================================
EMBEDDING_MODEL = "embeddinggemma:latest"

# ============================================================
# LLM
#
# Fonte única. pipeline.py, server.py e qualquer outro lugar
# devem IMPORTAR daqui, nunca reescrever essa lista.
# ============================================================
LLM_MODELS = [
    "gemma4:26b-a4b-it-q4_K_M",
    "qwen3.6:35b",
    "qwen3.6:27b",
    "gemma4:12b",
    "gemma4:e4b",
    "gemma3:12b",
    "gemma3:4b",
]
DEFAULT_LLM_MODEL = "gemma3:12b"

# ============================================================
# QDRANT
# ============================================================
QDRANT_URL = os.getenv("RAG_QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("RAG_QDRANT_COLLECTION", "rag_documents")

# ============================================================
# OLLAMA
# ============================================================
OLLAMA_BASE_URL = os.getenv("RAG_OLLAMA_URL", "http://localhost:11434")
OLLAMA_EMBED_URL = f"{OLLAMA_BASE_URL}/api/embed"
OLLAMA_CHAT_URL = f"{OLLAMA_BASE_URL}/api/chat"
OLLAMA_EMBED_TIMEOUT = int(os.getenv("RAG_EMBED_TIMEOUT", "120"))
OLLAMA_CHAT_TIMEOUT = int(os.getenv("RAG_CHAT_TIMEOUT", "600"))

# ============================================================
# RETRIEVAL
# ============================================================
TOP_K = 12
FINAL_CONTEXTS = 10
THRESHOLD = 0.30
RRF_K = 60

# NOTA: existia no config original mas não estava sendo usada em
# nenhum lugar do retriever.py. Não implementei uma versão
# "adivinhada" da lógica (ver nota em retriever.py) — confirme a
# intenção original antes de ativar isso de fato.
MIN_DOMINANCE_RATIO = 0.60

# ============================================================
# QUERY EXPANSION
# ============================================================
QUERY_EXPANSION_ENABLED = False
QUERY_EXPANSION_MODEL = "gemma4:e4b"
QUERY_EXPANSION_COUNT = 3
QUERY_EXPANSION_TEMPERATURE = 0.2
EXPANSION_TOP_K = TOP_K

# ============================================================
# GERAÇÃO
# ============================================================
GENERATION_TEMPERATURE = 0.1

# Orçamento de contexto por CARACTERES (aproximação, não é
# contagem real de tokens do modelo — Gemma e Qwen tokenizam
# diferente). Serve como trava de segurança contra estourar a
# janela de contexto silenciosamente.
MAX_CONTEXT_CHARS = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "24000"))

# ============================================================
# AUTENTICAÇÃO
#
# Mantido como estava: login/senha local com fallback padrão,
# só para a fase de testes. Em produção isso vai ser substituído
# pelo LDAP da empresa (AUTH_MODE="ldap") — ver placeholder em
# auth.py, ainda não implementado.
# ============================================================
AUTH_MODE = os.getenv("RAG_AUTH_MODE", "local")
LOCAL_USERNAME = os.getenv("RAG_LOCAL_USERNAME", "admin")
LOCAL_PASSWORD = os.getenv("RAG_LOCAL_PASSWORD", "admin123")

SESSION_COOKIE = "rag_session"
SESSION_TTL_SECONDS = int(os.getenv("RAG_SESSION_TTL", "28800"))  # 8h
# Defina RAG_COOKIE_SECURE=true se o serviço só for acessado via HTTPS
# (ex.: atrás de um proxy reverso com TLS).
SESSION_COOKIE_SECURE = os.getenv("RAG_COOKIE_SECURE", "false").lower() == "true"
