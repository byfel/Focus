"""
Servidor Web FastAPI para RAG v2.
Interface moderna estilo ChatGPT/Claude/Gemini com identidade visual inspirada na Globo.
Opera por padrão na porta 8001 (separada da v1).
"""
import os
import sys
import logging
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Garante inclusão de rag_v2 no path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.pipeline import ask
from core.config import (
    DEFAULT_LLM_MODEL,
    SUPPORTED_MODELS,
    EMBEDDING_MODEL,
    QDRANT_COLLECTION,
    SESSION_COOKIE,
    SESSION_COOKIE_SECURE,
    SESSION_TTL_SECONDS,
    WEB_HOST,
    WEB_PORT,
)
try:
    from api import auth
    from api import conversations
except ImportError:
    from . import auth
    from . import conversations

logger = logging.getLogger("rag_v2.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# Inicializa banco de dados de conversas da v2
conversations.init_db()

app = FastAPI(
    title="FOCUS AI - RAG v2",
    description="Assistente de Documentação Técnica com nomic-embed-text",
    version="2.0.0",
)

# Caminhos dos templates e estáticos
WEB_DIR = PROJECT_ROOT / "web"
TEMPLATES_DIR = WEB_DIR / "templates"
STATIC_DIR = WEB_DIR / "static"

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# Modelos Pydantic
class LoginRequest(BaseModel):
    username: str
    password: str


class QuestionRequest(BaseModel):
    question: str
    model: Optional[str] = None
    conversation_id: str


class ConversationCreateRequest(BaseModel):
    title: Optional[str] = None
    model: Optional[str] = None


class ConversationRenameRequest(BaseModel):
    title: str


def _get_authenticated_user(request: Request) -> Optional[str]:
    """Recupera o usuário autenticado através do cookie de sessão."""
    token = request.cookies.get(SESSION_COOKIE)
    return auth.validate_session(token)


# ============================================================
# ROTAS DE PÁGINAS (HTML)
# ============================================================

@app.get("/")
def index_page(request: Request):
    """Página principal (Index / Chat) ou redireciona para Login se não autenticado."""
    username = _get_authenticated_user(request)
    if not username:
        return FileResponse(str(TEMPLATES_DIR / "login.html"))
    return FileResponse(str(TEMPLATES_DIR / "index.html"))


@app.get("/login")
def login_page():
    """Tela de login."""
    return FileResponse(str(TEMPLATES_DIR / "login.html"))


# ============================================================
# ROTAS DE AUTENTICAÇÃO
# ============================================================

@app.post("/login")
def login_endpoint(payload: LoginRequest):
    username = payload.username.strip()
    password = payload.password

    if not username or not password:
        raise HTTPException(status_code=400, detail="Usuário e senha são obrigatórios.")

    if not auth.authenticate_user(username, password):
        raise HTTPException(status_code=401, detail="Usuário ou senha inválidos.")

    token = auth.create_session(username)

    response = JSONResponse({"status": "ok", "username": username})
    response.set_cookie(
        key=SESSION_COOKIE,
        value=token,
        httponly=True,
        secure=SESSION_COOKIE_SECURE,
        samesite="lax",
        max_age=SESSION_TTL_SECONDS,
    )
    return response


@app.post("/logout")
def logout_endpoint(request: Request):
    token = request.cookies.get(SESSION_COOKIE)
    auth.destroy_session(token)

    response = JSONResponse({"status": "ok"})
    response.delete_cookie(SESSION_COOKIE)
    return response


@app.get("/me")
def me_endpoint(request: Request):
    username = _get_authenticated_user(request)
    if not username:
        raise HTTPException(status_code=401, detail="Não autenticado.")
    return {"username": username}


# ============================================================
# ROTAS DO SISTEMA & RAG
# ============================================================

@app.get("/health")
def health_endpoint():
    return {
        "status": "ok",
        "version": "2.0.0",
        "embedding_model": EMBEDDING_MODEL,
        "collection": QDRANT_COLLECTION,
    }


@app.get("/models")
def models_endpoint():
    return {
        "models": SUPPORTED_MODELS,
        "default": DEFAULT_LLM_MODEL,
    }


@app.post("/ask")
def ask_endpoint(request: Request, body: QuestionRequest):
    username = _get_authenticated_user(request)
    if not username:
        raise HTTPException(status_code=401, detail="Não autenticado.")

    model_to_use = body.model or DEFAULT_LLM_MODEL
    if model_to_use not in SUPPORTED_MODELS:
        # Se o modelo solicitado não estiver na lista estrita, tenta usar com aviso ou fallback
        logger.warning("Modelo %s fora da lista SUPPORTED_MODELS. Tentando executar.", model_to_use)

    # Registra a mensagem do usuário no histórico
    try:
        conversations.add_message(
            body.conversation_id, username, "user", body.question
        )
    except conversations.ConversationNotFound:
        raise HTTPException(status_code=404, detail="Conversa não encontrada.")

    # Executa a busca e geração no pipeline v2
    try:
        result = ask(body.question, model=model_to_use)
    except Exception as e:
        logger.exception("Falha ao processar pergunta no pipeline v2: %s", e)
        raise HTTPException(
            status_code=503,
            detail=f"Erro ao processar com o modelo {model_to_use}: {str(e)}",
        )

    # Grava resposta do assistente no banco
    conversations.add_message(
        body.conversation_id,
        username,
        "assistant",
        result["answer"],
        result.get("sources", []),
    )

    # Define o primeiro texto como título da conversa caso seja o default
    conversations.set_first_message_as_title(
        body.conversation_id, username, body.question
    )

    return {
        "question": body.question,
        "model": model_to_use,
        "username": username,
        "answer": result["answer"],
        "sources": result.get("sources", []),
        "conversation_id": body.conversation_id,
    }


# ============================================================
# ROTAS DE CONVERSAS (SQLite)
# ============================================================

@app.get("/conversations")
def list_conversations_endpoint(request: Request):
    username = _get_authenticated_user(request)
    if not username:
        raise HTTPException(status_code=401, detail="Não autenticado.")
    return {"conversations": conversations.list_conversations(username)}


@app.post("/conversation")
def create_conversation_endpoint(request: Request, body: ConversationCreateRequest):
    username = _get_authenticated_user(request)
    if not username:
        raise HTTPException(status_code=401, detail="Não autenticado.")
    return conversations.create_conversation(username, body.title, body.model)


@app.get("/conversation/{conversation_id}")
def get_conversation_endpoint(conversation_id: str, request: Request):
    username = _get_authenticated_user(request)
    if not username:
        raise HTTPException(status_code=401, detail="Não autenticado.")

    try:
        return conversations.get_conversation(conversation_id, username)
    except conversations.ConversationNotFound:
        raise HTTPException(status_code=404, detail="Conversa não encontrada.")


@app.patch("/conversation/{conversation_id}")
def rename_conversation_endpoint(
    conversation_id: str, body: ConversationRenameRequest, request: Request
):
    username = _get_authenticated_user(request)
    if not username:
        raise HTTPException(status_code=401, detail="Não autenticado.")

    try:
        conversations.rename_conversation(conversation_id, username, body.title)
    except conversations.ConversationNotFound:
        raise HTTPException(status_code=404, detail="Conversa não encontrada.")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return {"status": "ok"}


@app.delete("/conversation/{conversation_id}")
def delete_conversation_endpoint(conversation_id: str, request: Request):
    username = _get_authenticated_user(request)
    if not username:
        raise HTTPException(status_code=401, detail="Não autenticado.")

    try:
        conversations.delete_conversation(conversation_id, username)
    except conversations.ConversationNotFound:
        raise HTTPException(status_code=404, detail="Conversa não encontrada.")

    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    print(f"🚀 Iniciando FOCUS AI v2 em http://{WEB_HOST}:{WEB_PORT}")
    uvicorn.run("server:app", host=WEB_HOST, port=WEB_PORT, reload=True)
