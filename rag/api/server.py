import logging

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from rag.pipeline import ask, SUPPORTED_MODELS
from rag.config import (
    DEFAULT_LLM_MODEL,
    SESSION_COOKIE,
    SESSION_COOKIE_SECURE,
    SESSION_TTL_SECONDS,
)

from . import conversations
from .auth import (
    authenticate_ldap,
    create_session,
    validate_session,
    destroy_session,
)

logger = logging.getLogger(__name__)

conversations.init_db()

app = FastAPI(
    title="SUPORTE AI RAG",
    description="Assistente técnico baseado em documentação",
    version="1.0.0",
)


class LoginRequest(BaseModel):
    username: str
    password: str


class Question(BaseModel):
    question: str
    model: str = DEFAULT_LLM_MODEL
    conversation_id: str


class ConversationCreateRequest(BaseModel):
    title: str | None = None


class ConversationRenameRequest(BaseModel):
    title: str


def _current_user(request: Request) -> str | None:
    token = request.cookies.get(SESSION_COOKIE)
    return validate_session(token)


@app.get("/")
def login_page():
    return FileResponse("/opt/ai/rag/web/login.html")


@app.get("/app")
def application_page(request: Request):
    username = _current_user(request)
    if not username:
        return FileResponse("/opt/ai/rag/web/login.html")
    return FileResponse("/opt/ai/rag/web/index.html")


@app.post("/login")
def login(request: LoginRequest):
    username = request.username.strip()
    password = request.password

    if not username or not password:
        raise HTTPException(
            status_code=400, detail="Usuário e senha são obrigatórios."
        )

    if not authenticate_ldap(username, password):
        raise HTTPException(
            status_code=401, detail="Usuário ou senha inválidos."
        )

    token = create_session(username)

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


@app.get("/me")
def current_user(request: Request):
    username = _current_user(request)
    if not username:
        raise HTTPException(status_code=401, detail="Não autenticado.")
    return {"username": username}


@app.post("/logout")
def logout(request: Request):
    token = request.cookies.get(SESSION_COOKIE)
    destroy_session(token)

    response = JSONResponse({"status": "ok"})
    response.delete_cookie(SESSION_COOKIE)
    return response


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models")
def models():
    return {"models": SUPPORTED_MODELS}


@app.post("/ask")
def ask_question(request: Request, question: Question):
    # Mantido síncrono de propósito: a cadeia inteira (embeddings,
    # Qdrant, geração no Ollama) usa `requests`, que é bloqueante.
    # Um `async def` aqui sem trocar isso por httpx.AsyncClient
    # travaria o event loop inteiro em vez de só uma thread do
    # pool — seria pior, não melhor.
    username = _current_user(request)
    if not username:
        raise HTTPException(status_code=401, detail="Não autenticado.")

    if question.model not in SUPPORTED_MODELS:
        raise HTTPException(status_code=400, detail="Modelo não suportado.")

    try:
        conversations.add_message(
            question.conversation_id, username, "user", question.question
        )
    except conversations.ConversationNotFound:
        raise HTTPException(status_code=404, detail="Conversa não encontrada.")

    try:
        result = ask(question.question, question.model)
    except Exception:
        logger.exception(
            "Falha ao processar pergunta (usuário=%s, modelo=%s)",
            username,
            question.model,
        )
        raise HTTPException(
            status_code=503,
            detail=(
                "Não consegui processar a pergunta agora (modelo "
                "indisponível ou serviço fora do ar). Tente novamente "
                "em instantes."
            ),
        )

    conversations.add_message(
        question.conversation_id,
        username,
        "assistant",
        result["answer"],
        result["sources"],
    )
    conversations.set_first_message_as_title(
        question.conversation_id, username, question.question
    )

    return {
        "question": question.question,
        "model": question.model,
        "username": username,
        "answer": result["answer"],
        "sources": result["sources"],
        "conversation_id": question.conversation_id,
    }


# ============================================================
# CONVERSAS
# ============================================================
@app.get("/conversations")
def list_conversations_endpoint(request: Request):
    username = _current_user(request)
    if not username:
        raise HTTPException(status_code=401, detail="Não autenticado.")

    return {"conversations": conversations.list_conversations(username)}


@app.post("/conversation")
def create_conversation_endpoint(request: Request, body: ConversationCreateRequest):
    username = _current_user(request)
    if not username:
        raise HTTPException(status_code=401, detail="Não autenticado.")

    return conversations.create_conversation(username, body.title)


@app.get("/conversation/{conversation_id}")
def get_conversation_endpoint(conversation_id: str, request: Request):
    username = _current_user(request)
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
    username = _current_user(request)
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
    username = _current_user(request)
    if not username:
        raise HTTPException(status_code=401, detail="Não autenticado.")

    try:
        conversations.delete_conversation(conversation_id, username)
    except conversations.ConversationNotFound:
        raise HTTPException(status_code=404, detail="Conversa não encontrada.")

    return {"status": "ok"}
