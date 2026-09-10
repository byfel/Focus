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

from rag.conversation import (
    create_conversation,
    list_conversations,
    get_conversation,
    get_messages,
    add_message,
    delete_conversation,
    update_conversation_title,
)

from .auth import (
    authenticate_ldap,
    create_session,
    validate_session,
    destroy_session,
)


logger = logging.getLogger(__name__)


app = FastAPI(
    title="SUPORTE AI RAG",
    description="Assistente técnico baseado em documentação",
    version="1.0.0",
)


# ============================================================
# MODELOS
# ============================================================

class LoginRequest(BaseModel):
    username: str
    password: str


class Question(BaseModel):
    question: str
    model: str = DEFAULT_LLM_MODEL
    conversation_id: str | None = None


class ConversationCreate(BaseModel):
    title: str | None = None


# ============================================================
# AUTENTICAÇÃO
# ============================================================

def _current_user(request: Request) -> str | None:
    token = request.cookies.get(SESSION_COOKIE)
    return validate_session(token)


# ============================================================
# PÁGINAS
# ============================================================

@app.get("/")
def login_page():
    return FileResponse("/opt/ai/rag/web/login.html")


@app.get("/app")
def application_page(request: Request):
    username = _current_user(request)

    if not username:
        return FileResponse("/opt/ai/rag/web/login.html")

    return FileResponse("/opt/ai/rag/web/index.html")


# ============================================================
# LOGIN
# ============================================================

@app.post("/login")
def login(request: LoginRequest):

    username = request.username.strip()
    password = request.password

    if not username or not password:
        raise HTTPException(
            status_code=400,
            detail="Usuário e senha são obrigatórios.",
        )

    if not authenticate_ldap(username, password):
        raise HTTPException(
            status_code=401,
            detail="Usuário ou senha inválidos.",
        )

    token = create_session(username)

    response = JSONResponse(
        {
            "status": "ok",
            "username": username,
        }
    )

    response.set_cookie(
        key=SESSION_COOKIE,
        value=token,
        httponly=True,
        secure=SESSION_COOKIE_SECURE,
        samesite="lax",
        max_age=SESSION_TTL_SECONDS,
    )

    return response


# ============================================================
# USUÁRIO ATUAL
# ============================================================

@app.get("/me")
def current_user(request: Request):

    username = _current_user(request)

    if not username:
        raise HTTPException(
            status_code=401,
            detail="Não autenticado.",
        )

    return {
        "username": username,
    }


# ============================================================
# LOGOUT
# ============================================================

@app.post("/logout")
def logout(request: Request):

    token = request.cookies.get(SESSION_COOKIE)

    destroy_session(token)

    response = JSONResponse(
        {
            "status": "ok",
        }
    )

    response.delete_cookie(SESSION_COOKIE)

    return response


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
    }


# ============================================================
# MODELOS
# ============================================================

@app.get("/models")
def models():
    return {
        "models": SUPPORTED_MODELS,
    }


# ============================================================
# CONVERSAS
# ============================================================

@app.post("/conversation")
def create_new_conversation(
    request: Request,
    conversation: ConversationCreate,
):
    """
    Cria uma nova conversa para o usuário autenticado.
    """

    username = _current_user(request)

    if not username:
        raise HTTPException(
            status_code=401,
            detail="Não autenticado.",
        )

    conversation_id = create_conversation(
        username=username,
        title=conversation.title,
    )

    result = get_conversation(
        conversation_id,
        username,
    )

    return result


@app.get("/conversations")
def get_conversations(request: Request):
    """
    Lista as conversas do usuário autenticado.
    """

    username = _current_user(request)

    if not username:
        raise HTTPException(
            status_code=401,
            detail="Não autenticado.",
        )

    return {
        "conversations": list_conversations(username),
    }


@app.get("/conversation/{conversation_id}")
def get_conversation_detail(
    request: Request,
    conversation_id: str,
):
    """
    Retorna uma conversa e suas mensagens.
    """

    username = _current_user(request)

    if not username:
        raise HTTPException(
            status_code=401,
            detail="Não autenticado.",
        )

    conversation = get_conversation(
        conversation_id,
        username,
    )

    if conversation is None:
        raise HTTPException(
            status_code=404,
            detail="Conversa não encontrada.",
        )

    messages = get_messages(
        conversation_id,
        username,
        limit=100,
    )

    return {
        "conversation": conversation,
        "messages": messages,
    }


@app.delete("/conversation/{conversation_id}")
def remove_conversation(
    request: Request,
    conversation_id: str,
):
    """
    Remove uma conversa do usuário.
    """

    username = _current_user(request)

    if not username:
        raise HTTPException(
            status_code=401,
            detail="Não autenticado.",
        )

    deleted = delete_conversation(
        conversation_id,
        username,
    )

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="Conversa não encontrada.",
        )

    return {
        "status": "ok",
    }


@app.patch("/conversation/{conversation_id}")
def rename_conversation(
    request: Request,
    conversation_id: str,
    conversation: ConversationCreate,
):
    """
    Altera o título de uma conversa.
    """

    username = _current_user(request)

    if not username:
        raise HTTPException(
            status_code=401,
            detail="Não autenticado.",
        )

    if not conversation.title:
        raise HTTPException(
            status_code=400,
            detail="Título não informado.",
        )

    updated = update_conversation_title(
        conversation_id,
        username,
        conversation.title.strip(),
    )

    if not updated:
        raise HTTPException(
            status_code=404,
            detail="Conversa não encontrada.",
        )

    return get_conversation(
        conversation_id,
        username,
    )


# ============================================================
# PERGUNTA / RAG
# ============================================================

@app.post("/ask")
def ask_question(
    request: Request,
    question: Question,
):
    """
    Processa uma pergunta dentro de uma conversa.

    Fluxo:

        usuário
            ↓
        salva pergunta
            ↓
        RAG
            ↓
        salva resposta
            ↓
        retorna resultado
    """

    username = _current_user(request)

    if not username:
        raise HTTPException(
            status_code=401,
            detail="Não autenticado.",
        )

    if question.model not in SUPPORTED_MODELS:
        raise HTTPException(
            status_code=400,
            detail="Modelo não suportado.",
        )

    text = question.question.strip()

    if not text:
        raise HTTPException(
            status_code=400,
            detail="A pergunta não pode estar vazia.",
        )

    # --------------------------------------------------------
    # CONVERSA
    # --------------------------------------------------------

    conversation_id = question.conversation_id

    if conversation_id:

        conversation = get_conversation(
            conversation_id,
            username,
        )

        if conversation is None:
            raise HTTPException(
                status_code=404,
                detail="Conversa não encontrada.",
            )

    else:

        conversation_id = create_conversation(
            username=username,
            title=text[:80],
        )

    # --------------------------------------------------------
    # SALVA PERGUNTA
    # --------------------------------------------------------

    try:

        message_id = add_message(
            conversation_id=conversation_id,
            username=username,
            role="user",
            content=text,
        )

    except Exception:
        logger.exception(
            "Falha ao salvar pergunta "
            "(usuário=%s, conversa=%s)",
            username,
            conversation_id,
        )

        raise HTTPException(
            status_code=500,
            detail="Não foi possível salvar a pergunta.",
        )

    # --------------------------------------------------------
    # EXECUTA RAG
    # --------------------------------------------------------

    try:

        result = ask(
            text,
            question.model,
        )

    except Exception:

        logger.exception(
            "Falha ao processar pergunta "
            "(usuário=%s, modelo=%s, conversa=%s)",
            username,
            question.model,
            conversation_id,
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "Não consegui processar a pergunta agora "
                "(modelo indisponível ou serviço fora do ar). "
                "Tente novamente em instantes."
            ),
        )

    # --------------------------------------------------------
    # SALVA RESPOSTA
    # --------------------------------------------------------

    try:

        add_message(
            conversation_id=conversation_id,
            username=username,
            role="assistant",
            content=result["answer"],
        )

    except Exception:

        logger.exception(
            "Falha ao salvar resposta "
            "(usuário=%s, conversa=%s)",
            username,
            conversation_id,
        )

        # A resposta ainda pode ser devolvida ao usuário.
        # O erro é registrado no log.

    # --------------------------------------------------------
    # RESPOSTA
    # --------------------------------------------------------

    return {
        "conversation_id": conversation_id,
        "message_id": message_id,
        "question": text,
        "model": question.model,
        "username": username,
        "answer": result["answer"],
        "sources": result["sources"],
    }