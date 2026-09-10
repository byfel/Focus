from fastapi import (
    FastAPI,
    Request,
    HTTPException
)

from fastapi.responses import (
    FileResponse,
    JSONResponse
)

from pydantic import BaseModel

from rag.pipeline import (
    ask,
    SUPPORTED_MODELS
)

from .auth import (
    authenticate_ldap,
    create_session,
    validate_session,
    SESSION_COOKIE
)


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="SUPORTE AI RAG",
    description="Assistente técnico baseado em documentação",
    version="1.0.0"
)


# ============================================================
# MODELOS
# ============================================================

class LoginRequest(BaseModel):

    username: str
    password: str


class Question(BaseModel):

    question: str
    model: str = "gemma4:12b"


# ============================================================
# PÁGINA DE LOGIN
# ============================================================

@app.get("/")
def login_page():

    return FileResponse(
        "/opt/ai/rag/web/login.html"
    )


# ============================================================
# APLICAÇÃO PRINCIPAL
# ============================================================

@app.get("/app")
def application_page(
    request: Request
):

    token = request.cookies.get(
        SESSION_COOKIE
    )

    username = validate_session(
        token
    )

    print("DEBUG /app")
    print("COOKIE:", token)
    print("USERNAME:", username)

    if not username:

        return FileResponse(
            "/opt/ai/rag/web/login.html"
        )

    return FileResponse(
        "/opt/ai/rag/web/index.html"
    )


# ============================================================
# LOGIN API
# ============================================================

@app.post("/login")
def login(
    request: LoginRequest
):

    username = request.username.strip()

    password = request.password

    if not username or not password:

        raise HTTPException(
            status_code=400,
            detail="Usuário e senha são obrigatórios."
        )

    authenticated = authenticate_ldap(
        username,
        password
    )

    if not authenticated:

        raise HTTPException(
            status_code=401,
            detail="Usuário ou senha inválidos."
        )

    token = create_session(
        username
    )

    response = JSONResponse({
        "status": "ok",
        "username": username
    })

    response.set_cookie(
        key=SESSION_COOKIE,
        value=token,
        httponly=True,
        secure=False,
        samesite="lax",
        max_age=28800
    )

    return response


# ============================================================
# USUÁRIO ATUAL
# ============================================================

@app.get("/me")
def current_user(
    request: Request
):

    token = request.cookies.get(
        SESSION_COOKIE
    )

    username = validate_session(
        token
    )

    if not username:

        raise HTTPException(
            status_code=401,
            detail="Não autenticado."
        )

    return {
        "username": username
    }


# ============================================================
# LOGOUT
# ============================================================

@app.post("/logout")
def logout():

    response = JSONResponse({
        "status": "ok"
    })

    response.delete_cookie(
        SESSION_COOKIE
    )

    return response


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok"
    }


# ============================================================
# MODELOS DISPONÍVEIS
# ============================================================

@app.get("/models")
def models():

    return {
        "models": SUPPORTED_MODELS
    }


# ============================================================
# RAG
# ============================================================

@app.post("/ask")
def ask_question(
    request: Request,
    question: Question
):

    token = request.cookies.get(
        SESSION_COOKIE
    )

    username = validate_session(
        token
    )

    if not username:

        raise HTTPException(
            status_code=401,
            detail="Não autenticado."
        )

    if question.model not in SUPPORTED_MODELS:

        raise HTTPException(
            status_code=400,
            detail="Modelo não suportado."
        )

    result = ask(
        question.question,
        question.model
    )

    return {
        "question": question.question,
        "model": question.model,
        "username": username,
        "answer": result["answer"],
        "sources": result["sources"]
    }