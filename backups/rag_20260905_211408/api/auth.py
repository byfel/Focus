import secrets
import os


# ============================================================
# CONFIGURAÇÃO
# ============================================================

AUTH_MODE = os.getenv(
    "RAG_AUTH_MODE",
    "local"
)

LOCAL_USERNAME = os.getenv(
    "RAG_LOCAL_USERNAME",
    "admin"
)

LOCAL_PASSWORD = os.getenv(
    "RAG_LOCAL_PASSWORD",
    "admin123"
)

SESSION_COOKIE = "rag_session"


# ============================================================
# SESSÕES
# ============================================================

SESSIONS = {}


# ============================================================
# AUTENTICAÇÃO
# ============================================================

def authenticate_ldap(username, password):

    # Modo temporário sem LDAP
    if AUTH_MODE == "local":

        return (
            secrets.compare_digest(
                username,
                LOCAL_USERNAME
            )
            and
            secrets.compare_digest(
                password,
                LOCAL_PASSWORD
            )
        )

    # Futuramente:
    # autenticação LDAP da empresa

    if AUTH_MODE == "ldap":

        return False

    return False


# ============================================================
# CRIAR SESSÃO
# ============================================================

def create_session(username):

    token = secrets.token_urlsafe(32)

    SESSIONS[token] = username

    return token


# ============================================================
# VALIDAR SESSÃO
# ============================================================

def validate_session(token):

    if not token:
        return None

    return SESSIONS.get(token)