import logging
import secrets
import time

from rag.config import (
    AUTH_MODE,
    LOCAL_USERNAME,
    LOCAL_PASSWORD,
    SESSION_TTL_SECONDS,
)

logger = logging.getLogger(__name__)

# token -> {"username": str, "expires_at": float}
# NOTA: ainda em memória — sessões somem se o processo reiniciar.
# Se isso virar um problema (deploy frequente, múltiplos workers),
# mover para Redis ou um arquivo é o próximo passo natural.
SESSIONS = {}


def authenticate_ldap(username: str, password: str) -> bool:
    if AUTH_MODE == "local":
        return secrets.compare_digest(
            username, LOCAL_USERNAME
        ) and secrets.compare_digest(password, LOCAL_PASSWORD)

    if AUTH_MODE == "ldap":
        # Placeholder: autenticação LDAP real ainda não implementada.
        # Nega explicitamente em vez de deixar alguém achar que está
        # funcionando.
        logger.error(
            "RAG_AUTH_MODE=ldap configurado, mas a autenticação LDAP "
            "não está implementada. Negando acesso."
        )
        return False

    logger.warning("RAG_AUTH_MODE desconhecido: %r", AUTH_MODE)
    return False


def create_session(username: str) -> str:
    token = secrets.token_urlsafe(32)
    SESSIONS[token] = {
        "username": username,
        "expires_at": time.time() + SESSION_TTL_SECONDS,
    }
    return token


def validate_session(token: str | None) -> str | None:
    if not token:
        return None

    session = SESSIONS.get(token)
    if not session:
        return None

    if time.time() > session["expires_at"]:
        SESSIONS.pop(token, None)
        return None

    return session["username"]


def destroy_session(token: str | None) -> None:
    """Invalida a sessão no servidor. Antes, o logout só apagava o
    cookie do navegador — o token continuava válido para sempre."""
    if token:
        SESSIONS.pop(token, None)