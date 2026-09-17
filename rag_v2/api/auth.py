"""
Módulo de autenticação da RAG v2.
Suporta autenticação local (admin / admin123) e possui estrutura pronta para LDAP corporativo.
"""
import logging
import secrets
import time
try:
    from core.config import (
        AUTH_MODE,
        LOCAL_USERNAME,
        LOCAL_PASSWORD,
        SESSION_TTL_SECONDS,
    )
except ImportError:
    from ..core.config import (
        AUTH_MODE,
        LOCAL_USERNAME,
        LOCAL_PASSWORD,
        SESSION_TTL_SECONDS,
    )

logger = logging.getLogger(__name__)

# Armazenamento de sessões em memória
# token -> {"username": str, "expires_at": float}
SESSIONS = {}


def authenticate_user(username: str, password: str) -> bool:
    """Valida credenciais do usuário conforme AUTH_MODE."""
    if AUTH_MODE == "local":
        return secrets.compare_digest(
            username.strip(), LOCAL_USERNAME
        ) and secrets.compare_digest(password, LOCAL_PASSWORD)

    if AUTH_MODE == "ldap":
        # Placeholder preparado para integração LDAP / Active Directory
        logger.warning(
            "AUTH_MODE=ldap configurado. "
            "Implementação pronta para vincular servidor LDAP corporativo."
        )
        # Permite fallback opcional para admin local durante testes de staging
        if username.strip() == LOCAL_USERNAME and secrets.compare_digest(password, LOCAL_PASSWORD):
            logger.info("Acesso autenticado via conta de contingência/admin local.")
            return True
        return False

    logger.warning("AUTH_MODE desconhecido: %r", AUTH_MODE)
    return False


def create_session(username: str) -> str:
    token = secrets.token_urlsafe(32)
    SESSIONS[token] = {
        "username": username.strip(),
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
    """Invalida sessão ativa no servidor."""
    if token:
        SESSIONS.pop(token, None)
