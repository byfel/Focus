"""
Testes de auth.py — sessão, expiração e credenciais locais.
Não precisam de Qdrant nem Ollama rodando.

Ajuste o import abaixo conforme o caminho real do pacote no seu
projeto (ex.: `from rag.api import auth` ou `from api import auth`,
dependendo de como o `auth.py` está posicionado no seu /opt/ai/rag).

Rodar com:
    pip install pytest --break-system-packages
    pytest tests/test_auth.py -v
"""
import time

import pytest

from api import auth  # <-- ajuste este import se necessário


@pytest.fixture(autouse=True)
def clear_sessions():
    """Garante que um teste não vaze sessão pro próximo."""
    auth.SESSIONS.clear()
    yield
    auth.SESSIONS.clear()


def test_create_and_validate_session():
    token = auth.create_session("felipe")
    assert auth.validate_session(token) == "felipe"


def test_validate_session_unknown_token():
    assert auth.validate_session("token-que-nao-existe") is None


def test_validate_session_empty_or_none_token():
    assert auth.validate_session(None) is None
    assert auth.validate_session("") is None


def test_session_expires_and_is_removed():
    token = auth.create_session("felipe")
    # Força expiração sem precisar de time.sleep real
    auth.SESSIONS[token]["expires_at"] = time.time() - 1

    assert auth.validate_session(token) is None
    # A sessão expirada deve ser limpa do dicionário, não só ignorada
    assert token not in auth.SESSIONS


def test_destroy_session_invalidates_immediately():
    """Este é o teste do bug que corrigimos: antes, logout só
    apagava o cookie do navegador e o token continuava válido
    para sempre em SESSIONS."""
    token = auth.create_session("felipe")
    auth.destroy_session(token)
    assert auth.validate_session(token) is None


def test_destroy_session_with_none_does_not_raise():
    auth.destroy_session(None)  # não deve levantar exceção


def test_authenticate_local_mode_correct_and_wrong_credentials(monkeypatch):
    monkeypatch.setattr(auth, "AUTH_MODE", "local")
    monkeypatch.setattr(auth, "LOCAL_USERNAME", "admin")
    monkeypatch.setattr(auth, "LOCAL_PASSWORD", "admin123")

    assert auth.authenticate_ldap("admin", "admin123") is True
    assert auth.authenticate_ldap("admin", "senha_errada") is False
    assert auth.authenticate_ldap("outro_user", "admin123") is False


def test_authenticate_ldap_mode_denies_explicitly(monkeypatch):
    """Enquanto o LDAP real não estiver implementado, esse modo
    deve negar acesso — não travar, não aceitar por engano."""
    monkeypatch.setattr(auth, "AUTH_MODE", "ldap")
    assert auth.authenticate_ldap("qualquer", "coisa") is False
