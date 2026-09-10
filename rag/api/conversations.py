"""
Persistência de conversas (histórico de chat) em SQLite.

Cada conversa pertence a um `username` (o mesmo username da sessão
autenticada) — todas as operações validam essa posse antes de ler,
alterar ou apagar, então um usuário nunca acessa a conversa de outro.
"""
import json
import logging
import sqlite3
import time
import uuid
from pathlib import Path

from rag.config import CONVERSATIONS_DB_PATH, CONVERSATION_TITLE_MAX_CHARS

logger = logging.getLogger(__name__)

DEFAULT_TITLE = "Nova conversa"


class ConversationNotFound(Exception):
    """Conversa não existe ou não pertence a esse usuário."""


def _get_connection() -> sqlite3.Connection:
    Path(CONVERSATIONS_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(CONVERSATIONS_DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    """Cria as tabelas se ainda não existirem. Chamado uma vez na
    inicialização da API (server.py)."""
    conn = _get_connection()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                title TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL
                    REFERENCES conversations(id) ON DELETE CASCADE,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                sources_json TEXT,
                created_at REAL NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_conversations_username "
            "ON conversations(username)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_conversation "
            "ON messages(conversation_id)"
        )
        conn.commit()
        logger.info("Banco de conversas pronto em %s", CONVERSATIONS_DB_PATH)
    finally:
        conn.close()


def _belongs_to_user(conn: sqlite3.Connection, conversation_id: str, username: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM conversations WHERE id = ? AND username = ?",
        (conversation_id, username),
    ).fetchone()
    return row is not None


def create_conversation(username: str, title: str | None = None) -> dict:
    conversation_id = str(uuid.uuid4())
    now = time.time()
    title = (title or DEFAULT_TITLE).strip()[:CONVERSATION_TITLE_MAX_CHARS] or DEFAULT_TITLE

    conn = _get_connection()
    try:
        conn.execute(
            "INSERT INTO conversations (id, username, title, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (conversation_id, username, title, now, now),
        )
        conn.commit()
    finally:
        conn.close()

    return {"id": conversation_id, "title": title, "created_at": now, "updated_at": now}


def list_conversations(username: str) -> list[dict]:
    conn = _get_connection()
    try:
        rows = conn.execute(
            "SELECT id, title, created_at, updated_at FROM conversations "
            "WHERE username = ? ORDER BY updated_at DESC",
            (username,),
        ).fetchall()
    finally:
        conn.close()

    return [dict(row) for row in rows]


def get_conversation(conversation_id: str, username: str) -> dict:
    conn = _get_connection()
    try:
        conv = conn.execute(
            "SELECT id, title, created_at, updated_at FROM conversations "
            "WHERE id = ? AND username = ?",
            (conversation_id, username),
        ).fetchone()

        if not conv:
            raise ConversationNotFound(conversation_id)

        messages = conn.execute(
            "SELECT role, content, sources_json, created_at FROM messages "
            "WHERE conversation_id = ? ORDER BY id ASC",
            (conversation_id,),
        ).fetchall()
    finally:
        conn.close()

    return {
        "id": conv["id"],
        "title": conv["title"],
        "created_at": conv["created_at"],
        "updated_at": conv["updated_at"],
        "messages": [
            {
                "role": m["role"],
                "content": m["content"],
                "sources": json.loads(m["sources_json"]) if m["sources_json"] else [],
                "created_at": m["created_at"],
            }
            for m in messages
        ],
    }


def add_message(
    conversation_id: str,
    username: str,
    role: str,
    content: str,
    sources: list | None = None,
) -> None:
    now = time.time()
    sources_json = json.dumps(sources) if sources else None

    conn = _get_connection()
    try:
        if not _belongs_to_user(conn, conversation_id, username):
            raise ConversationNotFound(conversation_id)

        conn.execute(
            "INSERT INTO messages (conversation_id, role, content, sources_json, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (conversation_id, role, content, sources_json, now),
        )
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (now, conversation_id),
        )
        conn.commit()
    finally:
        conn.close()


def set_first_message_as_title(conversation_id: str, username: str, question: str) -> None:
    """Se a conversa ainda estiver com o título padrão, usa a
    primeira pergunta (truncada) como título automático."""
    title = " ".join(question.strip().split())[:CONVERSATION_TITLE_MAX_CHARS]
    if not title:
        return

    conn = _get_connection()
    try:
        conv = conn.execute(
            "SELECT title FROM conversations WHERE id = ? AND username = ?",
            (conversation_id, username),
        ).fetchone()

        if not conv:
            raise ConversationNotFound(conversation_id)

        if conv["title"] == DEFAULT_TITLE:
            conn.execute(
                "UPDATE conversations SET title = ? WHERE id = ?",
                (title, conversation_id),
            )
            conn.commit()
    finally:
        conn.close()


def rename_conversation(conversation_id: str, username: str, new_title: str) -> None:
    new_title = " ".join(new_title.strip().split())[:CONVERSATION_TITLE_MAX_CHARS]
    if not new_title:
        raise ValueError("O título não pode ser vazio.")

    conn = _get_connection()
    try:
        if not _belongs_to_user(conn, conversation_id, username):
            raise ConversationNotFound(conversation_id)

        conn.execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
            (new_title, time.time(), conversation_id),
        )
        conn.commit()
    finally:
        conn.close()


def delete_conversation(conversation_id: str, username: str) -> None:
    conn = _get_connection()
    try:
        if not _belongs_to_user(conn, conversation_id, username):
            raise ConversationNotFound(conversation_id)

        # ON DELETE CASCADE (PRAGMA foreign_keys=ON) já apaga as
        # mensagens junto.
        conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        conn.commit()
    finally:
        conn.close()
