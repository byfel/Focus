import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path


# ============================================================
# CONFIGURAÇÃO
# ============================================================

BASE_DIR = Path("/opt/ai/rag")
DATABASE_DIR = BASE_DIR / "database"
DB_FILE = DATABASE_DIR / "conversations.db"


# ============================================================
# UTILITÁRIOS
# ============================================================

def _utc_now() -> str:
    """
    Retorna timestamp UTC em formato ISO 8601.
    """
    return datetime.now(timezone.utc).isoformat()


def _get_connection() -> sqlite3.Connection:
    """
    Abre conexão com o banco SQLite.
    """
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row

    return conn


# ============================================================
# BANCO DE DADOS
# ============================================================

def init_db() -> None:
    """
    Cria as tabelas necessárias caso ainda não existam.
    """

    with _get_connection() as conn:

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                title TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,

                FOREIGN KEY (conversation_id)
                    REFERENCES conversations(id)
                    ON DELETE CASCADE
            )
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_conversations_username
            ON conversations(username)
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_messages_conversation
            ON messages(conversation_id)
            """
        )

        conn.commit()


# ============================================================
# CONVERSAS
# ============================================================

def create_conversation(
    username: str,
    title: str | None = None,
) -> str:
    """
    Cria uma nova conversa.

    Retorna:
        ID da conversa
    """

    conversation_id = str(uuid.uuid4())
    now = _utc_now()

    if title is None:
        title = "Nova conversa"

    with _get_connection() as conn:

        conn.execute(
            """
            INSERT INTO conversations (
                id,
                username,
                title,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                conversation_id,
                username,
                title,
                now,
                now,
            ),
        )

        conn.commit()

    return conversation_id


def list_conversations(username: str) -> list[dict]:
    """
    Retorna todas as conversas pertencentes ao usuário.

    Ordenação:
        conversa mais recentemente atualizada primeiro.
    """

    with _get_connection() as conn:

        rows = conn.execute(
            """
            SELECT
                id,
                username,
                title,
                created_at,
                updated_at
            FROM conversations
            WHERE username = ?
            ORDER BY updated_at DESC
            """,
            (username,),
        ).fetchall()

    return [dict(row) for row in rows]


def get_conversation(
    conversation_id: str,
    username: str,
) -> dict | None:
    """
    Busca uma conversa garantindo que ela pertence ao usuário.
    """

    with _get_connection() as conn:

        row = conn.execute(
            """
            SELECT
                id,
                username,
                title,
                created_at,
                updated_at
            FROM conversations
            WHERE id = ?
              AND username = ?
            """,
            (
                conversation_id,
                username,
            ),
        ).fetchone()

    if row is None:
        return None

    return dict(row)


def delete_conversation(
    conversation_id: str,
    username: str,
) -> bool:
    """
    Remove uma conversa pertencente ao usuário.

    Retorna:
        True  -> removida
        False -> não encontrada
    """

    with _get_connection() as conn:

        cursor = conn.execute(
            """
            DELETE FROM conversations
            WHERE id = ?
              AND username = ?
            """,
            (
                conversation_id,
                username,
            ),
        )

        conn.commit()

    return cursor.rowcount > 0


def update_conversation_title(
    conversation_id: str,
    username: str,
    title: str,
) -> bool:
    """
    Atualiza o título de uma conversa.
    """

    now = _utc_now()

    with _get_connection() as conn:

        cursor = conn.execute(
            """
            UPDATE conversations
            SET
                title = ?,
                updated_at = ?
            WHERE id = ?
              AND username = ?
            """,
            (
                title,
                now,
                conversation_id,
                username,
            ),
        )

        conn.commit()

    return cursor.rowcount > 0


# ============================================================
# MENSAGENS
# ============================================================

def add_message(
    conversation_id: str,
    username: str,
    role: str,
    content: str,
) -> str:
    """
    Adiciona uma mensagem à conversa.

    role esperado:
        user
        assistant

    Retorna:
        ID da mensagem
    """

    if role not in {"user", "assistant"}:
        raise ValueError(
            f"Role inválido: {role}. "
            "Use 'user' ou 'assistant'."
        )

    # Primeiro verificamos se a conversa pertence ao usuário.
    conversation = get_conversation(
        conversation_id,
        username,
    )

    if conversation is None:
        raise ValueError(
            "Conversa não encontrada ou não pertence ao usuário."
        )

    message_id = str(uuid.uuid4())
    now = _utc_now()

    with _get_connection() as conn:

        conn.execute(
            """
            INSERT INTO messages (
                id,
                conversation_id,
                role,
                content,
                created_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                message_id,
                conversation_id,
                role,
                content,
                now,
            ),
        )

        conn.execute(
            """
            UPDATE conversations
            SET updated_at = ?
            WHERE id = ?
            """,
            (
                now,
                conversation_id,
            ),
        )

        conn.commit()

    return message_id


def get_messages(
    conversation_id: str,
    username: str,
    limit: int = 20,
) -> list[dict]:
    """
    Retorna as mensagens de uma conversa.

    As mensagens mais recentes são consideradas primeiro
    para aplicar o limite, mas o resultado final retorna
    na ordem cronológica normal.
    """

    # Garante que o usuário possui acesso à conversa.
    conversation = get_conversation(
        conversation_id,
        username,
    )

    if conversation is None:
        return []

    if limit <= 0:
        return []

    with _get_connection() as conn:

        rows = conn.execute(
            """
            SELECT
                id,
                conversation_id,
                role,
                content,
                created_at
            FROM messages
            WHERE conversation_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (
                conversation_id,
                limit,
            ),
        ).fetchall()

    messages = [dict(row) for row in rows]

    # Voltamos para ordem cronológica.
    messages.reverse()

    return messages


# ============================================================
# INICIALIZAÇÃO AUTOMÁTICA
# ============================================================

init_db()
