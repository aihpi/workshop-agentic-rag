"""Per-user knowledge bases, system prompt, and starter questions.

Extends the SQLite database at CHAT_DB_PATH with four tables:

    knowledge_bases  — one row per named KB owned by a user
    kb_documents     — one row per uploaded document inside a KB
    user_settings    — per-user system prompt override
    user_starters    — per-user starter questions (replaces env default)
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Stable pseudo-id for the legacy shared collection so it can be surfaced as a
# read-only KB in every user's KB list without a real DB row.
SHARED_KB_ID = "shared"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_user_kb_db(db_path: Path) -> None:
    with _connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS knowledge_bases (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                qdrant_collection TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                UNIQUE(user_id, name)
            );

            CREATE INDEX IF NOT EXISTS idx_knowledge_bases_user
            ON knowledge_bases(user_id);

            CREATE TABLE IF NOT EXISTS kb_documents (
                id TEXT PRIMARY KEY,
                kb_id TEXT NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
                file_name TEXT NOT NULL,
                size_bytes INTEGER NOT NULL DEFAULT 0,
                chunk_count INTEGER NOT NULL DEFAULT 0,
                uploaded_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_kb_documents_kb
            ON kb_documents(kb_id);

            CREATE TABLE IF NOT EXISTS user_settings (
                user_id TEXT PRIMARY KEY,
                system_prompt TEXT,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS user_starters (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                label TEXT NOT NULL,
                message TEXT NOT NULL,
                position INTEGER NOT NULL,
                UNIQUE(user_id, position)
            );

            CREATE INDEX IF NOT EXISTS idx_user_starters_user
            ON user_starters(user_id, position);
            """
        )
        conn.commit()


# ---------------------------------------------------------------------------
# Knowledge bases
# ---------------------------------------------------------------------------

def _kb_collection_name(kb_id: str) -> str:
    return f"kb_{kb_id.replace('-', '')}"


def create_kb(
    db_path: Path,
    user_id: str,
    name: str,
    description: str = "",
) -> dict[str, Any]:
    kb_id = str(uuid.uuid4())
    collection = _kb_collection_name(kb_id)
    now = _utc_now_iso()
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO knowledge_bases (id, user_id, name, description, qdrant_collection, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (kb_id, user_id, name, description, collection, now),
        )
        conn.commit()
    return {
        "id": kb_id,
        "user_id": user_id,
        "name": name,
        "description": description,
        "qdrant_collection": collection,
        "created_at": now,
        "readonly": False,
    }


def list_kbs(db_path: Path, user_id: str) -> list[dict[str, Any]]:
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT kb.id, kb.user_id, kb.name, kb.description, kb.qdrant_collection, kb.created_at,
                   COUNT(d.id) AS document_count
            FROM knowledge_bases kb
            LEFT JOIN kb_documents d ON d.kb_id = kb.id
            WHERE kb.user_id = ?
            GROUP BY kb.id
            ORDER BY kb.created_at ASC
            """,
            (user_id,),
        ).fetchall()
    out = []
    for row in rows:
        item = dict(row)
        item["readonly"] = False
        out.append(item)
    return out


def get_kb(db_path: Path, kb_id: str) -> dict[str, Any] | None:
    with _connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT id, user_id, name, description, qdrant_collection, created_at
            FROM knowledge_bases
            WHERE id = ?
            """,
            (kb_id,),
        ).fetchone()
    if not row:
        return None
    item = dict(row)
    item["readonly"] = False
    return item


def update_kb(
    db_path: Path,
    kb_id: str,
    *,
    name: str | None = None,
    description: str | None = None,
) -> None:
    if name is None and description is None:
        return
    fields: list[str] = []
    params: list[Any] = []
    if name is not None:
        fields.append("name = ?")
        params.append(name)
    if description is not None:
        fields.append("description = ?")
        params.append(description)
    params.append(kb_id)
    with _connect(db_path) as conn:
        conn.execute(
            f"UPDATE knowledge_bases SET {', '.join(fields)} WHERE id = ?",
            params,
        )
        conn.commit()


def delete_kb(db_path: Path, kb_id: str) -> dict[str, Any] | None:
    """Delete the KB row (cascades to documents). Returns the deleted row."""
    kb = get_kb(db_path, kb_id)
    if kb is None:
        return None
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM knowledge_bases WHERE id = ?", (kb_id,))
        conn.commit()
    return kb


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

def record_document(
    db_path: Path,
    kb_id: str,
    file_name: str,
    size_bytes: int,
    chunk_count: int,
) -> dict[str, Any]:
    doc_id = str(uuid.uuid4())
    now = _utc_now_iso()
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO kb_documents (id, kb_id, file_name, size_bytes, chunk_count, uploaded_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (doc_id, kb_id, file_name, size_bytes, chunk_count, now),
        )
        conn.commit()
    return {
        "id": doc_id,
        "kb_id": kb_id,
        "file_name": file_name,
        "size_bytes": size_bytes,
        "chunk_count": chunk_count,
        "uploaded_at": now,
    }


def list_documents(db_path: Path, kb_id: str) -> list[dict[str, Any]]:
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT id, kb_id, file_name, size_bytes, chunk_count, uploaded_at
            FROM kb_documents
            WHERE kb_id = ?
            ORDER BY uploaded_at DESC
            """,
            (kb_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_document(db_path: Path, doc_id: str) -> dict[str, Any] | None:
    with _connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT id, kb_id, file_name, size_bytes, chunk_count, uploaded_at
            FROM kb_documents
            WHERE id = ?
            """,
            (doc_id,),
        ).fetchone()
    return dict(row) if row else None


def delete_document(db_path: Path, doc_id: str) -> dict[str, Any] | None:
    doc = get_document(db_path, doc_id)
    if doc is None:
        return None
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM kb_documents WHERE id = ?", (doc_id,))
        conn.commit()
    return doc


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

def get_user_system_prompt(db_path: Path, user_id: str) -> str | None:
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT system_prompt FROM user_settings WHERE user_id = ?",
            (user_id,),
        ).fetchone()
    if not row:
        return None
    value = row["system_prompt"]
    return value if isinstance(value, str) and value.strip() else None


def set_user_system_prompt(db_path: Path, user_id: str, prompt: str | None) -> None:
    now = _utc_now_iso()
    normalized = prompt.strip() if isinstance(prompt, str) else ""
    stored: str | None = normalized if normalized else None
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO user_settings (user_id, system_prompt, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                system_prompt = excluded.system_prompt,
                updated_at = excluded.updated_at
            """,
            (user_id, stored, now),
        )
        conn.commit()


# ---------------------------------------------------------------------------
# Starter questions
# ---------------------------------------------------------------------------

def list_user_starters(db_path: Path, user_id: str) -> list[dict[str, Any]]:
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT id, label, message, position
            FROM user_starters
            WHERE user_id = ?
            ORDER BY position ASC
            """,
            (user_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def replace_user_starters(
    db_path: Path,
    user_id: str,
    starters: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """Replace all starters for a user in one transaction. Empty list clears them."""
    cleaned: list[tuple[str, str, str, int]] = []
    for idx, item in enumerate(starters):
        message = (item.get("message") or "").strip()
        if not message:
            continue
        label = (item.get("label") or "").strip() or message
        cleaned.append((str(uuid.uuid4()), label, message, idx))

    with _connect(db_path) as conn:
        conn.execute("DELETE FROM user_starters WHERE user_id = ?", (user_id,))
        for starter_id, label, message, position in cleaned:
            conn.execute(
                """
                INSERT INTO user_starters (id, user_id, label, message, position)
                VALUES (?, ?, ?, ?, ?)
                """,
                (starter_id, user_id, label, message, position),
            )
        conn.commit()
    return list_user_starters(db_path, user_id)
