from __future__ import annotations

import csv
import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_chat_db(db_path: Path) -> None:
    with _connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id TEXT PRIMARY KEY,
                title TEXT,
                user_id TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
                role TEXT NOT NULL CHECK(role IN ('system', 'user', 'assistant', 'tool')),
                content TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_chat_messages_session_created
            ON chat_messages(session_id, created_at);

            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                topics_json TEXT NOT NULL DEFAULT '[]',
                topic_embeddings_json TEXT NOT NULL DEFAULT '[]',
                excluded_bausteine_json TEXT NOT NULL DEFAULT '[]',
                selected_chat_profile TEXT,
                message_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_id
            ON chat_sessions(user_id);
            """
        )
        # Migration: add selected_chat_profile column if it doesn't exist
        cursor = conn.execute("PRAGMA table_info(user_profiles)")
        columns = [row[1] for row in cursor.fetchall()]
        if "selected_chat_profile" not in columns:
            conn.execute(
                "ALTER TABLE user_profiles ADD COLUMN selected_chat_profile TEXT"
            )
        conn.commit()


def create_chat_session(
    db_path: Path,
    session_id: str,
    *,
    title: str | None = None,
    user_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    now = _utc_now_iso()
    meta_json = json.dumps(metadata or {}, ensure_ascii=False)
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO chat_sessions (id, title, user_id, metadata_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (session_id, title, user_id, meta_json, now, now),
        )
        conn.commit()


def update_chat_session_metadata(
    db_path: Path,
    session_id: str,
    metadata: dict[str, Any] | None,
) -> None:
    now = _utc_now_iso()
    with _connect(db_path) as conn:
        conn.execute(
            """
            UPDATE chat_sessions
            SET metadata_json = ?, updated_at = ?
            WHERE id = ?
            """,
            (json.dumps(metadata or {}, ensure_ascii=False), now, session_id),
        )
        conn.commit()


def set_session_title_if_missing(db_path: Path, session_id: str, title: str) -> None:
    cleaned = title.strip()
    if not cleaned:
        return
    with _connect(db_path) as conn:
        conn.execute(
            """
            UPDATE chat_sessions
            SET title = COALESCE(NULLIF(title, ''), ?), updated_at = ?
            WHERE id = ?
            """,
            (cleaned, _utc_now_iso(), session_id),
        )
        conn.commit()


def add_chat_message(
    db_path: Path,
    session_id: str,
    role: str,
    content: str,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    now = _utc_now_iso()
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO chat_messages (session_id, role, content, metadata_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, role, content, json.dumps(metadata or {}, ensure_ascii=False), now),
        )
        conn.execute(
            """
            UPDATE chat_sessions
            SET updated_at = ?
            WHERE id = ?
            """,
            (now, session_id),
        )
        conn.commit()


def list_chat_sessions(db_path: Path, limit: int = 20) -> list[dict[str, Any]]:
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT
                s.id,
                COALESCE(NULLIF(s.title, ''), '(ohne Titel)') AS title,
                s.user_id,
                s.created_at,
                s.updated_at,
                COUNT(m.id) AS message_count
            FROM chat_sessions s
            LEFT JOIN chat_messages m ON m.session_id = s.id
            GROUP BY s.id
            ORDER BY s.updated_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_session_messages(db_path: Path, session_id: str) -> list[dict[str, Any]]:
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT id, session_id, role, content, metadata_json, created_at
            FROM chat_messages
            WHERE session_id = ?
            ORDER BY created_at ASC
            """,
            (session_id,),
        ).fetchall()
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        raw_meta = item.get("metadata_json", "{}")
        try:
            item["metadata"] = json.loads(raw_meta) if isinstance(raw_meta, str) else {}
        except json.JSONDecodeError:
            item["metadata"] = {}
        item.pop("metadata_json", None)
        out.append(item)
    return out


def get_chat_session(db_path: Path, session_id: str) -> dict[str, Any] | None:
    with _connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT id, title, user_id, metadata_json, created_at, updated_at
            FROM chat_sessions
            WHERE id = ?
            """,
            (session_id,),
        ).fetchone()
    if not row:
        return None
    item = dict(row)
    raw_meta = item.get("metadata_json", "{}")
    try:
        item["metadata"] = json.loads(raw_meta) if isinstance(raw_meta, str) else {}
    except json.JSONDecodeError:
        item["metadata"] = {}
    item.pop("metadata_json", None)
    return item


def export_session_json(db_path: Path, session_id: str, out_path: Path) -> Path:
    session = get_chat_session(db_path, session_id)
    if not session:
        raise ValueError(f"Session not found: {session_id}")
    payload = {
        "session": session,
        "messages": get_session_messages(db_path, session_id),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def export_all_sessions_jsonl(db_path: Path, out_path: Path) -> Path:
    sessions = list_chat_sessions(db_path, limit=1_000_000)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for session in sessions:
            session_id = session["id"]
            payload = {
                "session": get_chat_session(db_path, session_id),
                "messages": get_session_messages(db_path, session_id),
            }
            f.write(json.dumps(payload, ensure_ascii=False))
            f.write("\n")
    return out_path


def export_all_messages_csv(db_path: Path, out_path: Path) -> Path:
    sessions = list_chat_sessions(db_path, limit=1_000_000)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "session_id",
                "session_title",
                "session_created_at",
                "session_updated_at",
                "message_id",
                "message_created_at",
                "role",
                "content",
            ],
        )
        writer.writeheader()
        for session in sessions:
            messages = get_session_messages(db_path, session["id"])
            for msg in messages:
                writer.writerow(
                    {
                        "session_id": session["id"],
                        "session_title": session["title"],
                        "session_created_at": session["created_at"],
                        "session_updated_at": session["updated_at"],
                        "message_id": msg["id"],
                        "message_created_at": msg["created_at"],
                        "role": msg["role"],
                        "content": msg["content"],
                    }
                )
    return out_path


def _to_openai_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for msg in messages:
        role = str(msg.get("role") or "").strip().lower()
        # Export for evaluation: keep user/assistant and RAG tool outputs, drop system.
        if role not in {"user", "assistant", "tool"}:
            continue
        content = msg.get("content")
        if not isinstance(content, str):
            content = str(content or "")
        out.append({"role": role, "content": content})
    return out


def export_session_openai_json(db_path: Path, session_id: str, out_path: Path) -> Path:
    session = get_chat_session(db_path, session_id)
    if not session:
        raise ValueError(f"Session not found: {session_id}")
    messages = get_session_messages(db_path, session_id)
    payload = {
        "session_id": session_id,
        "messages": _to_openai_messages(messages),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def export_all_sessions_openai_jsonl(db_path: Path, out_path: Path) -> Path:
    sessions = list_chat_sessions(db_path, limit=1_000_000)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for session in sessions:
            session_id = session["id"]
            messages = get_session_messages(db_path, session_id)
            payload = {
                "session_id": session_id,
                "messages": _to_openai_messages(messages),
            }
            f.write(json.dumps(payload, ensure_ascii=False))
            f.write("\n")
    return out_path


# ---------------------------------------------------------------------------
# User Profile Management for Personalization
# ---------------------------------------------------------------------------


def get_user_profile(db_path: Path, user_id: str) -> dict[str, Any] | None:
    """Retrieve user profile with extracted topics and preferences."""
    with _connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT user_id, topics_json, topic_embeddings_json, excluded_bausteine_json,
                   message_count, created_at, updated_at
            FROM user_profiles
            WHERE user_id = ?
            """,
            (user_id,),
        ).fetchone()
    if not row:
        return None
    item = dict(row)
    for field in ("topics_json", "topic_embeddings_json", "excluded_bausteine_json"):
        raw = item.get(field, "[]")
        try:
            item[field.replace("_json", "")] = json.loads(raw) if isinstance(raw, str) else []
        except json.JSONDecodeError:
            item[field.replace("_json", "")] = []
        item.pop(field, None)
    return item


def upsert_user_profile(
    db_path: Path,
    user_id: str,
    *,
    topics: list[str] | None = None,
    topic_embeddings: list[list[float]] | None = None,
    excluded_bausteine: list[str] | None = None,
    selected_chat_profile: str | None = None,
    message_count: int | None = None,
) -> None:
    """Create or update user profile with extracted topics and embeddings."""
    now = _utc_now_iso()
    with _connect(db_path) as conn:
        existing = conn.execute(
            "SELECT user_id FROM user_profiles WHERE user_id = ?", (user_id,)
        ).fetchone()

        if existing:
            updates: list[str] = ["updated_at = ?"]
            params: list[Any] = [now]
            if topics is not None:
                updates.append("topics_json = ?")
                params.append(json.dumps(topics, ensure_ascii=False))
            if topic_embeddings is not None:
                updates.append("topic_embeddings_json = ?")
                params.append(json.dumps(topic_embeddings, ensure_ascii=False))
            if excluded_bausteine is not None:
                updates.append("excluded_bausteine_json = ?")
                params.append(json.dumps(excluded_bausteine, ensure_ascii=False))
            if selected_chat_profile is not None:
                updates.append("selected_chat_profile = ?")
                params.append(selected_chat_profile)
            if message_count is not None:
                updates.append("message_count = ?")
                params.append(message_count)
            params.append(user_id)
            conn.execute(
                f"UPDATE user_profiles SET {', '.join(updates)} WHERE user_id = ?",
                params,
            )
        else:
            conn.execute(
                """
                INSERT INTO user_profiles (user_id, topics_json, topic_embeddings_json,
                    excluded_bausteine_json, selected_chat_profile, message_count, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    json.dumps(topics or [], ensure_ascii=False),
                    json.dumps(topic_embeddings or [], ensure_ascii=False),
                    json.dumps(excluded_bausteine or [], ensure_ascii=False),
                    selected_chat_profile,
                    message_count or 0,
                    now,
                    now,
                ),
            )
        conn.commit()


def get_user_message_history(
    db_path: Path,
    user_id: str,
    limit: int = 100,
    role_filter: str | None = "user",
) -> list[dict[str, Any]]:
    """Get recent messages for a user across all their sessions for profile extraction."""
    with _connect(db_path) as conn:
        if role_filter:
            rows = conn.execute(
                """
                SELECT m.content, m.role, m.created_at, s.id as session_id, s.title as session_title
                FROM chat_messages m
                JOIN chat_sessions s ON m.session_id = s.id
                WHERE s.user_id = ? AND m.role = ?
                ORDER BY m.created_at DESC
                LIMIT ?
                """,
                (user_id, role_filter, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT m.content, m.role, m.created_at, s.id as session_id, s.title as session_title
                FROM chat_messages m
                JOIN chat_sessions s ON m.session_id = s.id
                WHERE s.user_id = ?
                ORDER BY m.created_at DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
    return [dict(r) for r in rows]


def get_user_message_count(db_path: Path, user_id: str) -> int:
    """Count total messages for a user to determine if profile extraction should run."""
    with _connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) as cnt
            FROM chat_messages m
            JOIN chat_sessions s ON m.session_id = s.id
            WHERE s.user_id = ? AND m.role = 'user'
            """,
            (user_id,),
        ).fetchone()
    return row["cnt"] if row else 0


def get_user_selected_chat_profile(db_path: Path, user_id: str) -> str | None:
    """Get the user's persisted chat profile selection.

    Requires init_chat_db() to have run at startup (handles schema migration).
    """
    try:
        with _connect(db_path) as conn:
            row = conn.execute(
                "SELECT selected_chat_profile FROM user_profiles WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        return row["selected_chat_profile"] if row else None
    except sqlite3.Error as e:
        logger.error(
            "Failed to query selected_chat_profile for user_id=%s: %s",
            user_id,
            e,
        )
        return None


def set_user_selected_chat_profile(
    db_path: Path, user_id: str, profile_name: str | None
) -> None:
    """Set the user's chat profile selection persistently."""
    upsert_user_profile(db_path, user_id, selected_chat_profile=profile_name)


def delete_user_profile(db_path: Path, user_id: str) -> bool:
    """Delete user profile (GDPR compliance)."""
    with _connect(db_path) as conn:
        cursor = conn.execute(
            "DELETE FROM user_profiles WHERE user_id = ?", (user_id,)
        )
        conn.commit()
        return cursor.rowcount > 0


def hard_delete_user_data(db_path: Path, user_id: str) -> dict[str, int]:
    """Delete every SQLite row owned by user_id across all 7 user-scoped tables.

    Used by the user-deletion orchestrator (self-delete and admin-delete both
    call this). Qdrant collections + PDFs on disk must be cleaned up by the
    caller BEFORE this runs, because we need the KB rows to enumerate them.

    FKs (`ON DELETE CASCADE`) handle the children: deleting a knowledge_base
    row cascades to its kb_documents; deleting a chat_session cascades to
    its chat_messages. Order below is therefore safe even with foreign_keys
    enabled.
    """
    counts: dict[str, int] = {}
    with _connect(db_path) as conn:
        for table in (
            "user_starters",
            "user_settings",
            "user_profiles",
            "knowledge_bases",
            "chat_sessions",
        ):
            cursor = conn.execute(
                f"DELETE FROM {table} WHERE user_id = ?", (user_id,)
            )
            counts[table] = cursor.rowcount
        conn.commit()
    return counts
