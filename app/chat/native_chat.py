from __future__ import annotations

import csv
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import asyncpg


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE TABLE IF NOT EXISTS "User" (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  identifier TEXT UNIQUE NOT NULL,
  email TEXT UNIQUE,
  password_hash TEXT,
  metadata TEXT NOT NULL DEFAULT '{}',
  "createdAt" TIMESTAMP NOT NULL DEFAULT NOW(),
  "updatedAt" TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Add columns for existing tables (idempotent migrations)
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'User' AND column_name = 'email') THEN
    ALTER TABLE "User" ADD COLUMN email TEXT UNIQUE;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'User' AND column_name = 'password_hash') THEN
    ALTER TABLE "User" ADD COLUMN password_hash TEXT;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'User' AND column_name = 'status') THEN
    ALTER TABLE "User" ADD COLUMN status TEXT NOT NULL DEFAULT 'approved';
  END IF;
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'User' AND column_name = 'role') THEN
    ALTER TABLE "User" ADD COLUMN role TEXT NOT NULL DEFAULT 'user';
  END IF;
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'User' AND column_name = 'lastLoginAt') THEN
    ALTER TABLE "User" ADD COLUMN "lastLoginAt" TIMESTAMP NULL;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'User' AND column_name = 'acceptedTermsAt') THEN
    ALTER TABLE "User" ADD COLUMN "acceptedTermsAt" TIMESTAMP NULL;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'User' AND column_name = 'acceptedTermsVersion') THEN
    ALTER TABLE "User" ADD COLUMN "acceptedTermsVersion" TEXT NULL;
  END IF;
END $$;

CREATE TABLE IF NOT EXISTS "Thread" (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT,
  "userId" UUID REFERENCES "User"(id) ON DELETE SET NULL,
  metadata TEXT NOT NULL DEFAULT '{}',
  tags TEXT,
  "createdAt" TIMESTAMP NOT NULL DEFAULT NOW(),
  "updatedAt" TIMESTAMP NOT NULL DEFAULT NOW(),
  "deletedAt" TIMESTAMP NULL
);

CREATE TABLE IF NOT EXISTS "Step" (
  id UUID PRIMARY KEY,
  "threadId" UUID REFERENCES "Thread"(id) ON DELETE CASCADE,
  "parentId" UUID NULL,
  input TEXT NULL,
  metadata TEXT NOT NULL DEFAULT '{}',
  name TEXT NULL,
  output TEXT NULL,
  type TEXT NOT NULL,
  "startTime" TIMESTAMP NULL,
  "endTime" TIMESTAMP NULL,
  "showInput" TEXT NULL,
  "isError" BOOLEAN NOT NULL DEFAULT FALSE,
  "createdAt" TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS "Element" (
  id UUID PRIMARY KEY,
  "threadId" UUID REFERENCES "Thread"(id) ON DELETE CASCADE,
  "stepId" UUID REFERENCES "Step"(id) ON DELETE CASCADE,
  metadata TEXT NOT NULL DEFAULT '{}',
  mime TEXT NULL,
  name TEXT NULL,
  "objectKey" TEXT NULL,
  url TEXT NULL,
  "chainlitKey" TEXT NULL,
  display TEXT NULL,
  size BIGINT NULL,
  language TEXT NULL,
  page INTEGER NULL,
  "autoPlay" BOOLEAN NULL,
  "playerConfig" TEXT NULL,
  props TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS "Feedback" (
  id UUID PRIMARY KEY,
  "stepId" UUID REFERENCES "Step"(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  value DOUBLE PRECISION NULL,
  comment TEXT NULL
);

CREATE INDEX IF NOT EXISTS idx_thread_user_updated
  ON "Thread"("userId", "updatedAt" DESC);
CREATE INDEX IF NOT EXISTS idx_step_thread_start
  ON "Step"("threadId", "startTime");
CREATE INDEX IF NOT EXISTS idx_element_thread
  ON "Element"("threadId");
"""


async def ensure_native_schema(database_url: str) -> None:
    conn = await asyncpg.connect(database_url)
    try:
        await conn.execute(SCHEMA_SQL)
    finally:
        await conn.close()


async def create_user(
    database_url: str,
    username: str,
    email: str,
    password_hash: str,
) -> dict[str, Any] | None:
    """Create a new user with hashed password. Returns user dict or None if exists."""
    conn = await asyncpg.connect(database_url)
    try:
        row = await conn.fetchrow(
            """
            INSERT INTO "User" (identifier, email, password_hash, metadata)
            VALUES ($1, $2, $3, '{"provider": "local"}')
            ON CONFLICT (identifier) DO NOTHING
            ON CONFLICT (email) DO NOTHING
            RETURNING id, identifier, email, metadata, "createdAt"
            """,
            username,
            email,
            password_hash,
        )
        if row is None:
            return None
        return dict(row)
    finally:
        await conn.close()


async def get_user_by_identifier(
    database_url: str,
    identifier: str,
) -> dict[str, Any] | None:
    """Get user by username/identifier."""
    conn = await asyncpg.connect(database_url)
    try:
        row = await conn.fetchrow(
            """
            SELECT id, identifier, email, password_hash, metadata, "createdAt"
            FROM "User"
            WHERE identifier = $1
            """,
            identifier,
        )
        return dict(row) if row else None
    finally:
        await conn.close()


async def get_user_by_email(
    database_url: str,
    email: str,
) -> dict[str, Any] | None:
    """Get user by email."""
    conn = await asyncpg.connect(database_url)
    try:
        row = await conn.fetchrow(
            """
            SELECT id, identifier, email, password_hash, metadata, "createdAt"
            FROM "User"
            WHERE email = $1
            """,
            email,
        )
        return dict(row) if row else None
    finally:
        await conn.close()


async def upsert_user_on_login(
    database_url: str,
    *,
    identifier: str,
    provider: str,
    email: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
    admin_identifiers: list[str] | None = None,
) -> dict[str, Any]:
    """Insert or refresh a user row on every login.

    - Creates the row if missing (default status='approved', role='user').
    - Bumps lastLoginAt.
    - Auto-promotes to role='admin' if identifier is in admin_identifiers
      (only transitions up; never demotes an existing admin).
    - Returns the row so callers can enforce status/role checks.
    """
    admins = {a.strip() for a in (admin_identifiers or []) if a and a.strip()}
    should_admin = identifier in admins

    metadata_json = json.dumps(
        {"provider": provider, **(extra_metadata or {})}
    )

    conn = await asyncpg.connect(database_url)
    try:
        # Create if missing. Existing rows are left alone here so admin edits stick.
        await conn.execute(
            """
            INSERT INTO "User" (identifier, email, metadata, status, role)
            VALUES ($1, $2, $3, 'approved', $4)
            ON CONFLICT (identifier) DO NOTHING
            """,
            identifier,
            email,
            metadata_json,
            "admin" if should_admin else "user",
        )
        # Bump lastLoginAt, provider metadata, and promote to admin if env says so.
        row = await conn.fetchrow(
            """
            UPDATE "User"
            SET "lastLoginAt" = NOW(),
                "updatedAt" = NOW(),
                role = CASE
                    WHEN $2::boolean AND role <> 'admin' THEN 'admin'
                    ELSE role
                END
            WHERE identifier = $1
            RETURNING id, identifier, email, metadata, status, role,
                      "createdAt", "lastLoginAt"
            """,
            identifier,
            should_admin,
        )
        return dict(row) if row else {}
    finally:
        await conn.close()


async def list_all_users(database_url: str) -> list[dict[str, Any]]:
    conn = await asyncpg.connect(database_url)
    try:
        rows = await conn.fetch(
            """
            SELECT id, identifier, email, metadata, status, role,
                   "createdAt", "updatedAt", "lastLoginAt"
            FROM "User"
            ORDER BY "createdAt" DESC
            """
        )
        out: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            try:
                item["metadata"] = json.loads(item.get("metadata") or "{}")
            except (TypeError, json.JSONDecodeError):
                item["metadata"] = {}
            item["id"] = str(item["id"])
            for key in ("createdAt", "updatedAt", "lastLoginAt"):
                if item.get(key) is not None:
                    item[key] = item[key].isoformat()
            out.append(item)
        return out
    finally:
        await conn.close()


async def update_user_admin_fields(
    database_url: str,
    user_id: str,
    *,
    status: str | None = None,
    role: str | None = None,
) -> dict[str, Any] | None:
    if status is not None and status not in ("approved", "blocked"):
        raise ValueError(f"invalid status: {status}")
    if role is not None and role not in ("user", "admin"):
        raise ValueError(f"invalid role: {role}")
    if status is None and role is None:
        return None

    conn = await asyncpg.connect(database_url)
    try:
        row = await conn.fetchrow(
            """
            UPDATE "User"
            SET status = COALESCE($2, status),
                role = COALESCE($3, role),
                "updatedAt" = NOW()
            WHERE id = $1::uuid
            RETURNING id, identifier, email, status, role
            """,
            user_id,
            status,
            role,
        )
        if row is None:
            return None
        out = dict(row)
        out["id"] = str(out["id"])
        return out
    finally:
        await conn.close()


async def get_terms_status(
    database_url: str,
    identifier: str,
) -> dict[str, Any] | None:
    conn = await asyncpg.connect(database_url)
    try:
        row = await conn.fetchrow(
            '''SELECT "acceptedTermsAt", "acceptedTermsVersion"
               FROM "User" WHERE identifier = $1''',
            identifier,
        )
        if row is None:
            return None
        at = row["acceptedTermsAt"]
        return {
            "accepted_at": at.isoformat() if at else None,
            "accepted_version": row["acceptedTermsVersion"],
        }
    finally:
        await conn.close()


async def accept_terms(
    database_url: str,
    identifier: str,
    version: str,
) -> dict[str, Any] | None:
    conn = await asyncpg.connect(database_url)
    try:
        row = await conn.fetchrow(
            '''UPDATE "User"
               SET "acceptedTermsAt" = NOW(),
                   "acceptedTermsVersion" = $2,
                   "updatedAt" = NOW()
               WHERE identifier = $1
               RETURNING "acceptedTermsAt", "acceptedTermsVersion"''',
            identifier,
            version,
        )
        if row is None:
            return None
        return {
            "accepted_at": row["acceptedTermsAt"].isoformat(),
            "accepted_version": row["acceptedTermsVersion"],
        }
    finally:
        await conn.close()


async def get_user_role_status(
    database_url: str,
    identifier: str,
) -> tuple[str, str] | None:
    """Return (role, status) for the given identifier, or None if missing."""
    conn = await asyncpg.connect(database_url)
    try:
        row = await conn.fetchrow(
            'SELECT role, status FROM "User" WHERE identifier = $1',
            identifier,
        )
        if row is None:
            return None
        return row["role"], row["status"]
    finally:
        await conn.close()


async def check_user_exists(
    database_url: str,
    username: str | None = None,
    email: str | None = None,
) -> dict[str, bool]:
    """Check if username or email already exists."""
    conn = await asyncpg.connect(database_url)
    try:
        result = {"username_exists": False, "email_exists": False}
        if username:
            row = await conn.fetchrow(
                'SELECT 1 FROM "User" WHERE identifier = $1',
                username,
            )
            result["username_exists"] = row is not None
        if email:
            row = await conn.fetchrow(
                'SELECT 1 FROM "User" WHERE email = $1',
                email,
            )
            result["email_exists"] = row is not None
        return result
    finally:
        await conn.close()


async def export_all_chats_zip(
    *,
    database_url: str,
    out_dir: Path,
    user_id: str | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = _stamp()
    jsonl_path = out_dir / f"native-chat-export-all-{stamp}.jsonl"
    csv_path = out_dir / f"native-chat-export-all-{stamp}.csv"
    zip_path = out_dir / f"native-chat-export-all-{stamp}.zip"

    conn = await asyncpg.connect(database_url)
    try:
        where_clause = 'WHERE t."deletedAt" IS NULL'
        params: list[Any] = []
        if user_id:
            where_clause += ' AND t."userId" = $1::uuid'
            params.append(user_id)

        threads = await conn.fetch(
            f"""
            SELECT
              t.id,
              t.name,
              t.metadata,
              t."createdAt",
              t."updatedAt",
              t."userId",
              u.identifier AS user_identifier
            FROM "Thread" t
            LEFT JOIN "User" u ON t."userId" = u.id
            {where_clause}
            ORDER BY t."updatedAt" DESC
            """,
            *params,
        )

        with jsonl_path.open("w", encoding="utf-8") as jf, csv_path.open(
            "w", encoding="utf-8", newline=""
        ) as cf:
            writer = csv.DictWriter(
                cf,
                fieldnames=[
                    "thread_id",
                    "thread_name",
                    "user_identifier",
                    "thread_created_at",
                    "thread_updated_at",
                    "step_id",
                    "step_type",
                    "step_name",
                    "step_start",
                    "step_end",
                    "step_input",
                    "step_output",
                    "step_is_error",
                ],
            )
            writer.writeheader()

            for thread in threads:
                steps = await conn.fetch(
                    """
                    SELECT id, type, name, input, output, "isError", "startTime", "endTime", metadata
                    FROM "Step"
                    WHERE "threadId" = $1::uuid
                    ORDER BY "startTime" NULLS LAST, "createdAt" NULLS LAST
                    """,
                    thread["id"],
                )
                payload = {
                    "thread": {
                        "id": str(thread["id"]),
                        "name": thread["name"],
                        "userId": str(thread["userId"]) if thread["userId"] else None,
                        "userIdentifier": thread["user_identifier"],
                        "metadata": json.loads(thread["metadata"] or "{}"),
                        "createdAt": thread["createdAt"].isoformat() if thread["createdAt"] else None,
                        "updatedAt": thread["updatedAt"].isoformat() if thread["updatedAt"] else None,
                    },
                    "steps": [
                        {
                            "id": str(step["id"]),
                            "type": step["type"],
                            "name": step["name"],
                            "input": step["input"],
                            "output": step["output"],
                            "isError": bool(step["isError"]),
                            "startTime": step["startTime"].isoformat() if step["startTime"] else None,
                            "endTime": step["endTime"].isoformat() if step["endTime"] else None,
                            "metadata": json.loads(step["metadata"] or "{}"),
                        }
                        for step in steps
                    ],
                }
                jf.write(json.dumps(payload, ensure_ascii=False))
                jf.write("\n")

                for step in steps:
                    writer.writerow(
                        {
                            "thread_id": str(thread["id"]),
                            "thread_name": thread["name"] or "",
                            "user_identifier": thread["user_identifier"] or "",
                            "thread_created_at": (
                                thread["createdAt"].isoformat() if thread["createdAt"] else ""
                            ),
                            "thread_updated_at": (
                                thread["updatedAt"].isoformat() if thread["updatedAt"] else ""
                            ),
                            "step_id": str(step["id"]),
                            "step_type": step["type"] or "",
                            "step_name": step["name"] or "",
                            "step_start": step["startTime"].isoformat() if step["startTime"] else "",
                            "step_end": step["endTime"].isoformat() if step["endTime"] else "",
                            "step_input": step["input"] or "",
                            "step_output": step["output"] or "",
                            "step_is_error": bool(step["isError"]),
                        }
                    )
    finally:
        await conn.close()

    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(jsonl_path, arcname=jsonl_path.name)
        zf.write(csv_path, arcname=csv_path.name)
    return zip_path
