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
