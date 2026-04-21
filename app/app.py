from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote
from uuid import uuid4

import asyncpg
import bcrypt
import chainlit as cl
from chainlit.auth import get_current_user
from chainlit.input_widget import Select
from chainlit.types import Starter
from fastapi import Depends, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse, RedirectResponse
from pydantic import BaseModel

from chat.chat_history import (
    add_chat_message,
    create_chat_session,
    export_all_sessions_openai_jsonl,
    export_session_openai_json,
    get_chat_session,
    get_session_messages,
    get_user_message_count,
    get_user_selected_chat_profile,
    init_chat_db,
    list_chat_sessions,
    set_session_title_if_missing,
    set_user_selected_chat_profile,
    update_chat_session_metadata,
)
from langflow.langflow_client import LangflowError, run_langflow
from core.llm import chat, message_to_dict
from chat.native_chat import (
    check_user_exists,
    create_user,
    ensure_native_schema,
    export_all_chats_zip,
    get_user_by_identifier,
    get_user_role_status,
    list_all_users,
    update_user_admin_fields,
    upsert_user_on_login,
)
from kb.rag_tool import build_context, extract_page, extract_source_file, format_citations, retrieve, personalized_retrieve
from core.settings import (
    ADMIN_IDENTIFIERS,
    CHAT_DB_PATH,
    CHAT_EXPORT_DIR,
    CHAINLIT_AUTH_PASSWORD,
    CHAINLIT_AUTH_USERNAME,
    CHAINLIT_INIT_DB,
    DATA_KB_DOCS_DIR,
    DATA_RAW_DIR,
    DATABASE_URL,
    EMBED_MODEL,
    LANGFLOW_ENABLED,
    MAX_FILE_SIZE_MB,
    MAX_TOP_K,
    MAX_SOURCE_LINKS,
    PERSONALIZATION_ENABLED,
    PERSONALIZED_FOLLOWUPS_COUNT,
    PROFILE_MIN_MESSAGES,
    QDRANT_COLLECTION,
    STARTER_QUESTIONS,
    SYSTEM_PROMPT_PATH,
    TOP_K,
)
from kb.user_kb import (
    SHARED_KB_ID,
    get_document,
    get_kb,
    get_user_system_prompt,
    init_user_kb_db,
    list_kbs,
    list_user_starters,
)
from kb.user_profile import (
    determine_balance,
    load_user_profile,
    update_user_profile,
    UserProfile,
)


def _load_system_prompt(path: Path) -> str | None:
    if path.is_file():
        content = path.read_text(encoding="utf-8").strip()
        return content or None
    return None


SYSTEM_PROMPT = _load_system_prompt(SYSTEM_PROMPT_PATH)


SHARED_KB_ENTRY: dict[str, Any] = {
    "id": SHARED_KB_ID,
    "name": "IT-Grundschutz (gemeinsam)",
    "description": (
        "Gemeinsame, schreibgeschützte Wissensbasis mit dem IT-Grundschutz-Kompendium "
        "und den BSI-Standards (200-1/2/3/4)."
    ),
    "qdrant_collection": QDRANT_COLLECTION,
    "readonly": True,
    "document_count": None,
}


def _build_user_kb_catalog(user_id: str | None) -> dict[str, dict[str, Any]]:
    """Return {kb_id: kb_entry} map the user can retrieve against.

    Always includes the shared IT-Grundschutz KB. Private per-user KBs are
    loaded from SQLite when a user_id is available.
    """
    catalog: dict[str, dict[str, Any]] = {SHARED_KB_ID: dict(SHARED_KB_ENTRY)}
    if not user_id:
        return catalog
    for kb in list_kbs(CHAT_DB_PATH, user_id):
        catalog[kb["id"]] = kb
    return catalog


def _resolve_kb_selection(
    requested_kb_id: str,
    catalog: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Pick the best KB match for an LLM tool-call arg.

    Falls back to the shared KB if the id is empty or unknown, so retrieval
    always has a target collection.
    """
    if requested_kb_id and requested_kb_id in catalog:
        return catalog[requested_kb_id]
    # Case-insensitive name match as a resilience net — models sometimes
    # pass the display name instead of the id.
    if requested_kb_id:
        lowered = requested_kb_id.strip().lower()
        for entry in catalog.values():
            if str(entry.get("name", "")).strip().lower() == lowered:
                return entry
    return catalog.get(SHARED_KB_ID, SHARED_KB_ENTRY)


def _format_kb_catalog_for_prompt(catalog: dict[str, dict[str, Any]]) -> str:
    if not catalog:
        return ""
    lines: list[str] = ["## VERFÜGBARE WISSENSDATENBANKEN"]
    lines.append(
        "Ruf `rag_retrieve` mit dem Argument `knowledge_base` gesetzt auf die passende Id aus dieser Liste auf:"
    )
    for entry in catalog.values():
        kb_id = entry.get("id", "")
        name = entry.get("name", kb_id)
        description = (entry.get("description") or "").strip()
        tags: list[str] = []
        if entry.get("readonly"):
            tags.append("gemeinsam, schreibgeschützt")
        doc_count = entry.get("document_count")
        if isinstance(doc_count, int):
            tags.append(f"{doc_count} Dokumente")
        suffix = f" ({'; '.join(tags)})" if tags else ""
        line = f"- id={kb_id}: {name}{suffix}"
        if description:
            line += f" — {description}"
        lines.append(line)
    return "\n".join(lines)

CITATION_PANEL_CACHE: dict[str, str] = {}
CITATION_SIDEBAR_TITLE = "Quellen & Belegstellen"
CITATION_HISTORY_SIDEBAR_TITLE = "Quellen & Belegstellen (Verlauf)"


def _allowed_source_pdf_names() -> set[str]:
    if not DATA_RAW_DIR.is_dir():
        return set()
    try:
        return {
            entry.name
            for entry in DATA_RAW_DIR.iterdir()
            if entry.is_file() and entry.suffix.lower() == ".pdf"
        }
    except OSError:
        return set()


def _resolve_source_pdf_path(file_name: str, allowed_names: set[str] | None = None) -> Path | None:
    if not file_name or file_name != Path(file_name).name:
        return None

    candidates = allowed_names if allowed_names is not None else _allowed_source_pdf_names()
    if file_name not in candidates:
        return None

    data_root = DATA_RAW_DIR.resolve()
    file_path = (DATA_RAW_DIR / file_name).resolve()
    try:
        file_path.relative_to(data_root)
    except ValueError:
        return None

    if not file_path.is_file() or file_path.suffix.lower() != ".pdf":
        return None
    return file_path


def _source_pdf_url(file_name: str, document_id: str | None = None) -> str:
    """Citation URL. Prefers the per-document id (kb_documents.id) when available,
    falls back to bare filename so legacy shared-KB chunks keep working."""
    if isinstance(document_id, str) and document_id.strip():
        return f"/sources/pdf/{quote(document_id.strip(), safe='')}"
    return f"/sources/pdf/{quote(file_name, safe='')}"


def _citation_panel_url(step_id: str) -> str:
    return f"/sources/citations/{quote(step_id, safe='')}"


async def _load_citation_panel_content(step_id: str) -> str | None:
    if not isinstance(step_id, str) or not re.fullmatch(r"[0-9a-fA-F-]{36}", step_id):
        return None
    cached = CITATION_PANEL_CACHE.get(step_id)
    if isinstance(cached, str) and cached.strip():
        return cached

    if not DATABASE_URL:
        return None

    conn: asyncpg.Connection | None = None
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        row = await conn.fetchrow(
            'SELECT metadata FROM "Step" WHERE id = $1::uuid',
            step_id,
        )
        if row is None:
            return None
        raw_metadata = row.get("metadata")
        if not isinstance(raw_metadata, str):
            return None
        metadata = json.loads(raw_metadata)
        if not isinstance(metadata, dict):
            return None
        panel_content = metadata.get("citation_panel_content")
        if isinstance(panel_content, str) and panel_content.strip():
            _cache_citation_panel_content(step_id, panel_content)
            return panel_content
        return None
    except Exception:
        return None
    finally:
        if conn is not None:
            await conn.close()


def _cache_citation_panel_content(step_id: str, panel_content: str, *, max_items: int = 512) -> None:
    if not isinstance(step_id, str) or not step_id.strip():
        return
    if not isinstance(panel_content, str) or not panel_content.strip():
        return
    CITATION_PANEL_CACHE[step_id] = panel_content
    while len(CITATION_PANEL_CACHE) > max_items:
        oldest_key = next(iter(CITATION_PANEL_CACHE))
        CITATION_PANEL_CACHE.pop(oldest_key, None)


def _ensure_route_precedes_catch_all(fastapi_app: Any, route_path: str) -> None:
    routes = getattr(getattr(fastapi_app, "router", None), "routes", None)
    if not isinstance(routes, list):
        return

    # Move every route with this path above the catch-all, preserving order.
    # (Multi-method paths like GET+POST /api/kbs register as separate APIRoutes.)
    while True:
        catch_all_idx = next(
            (
                i
                for i, route in enumerate(routes)
                if isinstance(getattr(route, "path", None), str)
                and str(getattr(route, "path")).endswith("/{full_path:path}")
            ),
            None,
        )
        if catch_all_idx is None:
            return

        route_idx = next(
            (
                i
                for i, route in enumerate(routes)
                if i > catch_all_idx and getattr(route, "path", None) == route_path
            ),
            None,
        )
        if route_idx is None:
            return

        route = routes.pop(route_idx)
        routes.insert(catch_all_idx, route)

# Chat profiles configuration
CHAT_PROFILES_PATH = Path(__file__).parent / "chat_profiles.json"


def _load_chat_profiles() -> dict[str, Any]:
    """Load chat profiles configuration from JSON file."""
    if CHAT_PROFILES_PATH.is_file():
        try:
            return json.loads(CHAT_PROFILES_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            print(f"[WARN] Failed to load chat_profiles.json: {e}")
    return {"profiles": [], "default_profile": None}


CHAT_PROFILES_CONFIG = _load_chat_profiles()


def _get_profile_by_name(profile_name: str) -> dict[str, Any] | None:
    """Get a profile configuration by its name."""
    for profile in CHAT_PROFILES_CONFIG.get("profiles", []):
        if profile.get("name") == profile_name:
            return profile
    return None


TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "rag_retrieve",
            "description": (
                "Suche relevante Dokumente in einer Wissensdatenbank. "
                "Die verfügbaren Wissensdatenbanken und ihre Ids sind im Systemprompt "
                "unter 'VERFÜGBARE WISSENSDATENBANKEN' aufgelistet. Wähle die passende "
                "Wissensdatenbank basierend auf der Nutzerfrage."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Die Nutzerfrage oder Suchanfrage."},
                    "knowledge_base": {
                        "type": "string",
                        "description": (
                            "Id der zu durchsuchenden Wissensdatenbank (siehe Systemprompt). "
                            "Verwende 'shared' für den gemeinsamen IT-Grundschutz-Korpus."
                        ),
                    },
                    "top_k": {"type": "integer", "description": "Anzahl der Treffer.", "default": 5},
                },
                "required": ["query", "knowledge_base"],
            },
        },
    }
]


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _truncate(text: str, max_len: int = 120) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 3].rstrip() + "..."


def _build_personalization_prompt(user_profile: UserProfile) -> str:
    """Build personalization context for the system prompt."""
    if not user_profile or not user_profile.topics:
        return ""

    topics_str = ", ".join(user_profile.topics)
    personalized_followups = PERSONALIZED_FOLLOWUPS_COUNT

    return f"""## PERSONALISIERTER KONTEXT
Der Nutzer hat sich häufig mit folgenden Themen beschäftigt: {topics_str}

## PERSONALISIERTE ANTWORT-SEKTION
- Füge nach der Hauptantwort eine kurze Sektion hinzu mit dem Header: "**Bezug zu Ihren Interessen:**"
- Beziehe die Antwort kurz auf die bekannten Interessen des Nutzers (max 50 Wörter)
- Diese Sektion soll nur erscheinen, wenn ein sinnvoller Bezug herstellbar ist

## PERSONALISIERTE ANSCHLUSSFRAGEN
- {personalized_followups} der 3 Anschlussfragen sollten sich auf die Nutzerinteressen beziehen
- Beispiel: Wenn der Nutzer sich für Webserver interessiert, könnte eine Anschlussfrage lauten: "Welche speziellen Anforderungen gelten für Webserver in diesem Kontext?"
"""


def _effective_system_prompt(messages: list[dict[str, Any]]) -> str:
    if not messages:
        return ""
    first = messages[0]
    if not isinstance(first, dict) or first.get("role") != "system":
        return ""
    content = first.get("content")
    return content.strip() if isinstance(content, str) else ""


def _current_chat_profile_prompt() -> str:
    config = cl.user_session.get("chat_profile_config")
    if not isinstance(config, dict):
        return ""
    prompt_context = config.get("prompt_context")
    return prompt_context.strip() if isinstance(prompt_context, str) else ""


def _current_personalization_context() -> str:
    user_profile = cl.user_session.get("user_profile")
    if isinstance(user_profile, UserProfile) and user_profile.topics:
        return _build_personalization_prompt(user_profile)
    return ""


def _format_langflow_chat_history(messages: list[dict[str, Any]], *, max_messages: int = 12, max_chars: int = 6000) -> str:
    rendered: list[str] = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        if role not in {"user", "assistant"}:
            continue
        content = item.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        rendered.append(f"{role.capitalize()}: {content.strip()}")
    joined = "\n\n".join(rendered[-max_messages:])
    if len(joined) <= max_chars:
        return joined
    return joined[-max_chars:]


def _langflow_global_vars(messages: list[dict[str, Any]]) -> dict[str, str]:
    return {
        "SYSTEM_PROMPT": _effective_system_prompt(messages),
        "CHAT_PROFILE": str(cl.user_session.get("chat_profile") or ""),
        "CHAT_PROFILE_PROMPT": _current_chat_profile_prompt(),
        "PERSONALIZATION_CONTEXT": _current_personalization_context(),
        "CHAT_HISTORY": _format_langflow_chat_history(messages),
    }


def _current_chat_session_id() -> str | None:
    value = cl.user_session.get("chat_history_session_id")
    return value if isinstance(value, str) and value.strip() else None


def _empty_source_catalog() -> dict[str, Any]:
    return {"next_id": 1, "key_to_id": {}, "entries": {}}


def _clean_section_title(section_title: str | None) -> str | None:
    if not isinstance(section_title, str):
        return None
    cleaned = re.sub(r"\s+", " ", section_title).strip()
    return cleaned or None


def _source_catalog_key(
    file_name: str,
    page_start: int | None,
    page_end: int | None,
    section_title: str | None,
) -> str:
    payload = {
        "file": file_name.strip().lower(),
        "page_start": page_start if isinstance(page_start, int) else None,
        "page_end": page_end if isinstance(page_end, int) else None,
        "section": (_clean_section_title(section_title) or "").lower(),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _sanitize_source_catalog(raw_catalog: Any) -> dict[str, Any]:
    if not isinstance(raw_catalog, dict):
        return _empty_source_catalog()

    key_to_id_raw = raw_catalog.get("key_to_id")
    entries_raw = raw_catalog.get("entries")
    next_id_raw = raw_catalog.get("next_id")

    key_to_id: dict[str, int] = {}
    if isinstance(key_to_id_raw, dict):
        for key, value in key_to_id_raw.items():
            if not isinstance(key, str):
                continue
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                continue
            if parsed > 0:
                key_to_id[key] = parsed

    entries: dict[str, dict[str, Any]] = {}
    if isinstance(entries_raw, dict):
        for source_id_raw, entry_raw in entries_raw.items():
            if not isinstance(source_id_raw, str) or not isinstance(entry_raw, dict):
                continue
            try:
                source_id = int(source_id_raw)
            except (TypeError, ValueError):
                continue
            if source_id <= 0:
                continue
            file_name = entry_raw.get("file")
            if not isinstance(file_name, str) or not file_name.strip():
                continue
            page_start = entry_raw.get("page_start")
            page_end = entry_raw.get("page_end")
            section = _clean_section_title(entry_raw.get("section"))
            normalized_entry: dict[str, Any] = {"file": file_name}
            if isinstance(page_start, int):
                normalized_entry["page_start"] = page_start
            if isinstance(page_end, int):
                normalized_entry["page_end"] = page_end
            if section:
                normalized_entry["section"] = section
            entries[str(source_id)] = normalized_entry

    valid_ids = {int(source_id) for source_id in entries}
    key_to_id = {key: source_id for key, source_id in key_to_id.items() if source_id in valid_ids}

    max_id = max(valid_ids, default=0)
    try:
        next_id = int(next_id_raw)
    except (TypeError, ValueError):
        next_id = 1
    if next_id <= max_id:
        next_id = max_id + 1
    if next_id < 1:
        next_id = 1

    return {"next_id": next_id, "key_to_id": key_to_id, "entries": entries}


def _load_session_source_catalog(session_id: str | None) -> dict[str, Any]:
    if not isinstance(session_id, str) or not session_id.strip():
        return _empty_source_catalog()
    session = get_chat_session(CHAT_DB_PATH, session_id)
    if not isinstance(session, dict):
        return _empty_source_catalog()
    metadata = session.get("metadata")
    if not isinstance(metadata, dict):
        return _empty_source_catalog()
    return _sanitize_source_catalog(metadata.get("source_catalog"))


def _persist_session_source_catalog(session_id: str | None, catalog: dict[str, Any]) -> None:
    if not isinstance(session_id, str) or not session_id.strip():
        return
    session = get_chat_session(CHAT_DB_PATH, session_id)
    if not isinstance(session, dict):
        return
    metadata = session.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    metadata["source_catalog"] = _sanitize_source_catalog(catalog)
    update_chat_session_metadata(CHAT_DB_PATH, session_id, metadata)


def _register_source_in_catalog(
    catalog: dict[str, Any],
    *,
    file_name: str,
    page_start: int | None,
    page_end: int | None,
    section_title: str | None,
) -> tuple[int, dict[str, Any], bool]:
    sanitized = _sanitize_source_catalog(catalog)
    if sanitized is not catalog:
        catalog.clear()
        catalog.update(sanitized)

    key_to_id = catalog["key_to_id"]
    entries = catalog["entries"]

    source_key = _source_catalog_key(file_name, page_start, page_end, section_title)
    existing_id = key_to_id.get(source_key)
    normalized_section = _clean_section_title(section_title)

    if isinstance(existing_id, int) and existing_id > 0:
        entry = entries.get(str(existing_id))
        changed = False
        if not isinstance(entry, dict):
            entry = {"file": file_name}
            entries[str(existing_id)] = entry
            changed = True
        if isinstance(page_start, int) and not isinstance(entry.get("page_start"), int):
            entry["page_start"] = page_start
            changed = True
        if isinstance(page_end, int) and not isinstance(entry.get("page_end"), int):
            entry["page_end"] = page_end
            changed = True
        if normalized_section and not isinstance(entry.get("section"), str):
            entry["section"] = normalized_section
            changed = True
        if not isinstance(entry.get("file"), str) or not entry["file"].strip():
            entry["file"] = file_name
            changed = True
        return existing_id, entry, changed

    next_id = 1
    while str(next_id) in entries:
        next_id += 1

    key_to_id[source_key] = next_id
    entry: dict[str, Any] = {"file": file_name}
    if isinstance(page_start, int):
        entry["page_start"] = page_start
    if isinstance(page_end, int):
        entry["page_end"] = page_end
    if normalized_section:
        entry["section"] = normalized_section
    entries[str(next_id)] = entry
    catalog["next_id"] = next_id + 1
    return next_id, entry, True


def _source_ids_from_citation_history(raw_history: Any) -> set[int]:
    ids: set[int] = set()
    for item in _sanitize_citation_history(raw_history):
        rows = item.get("source_rows")
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            source_id = row.get("source_id")
            if isinstance(source_id, int) and source_id > 0:
                ids.add(source_id)
    return ids


def _prune_source_catalog(catalog: dict[str, Any], keep_ids: set[int]) -> bool:
    sanitized = _sanitize_source_catalog(catalog)
    changed = sanitized is not catalog
    if sanitized is not catalog:
        catalog.clear()
        catalog.update(sanitized)

    wanted_ids = {source_id for source_id in keep_ids if isinstance(source_id, int) and source_id > 0}
    entries = catalog.get("entries", {})
    key_to_id = catalog.get("key_to_id", {})

    pruned_entries = {
        source_id_str: entry
        for source_id_str, entry in entries.items()
        if isinstance(source_id_str, str)
        and source_id_str.isdigit()
        and int(source_id_str) in wanted_ids
        and isinstance(entry, dict)
    }
    pruned_key_to_id = {
        source_key: source_id
        for source_key, source_id in key_to_id.items()
        if isinstance(source_key, str) and isinstance(source_id, int) and source_id in wanted_ids
    }

    if pruned_entries != entries:
        catalog["entries"] = pruned_entries
        changed = True
    if pruned_key_to_id != key_to_id:
        catalog["key_to_id"] = pruned_key_to_id
        changed = True

    next_id = 1
    while str(next_id) in catalog["entries"]:
        next_id += 1
    if catalog.get("next_id") != next_id:
        catalog["next_id"] = next_id
        changed = True

    return changed


def _format_history_overview(limit: int = 15) -> str:
    sessions = list_chat_sessions(CHAT_DB_PATH, limit=limit)
    if not sessions:
        return "Keine gespeicherten Chats gefunden."
    lines = ["## Gespeicherte Chats", ""]
    for item in sessions:
        lines.append(
            "- "
            f"`{item['id']}` | {item['title']} | {item['message_count']} Nachrichten | "
            f"zuletzt: {item['updated_at']}"
        )
    lines.append("")
    lines.append("Nutze `/history <session_id>` für den Verlauf oder `/export <session_id>` für JSON.")
    return "\n".join(lines)


def _format_session_messages(session_id: str, limit: int = 20) -> str:
    messages = get_session_messages(CHAT_DB_PATH, session_id)
    if not messages:
        return f"Keine Nachrichten für Session `{session_id}` gefunden."
    tail = messages[-limit:]
    lines = [f"## Verlauf `{session_id}` (letzte {len(tail)} Nachrichten)", ""]
    for msg in tail:
        role = msg.get("role", "unknown")
        content = _truncate(str(msg.get("content", "")), max_len=280)
        created_at = msg.get("created_at", "")
        lines.append(f"- **{role}** ({created_at}): {content}")
    return "\n".join(lines)


async def _handle_control_message(message: cl.Message) -> bool:
    text = (message.content or "").strip()
    if not text.startswith("/"):
        return False

    parts = text.split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd == "/history":
        if arg:
            await cl.Message(content=_format_session_messages(arg)).send()
        else:
            await cl.Message(content=_format_history_overview()).send()
        return True

    if cmd == "/export":
        CHAT_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = _utc_stamp()
        if arg.lower() == "all":
            out_jsonl = CHAT_EXPORT_DIR / f"chat-export-openai-all-{stamp}.jsonl"
            export_all_sessions_openai_jsonl(CHAT_DB_PATH, out_jsonl)
            await cl.Message(
                content="OpenAI-Export für alle Chats erstellt (JSONL).",
                elements=[
                    cl.File(name=out_jsonl.name, path=str(out_jsonl), display="inline"),
                ],
            ).send()
            return True

        session_id = arg or _current_chat_session_id()
        if not session_id:
            await cl.Message(content="Keine aktive Session gefunden. Nutze `/export <session_id>` oder `/export all`.").send()
            return True
        out_json = CHAT_EXPORT_DIR / f"chat-export-openai-{session_id}-{stamp}.json"
        try:
            export_session_openai_json(CHAT_DB_PATH, session_id, out_json)
        except ValueError:
            await cl.Message(content=f"Session nicht gefunden: `{session_id}`").send()
            return True
        await cl.Message(
            content=f"OpenAI-Export erstellt für Session `{session_id}`.",
            elements=[cl.File(name=out_json.name, path=str(out_json), display="inline")],
        ).send()
        return True

    if cmd in {"/help-history", "/help"}:
        await cl.Message(
            content=(
                "Verfügbare Befehle:\n"
                "- `/history` zeigt gespeicherte Chats\n"
                "- `/history <session_id>` zeigt die letzten Nachrichten einer Session\n"
                "- `/export` exportiert den aktuellen Chat im OpenAI-Format (JSON)\n"
                "- `/export <session_id>` exportiert eine bestimmte Session im OpenAI-Format (JSON)\n"
                "- `/export all` exportiert alle Chats im OpenAI-Format (JSONL)"
            )
        ).send()
        return True

    return False

def _first_sentence(text: str, max_len: int = 220) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", cleaned, maxsplit=1)
    sentence = parts[0]
    if len(sentence) > max_len:
        sentence = sentence[: max_len - 3].rstrip() + "..."
    return sentence


def _extractive_answer_from_results(question: str, results: list[Any], max_points: int = 5) -> str:
    if not results:
        return "Im bereitgestellten Kontext nicht enthalten"

    seen: set[str] = set()
    bullets: list[str] = []
    for idx, result in enumerate(results, start=1):
        if len(bullets) >= max_points:
            break
        sentence = _first_sentence(getattr(result, "text", "") or "", max_len=280)
        if not sentence:
            continue
        key = re.sub(r"\s+", " ", sentence).strip().lower()
        if key in seen:
            continue
        seen.add(key)
        bullets.append(f"- {sentence} [{idx}]")

    if not bullets:
        bullets = [
            f"- {(getattr(r, 'text', '') or '').strip()[:220]} [{i}]"
            for i, r in enumerate(results[:max_points], start=1)
            if (getattr(r, "text", "") or "").strip()
        ]
    if not bullets:
        bullets = ["- Relevante Fundstellen vorhanden, aber kein extrahierbarer Kurzsatz. [1]"]

    return (
        f"Ich habe relevante Inhalte im Kontext zur Frage \"{question}\" gefunden.\n\n"
        "Kernaussagen aus den Trefferstellen:\n"
        + "\n".join(bullets)
    )


def _strip_model_source_blocks(text: str) -> str:
    text = text.rstrip()
    patterns = [
        r"\n+\*\*Quellen\*\*[\s\S]*$",
        r"\n+Quellen[\s\S]*$",
        r"\n+Sources[\s\S]*$",
    ]
    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE).rstrip()
    return text


def _page_label(page_start: int | None, page_end: int | None) -> str:
    # Very large ranges are usually fallback mappings; show a compact page anchor.
    if page_start and page_end and page_end >= page_start and (page_end - page_start) > 60:
        return f"S.{page_start}+"
    if page_start and page_end and page_end != page_start:
        return f"S.{page_start}-{page_end}"
    if page_start:
        return f"S.{page_start}"
    return "S.?"


def _source_alias(source_number: int, section_title: str | None, page_start: int | None, page_end: int | None) -> str:
    section = (section_title or "Abschnitt unbekannt").strip()
    section = re.sub(r"\s+", " ", section)
    if len(section) > 48:
        section = section[:45].rstrip() + "..."
    return f"Quelle {source_number}: {section} ({_page_label(page_start, page_end)})"


def _markdown_link(label: str, url: str) -> str:
    clean_label = re.sub(r"\s+", " ", label).strip()
    # Escape markdown control chars in link text so aliases like "[...]" remain clickable.
    clean_label = clean_label.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")
    return f"[{clean_label}]({url})"


def _resolve_section_title(metadata: dict[str, Any]) -> str | None:
    explicit = metadata.get("section_title")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()

    anforderung_id = metadata.get("anforderung_id")
    if isinstance(anforderung_id, str) and anforderung_id.strip():
        return anforderung_id.strip()

    baustein_id = metadata.get("baustein_id")
    if isinstance(baustein_id, str) and baustein_id.strip():
        doc_type = metadata.get("doc_type")
        if doc_type == "baustein_beschreibung":
            return f"{baustein_id} Beschreibung"
        if doc_type == "baustein_gefaehrdungslage":
            return f"{baustein_id} Gefaehrdungslage"
        return baustein_id

    title = metadata.get("title")
    if isinstance(title, str) and title.strip():
        return title.strip()
    return None


def _inject_clickable_refs(
    text: str,
    alias_by_index: dict[int, str],
    alias_by_number: dict[int, str] | None = None,
    url_by_index: dict[int, str] | None = None,
    url_by_number: dict[int, str] | None = None,
) -> str:
    if not text or (not alias_by_index and not alias_by_number):
        return text

    def repl(match: re.Match) -> str:
        idx = int(match.group(1))
        alias = alias_by_index.get(idx) or (alias_by_number or {}).get(idx)
        if not alias:
            return match.group(0)
        # Bare alias text — Chainlit's frontend matches this against an inline
        # cl.Pdf element name and renders a side-panel opener. Wrapping as a
        # markdown link would suppress that and revert to a new-tab navigation.
        return alias

    # Covers citations like: 【1†L1-L4】 and [1†L1-L4]
    text = re.sub(r"【(\d+)[^】]*】", repl, text)
    text = re.sub(r"\[(\d+)†[^\]]*\]", repl, text)
    # Covers citations like: [1]
    text = re.sub(r"\[(\d+)\]", repl, text)
    return text


def _alias_number_map(source_rows: list[tuple[int, str, str, int | None, int | None, str | None, str]]) -> dict[int, str]:
    alias_by_number: dict[int, str] = {}
    for src_idx, alias, *_ in source_rows:
        if isinstance(src_idx, int):
            alias_by_number[src_idx] = alias
        match = re.match(r"^\s*Quelle\s+(\d+)\s*:", alias, flags=re.IGNORECASE)
        if not match:
            continue
        alias_by_number[int(match.group(1))] = alias
    return alias_by_number


def _inject_named_source_refs(
    text: str,
    source_rows: list[tuple[int, str, str, int | None, int | None, str | None, str]],
) -> str:
    if not text or not source_rows:
        return text

    def norm(s: str) -> str:
        s = s.lower()
        s = re.sub(r"[^a-z0-9äöüß]+", " ", s)
        return re.sub(r"\s+", " ", s).strip()

    entries = []
    for _, alias, file_name, page_start, page_end, section_title, _ in source_rows:
        file_stem = file_name.lower().removesuffix(".pdf")
        entries.append(
            {
                "alias": alias,
                "file_stem": norm(file_stem),
                "section": norm(section_title or ""),
                "page_start": page_start,
                "page_end": page_end,
                "is_kompendium": "kompendium" in file_stem,
            }
        )

    def replace_bracket(match: re.Match) -> str:
        raw = match.group(1).strip()
        # Keep pure numeric references for the numeric pass.
        if re.fullmatch(r"\d+", raw):
            return match.group(0)
        if "quelle " in raw.lower():
            return match.group(0)

        rnorm = norm(raw)
        if not rnorm:
            return match.group(0)

        page_match = re.search(r"(?:s\.?|seite)\s*(\d+)", raw, flags=re.IGNORECASE)
        wanted_page = int(page_match.group(1)) if page_match else None

        best_alias = None
        best_score = 0
        for entry in entries:
            score = 0
            if entry["file_stem"] and entry["file_stem"] in rnorm:
                score += 4
            if entry["section"] and any(tok in rnorm for tok in entry["section"].split()[:4]):
                score += 2
            if "kompendium" in rnorm and entry["is_kompendium"]:
                score += 2
            if "standard 200 2" in rnorm and "standard 200 2" in entry["file_stem"]:
                score += 3
            if wanted_page is not None:
                start = entry["page_start"]
                end = entry["page_end"] or start
                if isinstance(start, int) and isinstance(end, int) and start <= wanted_page <= end:
                    score += 2
            if score > best_score:
                best_score = score
                best_alias = entry["alias"]

        return best_alias if best_alias and best_score >= 3 else match.group(0)

    return re.sub(r"\[([^\[\]]{3,140})\]", replace_bracket, text)


def _normalize_source_alias_mentions(
    text: str,
    alias_by_index: dict[int, str],
    alias_by_number: dict[int, str] | None = None,
) -> str:
    if not text or (not alias_by_index and not alias_by_number):
        return text

    def repl(match: re.Match) -> str:
        idx = int(match.group(1))
        return alias_by_index.get(idx) or (alias_by_number or {}).get(idx, match.group(0))

    # Normalize free-form mentions like
    # "Quelle 1: APP.3.2.A20 ... (S) [Zentrale Verwaltung] (S.397)" or
    # "Quelle 2: Einleitung ... (Seite 12)" to exact alias token.
    text = re.sub(
        r"Quelle\s*([0-9]+)\s*:\s*[^\n]*?\((?:S\.?|Seite)\s*[^)\n]+\)",
        repl,
        text,
        flags=re.IGNORECASE,
    )
    # Also normalize bracket-wrapped alias mentions so they become clickable tokens.
    text = re.sub(
        r"【\s*Quelle\s*([0-9]+)\s*:\s*[^】]+\s*】",
        repl,
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\[\s*Quelle\s*([0-9]+)\s*:\s*[^\]]+\s*\]",
        repl,
        text,
        flags=re.IGNORECASE,
    )
    # Normalize plain mentions like "Quelle 2" (without trailing ": ...").
    text = re.sub(
        r"\bQuelle\s*([0-9]+)\b(?!\s*:)",
        repl,
        text,
        flags=re.IGNORECASE,
    )
    return text


def _normalize_source_mentions_by_content(
    text: str,
    source_rows: list[tuple[int, str, str, int | None, int | None, str | None, str]],
) -> str:
    if not text or not source_rows:
        return text

    def norm(s: str) -> str:
        s = s.lower()
        s = re.sub(r"[^a-z0-9äöüß]+", " ", s)
        return re.sub(r"\s+", " ", s).strip()

    entries = []
    for _, alias, _, page_start, page_end, section_title, _ in source_rows:
        entries.append(
            {
                "alias": alias,
                "section": norm(section_title or ""),
                "page_start": page_start,
                "page_end": page_end,
            }
        )

    def repl(match: re.Match) -> str:
        raw = match.group(0)
        page_match = re.search(r"(?:S\.?|Seite)\s*(\d+)", raw, flags=re.IGNORECASE)
        wanted_page = int(page_match.group(1)) if page_match else None
        rnorm = norm(raw)

        best_alias = None
        best_score = 0
        for entry in entries:
            score = 0
            if wanted_page is not None:
                start = entry["page_start"]
                end = entry["page_end"] or start
                if isinstance(start, int) and isinstance(end, int) and start <= wanted_page <= end:
                    score += 3
            if entry["section"] and any(tok in rnorm for tok in entry["section"].split()[:5]):
                score += 2
            if score > best_score:
                best_score = score
                best_alias = entry["alias"]

        return best_alias if best_alias and best_score >= 2 else raw

    return re.sub(
        r"Quelle\s*\d+\s*:\s*[^\n]{1,260}?\((?:S\.?|Seite)\s*[^)\n]+\)",
        repl,
        text,
        flags=re.IGNORECASE,
    )


def _inject_source_alias_links(
    text: str,
    alias_by_number: dict[int, str],
    url_by_number: dict[int, str],
) -> str:
    # No-op: in-text "Quelle N: ..." mentions stay as bare alias text so Chainlit's
    # frontend can match them against inline cl.Pdf element names and render side-panel
    # openers. Markdown-link wrapping would revert behavior to new-tab navigation.
    return text


def _inject_naked_source_links(text: str) -> str:
    if not text:
        return text

    def repl(match: re.Match) -> str:
        label = match.group("label")
        if not isinstance(label, str):
            return match.group(0)
        # Strip the trailing "(/sources/pdf/...)" and keep the alias as bare text so
        # Chainlit's frontend matches it against an inline cl.Pdf element name.
        return label

    return re.sub(
        r"(?P<label>Quelle\s*\d+\s*:[^\n]{1,260}?\((?:S\.?|Seite)\s*[^)\n]+\))\((?:https?://[^\s)]+|/sources/pdf/[^)\s]+)\)",
        repl,
        text,
        flags=re.IGNORECASE,
    )


def _compact_visible_source_numbering(
    content: str,
    source_rows: list[tuple[int, str, str, int | None, int | None, str | None, str]],
    source_rows_for_session: list[dict[str, Any]],
) -> tuple[str, list[tuple[int, str, str, int | None, int | None, str | None, str]], list[dict[str, Any]]]:
    if not source_rows:
        return content, source_rows, source_rows_for_session

    alias_remap: dict[str, str] = {}
    remapped_rows: list[tuple[int, str, str, int | None, int | None, str | None, str]] = []
    for display_idx, row in enumerate(source_rows, start=1):
        source_id, old_alias, file_name, page_start, page_end, section_title, evidence = row
        new_alias = _source_alias(display_idx, section_title, page_start, page_end)
        alias_remap[old_alias] = new_alias
        remapped_rows.append(
            (
                source_id,
                new_alias,
                file_name,
                page_start,
                page_end,
                section_title,
                evidence,
            )
        )

    remapped_session_rows: list[dict[str, Any]] = []
    for row in source_rows_for_session:
        if not isinstance(row, dict):
            continue
        old_alias = row.get("alias")
        new_alias = alias_remap.get(old_alias) if isinstance(old_alias, str) else None
        if not isinstance(new_alias, str):
            remapped_session_rows.append(dict(row))
            continue
        updated = dict(row)
        updated["alias"] = new_alias
        remapped_session_rows.append(updated)

    updated_content = content
    for old_alias, new_alias in alias_remap.items():
        if old_alias == new_alias:
            continue
        escaped_old = old_alias.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")
        escaped_new = new_alias.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")
        updated_content = updated_content.replace(f"[{escaped_old}](", f"[{escaped_new}](")
        updated_content = updated_content.replace(old_alias, new_alias)
        updated_content = updated_content.replace(escaped_old, escaped_new)

    return updated_content, remapped_rows, remapped_session_rows


def _align_aliases_to_source_ids(
    content: str,
    source_rows: list[tuple[int, str, str, int | None, int | None, str | None, str]],
    source_rows_for_session: list[dict[str, Any]],
) -> tuple[str, list[tuple[int, str, str, int | None, int | None, str | None, str]], list[dict[str, Any]]]:
    if not source_rows:
        return content, source_rows, source_rows_for_session

    alias_remap: dict[str, str] = {}
    remapped_rows: list[tuple[int, str, str, int | None, int | None, str | None, str]] = []
    for source_id, old_alias, file_name, page_start, page_end, section_title, evidence in source_rows:
        if isinstance(source_id, int) and source_id > 0:
            new_alias = _source_alias(source_id, section_title, page_start, page_end)
        else:
            new_alias = old_alias
        alias_remap[old_alias] = new_alias
        remapped_rows.append(
            (
                source_id,
                new_alias,
                file_name,
                page_start,
                page_end,
                section_title,
                evidence,
            )
        )

    remapped_session_rows: list[dict[str, Any]] = []
    for row in source_rows_for_session:
        if not isinstance(row, dict):
            continue
        old_alias = row.get("alias")
        new_alias = alias_remap.get(old_alias) if isinstance(old_alias, str) else None
        updated = dict(row)
        if isinstance(new_alias, str):
            updated["alias"] = new_alias
        remapped_session_rows.append(updated)

    updated_content = content
    for old_alias, new_alias in alias_remap.items():
        if old_alias == new_alias:
            continue
        escaped_old = old_alias.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")
        escaped_new = new_alias.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")
        updated_content = updated_content.replace(f"[{escaped_old}](", f"[{escaped_new}](")
        updated_content = updated_content.replace(old_alias, new_alias)
        updated_content = updated_content.replace(escaped_old, escaped_new)

        old_num_match = re.match(r"^\s*Quelle\s*(\d+)\s*:", old_alias, flags=re.IGNORECASE)
        new_num_match = re.match(r"^\s*Quelle\s*(\d+)\s*:", new_alias, flags=re.IGNORECASE)
        if old_num_match and new_num_match and old_num_match.group(1) != new_num_match.group(1):
            updated_content = re.sub(
                rf"\bQuelle\s*{re.escape(old_num_match.group(1))}\b",
                f"Quelle {new_num_match.group(1)}",
                updated_content,
                flags=re.IGNORECASE,
            )

    return updated_content, remapped_rows, remapped_session_rows


def _inject_alias_links_by_rows(text: str, source_rows: list[dict[str, Any]]) -> str:
    # No-op: aliases stay as bare text so Chainlit's frontend can match them against
    # inline cl.Pdf element names and open the side-panel viewer. The previous final
    # markdown-wrap pass is what produced new-tab links.
    return text


def _desired_source_count(text: str, available: int) -> int:
    if available <= 0:
        return 0
    refs: list[int] = []
    for pattern in (
        r"\[(\d+)\]",
        r"【(\d+)[^】]*】",
        r"\[(\d+)†[^\]]*\]",
        r"Quelle\s*(\d+)\s*:",
        r"\bQuelle\s*(\d+)\b",
    ):
        refs.extend(int(x) for x in re.findall(pattern, text or ""))
    if refs:
        return min(max(refs), available)
    return available


def _top_score(results: list[Any]) -> float:
    return max((float(getattr(r, "score", 0.0) or 0.0) for r in results), default=0.0)


def _is_weak_retrieval(results: list[Any], *, min_hits: int = 2, min_top_score: float = 0.22) -> bool:
    return len(results) < min_hits or _top_score(results) < min_top_score


def _is_strong_retrieval(results: list[Any], *, min_hits: int = 3, min_top_score: float = 0.45) -> bool:
    return len(results) >= min_hits and _top_score(results) >= min_top_score


def _is_context_abstention(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", (text or "")).strip().lower().rstrip(".!")
    if not normalized:
        return False
    if normalized.startswith("im bereitgestellten kontext nicht enthalten"):
        return True
    # Catch common model variants like "Die Information ist ... nicht enthalten."
    return "bereitgestellten kontext" in normalized and "nicht enthalten" in normalized


def _extract_standard_id(query: str) -> str | None:
    q = (query or "").lower()
    m = re.search(r"\b(?:bsi[- ]?standard\s*)?200[- ]?([1-4])\b", q)
    if not m:
        return None
    return f"standard_200_{m.group(1)}"


def _normalize_query_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _keyword_query_variant(query: str) -> str | None:
    tokens = re.findall(r"[A-Za-zÄÖÜäöüß0-9.\-]+", query or "")
    if not tokens:
        return None
    stopwords = {
        "welche",
        "welcher",
        "welches",
        "wie",
        "was",
        "sind",
        "ist",
        "der",
        "die",
        "das",
        "bei",
        "für",
        "im",
        "in",
        "nach",
        "und",
        "oder",
        "den",
        "dem",
        "des",
        "ein",
        "eine",
        "einen",
        "einer",
        "sinnvoll",
        "grundsätzlich",
    }
    kept = [t for t in tokens if len(t) > 2 and t.lower() not in stopwords]
    if len(kept) < 2:
        return None
    return " ".join(kept)


def _extract_baustein_id(query: str) -> str | None:
    m = re.search(r"\b([A-Z]{2,4}\.\d+(?:\.\d+){1,2})\b", (query or "").upper())
    if not m:
        return None
    return m.group(1)


def _build_query_variants(query: str, standard_id: str | None) -> list[str]:
    base = _normalize_query_text(query)
    if not base:
        return []

    variants: list[str] = [base]
    lower = base.lower()
    keyword_variant = _keyword_query_variant(base)

    if standard_id:
        std_label = standard_id.replace("standard_", "BSI-Standard ").replace("_", "-")
        if "basis-absicherung" in lower or "basis absicherung" in lower:
            variants.append(f"{std_label} Basis-Absicherung Schritte Vorgehensweise")
        if keyword_variant:
            variants.append(f"{std_label} {keyword_variant}")
    else:
        elevated_need = ("erhöht" in lower or "erhoeh" in lower) and "schutzbedarf" in lower
        if elevated_need:
            variants.append(f"{base} (H) Anforderungen bei erhöhtem Schutzbedarf")
            baustein_id = _extract_baustein_id(base)
            if not baustein_id and "webserver" in lower:
                baustein_id = "APP.3.2"
            if baustein_id:
                variants.append(f"{baustein_id} Anforderungen bei erhöhtem Schutzbedarf (H) Redundanz DDoS")
            else:
                variants.append("Anforderungen bei erhöhtem Schutzbedarf (H) Maßnahmen")
        elif keyword_variant:
            variants.append(f"IT-Grundschutz {keyword_variant}")

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in variants:
        normalized = _normalize_query_text(candidate)
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
        if len(deduped) >= 3:
            break
    return deduped


def _result_key(item: Any) -> tuple[str, int | None, str]:
    metadata = getattr(item, "metadata", {}) or {}
    file_name = extract_source_file(metadata) or ""
    page = extract_page(metadata)
    snippet = re.sub(r"\s+", " ", (getattr(item, "text", "") or "").strip())[:120]
    return (file_name, page, snippet)


def _fuse_results(result_sets: list[list[Any]], max_items: int) -> list[Any]:
    fused: dict[tuple[str, int | None, str], dict[str, Any]] = {}
    for results in result_sets:
        for rank, item in enumerate(results, start=1):
            key = _result_key(item)
            score = float(getattr(item, "score", 0.0) or 0.0)
            state = fused.get(key)
            if state is None:
                state = {"item": item, "rrf": 0.0, "hits": 0, "best_score": score}
                fused[key] = state
            state["rrf"] += 1.0 / (60.0 + rank)
            state["hits"] += 1
            if score > state["best_score"]:
                state["item"] = item
                state["best_score"] = score

    ranked = sorted(
        fused.values(),
        key=lambda s: (float(s["rrf"]), int(s["hits"]), float(s["best_score"])),
        reverse=True,
    )
    return [entry["item"] for entry in ranked[:max_items]]


async def _retrieve_fused(
    *,
    query: str,
    top_k: int,
    source_scope: str | None,
    standard_id: str | None,
) -> tuple[list[Any], list[dict[str, Any]]]:
    variants = _build_query_variants(query, standard_id)
    if not variants:
        return [], []

    result_sets: list[list[Any]] = []
    variant_stats: list[dict[str, Any]] = []
    for v in variants:
        hits = await retrieve(
            query=v,
            top_k=top_k,
            source_scope=source_scope,
            standard_id=standard_id,
        )
        result_sets.append(hits)
        variant_stats.append(
            {
                "query": v,
                "hits": len(hits),
                "top_score": round(_top_score(hits), 4),
            }
        )

    fused = _fuse_results(result_sets, max_items=top_k)
    print(
        "[DEBUG] retrieve_fused",
        {
            "source_scope": source_scope,
            "standard_id": standard_id,
            "variants": len(variants),
            "variant_stats": variant_stats,
            "fused_hits": len(fused),
            "fused_top_score": _top_score(fused),
        },
    )
    return fused, variant_stats


def _merge_results(primary: list[Any], secondary: list[Any], max_items: int) -> list[Any]:
    merged: list[Any] = []
    seen: set[tuple[str, int | None, str]] = set()

    def key_for(item: Any) -> tuple[str, int | None, str]:
        return _result_key(item)

    for item in [*primary, *secondary]:
        k = key_for(item)
        if k in seen:
            continue
        seen.add(k)
        merged.append(item)
        if len(merged) >= max_items:
            break
    return merged


def _extract_followups(text: str) -> tuple[str, list[str]]:
    lines = text.splitlines()
    start_idx = None
    for i, line in enumerate(lines):
        if re.search(r"anschlussfragen|weitere fragen", line, flags=re.IGNORECASE):
            start_idx = i
            break
    if start_idx is None:
        return text, []

    questions: list[str] = []
    end_idx = len(lines)
    for j in range(start_idx + 1, len(lines)):
        raw = lines[j].strip()
        if not raw:
            if questions:
                end_idx = j
                break
            continue

        m = re.match(r"^(?:\d+[\).\:]|-|\*)\s+(.*)$", raw)
        candidate = m.group(1).strip() if m else raw
        candidate = re.sub(r"\s+", " ", candidate)
        # Prefer question-shaped lines, but keep numbered follow-ups even without '?'.
        if candidate.endswith("?") or re.match(r"^(?:\d+[\).\:]|-|\*)\s+", raw):
            questions.append(candidate)
        elif questions:
            end_idx = j
            break
        if len(questions) >= 3:
            end_idx = j + 1
            break

    # Fallback: collect up to 3 trailing numbered/bullet lines anywhere in the answer.
    if not questions:
        for raw in lines:
            m = re.match(r"^\s*(?:\d+[\).\:]|-|\*)\s+(.+)$", raw)
            if not m:
                continue
            candidate = re.sub(r"\s+", " ", m.group(1)).strip()
            if len(candidate) < 12:
                continue
            questions.append(candidate)
            if len(questions) >= 3:
                break

    if not questions:
        return text, []

    cleaned_lines = lines[:start_idx] + lines[end_idx:]
    cleaned_text = "\n".join(cleaned_lines).strip()
    return cleaned_text, questions


def _coerce_step_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        text = value.get("content")
        if isinstance(text, str):
            return text
    return str(value)


@cl.oauth_callback
async def oauth_callback(
    provider_id: str,
    token: str,
    raw_user_data: dict[str, Any],
    default_user: cl.User,
) -> cl.User | None:
    """Handle OAuth login (e.g., GitHub).

    Upserts the user in Postgres, rejects blocked users, and attaches
    role/status into the Chainlit session metadata.
    """
    if provider_id == "github":
        identifier = raw_user_data.get("login")
        if not identifier:
            return None
        extra = {
            "name": raw_user_data.get("name"),
            "email": raw_user_data.get("email"),
            "avatar_url": raw_user_data.get("avatar_url"),
            "github_id": str(raw_user_data.get("id")),
        }
        role, status = "user", "approved"
        if DATABASE_URL:
            row = await upsert_user_on_login(
                DATABASE_URL,
                identifier=identifier,
                provider="github",
                email=raw_user_data.get("email"),
                extra_metadata=extra,
                admin_identifiers=ADMIN_IDENTIFIERS,
            )
            role = row.get("role", "user")
            status = row.get("status", "approved")
        if status == "blocked":
            print(f"[AUTH] rejecting blocked user: {identifier}")
            return None
        return cl.User(
            identifier=identifier,
            metadata={"provider": "github", "role": role, "status": status, **extra},
        )
    # Accept all users from other configured OAuth providers
    return default_user


def _coerce_step_metadata(step: dict[str, Any]) -> dict[str, Any]:
    raw = step.get("metadata")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return {}
    return {}


def _sanitize_source_rows_payload(raw_rows: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_rows, list):
        return []
    cleaned: list[dict[str, Any]] = []
    for row in raw_rows:
        if not isinstance(row, dict):
            continue
        file_name = row.get("file")
        alias = row.get("alias")
        page = row.get("page")
        if not isinstance(file_name, str) or not isinstance(alias, str):
            continue
        clean_row: dict[str, Any] = {"file": file_name, "alias": alias}
        document_id = row.get("document_id")
        if isinstance(document_id, str) and document_id.strip():
            clean_row["document_id"] = document_id.strip()
        source_id = row.get("source_id")
        if isinstance(source_id, int) and source_id > 0:
            clean_row["source_id"] = source_id
        if isinstance(page, int):
            clean_row["page"] = page
            clean_row["page_start"] = page
        page_start = row.get("page_start")
        if isinstance(page_start, int):
            clean_row["page_start"] = page_start
        page_end = row.get("page_end")
        if isinstance(page_end, int):
            clean_row["page_end"] = page_end
        section = row.get("section")
        if isinstance(section, str) and section.strip():
            clean_row["section"] = re.sub(r"\s+", " ", section).strip()
        evidence = row.get("evidence")
        if isinstance(evidence, str) and evidence.strip():
            clean_row["evidence"] = re.sub(r"\s+", " ", evidence).strip()
        cleaned.append(clean_row)
    return cleaned


def _sanitize_citation_history(raw_history: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_history, list):
        return []
    cleaned: list[dict[str, Any]] = []
    for item in raw_history:
        if not isinstance(item, dict):
            continue
        panel_content = item.get("panel_content")
        source_rows = _sanitize_source_rows_payload(item.get("source_rows"))
        if not isinstance(panel_content, str) or not panel_content.strip():
            continue
        cleaned.append({"panel_content": panel_content, "source_rows": source_rows})
    return cleaned


def _append_citation_history(
    history: list[dict[str, Any]],
    panel_content: str | None,
    source_rows: list[dict[str, Any]],
    *,
    max_entries: int = 60,
) -> list[dict[str, Any]]:
    if not isinstance(panel_content, str) or not panel_content.strip():
        return history
    cleaned_rows = _sanitize_source_rows_payload(source_rows)
    entry = {"panel_content": panel_content, "source_rows": cleaned_rows}

    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in [*history, entry]:
        key = json.dumps(item, ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped[-max_entries:]


def _build_citation_history_view(history: list[dict[str, Any]]) -> tuple[str | None, list[dict[str, Any]]]:
    cleaned_history = _sanitize_citation_history(history)
    if not cleaned_history:
        return None, []

    lines = ["## Quellen & Belegstellen (Verlauf)", ""]
    merged_rows: list[dict[str, Any]] = []

    ordered_history = list(enumerate(cleaned_history, start=1))
    for answer_number, item in reversed(ordered_history):
        lines.append(f"## Antwort {answer_number}")
        answer_rows = _sanitize_source_rows_payload(item["source_rows"])
        if answer_rows:
            for row_idx, row in enumerate(answer_rows, start=1):
                page_start = row.get("page_start")
                if not isinstance(page_start, int):
                    page_start = row.get("page") if isinstance(row.get("page"), int) else None
                page_end = row.get("page_end") if isinstance(row.get("page_end"), int) else page_start
                section = row.get("section")
                section_for_alias = section.strip() if isinstance(section, str) and section.strip() else None
                if section_for_alias is None:
                    raw_alias = row.get("alias")
                    if isinstance(raw_alias, str) and raw_alias.strip():
                        section_for_alias = re.sub(r"^\s*Quelle\s*\d+\s*:\s*", "", raw_alias, flags=re.IGNORECASE).strip()
                        section_for_alias = re.sub(
                            r"\s*\((?:S\.?|Seite)\s*[^)]+\)\s*$",
                            "",
                            section_for_alias,
                            flags=re.IGNORECASE,
                        ).strip()

                source_id = row.get("source_id")
                if isinstance(source_id, int) and source_id > 0:
                    alias_display = _source_alias(source_id, section_for_alias, page_start, page_end)
                else:
                    alias = row.get("alias")
                    if isinstance(alias, str) and alias.strip():
                        alias_display = alias.strip()
                    else:
                        alias_display = _source_alias(row_idx, section_for_alias, page_start, page_end)
                lines.append(f"### {alias_display}")
                file_name = row.get("file")
                if isinstance(file_name, str) and file_name.strip():
                    lines.append(f"Datei: `{file_name}`")
                    document_id = row.get("document_id") if isinstance(row.get("document_id"), str) else None
                    pdf_url = _source_pdf_url(file_name, document_id)
                    page_for_link = page_start
                    if isinstance(page_for_link, int):
                        pdf_url = f"{pdf_url}#page={page_for_link}"
                    lines.append(f"PDF: [Öffnen]({pdf_url})")
                if isinstance(source_id, int) and source_id > 0:
                    lines.append(f"Quellen-ID: {source_id}")
                else:
                    lines.append(f"Quellen-ID: {row_idx}")
                if isinstance(page_start, int):
                    lines.append(f"Seiten: {_page_label(page_start, page_end if isinstance(page_end, int) else None)}")
                if isinstance(section, str) and section.strip():
                    lines.append(f"Abschnitt: {section.strip()}")
                evidence = row.get("evidence")
                if isinstance(evidence, str) and evidence.strip():
                    lines.append(f"Belegsnippet: \"{evidence.strip()}\"")
                lines.append("")
                merged_rows.append(row)
        else:
            parsed_aliases = re.findall(r"^###\s*\[\d+\]\s*(.+)$", item["panel_content"], flags=re.MULTILINE)
            if not parsed_aliases:
                parsed_aliases = re.findall(r"^###\s*(.+)$", item["panel_content"], flags=re.MULTILINE)
            if parsed_aliases:
                for alias in parsed_aliases:
                    normalized_alias = alias.strip()
                    if normalized_alias:
                        lines.append(f"### {normalized_alias}")
                    lines.append("")
            else:
                lines.append("Keine Zitierungen erkannt.")
        lines.append("")

    return "\n".join(lines).strip(), merged_rows


def _cited_source_numbers(text: str) -> set[int]:
    numbers: set[int] = set()
    if not isinstance(text, str) or not text.strip():
        return numbers
    for match in re.finditer(r"Quelle\s+(\d+)\s*:", text, flags=re.IGNORECASE):
        numbers.add(int(match.group(1)))
    for match in re.finditer(r"\[(\d+)\]", text):
        numbers.add(int(match.group(1)))
    for match in re.finditer(r"【(\d+)[^】]*】", text):
        numbers.add(int(match.group(1)))
    return numbers


def _build_source_rows_from_results(
    content: str,
    last_results: list[Any],
) -> tuple[
    list[tuple[int, str, str, int | None, int | None, str | None, str]],
    dict[int, str],
    dict[int, str],
    list[dict[str, Any]],
]:
    seen_links: set[tuple[str, int | None]] = set()
    source_rows: list[tuple[int, str, str, int | None, int | None, str | None, str]] = []
    alias_by_index: dict[int, str] = {}
    url_by_index: dict[int, str] = {}
    source_rows_for_session: list[dict[str, Any]] = []
    alias_to_url: dict[str, str] = {}
    desired_sources = _desired_source_count(content, len(last_results))
    if MAX_SOURCE_LINKS > 0:
        desired_sources = min(desired_sources, MAX_SOURCE_LINKS)
    allowed_pdf_names = _allowed_source_pdf_names()
    display_counter = 1

    for idx, result in enumerate(last_results, start=1):
        file_name = extract_source_file(result.metadata)
        if not file_name:
            continue
        page = extract_page(result.metadata)
        key = (file_name, page)
        if key in seen_links:
            existing_alias = next(
                (
                    alias
                    for _, alias, fname, pstart, _, _, _ in source_rows
                    if fname == file_name and pstart == page
                ),
                None,
            )
            if existing_alias:
                alias_by_index[idx] = existing_alias
                existing_url = alias_to_url.get(existing_alias)
                if isinstance(existing_url, str) and existing_url:
                    url_by_index[idx] = existing_url
            continue
        raw_doc_id = result.metadata.get("document_id") if isinstance(result.metadata, dict) else None
        document_id = raw_doc_id.strip() if isinstance(raw_doc_id, str) and raw_doc_id.strip() else None
        # Legacy shared-KB chunks have no document_id; they require a real file in DATA_RAW_DIR
        # for the link to resolve. User-uploaded KB chunks are served by document_id and don't.
        if document_id is None and _resolve_source_pdf_path(file_name, allowed_pdf_names) is None:
            continue
        page_end = result.metadata.get("page_end") if isinstance(result.metadata.get("page_end"), int) else None
        section_title = _resolve_section_title(result.metadata)
        page_start = extract_page(result.metadata)
        alias = _source_alias(display_counter, section_title, page_start, page_end)
        pdf_url = _source_pdf_url(file_name, document_id)
        if isinstance(page, int):
            pdf_url = f"{pdf_url}#page={page}"
        evidence_snippet = _first_sentence(result.text)
        alias_by_index[idx] = alias
        url_by_index[idx] = pdf_url
        alias_to_url[alias] = pdf_url
        source_rows.append(
            (
                display_counter,
                alias,
                file_name,
                page_start,
                page_end,
                section_title,
                evidence_snippet,
            )
        )
        source_rows_for_session.append(
            {
                "alias": alias,
                "file": file_name,
                "document_id": document_id,
                "page": page,
                "page_start": page_start if isinstance(page_start, int) else None,
                "page_end": page_end if isinstance(page_end, int) else None,
                "section": section_title if isinstance(section_title, str) else None,
                "evidence": evidence_snippet if isinstance(evidence_snippet, str) else None,
            }
        )
        display_counter += 1
        seen_links.add(key)
        if desired_sources and len(seen_links) >= desired_sources:
            break

    return source_rows, alias_by_index, url_by_index, source_rows_for_session


def _build_source_rows_from_langflow_citations(
    citations: list[dict[str, Any]],
) -> tuple[
    list[tuple[int, str, str, int | None, int | None, str | None, str]],
    dict[int, str],
    dict[int, str],
    list[dict[str, Any]],
]:
    source_rows: list[tuple[int, str, str, int | None, int | None, str | None, str]] = []
    alias_by_index: dict[int, str] = {}
    url_by_index: dict[int, str] = {}
    source_rows_for_session: list[dict[str, Any]] = []
    allowed_pdf_names = _allowed_source_pdf_names()
    seen_numbers: set[int] = set()

    for fallback_number, raw in enumerate(citations, start=1):
        if not isinstance(raw, dict):
            continue
        citation_number = raw.get("citation_number")
        if not isinstance(citation_number, int) or citation_number < 1:
            citation_number = fallback_number
        if citation_number in seen_numbers:
            continue

        meta: dict[str, Any] = {}
        for key in ("file", "page_start", "page_end"):
            value = raw.get(key)
            if isinstance(value, (str, int)):
                meta[key] = value
        section_title = raw.get("section_title")
        if isinstance(section_title, str) and section_title.strip():
            meta["section_title"] = section_title.strip()

        file_name = extract_source_file(meta)
        if not file_name:
            file_value = raw.get("file")
            file_name = file_value.strip().split("/")[-1] if isinstance(file_value, str) and file_value.strip() else None
        if not file_name:
            continue

        raw_doc_id = raw.get("document_id")
        document_id = raw_doc_id.strip() if isinstance(raw_doc_id, str) and raw_doc_id.strip() else None
        # Same gate as the native path: only legacy/no-doc-id chunks need a real file on disk.
        if document_id is None and _resolve_source_pdf_path(file_name, allowed_pdf_names) is None:
            continue

        page_start = raw.get("page_start") if isinstance(raw.get("page_start"), int) else extract_page(meta)
        page_end = raw.get("page_end") if isinstance(raw.get("page_end"), int) else None
        section_title = (
            raw.get("section_title").strip()
            if isinstance(raw.get("section_title"), str) and raw.get("section_title").strip()
            else _resolve_section_title(meta)
        )
        alias = _source_alias(citation_number, section_title, page_start, page_end)
        pdf_url = _source_pdf_url(file_name, document_id)
        if isinstance(page_start, int):
            pdf_url = f"{pdf_url}#page={page_start}"

        evidence = raw.get("evidence")
        evidence_snippet = _first_sentence(evidence) if isinstance(evidence, str) else ""
        alias_by_index[citation_number] = alias
        url_by_index[citation_number] = pdf_url
        source_rows.append(
            (
                citation_number,
                alias,
                file_name,
                page_start,
                page_end,
                section_title,
                evidence_snippet,
            )
        )
        source_rows_for_session.append(
            {
                "alias": alias,
                "file": file_name,
                "document_id": document_id,
                "page": page_start if isinstance(page_start, int) else None,
                "page_start": page_start if isinstance(page_start, int) else None,
                "page_end": page_end if isinstance(page_end, int) else None,
                "section": section_title if isinstance(section_title, str) else None,
                "evidence": evidence_snippet if evidence_snippet else None,
            }
        )
        seen_numbers.add(citation_number)

    return source_rows, alias_by_index, url_by_index, source_rows_for_session


async def _finalize_assistant_reply(
    *,
    session_id: str,
    messages: list[dict[str, Any]],
    content: str,
    source_rows: list[tuple[int, str, str, int | None, int | None, str | None, str]] | None = None,
    source_rows_for_session: list[dict[str, Any]] | None = None,
    alias_by_index: dict[int, str] | None = None,
    url_by_index: dict[int, str] | None = None,
) -> None:
    content = _strip_model_source_blocks(content or "")
    source_rows = list(source_rows or [])
    source_rows_for_session = _sanitize_source_rows_payload(source_rows_for_session or [])
    alias_by_index = dict(alias_by_index or {})
    url_by_index = dict(url_by_index or {})

    used_source_ids: list[int] = []
    citation_panel_content: str | None = None

    if source_rows:
        session_source_catalog = _sanitize_source_catalog(cl.user_session.get("source_catalog"))
        if not session_source_catalog.get("entries"):
            session_source_catalog = _load_session_source_catalog(session_id)
        source_catalog_changed = False
        if _prune_source_catalog(
            session_source_catalog,
            _source_ids_from_citation_history(cl.user_session.get("citation_history")),
        ):
            source_catalog_changed = True
        cl.user_session.set("source_catalog", session_source_catalog)

        alias_by_number = _alias_number_map(source_rows)
        url_by_number: dict[int, str] = {}
        alias_to_url = {alias: url_by_index.get(idx) for idx, alias in alias_by_index.items()}
        for src_idx, alias, *_ in source_rows:
            alias_url = alias_to_url.get(alias)
            if isinstance(alias_url, str) and alias_url:
                if isinstance(src_idx, int):
                    url_by_number[src_idx] = alias_url
                number_match = re.match(r"^\s*Quelle\s+(\d+)\s*:", alias, flags=re.IGNORECASE)
                if number_match:
                    url_by_number[int(number_match.group(1))] = alias_url

        alias_by_ref = {**alias_by_number, **alias_by_index}
        url_by_ref = {**url_by_number, **url_by_index}

        content = _inject_clickable_refs(
            content,
            alias_by_index,
            alias_by_ref,
            url_by_index,
            url_by_ref,
        )
        content = _inject_named_source_refs(content, source_rows)
        content = _inject_source_alias_links(content, alias_by_ref, url_by_ref)
        content = _normalize_source_alias_mentions(content, alias_by_index, alias_by_ref)
        content = _normalize_source_mentions_by_content(content, source_rows)
        content = _inject_naked_source_links(content)

        cited_aliases = set()
        for _, alias, *_ in source_rows:
            if not isinstance(alias, str) or not alias:
                continue
            escaped_alias = alias.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")
            if alias in content or escaped_alias in content:
                cited_aliases.add(alias)
        if cited_aliases:
            source_rows = [row for row in source_rows if row[1] in cited_aliases]
            source_rows_for_session = [
                row
                for row in source_rows_for_session
                if isinstance(row.get("alias"), str) and row["alias"] in cited_aliases
            ]

        resolved_source_rows: list[tuple[int, str, str, int | None, int | None, str | None, str]] = []
        resolved_source_rows_for_session: list[dict[str, Any]] = []
        for row, session_row in zip(source_rows, source_rows_for_session):
            _, alias, file_name, page_start, page_end, section_title, evidence = row
            source_id, _, catalog_changed = _register_source_in_catalog(
                session_source_catalog,
                file_name=file_name,
                page_start=page_start if isinstance(page_start, int) else None,
                page_end=page_end if isinstance(page_end, int) else None,
                section_title=section_title if isinstance(section_title, str) else None,
            )
            if catalog_changed:
                source_catalog_changed = True
            resolved_source_rows.append(
                (
                    source_id,
                    alias,
                    file_name,
                    page_start,
                    page_end,
                    section_title,
                    evidence,
                )
            )
            updated_session_row = dict(session_row)
            updated_session_row["source_id"] = source_id
            resolved_source_rows_for_session.append(updated_session_row)

        source_rows = resolved_source_rows
        source_rows_for_session = resolved_source_rows_for_session
        content, source_rows, source_rows_for_session = _align_aliases_to_source_ids(
            content,
            source_rows,
            source_rows_for_session,
        )

        if source_catalog_changed:
            sanitized_catalog = _sanitize_source_catalog(session_source_catalog)
            cl.user_session.set("source_catalog", sanitized_catalog)
            _persist_session_source_catalog(session_id, sanitized_catalog)

        used_source_ids = sorted(
            {
                source_id
                for source_id, *_ in source_rows
                if isinstance(source_id, int) and source_id > 0
            }
        )

        content = _inject_alias_links_by_rows(content, source_rows_for_session)

        box_lines: list[str] = []
        if source_rows:
            box_lines = ["## Quellen & Belegstellen", ""]
            # source_rows and source_rows_for_session are parallel arrays — zip to pull
            # document_id (only carried by the dict form) while still unpacking the tuple.
            for visible_idx, (row_tuple, row_dict) in enumerate(
                zip(source_rows, source_rows_for_session), start=1
            ):
                src_idx, alias, file_name, page_start, page_end, section_title, evidence = row_tuple
                page_label = _page_label(page_start, page_end)
                section_label = section_title or "Abschnitt unbekannt"
                document_id = row_dict.get("document_id") if isinstance(row_dict, dict) else None
                if not isinstance(document_id, str) or not document_id.strip():
                    document_id = None
                pdf_url = _source_pdf_url(file_name, document_id)
                page_for_link = page_start if isinstance(page_start, int) else None
                if isinstance(page_for_link, int):
                    pdf_url = f"{pdf_url}#page={page_for_link}"
                box_lines.append(f"### {alias}")
                box_lines.append(f"- Datei: `{file_name}`")
                box_lines.append(f"- PDF: [Öffnen]({pdf_url})")
                if isinstance(src_idx, int) and src_idx > 0:
                    box_lines.append(f"- Quellen-ID: {src_idx}")
                else:
                    box_lines.append(f"- Quellen-ID: {visible_idx}")
                box_lines.append(f"- Seiten: {page_label}")
                box_lines.append(f"- Abschnitt: {section_label}")
                if evidence:
                    box_lines.append(f"- Belegsnippet: \"{evidence}\"")
                box_lines.append("")
        detail_block = "\n".join(box_lines)
        citation_panel_content = detail_block or None
        if citation_panel_content:
            cl.user_session.set("citation_panel_content", citation_panel_content)
            cl.user_session.set("citation_source_rows", source_rows_for_session)
            citation_history = _sanitize_citation_history(cl.user_session.get("citation_history"))
            citation_history = _append_citation_history(
                citation_history,
                citation_panel_content,
                source_rows_for_session,
            )
            cl.user_session.set("citation_history", citation_history)
        else:
            cl.user_session.set("citation_panel_content", None)
            cl.user_session.set("citation_source_rows", [])
    else:
        cl.user_session.set("citation_panel_content", None)
        cl.user_session.set("citation_source_rows", [])

    content, followups = _extract_followups(content)
    followup_questions = _sanitize_followup_questions(followups)
    cl.user_session.set("followup_questions", followup_questions)

    message_metadata: dict[str, Any] = {
        "has_citations_panel": bool(citation_panel_content),
        "followup_count": len(followup_questions),
        "followup_questions": followup_questions,
        "used_source_ids": used_source_ids,
    }
    if citation_panel_content:
        message_metadata["citation_panel_content"] = citation_panel_content
        message_metadata["citation_source_rows"] = _sanitize_source_rows_payload(source_rows_for_session)

    assistant_reply = cl.Message(content=content, metadata=message_metadata)
    actions = _build_chat_actions(
        followup_questions=followup_questions,
        has_citations_panel=bool(citation_panel_content),
        source_step_id=assistant_reply.id,
        citation_panel_content=citation_panel_content,
        citation_source_rows=source_rows_for_session,
    )
    assistant_reply.actions = actions
    print("[DEBUG] followup_actions=", len(followup_questions), "total_actions=", len(actions))
    if citation_panel_content:
        _cache_citation_panel_content(assistant_reply.id, citation_panel_content)
        assistant_reply.elements = _build_citation_elements(
            citation_panel_content,
            source_rows_for_session,
            citation_step_id=assistant_reply.id,
        )

    # Attach one inline cl.Pdf(display="side") per cited source so clicking the alias
    # text in the message body opens the PDF in the right side panel instead of a new tab.
    inline_pdf_elements = _build_inline_pdf_elements(source_rows_for_session)
    if inline_pdf_elements:
        assistant_reply.elements = (assistant_reply.elements or []) + inline_pdf_elements

    await assistant_reply.send()
    if citation_panel_content:
        history_panel_content, history_rows = _build_citation_history_view(
            _sanitize_citation_history(cl.user_session.get("citation_history"))
        )
        use_history_sidebar = isinstance(history_panel_content, str) and history_panel_content.strip()
        sidebar_content = history_panel_content if use_history_sidebar else citation_panel_content
        sidebar_rows = history_rows if use_history_sidebar else source_rows_for_session
        if "/sources/pdf/" not in sidebar_content:
            sidebar_content = _append_source_links_to_panel(sidebar_content, sidebar_rows)
        await _show_citation_sidebar(
            sidebar_content,
            sidebar_rows,
            sidebar_title=(CITATION_HISTORY_SIDEBAR_TITLE if use_history_sidebar else CITATION_SIDEBAR_TITLE),
        )

    messages.append({"role": "assistant", "content": content})
    add_chat_message(
        CHAT_DB_PATH,
        session_id,
        "assistant",
        content,
        metadata=message_metadata,
    )


async def _handle_langflow_turn(message: cl.Message, messages: list[dict[str, Any]], session_id: str) -> bool:
    history_messages = messages[:-1] if messages else []
    try:
        langflow_result = await run_langflow(
            input_value=message.content,
            session_id=session_id,
            global_vars=_langflow_global_vars(history_messages),
        )
    except LangflowError as exc:
        print(f"[WARN] langflow_error: {exc}")
        await _finalize_assistant_reply(
            session_id=session_id,
            messages=messages,
            content=(
                "Langflow ist aktiviert, konnte diese Anfrage aber nicht verarbeiten.\n\n"
                "Bitte pruefen Sie den Flow, die API-Konfiguration und die Langflow-Logs."
            ),
        )
        return False

    content = str(langflow_result.get("answer_text") or "").strip()
    if not content:
        print("[WARN] langflow_error: empty answer_text")
        await _finalize_assistant_reply(
            session_id=session_id,
            messages=messages,
            content=(
                "Langflow ist aktiviert, hat aber keine Antwort zurueckgegeben.\n\n"
                "Bitte pruefen Sie die Ausgabe des Flows in Langflow."
            ),
        )
        return False

    raw_citations = langflow_result.get("citations")
    citations = raw_citations if isinstance(raw_citations, list) else []
    source_rows, alias_by_index, url_by_index, source_rows_for_session = _build_source_rows_from_langflow_citations(citations)
    cited_numbers = _cited_source_numbers(content)
    missing_numbers = sorted(number for number in cited_numbers if number not in alias_by_index)
    if missing_numbers:
        print(f"[WARN] langflow_missing_citations: missing structured citations for numbers={missing_numbers}")
        source_rows = []
        alias_by_index = {}
        url_by_index = {}
        source_rows_for_session = []
    if not source_rows and not _is_context_abstention(content):
        print("[WARN] langflow_missing_citations: no usable citations returned")

    await _finalize_assistant_reply(
        session_id=session_id,
        messages=messages,
        content=content,
        source_rows=source_rows,
        source_rows_for_session=source_rows_for_session,
        alias_by_index=alias_by_index,
        url_by_index=url_by_index,
    )
    return True


def _append_source_links_to_panel(panel_content: str, source_rows: list[dict[str, Any]]) -> str:
    if not isinstance(panel_content, str) or not panel_content.strip():
        return panel_content
    rows = _sanitize_source_rows_payload(source_rows)
    if not rows:
        return panel_content

    # TODO(citations-ux): Re-evaluate whether source access should be links only (current),
    # sidebar preview only, or dual mode. Links are currently preferred for reliability
    # across resume/reload and container restarts.
    base = re.sub(r"\n+### PDF öffnen[\s\S]*$", "", panel_content.strip(), flags=re.IGNORECASE).strip()

    lines: list[str] = []
    seen: set[tuple[str, int | None, str]] = set()
    for row in rows:
        file_name = row.get("file")
        alias = row.get("alias")
        page = row.get("page")
        if not isinstance(file_name, str) or not isinstance(alias, str):
            continue
        key = (file_name, page if isinstance(page, int) else None, alias)
        if key in seen:
            continue
        seen.add(key)
        document_id = row.get("document_id") if isinstance(row.get("document_id"), str) else None
        pdf_url = _source_pdf_url(file_name, document_id)
        if isinstance(page, int):
            pdf_url = f"{pdf_url}#page={page}"
        label = alias
        if isinstance(page, int) and not re.search(r"\(S\.?\s*\d", alias, flags=re.IGNORECASE):
            label = f"{alias} (S.{page})"
        lines.append(f"- {_markdown_link(label, pdf_url)}")

    if not lines:
        return base or panel_content

    return f"{base}\n\n### PDF öffnen\n" + "\n".join(lines)


def _build_citation_elements(
    panel_content: str,
    source_rows: list[dict[str, Any]],
    *,
    include_panel_text: bool = True,
    citation_step_id: str | None = None,
) -> list[Any]:
    elements: list[Any] = []
    if include_panel_text:
        if isinstance(citation_step_id, str) and citation_step_id.strip():
            elements.append(cl.Text(name="CITATIONS_PANEL", url=_citation_panel_url(citation_step_id), display="side"))
        else:
            elements.append(cl.Text(name="CITATIONS_PANEL", content=panel_content, display="side"))
    return elements


def _build_inline_pdf_elements(source_rows: list[dict[str, Any]] | None) -> list[Any]:
    """One cl.Pdf(display="side") per cited source so Chainlit matches the bare alias
    text in the message body and opens the PDF in the right side panel. De-dup on alias
    because the alias is what appears in the text; a doc_id-backed row and a legacy
    row both produce a valid url via _source_pdf_url."""
    elements: list[Any] = []
    if not source_rows:
        return elements
    seen: set[str] = set()
    for row in source_rows:
        if not isinstance(row, dict):
            continue
        alias = row.get("alias")
        file_name = row.get("file")
        if not isinstance(alias, str) or not alias.strip():
            continue
        if not isinstance(file_name, str) or not file_name.strip():
            continue
        if alias in seen:
            continue
        document_id = row.get("document_id") if isinstance(row.get("document_id"), str) else None
        # Legacy (non-doc_id) rows must still resolve against DATA_RAW_DIR.
        if not document_id and _resolve_source_pdf_path(file_name) is None:
            continue
        seen.add(alias)
        page = row.get("page_start") if isinstance(row.get("page_start"), int) else row.get("page")
        page_int = page if isinstance(page, int) else 1
        elements.append(
            cl.Pdf(
                name=alias,
                url=_source_pdf_url(file_name, document_id),
                page=page_int,
                display="side",
            )
        )
    return elements


def _sanitize_followup_questions(raw_followups: Any, *, max_items: int = 8) -> list[str]:
    if not isinstance(raw_followups, list):
        return []
    cleaned: list[str] = []
    seen: set[str] = set()
    for question in raw_followups:
        if not isinstance(question, str):
            continue
        normalized = question.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        cleaned.append(normalized)
        if len(cleaned) >= max_items:
            break
    return cleaned


def _build_chat_actions(
    *,
    followup_questions: list[str],
    has_citations_panel: bool,
    source_step_id: str,
    citation_panel_content: str | None = None,
    citation_source_rows: list[dict[str, Any]] | None = None,
) -> list[cl.Action]:
    normalized_followups = _sanitize_followup_questions(followup_questions)
    base_payload: dict[str, Any] = {
        "source_step_id": source_step_id,
        "followup_questions": normalized_followups,
        "has_citations_panel": has_citations_panel,
    }
    actions: list[cl.Action] = []
    if has_citations_panel:
        if isinstance(citation_panel_content, str) and citation_panel_content.strip():
            base_payload["citation_panel_content"] = citation_panel_content
        cleaned_source_rows = _sanitize_source_rows_payload(citation_source_rows or [])
        if cleaned_source_rows:
            base_payload["citation_source_rows"] = cleaned_source_rows
        actions.append(
            cl.Action(
                name="open_all_citations",
                label="Quellen anzeigen",
                tooltip="Alle Quellen erneut im Seitenpanel anzeigen",
                payload={
                    **base_payload,
                    "show_history": True,
                },
            )
        )
    for question in normalized_followups:
        actions.append(
            cl.Action(
                name="ask_followup",
                label=question,
                tooltip=question,
                payload={
                    **base_payload,
                    "question": question,
                },
            )
        )
    return actions


async def _restore_actions_for_step(
    step_id: str | None,
    *,
    followup_questions: list[str],
    has_citations_panel: bool,
    citation_panel_content: str | None = None,
    citation_source_rows: list[dict[str, Any]] | None = None,
) -> None:
    if not isinstance(step_id, str) or not step_id.strip():
        return
    actions = _build_chat_actions(
        followup_questions=followup_questions,
        has_citations_panel=has_citations_panel,
        source_step_id=step_id,
        citation_panel_content=citation_panel_content,
        citation_source_rows=citation_source_rows,
    )
    if not actions:
        return
    for action in actions:
        await action.send(for_id=step_id)


async def _show_citation_sidebar(
    panel_content: str,
    source_rows: list[dict[str, Any]],
    *,
    citation_step_id: str | None = None,
    sidebar_title: str = CITATION_SIDEBAR_TITLE,
) -> None:
    elements = _build_citation_elements(
        panel_content,
        source_rows,
        citation_step_id=citation_step_id,
    )
    if not elements:
        return
    await cl.ElementSidebar.set_title(sidebar_title)
    # Force a refresh even when the sidebar key is unchanged.
    await cl.ElementSidebar.set_elements([], key="citations_panel")
    await cl.ElementSidebar.set_elements(elements, key="citations_panel")


def _hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def _verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against a bcrypt hash."""
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


@cl.password_auth_callback
async def auth_callback(username: str, password: str) -> cl.User | None:
    # Try database authentication first (if DATABASE_URL is configured)
    if DATABASE_URL:
        user = await get_user_by_identifier(DATABASE_URL, username)
        if user and user.get("password_hash"):
            if not _verify_password(password, user["password_hash"]):
                return None  # Wrong password for existing user
            row = await upsert_user_on_login(
                DATABASE_URL,
                identifier=user["identifier"],
                provider="local",
                email=user.get("email"),
                admin_identifiers=ADMIN_IDENTIFIERS,
            )
            status = row.get("status", "approved")
            if status == "blocked":
                print(f"[AUTH] rejecting blocked user: {user['identifier']}")
                return None
            metadata = json.loads(user.get("metadata") or "{}")
            metadata["provider"] = "local"
            metadata["role"] = row.get("role", "user")
            metadata["status"] = status
            return cl.User(identifier=user["identifier"], metadata=metadata)

    # Fallback to environment variable authentication (for backwards compatibility / admin)
    expected_user = CHAINLIT_AUTH_USERNAME or "admin"
    expected_password = CHAINLIT_AUTH_PASSWORD
    if expected_password and username == expected_user and password == expected_password:
        # Upsert the env-admin so the admin UI can target them like any other user.
        if DATABASE_URL:
            await upsert_user_on_login(
                DATABASE_URL,
                identifier=expected_user,
                provider="password",
                admin_identifiers=[*ADMIN_IDENTIFIERS, expected_user],
            )
        return cl.User(
            identifier=expected_user,
            metadata={"provider": "password", "role": "admin", "status": "approved"},
        )

    return None


@cl.on_app_startup
async def on_app_startup() -> None:
    print(
        "[STARTUP] system_prompt_path=",
        str(SYSTEM_PROMPT_PATH),
        "exists=",
        SYSTEM_PROMPT_PATH.is_file(),
        "loaded=",
        bool(SYSTEM_PROMPT),
    )
    print(
        "[STARTUP] retrieval_tuning",
        "embed_model=",
        EMBED_MODEL,
        "top_k=",
        TOP_K,
        "| mode: simple_docling",
    )
    from chainlit.server import app as chainlit_fastapi_app

    if DATABASE_URL and CHAINLIT_INIT_DB:
        await ensure_native_schema(DATABASE_URL)

    init_chat_db(CHAT_DB_PATH)
    init_user_kb_db(CHAT_DB_PATH)
    CHAT_EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Register user-scoped API routes + static settings UI regardless of whether
    # the Postgres native chat schema is configured — they only depend on SQLite
    # + Chainlit auth.
    if not getattr(chainlit_fastapi_app.state, "user_api_routes_added", False):
        from api.api_routes import register_user_api_routes

        user_routes = register_user_api_routes(chainlit_fastapi_app, SYSTEM_PROMPT)

        static_dir = Path(__file__).parent / "static"
        static_dir.mkdir(parents=True, exist_ok=True)

        from fastapi import Cookie
        from chainlit.auth import authenticate_user as _auth_user

        @chainlit_fastapi_app.get("/settings/app")
        async def serve_settings_page(access_token: str | None = Cookie(default=None)):
            if not access_token:
                return RedirectResponse(url="/login", status_code=307)
            try:
                await _auth_user(access_token)
            except HTTPException:
                return RedirectResponse(url="/login", status_code=307)
            return FileResponse(str(static_dir / "settings.html"), media_type="text/html")

        @chainlit_fastapi_app.get("/admin/app")
        async def serve_admin_page(access_token: str | None = Cookie(default=None)):
            if not access_token:
                return RedirectResponse(url="/login", status_code=307)
            try:
                user = await _auth_user(access_token)
            except HTTPException:
                return RedirectResponse(url="/login", status_code=307)
            identifier = getattr(user, "identifier", None)
            if not identifier or not DATABASE_URL:
                return RedirectResponse(url="/", status_code=307)
            pair = await get_user_role_status(DATABASE_URL, identifier)
            if pair is None or pair[0] != "admin":
                return RedirectResponse(url="/", status_code=307)
            return FileResponse(str(static_dir / "admin.html"), media_type="text/html")

        for path in user_routes:
            _ensure_route_precedes_catch_all(chainlit_fastapi_app, path)
        _ensure_route_precedes_catch_all(chainlit_fastapi_app, "/settings/app")
        _ensure_route_precedes_catch_all(chainlit_fastapi_app, "/admin/app")

        chainlit_fastapi_app.state.user_api_routes_added = True
        print("[STARTUP] user-scoped API routes registered: /api/kbs, /api/settings/...")

    if not DATABASE_URL:
        return

    if getattr(chainlit_fastapi_app.state, "native_export_route_added", False):
        return

    @chainlit_fastapi_app.get("/sources/pdf/{identifier:path}")
    async def source_pdf(identifier: str, current_user=Depends(get_current_user)):
        if current_user is None:
            raise HTTPException(status_code=401, detail="Unauthorized")

        # First try: treat identifier as a kb_documents.id (user-uploaded file).
        # UUID check is loose on purpose — record_document generates uuid4s, but
        # we don't want to pre-reject other shapes if future docs use a different id.
        if re.fullmatch(r"[0-9a-fA-F-]{8,64}", identifier):
            doc = get_document(CHAT_DB_PATH, identifier)
            if doc is not None:
                kb = get_kb(CHAT_DB_PATH, doc["kb_id"])
                user_identifier = getattr(current_user, "identifier", None)
                if kb is None or kb.get("user_id") != user_identifier:
                    # 404 (not 403) to avoid leaking existence across tenants.
                    raise HTTPException(status_code=404, detail="Source PDF not found")
                ext = Path(doc["file_name"]).suffix.lower()
                doc_root = DATA_KB_DOCS_DIR.resolve()
                persisted = (DATA_KB_DOCS_DIR / f"{doc['id']}{ext}").resolve()
                try:
                    persisted.relative_to(doc_root)
                except ValueError:
                    raise HTTPException(status_code=404, detail="Source PDF not found")
                if not persisted.is_file() or persisted.suffix.lower() != ".pdf":
                    raise HTTPException(status_code=404, detail="Source PDF not found")
                return FileResponse(
                    path=str(persisted),
                    media_type="application/pdf",
                    headers={"Content-Disposition": "inline"},
                )

        # Fallback: legacy flat-file lookup in DATA_RAW_DIR (shared Grundschutz PDF
        # and any historical citation URLs that predate the doc_id scheme).
        file_path = _resolve_source_pdf_path(identifier)
        if file_path is None:
            raise HTTPException(status_code=404, detail="Source PDF not found")

        return FileResponse(
            path=str(file_path),
            media_type="application/pdf",
            headers={"Content-Disposition": "inline"},
        )

    @chainlit_fastapi_app.get("/sources/citations/{step_id}")
    async def source_citations(step_id: str, current_user=Depends(get_current_user)):
        if current_user is None:
            raise HTTPException(status_code=401, detail="Unauthorized")

        panel_content = await _load_citation_panel_content(step_id)
        if not isinstance(panel_content, str) or not panel_content.strip():
            raise HTTPException(status_code=404, detail="Citation panel not found")

        return PlainTextResponse(content=panel_content, media_type="text/plain; charset=utf-8")

    @chainlit_fastapi_app.get("/export/all-chats")
    async def export_all_chats(current_user=Depends(get_current_user)):
        if current_user is None:
            raise HTTPException(status_code=401, detail="Unauthorized")
        user_id = getattr(current_user, "id", None)
        bundle = await export_all_chats_zip(
            database_url=DATABASE_URL,
            out_dir=CHAT_EXPORT_DIR,
            user_id=str(user_id) if user_id else None,
        )
        return FileResponse(path=str(bundle), media_type="application/zip", filename=bundle.name)

    # Registration endpoint for self-registration
    class RegisterRequest(BaseModel):
        username: str
        email: str
        password: str

    @chainlit_fastapi_app.post("/auth/register")
    async def register_user(request: RegisterRequest):
        # Validate input
        if not request.username or len(request.username) < 3:
            raise HTTPException(status_code=400, detail="Benutzername muss mindestens 3 Zeichen haben")
        if not request.email or "@" not in request.email:
            raise HTTPException(status_code=400, detail="Ungültige E-Mail-Adresse")
        if not request.password or len(request.password) < 8:
            raise HTTPException(status_code=400, detail="Passwort muss mindestens 8 Zeichen haben")

        # Check if user/email already exists
        exists = await check_user_exists(DATABASE_URL, request.username, request.email)
        if exists["username_exists"]:
            raise HTTPException(status_code=409, detail="Benutzername bereits vergeben")
        if exists["email_exists"]:
            raise HTTPException(status_code=409, detail="E-Mail-Adresse bereits registriert")

        # Create user with hashed password
        password_hash = _hash_password(request.password)
        user = await create_user(DATABASE_URL, request.username, request.email, password_hash)
        if user is None:
            raise HTTPException(status_code=500, detail="Registrierung fehlgeschlagen")

        return {"message": "Registrierung erfolgreich", "username": user["identifier"]}

    _ensure_route_precedes_catch_all(chainlit_fastapi_app, "/sources/pdf/{identifier:path}")
    _ensure_route_precedes_catch_all(chainlit_fastapi_app, "/sources/citations/{step_id}")
    _ensure_route_precedes_catch_all(chainlit_fastapi_app, "/export/all-chats")
    _ensure_route_precedes_catch_all(chainlit_fastapi_app, "/auth/register")

    chainlit_fastapi_app.state.native_export_route_added = True
    print("[STARTUP] native export route registered at /export/all-chats")
    print("[STARTUP] registration route registered at /auth/register")


@cl.on_chat_resume
async def on_chat_resume(thread: dict[str, Any]):
    thread_id = thread.get("id")
    session_source_catalog = _empty_source_catalog()
    if isinstance(thread_id, str) and thread_id.strip():
        create_chat_session(CHAT_DB_PATH, thread_id)
        cl.user_session.set("chat_history_session_id", thread_id)
        session_source_catalog = _load_session_source_catalog(thread_id)

    messages: list[dict[str, Any]] = []
    restored_panel_content: str | None = None
    restored_source_rows: list[dict[str, Any]] = []
    restored_followup_questions: list[str] = []
    restored_citation_history: list[dict[str, Any]] = []
    latest_assistant_step_id: str | None = None
    latest_assistant_has_actions = False
    if SYSTEM_PROMPT:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})

    steps = thread.get("steps") or []
    sorted_steps = sorted(
        [s for s in steps if isinstance(s, dict)],
        key=lambda s: (s.get("start") or s.get("createdAt") or "", s.get("id") or ""),
    )
    for step in sorted_steps:
        step_type = str(step.get("type") or "").lower()
        if "user_message" in step_type:
            text = _coerce_step_text(step.get("output") or step.get("input"))
            if text:
                messages.append({"role": "user", "content": text})
        elif "assistant_message" in step_type:
            text = _coerce_step_text(step.get("output") or step.get("input"))
            if text:
                messages.append({"role": "assistant", "content": text})
            step_id = step.get("id")
            if isinstance(step_id, str) and step_id.strip():
                latest_assistant_step_id = step_id
                step_actions = step.get("actions")
                latest_assistant_has_actions = isinstance(step_actions, list) and len(step_actions) > 0
            metadata = _coerce_step_metadata(step)
            panel_content = metadata.get("citation_panel_content")
            source_rows = metadata.get("citation_source_rows")
            followup_questions = metadata.get("followup_questions")
            if isinstance(panel_content, str) and panel_content.strip():
                restored_panel_content = panel_content
            valid_rows = _sanitize_source_rows_payload(source_rows)
            if valid_rows:
                restored_source_rows = valid_rows
            restored_citation_history = _append_citation_history(
                restored_citation_history,
                panel_content if isinstance(panel_content, str) else None,
                valid_rows,
            )
            valid_followups = _sanitize_followup_questions(followup_questions)
            if valid_followups:
                restored_followup_questions = valid_followups

    cl.user_session.set("messages", messages)
    cl.user_session.set("citation_panel_content", restored_panel_content)
    cl.user_session.set("citation_source_rows", restored_source_rows)
    cl.user_session.set("followup_questions", restored_followup_questions)
    cl.user_session.set("citation_history", restored_citation_history)
    cl.user_session.set("source_catalog", session_source_catalog)

    citation_panel_for_actions: str | None = restored_panel_content
    citation_source_rows_for_actions: list[dict[str, Any]] = _sanitize_source_rows_payload(restored_source_rows)
    if isinstance(restored_panel_content, str) and restored_panel_content.strip():
        panel_with_links = restored_panel_content
        if "/sources/pdf/" not in panel_with_links:
            panel_with_links = _append_source_links_to_panel(restored_panel_content, citation_source_rows_for_actions)
        cl.user_session.set("citation_panel_content", panel_with_links)
        citation_panel_for_actions = panel_with_links

    # Intentionally do NOT auto-open the citation sidebar on resume.
    # ElementSidebar.set_title / set_elements force-open the sidebar as a side effect
    # of populating it, which would pop an unrequested panel every time the user
    # returns to a chat. The sidebar stays reachable via the per-step "Quellen
    # anzeigen" action (restored below). Inline cl.Pdf(display="side") elements are
    # also NOT reattached here: emitting them on historical steps would pop the
    # sidebar to the last alias. Fresh answers in the resumed session still get
    # inline PDFs via the hot path in main().

    if not latest_assistant_has_actions:
        await _restore_actions_for_step(
            latest_assistant_step_id,
            followup_questions=restored_followup_questions,
            has_citations_panel=bool(isinstance(citation_panel_for_actions, str) and citation_panel_for_actions.strip()),
            citation_panel_content=citation_panel_for_actions,
            citation_source_rows=citation_source_rows_for_actions,
        )

    # Defensive close: if anything in the restore path opened the side view,
    # set_elements([]) maps to setSideView(undefined) on the Chainlit frontend —
    # safe idempotent no-op when already closed.
    try:
        await cl.ElementSidebar.set_elements([], key="citations_panel")
    except Exception:  # noqa: BLE001
        pass


@cl.set_chat_profiles
async def set_chat_profiles():
    """Chat profiles are now managed via settings for persistence.
    
    We return an empty list to disable the startup profile selector.
    The profile can be changed in the chat settings (sidebar).
    """
    return []


def _build_chat_settings(current_profile: str | None = None):
    """Build ChatSettings with the per-session role selector."""
    profiles = CHAT_PROFILES_CONFIG.get("profiles", [])
    profile_names = [p.get("name", "") for p in profiles if p.get("name")]
    if not profile_names:
        return None

    initial_index = 0
    if current_profile and current_profile in profile_names:
        initial_index = profile_names.index(current_profile)

    return cl.ChatSettings([
        Select(
            id="chat_profile",
            label="Ihre Rolle",
            description="Wählen Sie Ihre Rolle für angepasste Antworten",
            values=profile_names,
            initial_index=initial_index,
        )
    ])


@cl.on_settings_update
async def on_settings_update(settings: dict[str, Any]):
    """Persist ChatSettings edits: only the per-session role selector."""
    user_id = cl.user_session.get("current_user_id")

    changed_messages: list[str] = []

    new_profile_name = settings.get("chat_profile")
    if new_profile_name:
        if user_id:
            set_user_selected_chat_profile(CHAT_DB_PATH, user_id, new_profile_name)
        chat_profile_config = _get_profile_by_name(new_profile_name)
        cl.user_session.set("chat_profile", new_profile_name)
        cl.user_session.set("chat_profile_config", chat_profile_config)
        changed_messages.append(f"Rolle: **{new_profile_name}**")

    # Rebuild the active system prompt from the new state so this chat picks
    # up the change immediately.
    user_system_prompt_override: str | None = None
    if user_id:
        user_system_prompt_override = get_user_system_prompt(CHAT_DB_PATH, user_id)
    base_system_prompt = user_system_prompt_override or SYSTEM_PROMPT
    system_prompt = base_system_prompt

    chat_profile_config = cl.user_session.get("chat_profile_config")
    if system_prompt and chat_profile_config:
        profile_prompt = chat_profile_config.get("prompt_context", "")
        if profile_prompt:
            system_prompt = f"{system_prompt}\n\n## ROLLENKONTEXT\n{profile_prompt}"

    user_profile = cl.user_session.get("user_profile")
    if system_prompt and user_profile and user_profile.topics:
        personalization_context = _build_personalization_prompt(user_profile)
        system_prompt = f"{system_prompt}\n\n{personalization_context}"

    # Refresh the KB catalog block (KB list may have changed).
    kb_catalog = _build_user_kb_catalog(user_id)
    cl.user_session.set("kb_catalog", kb_catalog)
    kb_block = _format_kb_catalog_for_prompt(kb_catalog)
    if system_prompt and kb_block:
        system_prompt = f"{system_prompt}\n\n{kb_block}"

    messages = cl.user_session.get("messages") or []
    if messages and isinstance(messages[0], dict) and messages[0].get("role") == "system":
        messages[0]["content"] = system_prompt
    elif system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})
    cl.user_session.set("messages", messages)

    if changed_messages:
        await cl.Message(
            author="System",
            content="Gespeichert: " + "; ".join(changed_messages) + ".",
        ).send()


@cl.on_chat_start
async def on_chat_start():
    existing_session_id = cl.user_session.get("chat_history_session_id")
    resume_session_id = existing_session_id if isinstance(existing_session_id, str) and existing_session_id.strip() else None
    # Use Chainlit's own thread_id (set by the websocket session before on_chat_start
    # fires and reused by on_chat_resume as thread["id"]) so our SQLite session_id
    # stays aligned with Chainlit's Postgres thread_id. A fresh uuid4() here would
    # orphan our source_catalog whenever the user leaves and returns to the chat —
    # resume would look up an empty row and restart numbering from 1.
    thread_id = getattr(getattr(cl.context, "session", None), "thread_id", None)
    session_id = resume_session_id or (thread_id if isinstance(thread_id, str) and thread_id.strip() else str(uuid4()))
    resumed_session = resume_session_id is not None

    # Get authenticated user ID if available
    # Chainlit stores user in session after auth callback
    user = cl.user_session.get("user")
    user_id = None
    if user:
        # Try different attribute names Chainlit might use
        user_id = getattr(user, "identifier", None) or getattr(user, "id", None)

    # Load persisted chat profile for authenticated users (persistent across sessions)
    chat_profile_name = None
    if user_id:
        chat_profile_name = get_user_selected_chat_profile(CHAT_DB_PATH, user_id)
    
    # Fall back to default profile if none persisted
    if not chat_profile_name:
        chat_profile_name = CHAT_PROFILES_CONFIG.get("default_profile")
        # Find the profile name for the default_profile id
        if chat_profile_name:
            for p in CHAT_PROFILES_CONFIG.get("profiles", []):
                if p.get("id") == chat_profile_name:
                    chat_profile_name = p.get("name")
                    break
    
    chat_profile_config = _get_profile_by_name(chat_profile_name) if chat_profile_name else None
    cl.user_session.set("chat_profile", chat_profile_name)
    cl.user_session.set("chat_profile_config", chat_profile_config)

    print(
        f"[DEBUG] on_chat_start: user={user}, user_id={user_id}, chat_profile={chat_profile_name}, "
        f"resumed_session={resumed_session}, session_id={session_id}"
    )

    create_chat_session(
        CHAT_DB_PATH,
        session_id,
        user_id=user_id,
        metadata={
            "system_prompt_loaded": bool(SYSTEM_PROMPT),
            "chat_profile": chat_profile_name,
            "source_catalog": _empty_source_catalog(),
        },
    )
    cl.user_session.set("chat_history_session_id", session_id)
    cl.user_session.set("current_user_id", user_id)
    cl.user_session.set("source_catalog", _load_session_source_catalog(session_id))

    # Load or initialize user profile for personalization
    user_profile: UserProfile | None = None
    if PERSONALIZATION_ENABLED and user_id:
        user_profile = await load_user_profile(user_id)
        if user_profile and user_profile.has_sufficient_history():
            print(f"[DEBUG] on_chat_start: loaded profile for {user_id}, topics={user_profile.topics}")
        else:
            # Check if user has enough messages to generate profile
            msg_count = get_user_message_count(CHAT_DB_PATH, user_id)
            if msg_count >= PROFILE_MIN_MESSAGES:
                print(f"[DEBUG] on_chat_start: generating profile for {user_id}, msg_count={msg_count}")
                user_profile = await update_user_profile(user_id)
    cl.user_session.set("user_profile", user_profile)

    # Build system prompt: per-user override falls back to SYSTEM_PROMPT from system.md
    user_system_prompt_override: str | None = None
    if user_id:
        user_system_prompt_override = get_user_system_prompt(CHAT_DB_PATH, user_id)
    base_system_prompt = user_system_prompt_override or SYSTEM_PROMPT
    system_prompt = base_system_prompt

    # Add chat profile context if selected
    if system_prompt and chat_profile_config:
        profile_prompt = chat_profile_config.get("prompt_context", "")
        if profile_prompt:
            system_prompt = f"{system_prompt}\n\n## ROLLENKONTEXT\n{profile_prompt}"

    # Add personalization context if available
    if system_prompt and user_profile and user_profile.topics:
        personalization_context = _build_personalization_prompt(user_profile)
        system_prompt = f"{system_prompt}\n\n{personalization_context}"

    # Build the user's KB catalog and append its description to the system prompt.
    kb_catalog = _build_user_kb_catalog(user_id)
    cl.user_session.set("kb_catalog", kb_catalog)
    kb_block = _format_kb_catalog_for_prompt(kb_catalog)
    if system_prompt and kb_block:
        system_prompt = f"{system_prompt}\n\n{kb_block}"

    existing_messages = cl.user_session.get("messages")
    messages: list[dict[str, Any]]
    if resumed_session and isinstance(existing_messages, list) and existing_messages:
        messages = existing_messages
        if system_prompt:
            if messages and isinstance(messages[0], dict) and messages[0].get("role") == "system":
                messages[0]["content"] = system_prompt
            else:
                messages.insert(0, {"role": "system", "content": system_prompt})
    else:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            add_chat_message(CHAT_DB_PATH, session_id, "system", system_prompt)
    cl.user_session.set("messages", messages)

    # Send chat settings with just the role selector (rest lives on /settings/app)
    chat_settings = _build_chat_settings(chat_profile_name)
    if chat_settings:
        await chat_settings.send()


@cl.set_starters
async def set_starters(user: cl.User | None = None) -> list[Starter]:
    starter_icons = [
        "/public/icons/shield.svg",
        "/public/icons/search.svg",
        "/public/icons/book.svg",
    ]

    # Prefer per-user starters when we can resolve a user id; otherwise fall back
    # to the env-backed defaults so the welcome screen still has questions for
    # unauthenticated / pre-login renders.
    user_id: str | None = None
    if user is not None:
        user_id = getattr(user, "identifier", None) or getattr(user, "id", None)

    rendered: list[tuple[str, str]] = []
    if user_id:
        try:
            for row in list_user_starters(CHAT_DB_PATH, user_id):
                label = (row.get("label") or "").strip()
                message = (row.get("message") or "").strip()
                if message:
                    rendered.append((label or message, message))
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] set_starters: load_user_starters failed user_id={user_id} err={exc}")

    if not rendered:
        rendered = [(q, q) for q in STARTER_QUESTIONS]

    starters: list[Starter] = []
    for i, (label, message) in enumerate(rendered[:6]):
        starters.append(
            Starter(
                label=label if len(label) <= 70 else label[:67].rstrip() + "...",
                message=message,
                icon=starter_icons[i % len(starter_icons)],
            )
        )
    return starters


@cl.action_callback("open_source_pdf")
async def open_source_pdf(action: cl.Action):
    file_name = action.payload.get("file")
    page = action.payload.get("page")
    if not isinstance(file_name, str):
        return
    file_path = _resolve_source_pdf_path(file_name)
    if file_path is None:
        await cl.Message(content=f"Datei nicht gefunden: {file_name}").send()
        return

    pdf_name = f"{file_name} (S.{page})" if isinstance(page, int) else file_name
    element = cl.Pdf(name=pdf_name, url=_source_pdf_url(file_name), page=page if isinstance(page, int) else 1, display="side")
    await cl.Message(content=f"Quelle geöffnet: {pdf_name}", elements=[element]).send()


@cl.action_callback("open_all_citations")
async def open_all_citations(action: cl.Action):
    payload = action.payload if isinstance(action.payload, dict) else {}
    show_history = bool(payload.get("show_history"))
    payload_panel_content = payload.get("citation_panel_content")
    payload_source_rows = payload.get("citation_source_rows")

    latest_panel_content = (
        payload_panel_content
        if isinstance(payload_panel_content, str) and payload_panel_content.strip()
        else cl.user_session.get("citation_panel_content")
    )
    latest_source_rows = _sanitize_source_rows_payload(payload_source_rows)
    if not latest_source_rows:
        latest_source_rows = _sanitize_source_rows_payload(cl.user_session.get("citation_source_rows"))

    panel_content: str | None = latest_panel_content if isinstance(latest_panel_content, str) else None
    source_rows: list[dict[str, Any]] = latest_source_rows
    sidebar_title = CITATION_SIDEBAR_TITLE
    if show_history:
        history_panel_content, history_rows = _build_citation_history_view(
            _sanitize_citation_history(cl.user_session.get("citation_history"))
        )
        if isinstance(history_panel_content, str) and history_panel_content.strip():
            panel_content = history_panel_content
            source_rows = history_rows
            sidebar_title = CITATION_HISTORY_SIDEBAR_TITLE

    if not isinstance(panel_content, str) or not panel_content.strip():
        await cl.Message(content="Keine Zitierungen verfügbar.").send()
        return

    panel_content_with_links = panel_content
    if "/sources/pdf/" not in panel_content_with_links:
        panel_content_with_links = _append_source_links_to_panel(panel_content, source_rows)
    if sidebar_title == CITATION_SIDEBAR_TITLE:
        cl.user_session.set("citation_panel_content", panel_content)
        cl.user_session.set("citation_source_rows", source_rows)

    await _show_citation_sidebar(
        panel_content_with_links,
        source_rows,
        sidebar_title=sidebar_title,
    )


@cl.action_callback("ask_followup")
async def ask_followup(action: cl.Action):
    payload = action.payload if isinstance(action.payload, dict) else {}
    question = payload.get("question")
    if not isinstance(question, str) or not question.strip():
        return
    await cl.Message(content=question, author="You", type="user_message").send()
    await main(cl.Message(content=question))


async def _maybe_update_profile() -> None:
    if not PERSONALIZATION_ENABLED:
        return
    user_id = cl.user_session.get("current_user_id")
    if not user_id:
        return
    try:
        current_profile = cl.user_session.get("user_profile")
        current_count = get_user_message_count(CHAT_DB_PATH, user_id)
        profile_count = current_profile.message_count if current_profile else 0
        if current_count >= PROFILE_MIN_MESSAGES and current_count - profile_count >= 10:
            print(f"[DEBUG] triggering profile update for {user_id}, new_messages={current_count - profile_count}")
            updated_profile = await update_user_profile(user_id)
            cl.user_session.set("user_profile", updated_profile)
    except Exception as e:
        print(f"[WARN] profile_update_failed for user_id={user_id}: {e.__class__.__name__}: {e}")


@cl.on_message
async def main(message: cl.Message):
    if await _handle_control_message(message):
        return

    messages = cl.user_session.get("messages") or []
    session_id = _current_chat_session_id()
    if not session_id:
        # Defensive fallback — normally set by on_chat_start / on_chat_resume.
        # Use Chainlit's thread_id so we don't orphan the catalog across reconnects.
        thread_id = getattr(getattr(cl.context, "session", None), "thread_id", None)
        session_id = thread_id if isinstance(thread_id, str) and thread_id.strip() else str(uuid4())
        create_chat_session(CHAT_DB_PATH, session_id)
        cl.user_session.set("chat_history_session_id", session_id)

    messages.append({"role": "user", "content": message.content})
    add_chat_message(CHAT_DB_PATH, session_id, "user", message.content)
    set_session_title_if_missing(CHAT_DB_PATH, session_id, _first_sentence(message.content, max_len=96))

    if LANGFLOW_ENABLED:
        langflow_ok = await _handle_langflow_turn(message, messages, session_id)
        cl.user_session.set("messages", messages)
        if langflow_ok:
            await _maybe_update_profile()
        return

    response = await chat(messages, tools=TOOLS, tool_choice="required")
    assistant_msg = response.choices[0].message
    print(
        "[DEBUG] first_call",
        "content_empty=",
        not bool(assistant_msg.content),
        "tool_calls=",
        bool(getattr(assistant_msg, "tool_calls", None)),
    )

    if not getattr(assistant_msg, "tool_calls", None):
        print("[WARN] first_call_without_tool_retrying")
        retry_messages = [
            *messages,
            {
                "role": "system",
                "content": "Rufe zuerst das Tool rag_retrieve auf, bevor du antwortest.",
            },
        ]
        retry_response = await chat(retry_messages, tools=TOOLS, tool_choice="required")
        retry_msg = retry_response.choices[0].message
        print(
            "[DEBUG] first_call_retry",
            "content_empty=",
            not bool(retry_msg.content),
            "tool_calls=",
            bool(getattr(retry_msg, "tool_calls", None)),
        )
        if getattr(retry_msg, "tool_calls", None):
            assistant_msg = retry_msg

    if getattr(assistant_msg, "tool_calls", None):
        citations_text: str | None = None
        last_results = []
        content = ""
        current_msg = assistant_msg
        aggregated_by_key: dict[tuple[str, int | None, str], Any] = {}
        cached_tool_payloads: dict[str, tuple[list[Any], dict[str, Any]]] = {}

        max_tool_rounds_raw = os.getenv("MAX_TOOL_CALL_ROUNDS", "12")
        try:
            max_tool_rounds = max(1, int(max_tool_rounds_raw))
        except ValueError:
            max_tool_rounds = 12
        tool_round = 0

        while getattr(current_msg, "tool_calls", None) and tool_round < max_tool_rounds:
            tool_round += 1
            messages.append(message_to_dict(current_msg))
            print(
                "[DEBUG] tool_round_start",
                "round=",
                tool_round,
                "tool_calls=",
                len(current_msg.tool_calls),
            )
            for tool_call in current_msg.tool_calls:
                function_name = getattr(getattr(tool_call, "function", None), "name", "")
                if function_name != "rag_retrieve":
                    tool_payload = {"error": f"Unsupported tool: {function_name}"}
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(tool_payload, ensure_ascii=False),
                        }
                    )
                    continue

                args = json.loads(tool_call.function.arguments or "{}")
                query = str(args.get("query") or message.content or "")
                raw_top_k = args.get("top_k")
                try:
                    requested_top_k = int(raw_top_k) if raw_top_k is not None else TOP_K
                except (TypeError, ValueError):
                    requested_top_k = TOP_K
                top_k = max(1, min(requested_top_k, MAX_TOP_K))

                requested_kb_id = str(args.get("knowledge_base") or "").strip()
                kb_catalog = cl.user_session.get("kb_catalog") or {}
                resolved_kb = _resolve_kb_selection(requested_kb_id, kb_catalog)
                target_collection = resolved_kb.get("qdrant_collection")
                resolved_kb_id = resolved_kb.get("id") or requested_kb_id

                signature_key = {
                    "query": query,
                    "top_k": top_k,
                    "knowledge_base": resolved_kb_id,
                }
                signature = f"{function_name}:{json.dumps(signature_key, ensure_ascii=False, sort_keys=True)}"
                if signature in cached_tool_payloads:
                    results, tool_payload = cached_tool_payloads[signature]
                    with cl.Step(name="rag_retrieve", type="tool") as step:
                        step.input = {
                            "query": query,
                            "top_k": top_k,
                            "knowledge_base": resolved_kb_id,
                            "cached": True,
                        }
                        step.output = {"hits": len(results), "cached": True}
                else:
                    with cl.Step(name="rag_retrieve", type="tool") as step:
                        # Get user profile for personalized retrieval
                        user_profile = cl.user_session.get("user_profile")
                        chat_profile_name = cl.user_session.get("chat_profile") or ""
                        balance = 1.0  # Default: no personalization

                        if PERSONALIZATION_ENABLED and user_profile:
                            # Dynamically determine balance based on query relevance to user interests
                            balance = await determine_balance(query, user_profile, user_role=chat_profile_name)
                            step.input = {
                                "query": query,
                                "top_k": top_k,
                                "knowledge_base": resolved_kb_id,
                                "collection": target_collection,
                                "personalized": True,
                                "balance": balance,
                            }
                        else:
                            step.input = {
                                "query": query,
                                "top_k": top_k,
                                "knowledge_base": resolved_kb_id,
                                "collection": target_collection,
                            }

                        # Use personalized retrieval if profile available
                        results = await personalized_retrieve(
                            query=query,
                            user_profile=user_profile,
                            balance=balance,
                            top_k=top_k,
                            collection=target_collection,
                        )
                        print(
                            "[DEBUG] rag_retrieve",
                            "hits=",
                            len(results),
                            "first_text_len=",
                            len(results[0].text) if results else 0,
                            "personalized=",
                            balance < 1.0,
                        )
                        step.output = {"hits": len(results), "balance": balance}

                    context = build_context(results)
                    citations_text = format_citations(results)
                    tool_payload = {
                        "query": query,
                        "context": context,
                        "citations": citations_text,
                    }
                    cached_tool_payloads[signature] = (results, tool_payload)

                for item in results:
                    key = _result_key(item)
                    existing = aggregated_by_key.get(key)
                    if existing is None:
                        aggregated_by_key[key] = item
                        continue
                    if float(getattr(item, "score", 0.0) or 0.0) > float(getattr(existing, "score", 0.0) or 0.0):
                        aggregated_by_key[key] = item

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(tool_payload, ensure_ascii=False),
                    }
                )
                add_chat_message(
                    CHAT_DB_PATH,
                    session_id,
                    "tool",
                    json.dumps(tool_payload, ensure_ascii=False),
                    metadata={"tool_name": "rag_retrieve"},
                )

            followup = await chat(messages, tools=TOOLS, tool_choice="auto")
            current_msg = followup.choices[0].message
            print(
                "[DEBUG] tool_round_followup",
                "round=",
                tool_round,
                "content_empty=",
                not bool(current_msg.content),
                "tool_calls=",
                bool(getattr(current_msg, "tool_calls", None)),
            )

        last_results = sorted(
            aggregated_by_key.values(),
            key=lambda r: float(getattr(r, "score", 0.0) or 0.0),
            reverse=True,
        )

        if getattr(current_msg, "tool_calls", None):
            # Safety stop: avoid endless tool loops, force final answer from collected context.
            print(
                "[WARN] tool_round_limit_reached",
                "max_tool_rounds=",
                max_tool_rounds,
                "aggregated_hits=",
                len(last_results),
            )
            final_context = build_context(last_results[: max(TOP_K, 8)])
            forced_messages = [
                *messages,
                {
                    "role": "system",
                    "content": (
                        "Erstelle jetzt die finale Antwort ausschließlich aus dem Kontext. "
                        "Keine weiteren Tool-Aufrufe."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Frage: {message.content}\n\n"
                        f"Kontext:\n{final_context}\n\n"
                        "Antworte auf Deutsch mit Quellenhinweisen [1], [2], ..."
                    ),
                },
            ]
            forced_final = await chat(forced_messages)
            forced_final_msg = forced_final.choices[0].message
            content = forced_final_msg.content or ""
        else:
            content = current_msg.content or ""

        if not content.strip():
            if last_results:
                content = _extractive_answer_from_results(message.content, last_results)
            else:
                content = "Im bereitgestellten Kontext nicht enthalten"

        content = _strip_model_source_blocks(content)

        # Attach source PDFs as endpoint URLs to avoid session-scoped file copies.
        session_source_catalog = _sanitize_source_catalog(cl.user_session.get("source_catalog"))
        if not session_source_catalog.get("entries"):
            session_source_catalog = _load_session_source_catalog(session_id)
        source_catalog_changed = False
        # Keep the catalog compact: drop IDs not referenced by persisted citation history.
        if _prune_source_catalog(
            session_source_catalog,
            _source_ids_from_citation_history(cl.user_session.get("citation_history")),
        ):
            source_catalog_changed = True
        cl.user_session.set("source_catalog", session_source_catalog)
        seen_links: set[tuple[str, int | None]] = set()
        source_rows: list[tuple[int, str, str, int | None, int | None, str | None, str]] = []
        alias_by_index: dict[int, str] = {}
        url_by_index: dict[int, str] = {}
        source_rows_for_session: list[dict[str, Any]] = []
        alias_to_url: dict[str, str] = {}
        desired_sources = _desired_source_count(content, len(last_results))
        if MAX_SOURCE_LINKS > 0:
            desired_sources = min(desired_sources, MAX_SOURCE_LINKS)
        allowed_pdf_names = _allowed_source_pdf_names()
        display_counter = 1
        for idx, result in enumerate(last_results, start=1):
            file_name = extract_source_file(result.metadata)
            if not file_name:
                continue
            page = extract_page(result.metadata)
            key = (file_name, page)
            if key in seen_links:
                existing_alias = next((alias for _, alias, fname, pstart, _, _, _ in source_rows if fname == file_name and pstart == page), None)
                if existing_alias:
                    alias_by_index[idx] = existing_alias
                    existing_url = alias_to_url.get(existing_alias)
                    if isinstance(existing_url, str) and existing_url:
                        url_by_index[idx] = existing_url
                continue
            raw_doc_id = result.metadata.get("document_id") if isinstance(result.metadata, dict) else None
            document_id = raw_doc_id.strip() if isinstance(raw_doc_id, str) and raw_doc_id.strip() else None
            # Legacy shared-KB chunks must resolve to a real file; user-uploaded chunks
            # are served via document_id and don't need a DATA_RAW_DIR hit.
            file_ok = document_id is not None or _resolve_source_pdf_path(file_name, allowed_pdf_names) is not None
            if file_ok:
                page_end = result.metadata.get("page_end") if isinstance(result.metadata.get("page_end"), int) else None
                section_title = _resolve_section_title(result.metadata)
                page_start = extract_page(result.metadata)
                alias = _source_alias(display_counter, section_title, page_start, page_end)
                pdf_url = _source_pdf_url(file_name, document_id)
                if isinstance(page, int):
                    pdf_url = f"{pdf_url}#page={page}"
                evidence_snippet = _first_sentence(result.text)
                alias_by_index[idx] = alias
                url_by_index[idx] = pdf_url
                alias_to_url[alias] = pdf_url
                source_rows.append(
                    (
                        display_counter,
                        alias,
                        file_name,
                        page_start,
                        page_end,
                        section_title,
                        evidence_snippet,
                    )
                )
                source_rows_for_session.append(
                    {
                        "alias": alias,
                        "file": file_name,
                        "document_id": document_id,
                        "page": page,
                        "page_start": page_start if isinstance(page_start, int) else None,
                        "page_end": page_end if isinstance(page_end, int) else None,
                        "section": section_title if isinstance(section_title, str) else None,
                        "evidence": evidence_snippet if isinstance(evidence_snippet, str) else None,
                    }
                )
                display_counter += 1
                seen_links.add(key)
            if desired_sources and len(seen_links) >= desired_sources:
                break

        alias_by_number = _alias_number_map(source_rows)
        url_by_number: dict[int, str] = {}
        for src_idx, alias, *_ in source_rows:
            alias_url = alias_to_url.get(alias)
            if isinstance(alias_url, str) and alias_url:
                if isinstance(src_idx, int):
                    url_by_number[src_idx] = alias_url
                number_match = re.match(r"^\s*Quelle\s+(\d+)\s*:", alias, flags=re.IGNORECASE)
                if number_match:
                    url_by_number[int(number_match.group(1))] = alias_url

        alias_by_ref = {**alias_by_number, **alias_by_index}
        url_by_ref = {**url_by_number, **url_by_index}

        # Make in-text citations clickable (supports [1], [1†...], 【1†...】).
        content = _inject_clickable_refs(
            content,
            alias_by_index,
            alias_by_ref,
            url_by_index,
            url_by_ref,
        )
        # Also map named refs like [standard_200_2.pdf, S. 2] to known source aliases.
        content = _inject_named_source_refs(content, source_rows)
        # Link explicit alias mentions like "Quelle 3: ... (S.312-313)" early,
        # before normalization potentially removes the numeric anchor.
        content = _inject_source_alias_links(content, alias_by_ref, url_by_ref)
        # Normalize model-written "Quelle n: ..." strings to exact alias values.
        content = _normalize_source_alias_mentions(content, alias_by_index, alias_by_ref)
        # Fallback: if model index does not match retrieved order, map by title/page similarity.
        content = _normalize_source_mentions_by_content(content, source_rows)
        # Repair model outputs like: "Quelle 1: ... (S.30)(/sources/pdf/...)" to markdown links.
        content = _inject_naked_source_links(content)

        cited_aliases = set()
        for _, alias, *_ in source_rows:
            if not isinstance(alias, str) or not alias:
                continue
            escaped_alias = alias.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")
            if alias in content or escaped_alias in content:
                cited_aliases.add(alias)
        if cited_aliases:
            source_rows = [row for row in source_rows if row[1] in cited_aliases]
            source_rows_for_session = [
                row
                for row in source_rows_for_session
                if isinstance(row.get("alias"), str) and row["alias"] in cited_aliases
            ]

        # Assign persistent IDs only for sources that remain in the final assistant message.
        resolved_source_rows: list[tuple[int, str, str, int | None, int | None, str | None, str]] = []
        resolved_source_rows_for_session: list[dict[str, Any]] = []
        for row, session_row in zip(source_rows, source_rows_for_session):
            _, alias, file_name, page_start, page_end, section_title, evidence = row
            source_id, _, catalog_changed = _register_source_in_catalog(
                session_source_catalog,
                file_name=file_name,
                page_start=page_start if isinstance(page_start, int) else None,
                page_end=page_end if isinstance(page_end, int) else None,
                section_title=section_title if isinstance(section_title, str) else None,
            )
            if catalog_changed:
                source_catalog_changed = True
            resolved_source_rows.append(
                (
                    source_id,
                    alias,
                    file_name,
                    page_start,
                    page_end,
                    section_title,
                    evidence,
                )
            )
            updated_session_row = dict(session_row)
            updated_session_row["source_id"] = source_id
            resolved_source_rows_for_session.append(updated_session_row)

        source_rows = resolved_source_rows
        source_rows_for_session = resolved_source_rows_for_session
        content, source_rows, source_rows_for_session = _align_aliases_to_source_ids(
            content,
            source_rows,
            source_rows_for_session,
        )

        if source_catalog_changed:
            sanitized_catalog = _sanitize_source_catalog(session_source_catalog)
            cl.user_session.set("source_catalog", sanitized_catalog)
            _persist_session_source_catalog(session_id, sanitized_catalog)

        used_source_ids = sorted(
            {
                source_id
                for source_id, *_ in source_rows
                if isinstance(source_id, int) and source_id > 0
            }
        )

        # Final safety pass: ensure all plain "Quelle X: ..." aliases in answer text are clickable.
        content = _inject_alias_links_by_rows(content, source_rows_for_session)

        # Build a detailed source block for the citation panel.
        detail_block = ""
        if source_rows:
            box_lines = ["## Quellen & Belegstellen", ""]
            for visible_idx, (row, session_row) in enumerate(
                zip(source_rows, source_rows_for_session), start=1
            ):
                src_idx, alias, file_name, page_start, page_end, section_title, evidence = row
                page_label = _page_label(page_start, page_end)
                section_label = section_title or "Abschnitt unbekannt"
                document_id = session_row.get("document_id") if isinstance(session_row, dict) else None
                pdf_url = _source_pdf_url(file_name, document_id)
                page_for_link = page_start if isinstance(page_start, int) else None
                if isinstance(page_for_link, int):
                    pdf_url = f"{pdf_url}#page={page_for_link}"
                box_lines.append(f"### {alias}")
                box_lines.append(f"- Datei: `{file_name}`")
                box_lines.append(f"- PDF: [Öffnen]({pdf_url})")
                if isinstance(src_idx, int) and src_idx > 0:
                    box_lines.append(f"- Quellen-ID: {src_idx}")
                else:
                    box_lines.append(f"- Quellen-ID: {visible_idx}")
                box_lines.append(f"- Seiten: {page_label}")
                box_lines.append(f"- Abschnitt: {section_label}")
                if evidence:
                    box_lines.append(f"- Belegsnippet: \"{evidence}\"")
                box_lines.append("")
            detail_block = "\n".join(box_lines)

        # Put only the detailed evidence list into a separate side panel.
        citation_panel_content = detail_block
        if citation_panel_content:
            cl.user_session.set("citation_panel_content", citation_panel_content)
            cl.user_session.set("citation_source_rows", source_rows_for_session)
            citation_history = _sanitize_citation_history(cl.user_session.get("citation_history"))
            citation_history = _append_citation_history(
                citation_history,
                citation_panel_content,
                source_rows_for_session,
            )
            cl.user_session.set("citation_history", citation_history)
        else:
            cl.user_session.set("citation_panel_content", None)
            cl.user_session.set("citation_source_rows", [])

        content, followups = _extract_followups(content)
        followup_questions = _sanitize_followup_questions(followups)
        cl.user_session.set("followup_questions", followup_questions)
        render_content = content
        message_metadata: dict[str, Any] = {
            "has_citations_panel": bool(citation_panel_content),
            "followup_count": len(followup_questions),
            "followup_questions": followup_questions,
            "used_source_ids": used_source_ids,
        }
        if citation_panel_content:
            message_metadata["citation_panel_content"] = citation_panel_content
            message_metadata["citation_source_rows"] = _sanitize_source_rows_payload(source_rows_for_session)

        assistant_reply = cl.Message(
            content=render_content,
            metadata=message_metadata,
        )
        actions = _build_chat_actions(
            followup_questions=followup_questions,
            has_citations_panel=bool(citation_panel_content),
            source_step_id=assistant_reply.id,
            citation_panel_content=citation_panel_content,
            citation_source_rows=source_rows_for_session,
        )
        assistant_reply.actions = actions
        print("[DEBUG] followup_actions=", len(followup_questions), "total_actions=", len(actions))
        if citation_panel_content:
            _cache_citation_panel_content(assistant_reply.id, citation_panel_content)
            panel_elements = _build_citation_elements(
                citation_panel_content,
                source_rows_for_session,
                citation_step_id=assistant_reply.id,
            )
            assistant_reply.elements = panel_elements

        # Attach one inline cl.Pdf(display="side") per cited source so clicking the alias
        # text in the message body opens the PDF in the right side panel instead of a new tab.
        inline_pdf_elements = _build_inline_pdf_elements(source_rows_for_session)
        if inline_pdf_elements:
            assistant_reply.elements = (assistant_reply.elements or []) + inline_pdf_elements

        await assistant_reply.send()
        if citation_panel_content:
            history_panel_content, history_rows = _build_citation_history_view(
                _sanitize_citation_history(cl.user_session.get("citation_history"))
            )
            use_history_sidebar = isinstance(history_panel_content, str) and history_panel_content.strip()
            sidebar_content = (
                history_panel_content
                if use_history_sidebar
                else citation_panel_content
            )
            sidebar_rows = history_rows if use_history_sidebar else source_rows_for_session
            if "/sources/pdf/" not in sidebar_content:
                sidebar_content = _append_source_links_to_panel(sidebar_content, sidebar_rows)
            await _show_citation_sidebar(
                sidebar_content,
                sidebar_rows,
                sidebar_title=(
                    CITATION_HISTORY_SIDEBAR_TITLE if use_history_sidebar else CITATION_SIDEBAR_TITLE
                ),
            )
        messages.append({"role": "assistant", "content": content})
        add_chat_message(
            CHAT_DB_PATH,
            session_id,
            "assistant",
            content,
            metadata=message_metadata,
        )
    else:
        content = assistant_msg.content or ""
        content, followups = _extract_followups(content)
        followup_questions = _sanitize_followup_questions(followups)
        cl.user_session.set("followup_questions", followup_questions)
        assistant_reply = cl.Message(content=content)
        actions = _build_chat_actions(
            followup_questions=followup_questions,
            has_citations_panel=False,
            source_step_id=assistant_reply.id,
        )
        assistant_reply.actions = actions
        await assistant_reply.send()
        messages.append({"role": "assistant", "content": content})
        add_chat_message(
            CHAT_DB_PATH,
            session_id,
            "assistant",
            content,
            metadata={
                "followup_count": len(followup_questions),
                "followup_questions": followup_questions,
            },
        )

    cl.user_session.set("messages", messages)

    await _maybe_update_profile()
