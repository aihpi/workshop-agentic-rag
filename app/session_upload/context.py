"""Session-scoped document state + system-prompt assembly.

Holds the list of session-uploaded documents in cl.user_session under
SESSION_DOCS_KEY, and rebuilds messages[0]["content"] in place whenever
the list changes. The pre-session-docs system prompt is captured once in
on_chat_start under BASE_SYSTEM_PROMPT_KEY so we can keep regenerating a
clean system message without re-deriving the personalization / KB block.
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import chainlit as cl
import tiktoken


logger = logging.getLogger(__name__)


SESSION_DOCS_KEY = "session_documents"
BASE_SYSTEM_PROMPT_KEY = "base_system_prompt"
SESSION_TMP_FILES_KEY = "session_upload_tmp_files"


@dataclass
class SessionDocEntry:
    filename: str
    markdown: str
    token_count: int
    uploaded_at: str  # ISO8601


@functools.cache
def _tokenizer() -> Any:
    # cl100k_base matches the OpenAI / LiteLLM-compatible tokenizer family;
    # close enough for any model behind our LiteLLM gateway.
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(_tokenizer().encode(text))


def get_session_documents() -> list[SessionDocEntry]:
    raw = cl.user_session.get(SESSION_DOCS_KEY) or []
    return list(raw)


def append_session_document(entry: SessionDocEntry) -> None:
    docs = get_session_documents()
    docs.append(entry)
    cl.user_session.set(SESSION_DOCS_KEY, docs)


def clear_session_documents() -> None:
    cl.user_session.set(SESSION_DOCS_KEY, [])


def total_session_doc_tokens() -> int:
    return sum(d.token_count for d in get_session_documents())


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_entry(filename: str, markdown: str) -> SessionDocEntry:
    return SessionDocEntry(
        filename=filename,
        markdown=markdown,
        token_count=count_tokens(markdown),
        uploaded_at=now_iso(),
    )


def render_session_documents_block(docs: list[SessionDocEntry]) -> str:
    """Render the docs as an XML-ish block prefixed with a German instruction
    telling the model to ground its answers in this content when relevant.

    Empty list → empty string so the system prompt stays unchanged before
    any upload. Without the preamble the LLM sees raw XML and doesn't know
    whether to prioritise it over the BSI rag_retrieve tool output.
    """
    if not docs:
        return ""
    file_list = ", ".join(f'"{d.filename}"' for d in docs)
    parts: list[str] = [
        "## Vom Nutzer in dieser Chat-Session hochgeladene Dokumente",
        "",
        f"Der Nutzer hat in dieser Chat-Session die folgenden Dokumente "
        f"hochgeladen: {file_list}. Wenn sich eine Nutzerfrage auf diese "
        "Dokumente bezieht (etwa 'Fasse das PDF zusammen', 'Was steht in "
        "dem hochgeladenen Dokument zu X' oder eine inhaltliche Folgefrage "
        "darauf), antworte primär auf Basis des Inhalts dieser Dokumente "
        "und nicht aus der BSI-Wissensbasis. Zitiere konkret und gib den "
        "Dateinamen an. Greife nur dann zusätzlich auf die "
        "BSI-Wissensbasis zurück, wenn die Nutzerfrage explizit eine "
        "Verknüpfung zwischen dem Dokument und IT-Grundschutz herstellt.",
        "",
        "<session_documents>",
    ]
    for idx, doc in enumerate(docs, start=1):
        # Filenames are user-controlled; strip quotes/angle-brackets so a
        # malicious filename can't break out of the attribute.
        safe_name = (
            doc.filename.replace('"', "'").replace("<", "(").replace(">", ")")
        )
        parts.append(
            f'  <document filename="{safe_name}" upload_index="{idx}">'
        )
        parts.append(doc.markdown.strip())
        parts.append("  </document>")
    parts.append("</session_documents>")
    return "\n".join(parts)


def rebuild_system_message_with_docs() -> None:
    """Mutate messages[0]["content"] in place from base + session_documents.

    Idempotent: safe to call after every upload. If there are no session
    docs, the system message reverts to the base prompt.
    """
    base = cl.user_session.get(BASE_SYSTEM_PROMPT_KEY)
    if not isinstance(base, str):
        # on_chat_start always sets this; if it's missing we got called too
        # early or from a path that doesn't go through on_chat_start. Skip
        # gracefully instead of crashing the chat turn.
        logger.warning("session_upload.rebuild_skipped reason=no_base_prompt")
        return

    docs = get_session_documents()
    block = render_session_documents_block(docs)
    new_content = f"{base}\n\n{block}" if block else base

    messages = cl.user_session.get("messages")
    if not isinstance(messages, list) or not messages:
        cl.user_session.set("messages", [{"role": "system", "content": new_content}])
        return

    if isinstance(messages[0], dict) and messages[0].get("role") == "system":
        messages[0]["content"] = new_content
    else:
        messages.insert(0, {"role": "system", "content": new_content})
    cl.user_session.set("messages", messages)


def track_tmp_file(path: str) -> None:
    paths = cl.user_session.get(SESSION_TMP_FILES_KEY) or []
    paths.append(path)
    cl.user_session.set(SESSION_TMP_FILES_KEY, paths)


def consume_tmp_files() -> list[str]:
    paths = cl.user_session.get(SESSION_TMP_FILES_KEY) or []
    cl.user_session.set(SESSION_TMP_FILES_KEY, [])
    return list(paths)
