"""Source-catalog helpers for session-uploaded documents.

The KB pipeline keeps a per-chat `source_catalog` ({"next_id", "key_to_id",
"entries"}) where each entry is a citable source with a numeric id. The
LLM is told to cite as `Quelle <N>: ...` and the Chainlit frontend matches
that bare body text to inline `cl.Pdf`/`cl.Text` elements by name.

This module hooks session uploads into the same catalog so they share
numbering with KB sources — but with a `kind="session"` discriminator
plus a `session_token` that maps to the upload's temp path on disk.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

import chainlit as cl


logger = logging.getLogger(__name__)


SESSION_SOURCE_PATHS_KEY = "session_source_paths"
SESSION_DOC_ELEMENTS_KEY = "session_doc_elements"
SESSION_TOKEN_PREFIX = "session-"


def register_session_source(
    catalog: dict[str, Any],
    *,
    filename: str,
    path: Path,
    markdown: str,
) -> int:
    """Add a session-doc entry to the per-chat source catalog.

    Returns the assigned `Quelle N` integer.

    Unlike the KB registration which dedups on (file, page, section) so
    repeated retrieval of the same chunk reuses the same id, every
    session upload gets a fresh id — two uploads named `report.pdf` are
    distinct documents from the user's perspective.

    Side effects:
      - Mutates `catalog["next_id"]` and `catalog["entries"]`.
      - Writes the token → path mapping to `cl.user_session` for the
        `/sources/pdf/session-<token>` endpoint to resolve.
      - Stashes inline-element parameters in `cl.user_session` so each
        assistant message can pull fresh `cl.Pdf`/`cl.Text` instances
        without needing to know about the catalog.
    """
    if not isinstance(catalog, dict):
        raise TypeError("catalog must be a dict")

    entries = catalog.setdefault("entries", {})
    next_id = catalog.get("next_id", 1)
    while str(next_id) in entries:
        next_id += 1

    token = uuid.uuid4().hex[:16]
    ext = path.suffix.lower()

    entries[str(next_id)] = {
        "file": filename,
        "kind": "session",
        "session_token": token,
        "ext": ext,
    }
    catalog["next_id"] = next_id + 1

    # Live path-resolver map used by /sources/pdf/{session-<token>}.
    paths = cl.user_session.get(SESSION_SOURCE_PATHS_KEY) or {}
    paths[token] = str(path)
    cl.user_session.set(SESSION_SOURCE_PATHS_KEY, paths)

    # Inline-element scaffolding. Concrete cl.Pdf/cl.Text get instantiated
    # per assistant message; we just stash the data they need so the
    # builder doesn't have to re-derive aliases or the URL.
    alias = _session_alias(next_id, filename)
    elements = cl.user_session.get(SESSION_DOC_ELEMENTS_KEY) or []
    if ext == ".pdf":
        elements.append({
            "alias": alias,
            "kind": "pdf",
            "url": f"/sources/pdf/{SESSION_TOKEN_PREFIX}{token}",
        })
    else:
        # MD/TXT: the markdown content goes straight into a side-panel
        # cl.Text. No URL/endpoint needed; bytes never leave the pod.
        elements.append({
            "alias": alias,
            "kind": "text",
            "content": markdown,
        })
    cl.user_session.set(SESSION_DOC_ELEMENTS_KEY, elements)

    logger.info(
        "session_upload.source_registered quelle=%d filename=%s ext=%s token=%s",
        next_id, filename, ext, token,
    )
    return next_id


def _session_alias(quelle_n: int, filename: str) -> str:
    """Match Chainlit's body-text → element-name lookup. Must equal the
    exact string the LLM is told to emit (`Quelle N: <Dateiname>`)."""
    return f"Quelle {quelle_n}: {filename}"


def get_session_source_path(token: str) -> Path | None:
    """Resolve a `session-<token>` URL to the original temp path. Returns
    None if the token is unknown — never raises, never leaks layout."""
    paths = cl.user_session.get(SESSION_SOURCE_PATHS_KEY) or {}
    raw = paths.get(token)
    if not isinstance(raw, str) or not raw:
        return None
    candidate = Path(raw)
    if not candidate.is_file():
        return None
    return candidate


def clear_session_source_state() -> None:
    """Wipe per-session resolver map + inline-element stash. Called from
    on_chat_end alongside the existing temp-file unlinking."""
    cl.user_session.set(SESSION_SOURCE_PATHS_KEY, {})
    cl.user_session.set(SESSION_DOC_ELEMENTS_KEY, [])


def build_session_doc_inline_elements() -> list[Any]:
    """Construct fresh cl.Pdf/cl.Text elements per assistant turn.

    Reusing the same element objects across messages risks Chainlit
    element-id collisions, so we stash only the parameters and rebuild
    each time. Empty list if no session docs uploaded.
    """
    raw = cl.user_session.get(SESSION_DOC_ELEMENTS_KEY) or []
    elements: list[Any] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        alias = item.get("alias")
        kind = item.get("kind")
        if not isinstance(alias, str) or not alias.strip():
            continue
        if kind == "pdf":
            url = item.get("url")
            if isinstance(url, str) and url.strip():
                elements.append(
                    cl.Pdf(name=alias, url=url, page=1, display="side")
                )
        elif kind == "text":
            content = item.get("content")
            if isinstance(content, str):
                elements.append(
                    cl.Text(name=alias, content=content, display="side")
                )
    return elements
