"""Async client for the in-cluster Docling service.

Calls the same FastAPI /process endpoint that kb.ingestion_pipeline uses
for KB uploads, but stays async (the chat handler is already async, no
need for the to_thread hop the KB path takes). Returns markdown joined
from the section list so the caller can drop it straight into the
session-document context block.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import httpx

from core.settings import DOCLING_SERVICE_URL


logger = logging.getLogger(__name__)


class DoclingClientError(Exception):
    """Base for client-visible Docling failures."""


class ServiceUnavailableError(DoclingClientError):
    """Connection refused, timeout, or 5xx — service is down or overloaded."""


class ExtractionError(DoclingClientError):
    """Service responded but couldn't extract content (bad PDF, 4xx, malformed JSON)."""


def _sections_to_markdown(sections: list[dict[str, Any]]) -> str:
    """Flatten Docling's section list to a single Markdown string.

    Each section becomes "## {section_title}\\n\\n{text}\\n" — sections with
    empty titles get just the text, separated by blank lines. Page numbers
    are dropped here; the LLM doesn't need them for chat-context recall and
    they'd noise up the prompt.
    """
    parts: list[str] = []
    for sec in sections:
        if not isinstance(sec, dict):
            continue
        text = (sec.get("text") or "").strip()
        if not text:
            continue
        title = (sec.get("section_title") or "").strip()
        if title:
            parts.append(f"## {title}\n\n{text}")
        else:
            parts.append(text)
    return "\n\n".join(parts)


async def extract_markdown(
    file_path: Path,
    *,
    timeout_s: float = 120.0,
) -> str:
    """POST a PDF to the Docling service and return joined Markdown.

    Raises:
        ServiceUnavailableError: if DOCLING_SERVICE_URL is unset, the
            service is unreachable, times out, or returns 5xx.
        ExtractionError: if the service returns a 4xx or a payload we
            can't parse / has no usable sections.
    """
    if not DOCLING_SERVICE_URL:
        raise ServiceUnavailableError(
            "DOCLING_SERVICE_URL ist nicht konfiguriert."
        )

    url = f"{DOCLING_SERVICE_URL}/process"
    timeout = httpx.Timeout(connect=2.0, read=timeout_s, write=30.0, pool=5.0)

    try:
        with open(file_path, "rb") as fh:
            files = {"file": (file_path.name, fh, "application/pdf")}
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(url, files=files)
    except (httpx.ConnectError, httpx.TimeoutException, OSError) as exc:
        logger.warning(
            "session_upload.docling_unavailable url=%s reason=%s",
            url, type(exc).__name__,
        )
        raise ServiceUnavailableError(
            "Docling-Service nicht erreichbar."
        ) from exc

    if resp.status_code >= 500:
        logger.warning(
            "session_upload.docling_5xx url=%s status=%d", url, resp.status_code
        )
        raise ServiceUnavailableError(
            f"Docling-Service antwortete mit {resp.status_code}."
        )
    if resp.status_code >= 400:
        logger.warning(
            "session_upload.docling_4xx url=%s status=%d body=%s",
            url, resp.status_code, resp.text[:200],
        )
        raise ExtractionError(
            f"Docling lehnte die Datei ab (HTTP {resp.status_code})."
        )

    try:
        payload = resp.json()
    except ValueError as exc:
        logger.warning("session_upload.docling_bad_json url=%s", url)
        raise ExtractionError("Docling-Antwort war kein gültiges JSON.") from exc

    sections = payload.get("sections") if isinstance(payload, dict) else None
    if not isinstance(sections, list) or not sections:
        raise ExtractionError(
            "Docling lieferte keine extrahierten Abschnitte zurück."
        )

    markdown = _sections_to_markdown(sections)
    if not markdown.strip():
        raise ExtractionError("Aus dem Dokument konnte kein Text extrahiert werden.")

    logger.info(
        "session_upload.docling_ok url=%s sections=%d chars=%d",
        url, len(sections), len(markdown),
    )
    return markdown
