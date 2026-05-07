"""Async client for the in-cluster Docling service.

Routes PDFs through the same `parse_pdf_async` chunk-fairness queue that
KB uploads use ([app/kb/ingestion_pipeline.py](../kb/ingestion_pipeline.py)).
That gives session uploads (small, latency-sensitive) a fair shot at the
same shared docling-service replicas as KB uploads (potentially huge),
because the `asyncio.Semaphore` interleaves chunks across both paths.

Returns markdown joined from the merged section list so the caller can
drop it straight into the session-document context block.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import httpx

from core.settings import DOCLING_SERVICE_URL
from kb.ingestion_pipeline import parse_pdf_async


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
    """Run the PDF through `parse_pdf_async` and return joined Markdown.

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

    try:
        sections = await parse_pdf_async(file_path)
    except (httpx.ConnectError, httpx.TimeoutException, OSError) as exc:
        logger.warning(
            "session_upload.docling_unavailable file=%s reason=%s",
            file_path.name, type(exc).__name__,
        )
        raise ServiceUnavailableError(
            "Docling-Service nicht erreichbar."
        ) from exc
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code if exc.response is not None else 0
        logger.warning(
            "session_upload.docling_http_error file=%s status=%d",
            file_path.name, status,
        )
        if status >= 500:
            raise ServiceUnavailableError(
                f"Docling-Service antwortete mit {status}."
            ) from exc
        raise ExtractionError(
            f"Docling lehnte die Datei ab (HTTP {status})."
        ) from exc
    except ValueError as exc:
        # _post_chunk raises ValueError on malformed JSON / missing sections.
        logger.warning("session_upload.docling_bad_payload file=%s", file_path.name)
        raise ExtractionError(str(exc)) from exc

    if not sections:
        raise ExtractionError(
            "Docling lieferte keine extrahierten Abschnitte zurück."
        )

    markdown = _sections_to_markdown(sections)
    if not markdown.strip():
        raise ExtractionError("Aus dem Dokument konnte kein Text extrahiert werden.")

    logger.info(
        "session_upload.docling_ok file=%s sections=%d chars=%d",
        file_path.name, len(sections), len(markdown),
    )
    return markdown
