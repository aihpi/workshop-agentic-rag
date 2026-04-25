"""Per-element session-upload entrypoint.

The chat handler in app.py loops over message.elements and calls
handle_session_upload(element) for each. This module owns:
  - Extension / size / magic-byte / UTF-8 validation
  - Dispatch (PDF → Docling, MD/TXT → direct read)
  - Token budget check (cumulative across the session)

Returns a SessionDocResult that the caller turns into Chainlit UI: the
human-facing message text is in result.message, the entry to inject into
the system prompt (if any) is in result.entry.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from core.settings import (
    MAX_SESSION_FILE_SIZE_MB,
    SESSION_DOC_TOKEN_LIMIT,
)
from kb.ingestion_pipeline import validate_pdf_upload
from session_upload import context as ctx
from session_upload.docling_client import (
    ExtractionError,
    ServiceUnavailableError,
    extract_markdown,
)


logger = logging.getLogger(__name__)


_PDF_EXT = ".pdf"
_TEXT_EXTS = {".md", ".txt"}
_ALL_EXTS = {_PDF_EXT, *_TEXT_EXTS}


class SessionDocStatus(str, Enum):
    OK = "ok"
    UNSUPPORTED = "unsupported"
    INVALID = "invalid"
    TOO_LARGE_BYTES = "too_large_bytes"
    TOO_LARGE_TOKENS = "too_large_tokens"
    EXTRACTION_FAILED = "extraction_failed"
    SERVICE_UNAVAILABLE = "service_unavailable"


@dataclass
class SessionDocResult:
    status: SessionDocStatus
    filename: str
    message: str
    entry: ctx.SessionDocEntry | None = None
    token_count: int | None = None


def _filename(element: Any) -> str:
    raw = getattr(element, "name", None) or "unbenannte_datei"
    return str(raw)


def _path(element: Any) -> Path | None:
    raw = getattr(element, "path", None)
    if not raw:
        return None
    return Path(str(raw))


def _is_valid_utf8(path: Path) -> bool:
    try:
        path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return False
    return True


async def handle_session_upload(element: Any) -> SessionDocResult:
    filename = _filename(element)
    ext = Path(filename).suffix.lower()

    if ext not in _ALL_EXTS:
        return SessionDocResult(
            status=SessionDocStatus.UNSUPPORTED,
            filename=filename,
            message=(
                f"Dateityp {ext or '(leer)'} wird im Chat nicht unterstützt. "
                f"Erlaubt: {sorted(_ALL_EXTS)}. Für andere Formate bitte die "
                "Wissensdatenbank-Upload-Funktion verwenden."
            ),
        )

    path = _path(element)
    if path is None or not path.exists():
        return SessionDocResult(
            status=SessionDocStatus.INVALID,
            filename=filename,
            message="Hochgeladene Datei konnte serverseitig nicht gefunden werden.",
        )

    # Size cap. Chainlit's UI also caps via config.toml, but a malicious
    # client could bypass the UI; this is the authoritative check.
    size_bytes = path.stat().st_size
    max_bytes = MAX_SESSION_FILE_SIZE_MB * 1024 * 1024
    if size_bytes > max_bytes:
        return SessionDocResult(
            status=SessionDocStatus.TOO_LARGE_BYTES,
            filename=filename,
            message=(
                f"Datei überschreitet das Limit von {MAX_SESSION_FILE_SIZE_MB} MB "
                f"(tatsächlich {size_bytes // (1024 * 1024)} MB)."
            ),
        )

    # Track the temp file so on_chat_end can unlink it.
    ctx.track_tmp_file(str(path))

    # Format-specific validation + extraction.
    if ext == _PDF_EXT:
        try:
            validate_pdf_upload(path)
        except ValueError as exc:
            return SessionDocResult(
                status=SessionDocStatus.INVALID,
                filename=filename,
                message=str(exc),
            )
        try:
            markdown = await extract_markdown(path)
        except ServiceUnavailableError as exc:
            return SessionDocResult(
                status=SessionDocStatus.SERVICE_UNAVAILABLE,
                filename=filename,
                message=(
                    "Der Dokument-Extraktionsdienst ist gerade nicht "
                    "verfügbar. Bitte versuchen Sie es später erneut oder "
                    "verwenden Sie die Wissensdatenbank-Upload-Funktion."
                ),
            )
        except ExtractionError as exc:
            return SessionDocResult(
                status=SessionDocStatus.EXTRACTION_FAILED,
                filename=filename,
                message=f"Extraktion fehlgeschlagen: {exc}",
            )
    else:
        # .md / .txt — read directly, must be valid UTF-8.
        if not _is_valid_utf8(path):
            return SessionDocResult(
                status=SessionDocStatus.INVALID,
                filename=filename,
                message=(
                    "Die Datei ist keine gültige UTF-8-Textdatei. Bitte als "
                    "UTF-8 speichern und erneut hochladen."
                ),
            )
        try:
            markdown = path.read_text(encoding="utf-8")
        except OSError as exc:
            return SessionDocResult(
                status=SessionDocStatus.INVALID,
                filename=filename,
                message=f"Datei nicht lesbar: {exc}",
            )

    # Cumulative token budget. Refuse if adding this doc would exceed the
    # session-wide cap so a user can't sneak past with several smaller files.
    new_tokens = ctx.count_tokens(markdown)
    current_total = ctx.total_session_doc_tokens()
    if current_total + new_tokens > SESSION_DOC_TOKEN_LIMIT:
        return SessionDocResult(
            status=SessionDocStatus.TOO_LARGE_TOKENS,
            filename=filename,
            token_count=new_tokens,
            message=(
                f"Das Dokument hat {new_tokens} Tokens; zusammen mit den "
                f"bereits hochgeladenen ({current_total}) würde das Limit "
                f"von {SESSION_DOC_TOKEN_LIMIT} überschritten. Bitte "
                "verwenden Sie für größere Dokumente die "
                "Wissensdatenbank-Upload-Funktion."
            ),
        )

    entry = ctx.make_entry(filename=filename, markdown=markdown)
    logger.info(
        "session_upload.parse_ok filename=%s ext=%s tokens=%d total=%d",
        filename, ext, entry.token_count, current_total + entry.token_count,
    )
    return SessionDocResult(
        status=SessionDocStatus.OK,
        filename=filename,
        message=(
            f"Dokument '{filename}' zur Chat-Session hinzugefügt "
            f"({entry.token_count} Tokens)."
        ),
        entry=entry,
        token_count=entry.token_count,
    )
