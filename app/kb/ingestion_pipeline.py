"""Shared document ingestion pipeline: parse → chunk → embed → upsert.

Used by the per-user FastAPI upload endpoint (app/api/api_routes.py).
"""

from __future__ import annotations

import asyncio
import functools
import logging
import tempfile
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable

import httpx
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from core.llm import embed
from core.settings import (
    CHUNK_MAX_CHARS,
    CHUNK_OVERLAP,
    DOCLING_CHUNK_PAGES,
    DOCLING_CONCURRENCY,
    DOCLING_MAX_REPLICAS,
    DOCLING_SERVICE_URL,
    EMBED_BATCH_SIZE,
    EMBED_MAX_BATCH_CHARS,
    QDRANT_API_KEY,
    QDRANT_URL,
)
from kb import docling_scaler

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int], Awaitable[None]]

# User-facing upload endpoints (app + docling-service) accept PDFs only.
# The .txt/.md branch in parse_file() is dead code for now but kept in case
# we re-enable those formats later.
SUPPORTED_EXTENSIONS = {".pdf"}

_PDF_MAGIC = b"%PDF-"


def validate_pdf_upload(path: Path) -> None:
    """Raise ValueError if the file at `path` is not a valid PDF.

    Magic-byte check (not filename). Called after we've written bytes to a
    tempfile but before any downstream parser touches them, so renamed
    non-PDF files (e.g. foo.exe → foo.pdf) are rejected before they reach
    Docling.
    """
    try:
        with open(path, "rb") as fh:
            header = fh.read(len(_PDF_MAGIC))
    except OSError as exc:
        raise ValueError(f"PDF-Datei nicht lesbar: {exc}")
    if not header.startswith(_PDF_MAGIC):
        raise ValueError("Die Datei ist keine gültige PDF (falsche Magic Bytes).")


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    max_chars: int = CHUNK_MAX_CHARS,
    overlap: int = CHUNK_OVERLAP,
) -> Iterable[str]:
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_chars:
        yield cleaned
        return
    start = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + max_chars)
        chunk = cleaned[start:end]
        if chunk:
            yield chunk
        if end == len(cleaned):
            break
        start = max(0, end - overlap)


# ---------------------------------------------------------------------------
# Docling PDF extraction
# ---------------------------------------------------------------------------

def _extract_page_from_prov(prov: Any) -> int | None:
    if isinstance(prov, dict):
        pn = prov.get("page_no")
        return pn if isinstance(pn, int) else None
    if isinstance(prov, list):
        page_numbers = [
            p.get("page_no")
            for p in prov
            if isinstance(p, dict) and isinstance(p.get("page_no"), int)
        ]
        return min(page_numbers) if page_numbers else None
    return None


def _parse_ref_index(ref: str, prefix: str) -> int | None:
    marker = f"#/{prefix}/"
    if not ref.startswith(marker):
        return None
    raw = ref[len(marker):]
    return int(raw) if raw.isdigit() else None


def _collect_text_refs(ref: str, groups: list[dict[str, Any]], out: list[int]) -> None:
    text_idx = _parse_ref_index(ref, "texts")
    if text_idx is not None:
        out.append(text_idx)
        return
    group_idx = _parse_ref_index(ref, "groups")
    if group_idx is None or group_idx >= len(groups):
        return
    group = groups[group_idx]
    children = group.get("children")
    if not isinstance(children, list):
        return
    for child in children:
        if isinstance(child, dict):
            child_ref = child.get("$ref")
            if isinstance(child_ref, str):
                _collect_text_refs(child_ref, groups, out)


def _ordered_text_indices(data: dict[str, Any]) -> list[int]:
    body = data.get("body")
    if not isinstance(body, dict):
        return []
    children = body.get("children")
    if not isinstance(children, list):
        return []
    groups = data.get("groups")
    if not isinstance(groups, list):
        groups = []
    out: list[int] = []
    for child in children:
        if isinstance(child, dict):
            ref = child.get("$ref")
            if isinstance(ref, str):
                _collect_text_refs(ref, groups, out)
    return out


def _sections_from_docling_dict(data: dict[str, Any], file_name: str) -> list[dict[str, Any]]:
    texts = data.get("texts")
    if not isinstance(texts, list):
        return []

    ordered_indices = _ordered_text_indices(data) or list(range(len(texts)))
    sections: list[dict[str, Any]] = []
    current_title: str | None = None
    current_texts: list[str] = []
    current_pages: list[int] = []

    def flush() -> None:
        nonlocal current_title, current_texts, current_pages
        content = " ".join(current_texts).strip()
        if len(content) < 40:
            current_title = None
            current_texts = []
            current_pages = []
            return
        section_title = (current_title or "").strip()
        merged = content
        if section_title and not content.lower().startswith(section_title.lower()):
            merged = f"{section_title}\n\n{content}"
        sections.append({
            "text": merged,
            "file": file_name,
            "section_title": section_title,
            "page_start": min(current_pages) if current_pages else None,
            "page_end": max(current_pages) if current_pages else None,
        })
        current_title = None
        current_texts = []
        current_pages = []

    for text_idx in ordered_indices:
        if text_idx >= len(texts):
            continue
        item = texts[text_idx]
        if not isinstance(item, dict):
            continue
        if item.get("content_layer") == "furniture":
            continue
        text = item.get("canonical_text") or item.get("text")
        if not isinstance(text, str):
            continue
        cleaned = " ".join(text.split())
        if not cleaned:
            continue
        label = item.get("label")
        page_no = _extract_page_from_prov(item.get("prov"))
        if isinstance(page_no, int):
            current_pages.append(page_no)
        if label in {"section_header", "title", "chapter_title"}:
            flush()
            current_title = cleaned
            continue
        current_texts.append(cleaned)

    flush()
    return sections


def _sections_from_pages(document: Any, file_name: str) -> list[dict[str, Any]]:
    pages = getattr(document, "pages", None)
    if not pages:
        return []
    sections: list[dict[str, Any]] = []
    for page in pages:
        page_no = (
            getattr(page, "page_number", None)
            or getattr(page, "number", None)
            or getattr(page, "page_no", None)
        )
        text = None
        if hasattr(page, "export_to_markdown"):
            text = page.export_to_markdown()
        elif hasattr(page, "export_to_text"):
            text = page.export_to_text()
        elif hasattr(page, "text"):
            text = page.text() if callable(page.text) else page.text
        if isinstance(text, str) and text.strip():
            sections.append({
                "text": text.strip(),
                "file": file_name,
                "section_title": "",
                "page_start": int(page_no) if page_no is not None else None,
                "page_end": int(page_no) if page_no is not None else None,
            })
    return sections


def _parse_pdf_remote(path: Path, service_url: str) -> list[dict[str, Any]] | None:
    """POST the PDF to the GPU Docling service. Returns None on any failure
    (connection refused, timeout, non-2xx, malformed response) so the caller
    can fall back to local parsing. This is what makes scale-to-zero work:
    kubectl scale --replicas=0 → next upload gets a connect error in ~2s
    → local Docling takes over."""
    try:
        with open(path, "rb") as fh:
            files = {"file": (path.name, fh, "application/pdf")}
            resp = httpx.post(
                f"{service_url}/process",
                files=files,
                # Connect timeout short so scale-to-zero flips fast.
                # Read timeout generous for worst-case GPU parse on a big PDF.
                timeout=httpx.Timeout(connect=2.0, read=180.0, write=30.0, pool=5.0),
            )
    except (httpx.ConnectError, httpx.TimeoutException, OSError) as exc:
        logger.warning("docling.fallback service_url=%s reason=%s", service_url, type(exc).__name__)
        return None

    if resp.status_code >= 500:
        logger.warning("docling.fallback service_url=%s status=%d", service_url, resp.status_code)
        return None
    if resp.status_code != 200:
        # 4xx is a real error (bad request, unsupported format) — surface it.
        resp.raise_for_status()

    try:
        sections = resp.json().get("sections")
    except ValueError:
        logger.warning("docling.fallback service_url=%s reason=bad_json", service_url)
        return None

    if not isinstance(sections, list):
        logger.warning("docling.fallback service_url=%s reason=missing_sections", service_url)
        return None
    logger.info("docling.remote_ok service_url=%s sections=%d", service_url, len(sections))
    return sections


@functools.cache
def _get_pdf_converter() -> DocumentConverter:
    """Module-level DocumentConverter cache.

    DocumentConverter is cheap to construct, but internally it lazily builds
    a Pipeline on first `convert()` — that's where the layout / table models
    get loaded into memory (or onto the GPU). If we construct a fresh
    DocumentConverter per upload (as we used to), every parse re-runs that
    pipeline init and re-loads weights. Sharing one instance means the
    pipeline is built exactly once per process.
    """
    pdf_opts = PdfPipelineOptions(do_ocr=False)
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts)},
    )


def _parse_pdf_local(path: Path) -> list[dict[str, Any]]:
    """In-process Docling parse. Used by `parse_pdf` (when no remote URL)
    and by `parse_pdf_async` as the cold-start / unreachable fallback.

    Kept separate from `parse_pdf` so the async fallback path doesn't
    re-attempt the remote call (which `parse_pdf` would do via
    `_parse_pdf_remote`) — that would just add 2s of connect-timeout
    latency to every fallback.
    """
    converter = _get_pdf_converter()
    result = converter.convert(str(path))
    document = getattr(result, "document", None)
    if document is None:
        return []

    export_dict = None
    if hasattr(document, "export_to_dict"):
        export_dict = document.export_to_dict()
    sections = _sections_from_docling_dict(export_dict, path.name) if export_dict else []
    if sections:
        return sections

    sections = _sections_from_pages(document, path.name)
    if sections:
        return sections

    text = ""
    if hasattr(document, "export_to_markdown"):
        text = document.export_to_markdown()
    elif hasattr(document, "export_to_text"):
        text = document.export_to_text()
    if isinstance(text, str) and text.strip():
        return [{
            "text": text.strip(),
            "file": path.name,
            "section_title": "",
            "page_start": None,
            "page_end": None,
        }]
    return []


def parse_pdf(path: Path) -> list[dict[str, Any]]:
    if DOCLING_SERVICE_URL:
        remote_sections = _parse_pdf_remote(path, DOCLING_SERVICE_URL)
        if remote_sections is not None:
            return remote_sections
        # Fall through to local parsing on remote failure.
    return _parse_pdf_local(path)


# ---------------------------------------------------------------------------
# Fair chunked async parsing — split a PDF into page-batches, fire all
# batches concurrently through a shared semaphore, merge results.
#
# The semaphore *is* the queue. Every chunk from every user competes for
# the same DOCLING_CONCURRENCY slots; a 200-page upload's chunks get
# interleaved with a 10-page upload's single chunk instead of head-of-line
# blocking it.
# ---------------------------------------------------------------------------

# Module-level: shared across all callers in this process. Sized for the
# *maximum* replica count, not the current — chunks pile up at docling-service
# (k8s round-robins) until the in-process scaler catches up.
_DOCLING_SEMAPHORE: asyncio.Semaphore | None = None


def _get_docling_semaphore() -> asyncio.Semaphore:
    """Lazy-init so we bind to the running event loop, not import-time."""
    global _DOCLING_SEMAPHORE
    if _DOCLING_SEMAPHORE is None:
        _DOCLING_SEMAPHORE = asyncio.Semaphore(DOCLING_CONCURRENCY)
    return _DOCLING_SEMAPHORE


def _split_pdf_to_chunks(path: Path, out_dir: Path, pages_per_chunk: int) -> list[Path]:
    """Split `path` into ceil(N / pages_per_chunk) PDFs in `out_dir`.

    Returns the chunk paths in document order. Single-chunk shortcut: if
    the source has <= pages_per_chunk pages, copy nothing — return [path]
    so the caller doesn't pay split + reload for tiny PDFs.
    """
    from pypdf import PdfReader, PdfWriter

    reader = PdfReader(str(path))
    total_pages = len(reader.pages)
    if total_pages <= pages_per_chunk:
        return [path]

    chunk_paths: list[Path] = []
    for chunk_idx, start in enumerate(range(0, total_pages, pages_per_chunk)):
        end = min(start + pages_per_chunk, total_pages)
        writer = PdfWriter()
        for page_no in range(start, end):
            writer.add_page(reader.pages[page_no])
        chunk_path = out_dir / f"chunk_{chunk_idx:04d}.pdf"
        with open(chunk_path, "wb") as fh:
            writer.write(fh)
        chunk_paths.append(chunk_path)
    return chunk_paths


def _rewrite_section_pages(sections: list[dict[str, Any]], offset: int) -> None:
    """Add `offset` to every page_start/page_end in-place.

    Offset is `chunk_idx * DOCLING_CHUNK_PAGES` — pypdf produces a fresh
    PDF whose first page is page 1, and Docling's `page_no` in `prov` is
    1-indexed, so a section reporting page 3 in chunk 1 (chunk_idx=1,
    offset=20) is absolute page 23 = 3 + 20. NOT 24.
    """
    if offset == 0:
        return
    for sec in sections:
        ps = sec.get("page_start")
        if isinstance(ps, int):
            sec["page_start"] = ps + offset
        pe = sec.get("page_end")
        if isinstance(pe, int):
            sec["page_end"] = pe + offset


def _stitch_boundary_sections(merged: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Repair section-headers that got orphaned at chunk boundaries.

    A section spanning pages 19-22 split at the page-20 boundary becomes:
      - chunk N: title-only section (header found, body cut off)
      - chunk N+1: body-only section (no title, starts mid-content)
    Detect that pattern and concatenate. Cheap heuristic — title-only
    means the previous section's text is shorter than 200 chars; body-only
    means the next section has empty section_title and its page_start
    equals or directly follows the previous section's page_end.
    """
    if len(merged) < 2:
        return merged
    out: list[dict[str, Any]] = [merged[0]]
    for nxt in merged[1:]:
        prev = out[-1]
        prev_text = prev.get("text") or ""
        prev_title = (prev.get("section_title") or "").strip()
        nxt_title = (nxt.get("section_title") or "").strip()
        prev_end = prev.get("page_end")
        nxt_start = nxt.get("page_start")
        adjacent = (
            isinstance(prev_end, int)
            and isinstance(nxt_start, int)
            and nxt_start - prev_end <= 1
        )
        title_only = bool(prev_title) and len(prev_text) < 200
        body_only = not nxt_title
        if adjacent and title_only and body_only:
            stitched_text = (nxt.get("text") or "").strip()
            # Re-prepend the orphan title since `_sections_from_docling_dict`
            # would have done that on the un-split parse.
            if stitched_text and not stitched_text.lower().startswith(prev_title.lower()):
                stitched_text = f"{prev_title}\n\n{stitched_text}"
            prev["text"] = stitched_text
            prev["section_title"] = prev_title
            if isinstance(nxt.get("page_end"), int):
                prev["page_end"] = nxt["page_end"]
            continue
        out.append(nxt)
    return out


async def _post_chunk(
    client: httpx.AsyncClient,
    chunk_path: Path,
    chunk_idx: int,
    service_url: str,
    file_name: str,
) -> tuple[int, list[dict[str, Any]]]:
    """POST one chunk to /process under the shared semaphore.

    Page numbers in returned sections are rewritten to absolute (original
    document) coordinates before return. The chunk's source file_name is
    forced onto each section so citations show the upload's original name,
    not "chunk_0003.pdf".
    """
    sem = _get_docling_semaphore()
    offset = chunk_idx * DOCLING_CHUNK_PAGES
    async with sem:
        with open(chunk_path, "rb") as fh:
            files = {"file": (file_name, fh, "application/pdf")}
            resp = await client.post(f"{service_url}/process", files=files)
    if resp.status_code >= 500:
        raise httpx.HTTPStatusError(
            f"docling-service 5xx: {resp.status_code}", request=resp.request, response=resp,
        )
    resp.raise_for_status()
    payload = resp.json()
    sections = payload.get("sections") if isinstance(payload, dict) else None
    if not isinstance(sections, list):
        raise ValueError("docling-service returned no 'sections' list")
    _rewrite_section_pages(sections, offset)
    for sec in sections:
        if isinstance(sec, dict):
            sec["file"] = file_name
    return chunk_idx, sections


async def parse_pdf_async(
    path: Path,
    *,
    on_progress: Callable[[int, int], Awaitable[None]] | None = None,
) -> list[dict[str, Any]]:
    """Split `path` into page batches, fire concurrently, return merged sections.

    Falls back to local CPU `parse_pdf` if:
      - DOCLING_SERVICE_URL is unset (no remote service configured), or
      - cold-start scale-up doesn't yield a ready replica within 90s.

    If any *individual* chunk fails after the service is up, raises — no
    mixed mode. Local CPU was meant for "service unreachable", not
    "service overloaded".
    """
    if not DOCLING_SERVICE_URL:
        # No remote service — use the existing local in-process path.
        return await asyncio.to_thread(_parse_pdf_local, path)

    docling_scaler.mark_request_start()
    try:
        with tempfile.TemporaryDirectory(prefix="docling_chunks_") as tmp_dir:
            tmp_dir_path = Path(tmp_dir)
            chunk_paths = await asyncio.to_thread(
                _split_pdf_to_chunks, path, tmp_dir_path, DOCLING_CHUNK_PAGES,
            )

            await docling_scaler.ensure_capacity(
                desired=min(len(chunk_paths), DOCLING_MAX_REPLICAS),
            )
            ready = await docling_scaler.wait_for_ready_replicas(
                min_ready=1, timeout_s=90.0,
            )
            if not ready:
                logger.warning(
                    "docling.cold_start_timeout falling back to local CPU file=%s pages=%d",
                    path.name, len(chunk_paths),
                )
                return await asyncio.to_thread(_parse_pdf_local, path)

            timeout = httpx.Timeout(connect=5.0, read=300.0, write=30.0, pool=5.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                tasks = [
                    asyncio.create_task(
                        _post_chunk(client, cp, idx, DOCLING_SERVICE_URL, path.name)
                    )
                    for idx, cp in enumerate(chunk_paths)
                ]
                results: list[tuple[int, list[dict[str, Any]]]] = []
                done = 0
                total = len(tasks)
                try:
                    for coro in asyncio.as_completed(tasks):
                        chunk_idx, sections = await coro
                        results.append((chunk_idx, sections))
                        done += 1
                        if on_progress is not None:
                            await on_progress(done, total)
                except (httpx.ConnectError, httpx.TimeoutException, OSError) as exc:
                    # Cancel everything still in flight so we don't leave
                    # stragglers running after we've decided to fall back.
                    for t in tasks:
                        if not t.done():
                            t.cancel()
                    if not results:
                        # No chunk has succeeded yet → service is genuinely
                        # unreachable (local dev with broken URL, or scaled
                        # to 0 with scaler disabled). Honour the existing
                        # "service unreachable → local CPU" contract.
                        logger.warning(
                            "docling.first_chunk_unreachable falling back to local CPU file=%s reason=%s",
                            path.name, type(exc).__name__,
                        )
                        return await asyncio.to_thread(_parse_pdf_local, path)
                    # Partial success then a failure mid-stream: that's a
                    # real error, not a cold-start. Don't quietly degrade.
                    raise
                except Exception:
                    for t in tasks:
                        if not t.done():
                            t.cancel()
                    raise

        results.sort(key=lambda r: r[0])
        merged: list[dict[str, Any]] = []
        for _, sections in results:
            merged.extend(sections)
        merged = _stitch_boundary_sections(merged)
        logger.info(
            "docling.parse_async_ok file=%s chunks=%d sections=%d",
            path.name, total, len(merged),
        )
        return merged
    finally:
        docling_scaler.mark_request_end()


def parse_text_file(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return []
    return [{
        "text": text,
        "file": path.name,
        "section_title": "",
        "page_start": None,
        "page_end": None,
    }]


def parse_file(path: Path) -> list[dict[str, Any]]:
    """Dispatch to the right parser based on extension."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        return parse_pdf(path)
    if ext in {".txt", ".md"}:
        return parse_text_file(path)
    raise ValueError(f"Unsupported file extension: {ext}")


# ---------------------------------------------------------------------------
# Qdrant helpers
# ---------------------------------------------------------------------------

def _point_id(doc_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, doc_id))


def _get_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def _ensure_collection(client: QdrantClient, name: str, vector_size: int) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if name not in existing:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


async def ingest_sections(
    sections: list[dict[str, Any]],
    *,
    collection: str,
    kb_id: str | None = None,
    document_id: str | None = None,
    progress: ProgressCallback | None = None,
) -> int:
    """Chunk, embed, and upsert sections into a Qdrant collection.

    Args:
        sections: Output of parse_pdf / parse_text_file.
        collection: Target Qdrant collection name.
        kb_id: Knowledge base id, stored in payload for scoped retrieval.
        document_id: Stable document id, stored in payload so a whole upload
            can be deleted with a single filter.
        progress: Optional async callback (current, total) called during embed.

    Returns:
        Total number of points upserted.
    """
    if not sections:
        return 0

    chunks: list[tuple[str, dict[str, Any]]] = []
    for idx, section in enumerate(sections, start=1):
        text = section["text"]
        for chunk_idx, chunk in enumerate(chunk_text(text), start=1):
            doc_key_parts = [
                kb_id or "shared",
                document_id or section["file"],
                f"s{idx}",
                f"c{chunk_idx}",
            ]
            doc_key = ":".join(doc_key_parts)
            payload: dict[str, Any] = {
                "text": chunk,
                "file": section["file"],
                "source": section["file"],
                "section_title": section.get("section_title", ""),
                "page_start": section.get("page_start"),
                "page_end": section.get("page_end"),
            }
            if kb_id:
                payload["kb_id"] = kb_id
            if document_id:
                payload["document_id"] = document_id
            chunks.append((doc_key, payload))

    if not chunks:
        return 0

    client = _get_client()

    all_points: list[PointStruct] = []
    batch_start = 0
    total = len(chunks)
    while batch_start < total:
        batch: list[tuple[str, dict[str, Any]]] = []
        batch_chars = 0
        for doc_key, payload in chunks[batch_start: batch_start + EMBED_BATCH_SIZE]:
            doc_len = len(payload["text"])
            if batch and batch_chars + doc_len > EMBED_MAX_BATCH_CHARS:
                break
            batch.append((doc_key, payload))
            batch_chars += doc_len

        texts = [p["text"] for _, p in batch]
        vectors = await embed(texts)

        for (doc_key, payload), vec in zip(batch, vectors, strict=True):
            all_points.append(PointStruct(
                id=_point_id(doc_key),
                vector=vec,
                payload=payload,
            ))

        batch_start += len(batch)
        if progress is not None:
            await progress(min(batch_start, total), total)

    vector_size = len(all_points[0].vector)
    _ensure_collection(client, collection, vector_size)

    for i in range(0, len(all_points), 100):
        client.upsert(collection_name=collection, points=all_points[i: i + 100])

    return len(all_points)


def delete_document_points(collection: str, document_id: str) -> None:
    """Remove every vector belonging to a document_id from a collection."""
    from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue

    client = _get_client()
    try:
        client.delete(
            collection_name=collection,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
                )
            ),
        )
    except Exception as exc:  # noqa: BLE001
        # Collection might not exist yet (no uploads) — treat as no-op.
        print(f"[WARN] delete_document_points collection={collection} err={exc}")


def drop_collection(collection: str) -> None:
    client = _get_client()
    try:
        client.delete_collection(collection_name=collection)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] drop_collection collection={collection} err={exc}")
