"""Chainlit-based document ingestion UI.

Upload PDF or text documents and ingest them into a Qdrant vector collection
via Docling (parsing) and a hosted embedding model (via LiteLLM).
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Iterable

import chainlit as cl
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

import litellm

from settings import (
    CHUNK_MAX_CHARS,
    CHUNK_OVERLAP,
    EMBED_BATCH_SIZE,
    EMBED_MAX_BATCH_CHARS,
    EMBED_MODEL,
    LITELLM_API_KEY,
    LITELLM_BASE_URL,
    MAX_FILE_SIZE_MB,
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
    QDRANT_URL,
)

# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------

def _llm_kwargs() -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if LITELLM_BASE_URL:
        kwargs["api_base"] = LITELLM_BASE_URL
    if LITELLM_API_KEY:
        kwargs["api_key"] = LITELLM_API_KEY
    return kwargs


async def embed(texts: list[str]) -> list[list[float]]:
    response = await litellm.aembedding(
        model=EMBED_MODEL,
        input=texts,
        encoding_format="float",
        **_llm_kwargs(),
    )
    data = sorted(response.data, key=lambda item: item["index"])
    return [item["embedding"] for item in data]


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, max_chars: int = CHUNK_MAX_CHARS, overlap: int = CHUNK_OVERLAP) -> Iterable[str]:
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
            p.get("page_no") for p in prov
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
    """Extract sections from a Docling export dict (structured mode)."""
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
    """Fallback: extract page-by-page from a Docling document object."""
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


def parse_pdf(path: Path) -> list[dict[str, Any]]:
    """Parse a PDF via Docling and return a list of section dicts."""
    pdf_opts = PdfPipelineOptions(do_ocr=False)
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts)},
    )
    result = converter.convert(str(path))
    document = getattr(result, "document", None)
    if document is None:
        return []

    # Try structured export first
    export_dict = None
    if hasattr(document, "export_to_dict"):
        export_dict = document.export_to_dict()
    sections = _sections_from_docling_dict(export_dict, path.name) if export_dict else []
    if sections:
        return sections

    # Fallback: page-level extraction
    sections = _sections_from_pages(document, path.name)
    if sections:
        return sections

    # Last resort: full document text
    text = ""
    if hasattr(document, "export_to_markdown"):
        text = document.export_to_markdown()
    elif hasattr(document, "export_to_text"):
        text = document.export_to_text()
    if isinstance(text, str) and text.strip():
        return [{"text": text.strip(), "file": path.name, "section_title": "", "page_start": None, "page_end": None}]
    return []


def parse_text_file(path: Path) -> list[dict[str, Any]]:
    """Read a plain text / markdown file and return it as a single section."""
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return []
    return [{"text": text, "file": path.name, "section_title": "", "page_start": None, "page_end": None}]


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
    collection: str = QDRANT_COLLECTION,
    msg: cl.Message | None = None,
) -> int:
    """Chunk, embed, and upsert sections into Qdrant. Returns total point count."""
    if not sections:
        return 0

    # Build chunks with metadata
    chunks: list[tuple[str, dict[str, Any]]] = []
    for idx, section in enumerate(sections, start=1):
        text = section["text"]
        for chunk_idx, chunk in enumerate(chunk_text(text), start=1):
            doc_id = f"upload:{section['file']}:s{idx}:c{chunk_idx}"
            payload = {
                "text": chunk,
                "file": section["file"],
                "source": section["file"],
                "section_title": section.get("section_title", ""),
                "page_start": section.get("page_start"),
                "page_end": section.get("page_end"),
            }
            chunks.append((doc_id, payload))

    if not chunks:
        return 0

    client = _get_client()

    # Embed in batches
    all_points: list[PointStruct] = []
    batch_start = 0
    total = len(chunks)
    while batch_start < total:
        batch: list[tuple[str, dict[str, Any]]] = []
        batch_chars = 0
        for doc_id, payload in chunks[batch_start: batch_start + EMBED_BATCH_SIZE]:
            doc_len = len(payload["text"])
            if batch and batch_chars + doc_len > EMBED_MAX_BATCH_CHARS:
                break
            batch.append((doc_id, payload))
            batch_chars += doc_len

        texts = [p["text"] for _, p in batch]
        vectors = await embed(texts)

        for (doc_id, payload), vec in zip(batch, vectors, strict=True):
            all_points.append(PointStruct(
                id=_point_id(doc_id),
                vector=vec,
                payload=payload,
            ))

        batch_start += len(batch)
        if msg:
            progress = min(batch_start, total)
            await msg.stream_token(f"\rEmbedding: {progress}/{total} chunks...")

    # Ensure collection exists
    vector_size = len(all_points[0].vector)
    _ensure_collection(client, collection, vector_size)

    # Upsert in batches of 100
    for i in range(0, len(all_points), 100):
        client.upsert(collection_name=collection, points=all_points[i: i + 100])

    return len(all_points)


# ---------------------------------------------------------------------------
# Chainlit handlers
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


@cl.on_chat_start
async def on_start():
    await cl.Message(
        content=(
            "**Document Ingestion**\n\n"
            f"Upload PDF, TXT, or Markdown files (max {MAX_FILE_SIZE_MB} MB each) "
            f"and I will parse, chunk, embed, and store them in the "
            f"`{QDRANT_COLLECTION}` Qdrant collection.\n\n"
            "Attach files to get started."
        ),
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    files = message.elements or []
    supported_files = [
        f for f in files
        if isinstance(f, cl.File) and Path(f.name).suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not supported_files:
        await cl.Message(
            content="Please attach at least one PDF, TXT, or Markdown file.",
        ).send()
        return

    total_points = 0
    results: list[str] = []

    for file in supported_files:
        file_path = Path(file.path)
        file_size_mb = file_path.stat().st_size / (1024 * 1024)

        if file_size_mb > MAX_FILE_SIZE_MB:
            results.append(f"**{file.name}**: skipped (exceeds {MAX_FILE_SIZE_MB} MB limit)")
            continue

        status_msg = cl.Message(content=f"Processing **{file.name}**...")
        await status_msg.send()

        try:
            # Parse
            ext = file_path.suffix.lower()
            if ext == ".pdf":
                sections = parse_pdf(file_path)
            else:
                sections = parse_text_file(file_path)

            if not sections:
                results.append(f"**{file.name}**: no text content extracted")
                await status_msg.update(content=f"**{file.name}**: no text found")
                continue

            await status_msg.update(
                content=f"**{file.name}**: extracted {len(sections)} sections, embedding..."
            )

            # Ingest
            count = await ingest_sections(sections, msg=status_msg)
            total_points += count
            results.append(f"**{file.name}**: {len(sections)} sections -> {count} chunks ingested")
            await status_msg.update(
                content=f"**{file.name}**: {count} chunks ingested"
            )

        except Exception as exc:
            results.append(f"**{file.name}**: error - {exc}")
            await status_msg.update(content=f"**{file.name}**: error - {exc}")

    summary = "\n".join(f"- {r}" for r in results)
    await cl.Message(
        content=f"**Ingestion complete**\n\n{summary}\n\nTotal: **{total_points}** chunks stored in `{QDRANT_COLLECTION}`.",
    ).send()
