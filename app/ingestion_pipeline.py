"""Shared document ingestion pipeline: parse → chunk → embed → upsert.

Used by the per-user FastAPI upload endpoint (app/api_routes.py).
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from llm import embed
from settings import (
    CHUNK_MAX_CHARS,
    CHUNK_OVERLAP,
    EMBED_BATCH_SIZE,
    EMBED_MAX_BATCH_CHARS,
    QDRANT_API_KEY,
    QDRANT_URL,
)

ProgressCallback = Callable[[int, int], Awaitable[None]]
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


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


def parse_pdf(path: Path) -> list[dict[str, Any]]:
    pdf_opts = PdfPipelineOptions(do_ocr=False)
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts)},
    )
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
