from __future__ import annotations

import argparse
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import OcrMacOptions, PdfPipelineOptions, TesseractOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from core.llm import embed
from core.settings import QDRANT_API_KEY, QDRANT_COLLECTION, QDRANT_URL


def _default_pdf_dir() -> Path:
    script = Path(__file__).resolve()
    # Local repo layout: <repo>/app/kb/ingest_docling.py
    if len(script.parents) >= 3:
        return script.parents[2] / "data" / "data_raw"
    # Container fallback: caller should usually pass --docling-json-dir explicitly.
    return Path.cwd() / "data" / "data_raw"


@dataclass
class Doc:
    id: str
    text: str
    payload: dict[str, Any]


def _point_id(doc_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, doc_id))


def _chunk_text(text: str, max_chars: int = 3000, overlap: int = 300) -> Iterable[str]:
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


def _ensure_collection(client: QdrantClient, name: str, vector_size: int, recreate: bool) -> None:
    if recreate:
        client.delete_collection(collection_name=name, timeout=60)
    existing = {c.name for c in client.get_collections().collections}
    if name not in existing:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


def _extract_pages(document: Any) -> list[tuple[int | None, str]]:
    pages: list[tuple[int | None, str]] = []

    doc_pages = getattr(document, "pages", None)
    if doc_pages:
        for page in doc_pages:
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
                pages.append((int(page_no) if page_no is not None else None, text.strip()))

    if pages:
        return pages

    # Fallback: export to dict/json and look for pages
    try:
        data = None
        if hasattr(document, "export_to_dict"):
            data = document.export_to_dict()
        elif hasattr(document, "export_to_json"):
            data = json.loads(document.export_to_json())
        if isinstance(data, dict):
            for page in data.get("pages", []) or []:
                page_no = page.get("number") or page.get("page_number") or page.get("page_no")
                text = page.get("text") or page.get("content")
                if isinstance(text, str) and text.strip():
                    pages.append((int(page_no) if page_no is not None else None, text.strip()))
    except Exception:  # noqa: BLE001
        return pages

    return pages


def _build_ocr_options(engine: str, lang: list[str], force_full_page_ocr: bool):
    if engine == "mac":
        return OcrMacOptions(lang=lang, force_full_page_ocr=force_full_page_ocr)
    return TesseractOcrOptions(lang=lang, force_full_page_ocr=force_full_page_ocr)


def _source_meta_from_name(name: str) -> tuple[str, str | None, str]:
    stem = Path(name).stem.lower()
    if stem.startswith("standard_200_"):
        return "standards", stem, "standard_abschnitt"
    return "grundschutz", None, "kompendium_abschnitt"


def _build_docs(
    pdf_dir: Path,
    device: str,
    ocr: bool,
    ocr_engine: str,
    ocr_lang: list[str],
    force_full_page_ocr: bool,
) -> Iterable[Doc]:
    pdf_opts = PdfPipelineOptions(
        accelerator_options=AcceleratorOptions(device=device, num_threads=4),
        do_ocr=ocr,
        ocr_batch_size=1,
        layout_batch_size=1,
        table_batch_size=1,
        ocr_options=_build_ocr_options(ocr_engine, ocr_lang, force_full_page_ocr),
    )
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts),
        }
    )
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    for pdf in pdfs:
        source_scope, standard_id, doc_type = _source_meta_from_name(pdf.name)
        result = converter.convert(str(pdf))
        if not getattr(result, "document", None):
            continue
        document = result.document
        pages = _extract_pages(document)
        if not pages:
            # Fallback to whole document
            if hasattr(document, "export_to_markdown"):
                text = document.export_to_markdown()
            elif hasattr(document, "export_to_text"):
                text = document.export_to_text()
            else:
                text = ""
            if isinstance(text, str) and text.strip():
                yield Doc(
                    id=f"docling:{pdf.name}",
                    text=text.strip(),
                    payload={
                        "file": pdf.name,
                        "page_start": None,
                        "source": pdf.name,
                        "source_scope": source_scope,
                        "standard_id": standard_id,
                        "doc_type": doc_type,
                    },
                )
            continue

        for idx, (page_no, text) in enumerate(pages, start=1):
            for chunk_idx, chunk in enumerate(_chunk_text(text), start=1):
                doc_id = f"docling:{pdf.name}:p{page_no or idx}:c{chunk_idx}"
                payload = {
                    "file": pdf.name,
                    "page_start": page_no,
                    "source": pdf.name,
                    "chunk_index": chunk_idx,
                    "source_scope": source_scope,
                    "standard_id": standard_id,
                    "doc_type": doc_type,
                }
                yield Doc(id=doc_id, text=chunk, payload=payload)


def _extract_page_from_prov(prov: Any) -> int | None:
    if isinstance(prov, dict):
        pn = prov.get("page_no")
        return pn if isinstance(pn, int) else None
    if isinstance(prov, list):
        page_numbers = [p.get("page_no") for p in prov if isinstance(p, dict) and isinstance(p.get("page_no"), int)]
        if page_numbers:
            return min(page_numbers)
    return None


def _walk_docling_json(
    node: Any,
    current_page: int | None,
    items: list[tuple[int | None, str, dict[str, Any]]],
) -> None:
    if isinstance(node, dict):
        page = current_page
        pn = _extract_page_from_prov(node.get("prov"))
        if pn is not None:
            page = pn

        # Collect textual leaves with best-known page context.
        for key in ("text", "content"):
            value = node.get(key)
            if isinstance(value, str):
                stripped = value.strip()
                if stripped:
                    items.append((page, stripped, {}))

        for value in node.values():
            _walk_docling_json(value, page, items)
        return

    if isinstance(node, list):
        for child in node:
            _walk_docling_json(child, current_page, items)


def _parse_ref_index(ref: str, prefix: str) -> int | None:
    if not isinstance(ref, str):
        return None
    marker = f"#/{prefix}/"
    if not ref.startswith(marker):
        return None
    raw = ref[len(marker) :]
    return int(raw) if raw.isdigit() else None


def _collect_text_refs(
    ref: str,
    groups: list[dict[str, Any]],
    out: list[int],
) -> None:
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


def _build_docs_from_docling_json(json_dir: Path) -> Iterable[Doc]:
    json_files = sorted(json_dir.glob("*.json"))
    for json_path in json_files:
        source_scope, standard_id, doc_type = _source_meta_from_name(json_path.name)
        data = json.loads(json_path.read_text(encoding="utf-8"))
        texts = data.get("texts")
        if not isinstance(texts, list):
            continue

        ordered_indices = _ordered_text_indices(data)
        if not ordered_indices:
            ordered_indices = list(range(len(texts)))

        sections: list[dict[str, Any]] = []
        current_title: str | None = None
        current_texts: list[str] = []
        current_pages: list[int] = []

        def flush_section() -> None:
            nonlocal current_title, current_texts, current_pages
            content = " ".join(current_texts).strip()
            if len(content) < 40:
                current_title = None
                current_texts = []
                current_pages = []
                return
            section_title = (current_title or "Untitled Section").strip()
            # Keep header and body together in one retrieval chunk until the next header starts.
            merged_text = content
            if section_title and not content.lower().startswith(section_title.lower()):
                merged_text = f"{section_title}\n\n{content}"
            sections.append(
                {
                    "title": section_title,
                    "text": merged_text,
                    "page_start": min(current_pages) if current_pages else None,
                    "page_end": max(current_pages) if current_pages else None,
                }
            )
            current_title = None
            current_texts = []
            current_pages = []

        for text_idx in ordered_indices:
            if text_idx >= len(texts):
                continue
            item = texts[text_idx]
            if not isinstance(item, dict):
                continue
            content_layer = item.get("content_layer")
            if content_layer == "furniture":
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

            # Section boundary at heading-like labels.
            if label in {"section_header", "title", "chapter_title"}:
                flush_section()
                current_title = cleaned
                continue

            current_texts.append(cleaned)

        flush_section()

        for idx, section in enumerate(sections, start=1):
            doc_id = f"docling-json:{json_path.stem}:s{idx}"
            pdf_name = f"{json_path.stem}.pdf"
            payload = {
                "text": section["text"],
                "file": pdf_name,
                "source": pdf_name,
                "source_scope": source_scope,
                "standard_id": standard_id,
                "doc_type": doc_type,
                "section_title": section["title"],
                "section_index": idx,
                "page_start": section["page_start"],
                "page_end": section["page_end"],
            }
            yield Doc(id=doc_id, text=section["text"], payload=payload)


async def _ingest(
    docs: list[Doc],
    collection: str,
    recreate: bool,
    batch_size: int,
    max_batch_chars: int,
) -> None:
    if not docs:
        print("No documents found to ingest.")
        return

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    first_vec = (await embed([docs[0].text]))[0]
    _ensure_collection(client, collection, len(first_vec), recreate=recreate)

    points = [PointStruct(id=_point_id(docs[0].id), vector=first_vec, payload=docs[0].payload)]
    client.upsert(collection_name=collection, points=points)

    start = 1
    current_batch_size = batch_size
    while start < len(docs):
        # Build a batch constrained by count and total char budget
        batch: list[Doc] = []
        total_chars = 0
        for doc in docs[start : start + current_batch_size]:
            doc_len = len(doc.text)
            if batch and total_chars + doc_len > max_batch_chars:
                break
            batch.append(doc)
            total_chars += doc_len

        vectors = await embed([d.text for d in batch])
        points = [
            PointStruct(id=_point_id(d.id), vector=vec, payload=d.payload)
            for d, vec in zip(batch, vectors, strict=True)
        ]
        try:
            client.upsert(collection_name=collection, points=points)
            start += len(batch)
        except Exception as exc:  # noqa: BLE001
            message = str(exc)
            if "Payload error" in message and current_batch_size > 1:
                current_batch_size = max(1, current_batch_size // 2)
                print(f"Batch too large, reducing batch size to {current_batch_size} and retrying...")
                continue
            raise

    print(f"Ingested {len(docs)} documents into '{collection}'.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest PDFs via Docling into Qdrant.")
    parser.add_argument(
        "--pdf-dir",
        default=str(_default_pdf_dir()),
        help="Directory containing PDFs.",
    )
    parser.add_argument(
        "--docling-json-dir",
        default="",
        help="Optional directory containing Docling JSON exports. If set, ingest from JSON instead of live PDF conversion.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "mps", "cuda", "auto"],
        default="cpu",
        help="Docling inference device for live PDF conversion.",
    )
    parser.add_argument(
        "--ocr",
        action="store_true",
        help="Enable OCR in live PDF conversion mode.",
    )
    parser.add_argument(
        "--ocr-engine",
        choices=["tesseract", "mac"],
        default="tesseract",
        help="OCR backend. Use 'mac' for native macOS Vision OCR.",
    )
    parser.add_argument(
        "--ocr-lang",
        nargs="+",
        default=["eng", "deu"],
        help="OCR languages for Tesseract, e.g. --ocr-lang eng deu.",
    )
    parser.add_argument(
        "--force-full-page-ocr",
        action="store_true",
        help="Force OCR on full page instead of auto text-region detection.",
    )
    parser.add_argument("--collection", default=QDRANT_COLLECTION, help="Target Qdrant collection.")
    parser.add_argument("--recreate", action="store_true", help="Recreate collection before ingesting.")
    parser.add_argument("--batch-size", type=int, default=64, help="Max embedding batch size.")
    parser.add_argument(
        "--max-batch-chars",
        type=int,
        default=20000,
        help="Max total characters per embedding request.",
    )
    args = parser.parse_args()

    if args.docling_json_dir:
        docs = list(_build_docs_from_docling_json(Path(args.docling_json_dir)))
    else:
        docs = list(
            _build_docs(
                Path(args.pdf_dir),
                args.device,
                args.ocr,
                args.ocr_engine,
                args.ocr_lang,
                args.force_full_page_ocr,
            )
        )

    import asyncio

    asyncio.run(_ingest(docs, args.collection, args.recreate, args.batch_size, args.max_batch_chars))


if __name__ == "__main__":
    main()
