from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
import uuid
from pathlib import Path
from typing import Any, Iterable

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PayloadSchemaType, PointStruct, VectorParams

from llm import embed
from settings import GRUNDSCHUTZ_SOURCE_PDF, QDRANT_API_KEY, QDRANT_COLLECTION, QDRANT_URL


@dataclass
class Doc:
    id: str
    text: str
    payload: dict[str, Any]


def _point_id(doc_id: str) -> str:
    # Qdrant expects int or UUID; derive stable UUID from our string id
    return str(uuid.uuid5(uuid.NAMESPACE_URL, doc_id))


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_entfallen_requirement(req: dict[str, Any]) -> bool:
    anforderung = req.get("anforderung")
    if not isinstance(anforderung, dict):
        return False
    titel = anforderung.get("titel")
    return isinstance(titel, str) and titel.strip().upper() == "ENTFALLEN"


def _extract_page_range(mapping: Any) -> tuple[int | None, int | None]:
    if not isinstance(mapping, dict):
        return None, None
    pages = mapping.get("pages")
    if isinstance(pages, list):
        page_list = sorted({p for p in pages if isinstance(p, int)})
        if page_list:
            # Keep the densest near-contiguous cluster to avoid huge ranges caused by sparse outliers.
            clusters: list[list[int]] = []
            current: list[int] = [page_list[0]]
            for p in page_list[1:]:
                if p - current[-1] <= 2:
                    current.append(p)
                else:
                    clusters.append(current)
                    current = [p]
            clusters.append(current)
            best = max(clusters, key=len)
            return best[0], best[-1]

    start = mapping.get("page_start")
    end = mapping.get("page_end")
    return (start if isinstance(start, int) else None, end if isinstance(end, int) else None)


def _beschreibung_text(desc: Any) -> str:
    if isinstance(desc, str):
        return desc.strip()
    if isinstance(desc, dict):
        parts: list[str] = []
        for key, value in desc.items():
            if isinstance(value, str) and value.strip():
                title = str(key).replace("_", " ").strip().title()
                parts.append(f"{title}\n{value.strip()}")
        return "\n\n".join(parts).strip()
    return ""


def _gefaehrdungslage_text(gefaehrdungslage: Any) -> str:
    if isinstance(gefaehrdungslage, str):
        return gefaehrdungslage.strip()
    if isinstance(gefaehrdungslage, list):
        parts: list[str] = []
        for item in gefaehrdungslage:
            if not isinstance(item, dict):
                continue
            titel = item.get("titel")
            beschreibung = item.get("beschreibung")
            block: list[str] = []
            if isinstance(titel, str) and titel.strip():
                block.append(titel.strip())
            if isinstance(beschreibung, str) and beschreibung.strip():
                block.append(beschreibung.strip())
            if block:
                parts.append("\n".join(block))
        return "\n\n".join(parts).strip()
    return ""


def _extract_page_from_prov(prov: Any) -> int | None:
    if isinstance(prov, dict):
        page = prov.get("page_no")
        return page if isinstance(page, int) else None
    if isinstance(prov, list):
        pages = [p.get("page_no") for p in prov if isinstance(p, dict) and isinstance(p.get("page_no"), int)]
        return min(pages) if pages else None
    return None


def _standards_docs_from_docling_json(json_dir: Path) -> Iterable[Doc]:
    json_files = sorted(json_dir.glob("standard_200_*.json"))
    for json_path in json_files:
        data = _load_json(json_path)
        texts = data.get("texts")
        if not isinstance(texts, list):
            continue

        sections: list[dict[str, Any]] = []
        current_title: str | None = None
        current_texts: list[str] = []
        current_pages: list[int] = []

        def flush_section() -> None:
            nonlocal current_title, current_texts, current_pages
            content = " ".join(current_texts).strip()
            if len(content) < 80:
                current_title = None
                current_texts = []
                current_pages = []
                return
            sections.append(
                {
                    "title": current_title or "Abschnitt",
                    "text": content,
                    "page_start": min(current_pages) if current_pages else None,
                    "page_end": max(current_pages) if current_pages else None,
                }
            )
            current_title = None
            current_texts = []
            current_pages = []

        for item in texts:
            if not isinstance(item, dict):
                continue
            if item.get("content_layer") == "furniture":
                continue

            raw = item.get("canonical_text") or item.get("text")
            if not isinstance(raw, str):
                continue
            cleaned = re.sub(r"\s+", " ", raw).strip()
            if not cleaned:
                continue

            page_no = _extract_page_from_prov(item.get("prov"))
            if isinstance(page_no, int):
                current_pages.append(page_no)

            label = item.get("label")
            if label in {"section_header", "title", "chapter_title"}:
                flush_section()
                current_title = cleaned
                continue

            current_texts.append(cleaned)

        flush_section()

        pdf_name = f"{json_path.stem}.pdf"
        for idx, section in enumerate(sections, start=1):
            doc_id = f"standard-docling:{json_path.stem}:s{idx}"
            text = section["text"]
            payload = {
                "text": text,
                "title": section["title"],
                "doc_type": "standard_abschnitt",
                "source": {"document": json_path.stem, "file": pdf_name},
                "file": pdf_name,
                "document_id": f"{json_path.stem}:s{idx}",
                "section_title": section["title"],
                "section_index": idx,
                "page_start": section["page_start"],
                "page_end": section["page_end"],
            }
            yield Doc(id=doc_id, text=text, payload=payload)


def _standards_docs_from_preprocessed(root: Path) -> Iterable[Doc]:
    for path in root.rglob("*.json"):
        if path.name == "_index.json":
            continue
        data = _load_json(path)
        if not isinstance(data, dict):
            continue
        content = (data.get("content") or "").strip()
        if not content:
            continue
        header = data.get("header") or ""
        text = f"{header}\n\n{content}".strip()
        doc_id = f"standards:{data.get('id') or path.stem}"
        source = data.get("source") or {}
        source_file = source.get("file") if isinstance(source, dict) else None
        pages = data.get("pages") or {}
        page_start = pages.get("start") if isinstance(pages, dict) else None
        page_end = pages.get("end") if isinstance(pages, dict) else None
        payload = {
            "text": text,
            "title": header,
            "doc_type": "standard_abschnitt",
            "source": data.get("source") or str(path),
            "file": source_file,
            "page": data.get("pages"),
            "page_start": page_start,
            "page_end": page_end,
            "header_hierarchy": data.get("header_hierarchy"),
            "document_id": data.get("id"),
        }
        yield Doc(id=doc_id, text=text, payload=payload)


def _grundschutz_docs(path: Path) -> Iterable[Doc]:
    data = _load_json(path)

    for item in data.get("elementare_gefaehrdungen", []):
        beschreibung = (item.get("beschreibung") or "").strip()
        beispiele = item.get("beispiele") or []
        extra = "\n".join(beispiele).strip()
        text = "\n\n".join([t for t in [beschreibung, extra] if t]).strip()
        if not text:
            continue
        doc_id = f"grundschutz:gefaehrdung:{item.get('id')}"
        payload = {
            "text": text,
            "title": item.get("titel"),
            "source": "grundschutz.json",
            "document_id": item.get("id"),
            "typ": "elementare_gefaehrdung",
        }
        yield Doc(id=doc_id, text=text, payload=payload)

    for schicht in data.get("schichten", []):
        for baustein in schicht.get("bausteine", []):
            base_payload = {
                "source": "grundschutz.json",
                "file": GRUNDSCHUTZ_SOURCE_PDF,
                "schicht_id": schicht.get("id"),
                "schicht_name": schicht.get("name"),
                "baustein_id": (baustein.get("id") or ""),
                "baustein_titel": baustein.get("titel"),
                "document_id": baustein.get("id"),
            }

            beschreibung = _beschreibung_text(baustein.get("beschreibung") or "")
            b_desc_start, b_desc_end = _extract_page_range(baustein.get("page_mapping_beschreibung"))
            if beschreibung:
                text = f"{baustein.get('titel') or ''}\n\n{beschreibung}".strip()
                yield Doc(
                    id=f"grundschutz:baustein:beschreibung:{baustein.get('id')}",
                    text=text,
                    payload={
                        "text": text,
                        "title": baustein.get("titel"),
                        **base_payload,
                        "doc_type": "baustein_beschreibung",
                        "page_start": b_desc_start,
                        "page_end": b_desc_end,
                    },
                )

            gefaehrdungslage = _gefaehrdungslage_text(baustein.get("gefaehrdungslage") or "")
            b_gef_start, b_gef_end = _extract_page_range(baustein.get("page_mapping_gefaehrdungslage"))
            if gefaehrdungslage:
                text = f"{baustein.get('titel') or ''}\n\n{gefaehrdungslage}".strip()
                yield Doc(
                    id=f"grundschutz:baustein:gefaehrdungslage:{baustein.get('id')}",
                    text=text,
                    payload={
                        "text": text,
                        "title": baustein.get("titel"),
                        **base_payload,
                        "doc_type": "baustein_gefaehrdungslage",
                        "page_start": b_gef_start,
                        "page_end": b_gef_end,
                    },
                )

            anforderungen = baustein.get("anforderungen") or {}
            for level in ("basis", "standard", "erhoeht"):
                for req in anforderungen.get(level, []) or []:
                    if _is_entfallen_requirement(req):
                        continue
                    inhalt = (req.get("inhalt") or "").strip()
                    if not inhalt:
                        continue
                    req_meta = req.get("anforderung") if isinstance(req.get("anforderung"), dict) else {}
                    req_title = req_meta.get("titel")
                    text = f"{req_title or ''}\n\n{inhalt}".strip()
                    req_id = req.get("id")
                    req_page_start, req_page_end = _extract_page_range(req.get("page_mapping"))
                    payload = {
                        "text": text,
                        "title": req_title,
                        **base_payload,
                        "doc_type": "anforderung",
                        "anforderung_id": req_id,
                        "anforderung_level": level,
                        "anforderung_typ": req_meta.get("typ"),
                        "anforderung_typ_lang": req_meta.get("typ_lang"),
                        "verantwortliche": req_meta.get("verantwortliche"),
                        "zustaendigkeiten": req.get("zustaendigkeiten"),
                        "modal_verben": req.get("modal_verben"),
                        "page_start": req_page_start,
                        "page_end": req_page_end,
                    }
                    yield Doc(
                        id=f"grundschutz:anforderung:{req_id}",
                        text=text,
                        payload=payload,
                    )


def _build_docs(source: str) -> list[Doc]:
    docs: list[Doc] = []
    env_root = os.getenv("DATA_PREPROCESSED_DIR")
    env_docling = os.getenv("INGEST_DOCLING_JSON_DIR")
    script_path = Path(__file__).resolve()
    candidates: list[Path] = []
    if env_root:
        candidates.append(Path(env_root))
    # Docker default mount in this project.
    candidates.append(Path("/data/data_preprocessed"))
    # Local repo layout fallback.
    if len(script_path.parents) >= 3:
        candidates.append(script_path.parents[2] / "data" / "data_preprocessed")
    # Last-resort fallback relative to script location.
    candidates.append(script_path.parent / "data" / "data_preprocessed")

    data_root = next((p for p in candidates if p.is_dir()), candidates[0])

    docling_candidates: list[Path] = []
    if env_docling:
        docling_candidates.append(Path(env_docling))
    docling_candidates.append(Path("/data/data_docling_json_ocr"))
    if len(script_path.parents) >= 3:
        docling_candidates.append(script_path.parents[2] / "data" / "data_docling_json_ocr")
    docling_root = next((p for p in docling_candidates if p.is_dir()), docling_candidates[0])

    if source in ("all", "standards"):
        standards_docs = list(_standards_docs_from_docling_json(docling_root))
        if standards_docs:
            docs.extend(standards_docs)
        else:
            docs.extend(_standards_docs_from_preprocessed(data_root / "standards"))

    if source in ("all", "grundschutz"):
        enriched = data_root / "grundschutz_with_pages.json"
        docs.extend(_grundschutz_docs(enriched if enriched.is_file() else (data_root / "grundschutz.json")))

    return docs


def _ensure_collection(client: QdrantClient, name: str, vector_size: int, recreate: bool) -> None:
    if recreate:
        client.recreate_collection(
            collection_name=name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        return

    existing = {c.name for c in client.get_collections().collections}
    if name not in existing:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
    # Index frequently-filtered payload fields for fast filtered retrieval.
    for field in ("doc_type", "schicht_id", "baustein_id", "anforderung_typ", "anforderung_level"):
        try:
            client.create_payload_index(
                collection_name=name,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass


def _collection_exists(client: QdrantClient, name: str) -> bool:
    existing = {c.name for c in client.get_collections().collections}
    return name in existing


async def _ingest(docs: list[Doc], collection: str, recreate: bool, batch_size: int) -> None:
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
        batch = docs[start : start + current_batch_size]
        vectors = await embed([d.text for d in batch])
        points = [
            PointStruct(id=_point_id(d.id), vector=vec, payload=d.payload)
            for d, vec in zip(batch, vectors, strict=True)
        ]
        try:
            client.upsert(collection_name=collection, points=points)
            start += current_batch_size
        except Exception as exc:  # noqa: BLE001
            message = str(exc)
            if "Payload error" in message and current_batch_size > 1:
                current_batch_size = max(1, current_batch_size // 2)
                print(f"Batch too large, reducing batch size to {current_batch_size} and retrying...")
                continue
            raise

    print(f"Ingested {len(docs)} documents into '{collection}'.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Grundschutz/Standards docs into Qdrant.")
    parser.add_argument(
        "--source",
        choices=["all", "grundschutz", "standards"],
        default="all",
        help="Which documents to ingest.",
    )
    parser.add_argument(
        "--collection",
        default=QDRANT_COLLECTION,
        help="Target Qdrant collection.",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate collection before ingesting.",
    )
    parser.add_argument(
        "--skip-if-exists",
        action="store_true",
        help="Exit successfully when target collection already exists (ignored when --recreate is set).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Embedding batch size.",
    )
    args = parser.parse_args()

    if args.skip_if_exists and not args.recreate:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        if _collection_exists(client, args.collection):
            print(f"Collection already exists: {args.collection}. Skipping ingestion.")
            return

    docs = _build_docs(args.source)

    import asyncio

    asyncio.run(_ingest(docs, args.collection, args.recreate, args.batch_size))


if __name__ == "__main__":
    main()
