from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import json
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from llm import embed
from settings import (
    CITATION_MAP_PATH,
    GRUNDSCHUTZ_SOURCE_PDF,
    PERSONALIZATION_ENABLED,
    PROFILE_RELEVANCE_THRESHOLD,
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
    QDRANT_URL,
    SCORE_THRESHOLD,
    TOP_K,
)

if TYPE_CHECKING:
    from user_profile import UserProfile


@dataclass
class RagResult:
    text: str
    score: float
    metadata: dict[str, Any]


_client: QdrantClient | None = None
_citation_map: dict[str, dict[str, str]] | None = None


def _canonical_pdf_from_text(value: str) -> str | None:
    raw = (value or "").strip()
    if not raw:
        return None
    lower = raw.lower()

    if lower.endswith(".pdf"):
        return raw.split("/")[-1]

    if "standard_200_1" in lower or "standard 200 1" in lower:
        return "standard_200_1.pdf"
    if "standard_200_2" in lower or "standard 200 2" in lower:
        return "standard_200_2.pdf"
    if "standard_200_3" in lower or "standard 200 3" in lower:
        return "standard_200_3.pdf"
    if "standard_200_4" in lower or "standard 200 4" in lower:
        return "standard_200_4.pdf"

    if "kompendium" in lower or "grundschutz" in lower:
        return GRUNDSCHUTZ_SOURCE_PDF

    return None


def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return _client


def _load_citation_map() -> dict[str, dict[str, str]]:
    global _citation_map
    if _citation_map is not None:
        return _citation_map
    try:
        if CITATION_MAP_PATH.is_file():
            _citation_map = json.loads(CITATION_MAP_PATH.read_text(encoding="utf-8"))
        else:
            _citation_map = {}
    except Exception:  # noqa: BLE001
        _citation_map = {}
    return _citation_map


def _extract_text(payload: dict[str, Any]) -> str:
    for key in ("text", "content", "chunk", "body"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _extract_citation(payload: dict[str, Any]) -> str:
    source = payload.get("source") or payload.get("document") or payload.get("title") or payload.get("file")
    page = payload.get("page") or payload.get("page_number") or payload.get("pages")
    module = payload.get("module") or payload.get("baustein")
    parts: list[str] = []
    if source:
        parts.append(str(source))
    if module:
        parts.append(f"Modul {module}")
    if page:
        parts.append(f"Seite {page}")
    if not parts:
        return "Quelle unbekannt"
    return " | ".join(parts)


def extract_source_file(payload: dict[str, Any]) -> str | None:
    value = payload.get("file")
    if isinstance(value, str):
        resolved = _canonical_pdf_from_text(value)
        if resolved:
            return resolved

    source = payload.get("source")
    if isinstance(source, dict):
        value = source.get("file")
        if isinstance(value, str):
            resolved = _canonical_pdf_from_text(value)
            if resolved:
                return resolved

        for key in ("document", "title", "source"):
            nested = source.get(key)
            if isinstance(nested, str):
                resolved = _canonical_pdf_from_text(nested)
                if resolved:
                    return resolved

    for key in ("source", "document", "title"):
        value = payload.get(key)
        if isinstance(value, str):
            resolved = _canonical_pdf_from_text(value)
            if resolved:
                return resolved

    # Fallback for Grundschutz chunks ingested from structured JSON without explicit PDF file.
    source = payload.get("source")
    if isinstance(source, str) and source.lower().endswith("grundschutz.json"):
        return GRUNDSCHUTZ_SOURCE_PDF
    doc_type = payload.get("doc_type")
    if isinstance(doc_type, str) and doc_type in {
        "anforderung",
        "baustein_beschreibung",
        "baustein_gefaehrdungslage",
    }:
        return GRUNDSCHUTZ_SOURCE_PDF
    return None


def extract_page(payload: dict[str, Any]) -> int | None:
    page_start = payload.get("page_start")
    if isinstance(page_start, int):
        return page_start

    page = payload.get("page")
    if isinstance(page, int):
        return page
    if isinstance(page, dict):
        start = page.get("start")
        if isinstance(start, int):
            return start
    return None


def _clean_text(text: str, max_len: int = 1200) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


async def retrieve(
    query: str,
    top_k: int | None = None,
    *,
    source_scope: str | None = None,
    standard_id: str | None = None,
    include_vectors: bool = False,
) -> list[RagResult]:
    """Retrieve documents matching the query.

    Args:
        query: Search query text
        top_k: Number of results to return
        source_scope: Optional filter by source scope
        standard_id: Optional filter by standard ID
        include_vectors: If True, include embedding vectors in results (for personalization)

    Returns:
        List of RagResult objects
    """
    client = _get_client()
    vector = (await embed([query]))[0]
    k = top_k or TOP_K
    must: list[FieldCondition] = []
    if source_scope:
        must.append(FieldCondition(key="source_scope", match=MatchValue(value=source_scope)))
    if standard_id:
        must.append(FieldCondition(key="standard_id", match=MatchValue(value=standard_id)))
    query_filter = Filter(must=must) if must else None
    print("[DEBUG] retrieve", {"top_k": k, "source_scope": source_scope, "standard_id": standard_id})
    response = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=vector,
        limit=k,
        score_threshold=SCORE_THRESHOLD,
        with_payload=True,
        with_vectors=include_vectors,
        query_filter=query_filter,
    )
    points = list(response.points or [])
    if not points and (source_scope or standard_id):
        # Compatibility fallback for older collections without new metadata fields.
        print("[WARN] filtered_retrieval_empty_fallback_unfiltered", {"top_k": k})
        response = client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=vector,
            limit=k,
            score_threshold=SCORE_THRESHOLD,
            with_payload=True,
            with_vectors=include_vectors,
        )
        points = list(response.points or [])

    hits: list[RagResult] = []
    for hit in points:
        payload = dict(hit.payload or {})
        text = _extract_text(payload)
        if not text:
            continue
        # Store embedding vector if requested (for personalization scoring)
        if include_vectors and hit.vector is not None:
            if isinstance(hit.vector, list):
                payload["_embedding"] = hit.vector
        hits.append(
            RagResult(
                text=_clean_text(text),
                score=float(hit.score),
                metadata=payload,
            )
        )
    return hits


async def personalized_retrieve(
    query: str,
    user_profile: "UserProfile | None",
    balance: float = 0.5,
    top_k: int | None = None,
    *,
    source_scope: str | None = None,
    standard_id: str | None = None,
) -> list[RagResult]:
    """Retrieve documents with personalization based on user profile.

    Implements dual retrieval:
    1. Standard retrieval (current implementation)
    2. User-profile-filtered retrieval (removes irrelevant chunks)

    The balance parameter controls the weighting:
    - balance = 1.0: Only standard retrieval
    - balance = 0.0: Only profile-filtered retrieval
    - balance = 0.5: Blend both mechanisms

    Args:
        query: Search query text
        user_profile: User profile with topics and embeddings
        balance: Weighting between standard (1.0) and personalized (0.0) retrieval
        top_k: Number of results to return
        source_scope: Optional filter by source scope
        standard_id: Optional filter by standard ID

    Returns:
        List of RagResult objects with personalized scoring
    """
    from user_profile import compute_profile_relevance

    # If personalization disabled or no profile, fall back to standard retrieval
    if not PERSONALIZATION_ENABLED or user_profile is None or balance >= 1.0:
        return await retrieve(
            query, top_k, source_scope=source_scope, standard_id=standard_id
        )

    # If no topic embeddings, can't personalize
    if not user_profile.topic_embeddings:
        print("[DEBUG] personalized_retrieve: no topic embeddings, using standard")
        return await retrieve(
            query, top_k, source_scope=source_scope, standard_id=standard_id
        )

    k = top_k or TOP_K

    # Retrieve more candidates for filtering (2x to account for filtering)
    extended_k = max(k, min(k * 2, 20))

    # Standard retrieval with embeddings for scoring
    results = await retrieve(
        query,
        extended_k,
        source_scope=source_scope,
        standard_id=standard_id,
        include_vectors=True,
    )

    print(f"[DEBUG] personalized_retrieve: balance={balance}, candidates={len(results)}")

    # Score and filter based on user profile
    scored_results: list[tuple[RagResult, float, float]] = []
    for result in results:
        base_score = result.score
        embedding = result.metadata.get("_embedding")

        if embedding and user_profile is not None:
            profile_relevance = compute_profile_relevance(embedding, user_profile)
        else:
            profile_relevance = 0.5  # Neutral if no embedding

        # Apply negative filter: skip chunks below relevance threshold when balance < 1
        if balance < 0.5 and profile_relevance < PROFILE_RELEVANCE_THRESHOLD:
            print(f"[DEBUG] filtered out chunk with relevance={profile_relevance:.3f}")
            continue

        # Blend scores
        blended_score = balance * base_score + (1 - balance) * profile_relevance

        # Store scores in metadata for debugging
        result.metadata["_original_score"] = base_score
        result.metadata["_profile_relevance"] = profile_relevance
        result.metadata["_blended_score"] = blended_score

        scored_results.append((result, blended_score, profile_relevance))

    # Sort by blended score
    scored_results.sort(key=lambda x: x[1], reverse=True)

    # Take top k and update scores
    final_results: list[RagResult] = []
    for result, blended_score, _ in scored_results[:k]:
        # Clean up embedding from metadata (large, not needed downstream)
        result.metadata.pop("_embedding", None)
        # Update score to blended score
        result.score = blended_score
        final_results.append(result)

    print(f"[DEBUG] personalized_retrieve: returning {len(final_results)} results")
    return final_results


def build_context(results: list[RagResult]) -> str:
    lines: list[str] = []
    for idx, result in enumerate(results, start=1):
        citation = _extract_citation(result.metadata)
        lines.append(f"[{idx}] {result.text}\nQuelle: {citation}")
    return "\n\n".join(lines)


def format_citations(results: list[RagResult]) -> str:
    citation_map = _load_citation_map()
    lines: list[str] = []
    for idx, result in enumerate(results, start=1):
        meta = result.metadata
        source = meta.get("source") or {}
        document = meta.get("document") or (source.get("document") if isinstance(source, dict) else None)
        file_name = extract_source_file(meta) or (source.get("file") if isinstance(source, dict) else None)
        page = meta.get("page") or meta.get("page_start")
        if isinstance(page, dict):
            start = page.get("start")
            end = page.get("end")
        else:
            start = page if isinstance(page, int) else None
            end = None

        # Grundschutz-specific fields
        baustein_id = meta.get("baustein")
        baustein_title = meta.get("baustein_titel")
        anforderung_id = meta.get("anforderung_id")

        doc_key = None
        if isinstance(document, str):
            doc_key = document
        elif isinstance(file_name, str):
            doc_key = file_name.replace(".pdf", "")
        meta_entry = citation_map.get(doc_key or "", {})

        author = meta_entry.get("author")
        year = meta_entry.get("year")
        title = meta_entry.get("title") or doc_key
        publisher = meta_entry.get("publisher")

        page_label = None
        if start is not None and end is not None and start != end:
            page_label = f"S. {start}–{end}"
        elif start is not None:
            page_label = f"S. {start}"
        elif end is not None:
            page_label = f"S. {end}"

        if baustein_id:
            parts = [f"Modul {baustein_id}"]
            if baustein_title:
                parts.append(str(baustein_title))
            if anforderung_id:
                parts.append(f"Anforderung {anforderung_id}")
            if page_label:
                parts.append(page_label)
            line = " | ".join(parts)
        elif author or year or title:
            parts = []
            if author:
                parts.append(author)
            if year:
                parts.append(f"({year}).")
            if title:
                parts.append(title + ".")
            if publisher:
                parts.append(publisher + ".")
            if page_label:
                parts.append(page_label + ".")
            line = " ".join(parts)
        else:
            line = _extract_citation(meta)
        lines.append(f"[{idx}] {line}")
    return "\n".join(lines)
