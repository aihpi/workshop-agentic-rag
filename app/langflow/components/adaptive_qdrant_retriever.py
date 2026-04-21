from __future__ import annotations

import math
import json
import re
from collections import Counter
from statistics import mean
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.messages import BaseMessage
from qdrant_client import QdrantClient

from lfx.base.vectorstores.model import LCVectorStoreComponent, check_cached_vector_store
from lfx.io import (
    BoolInput,
    DropdownInput,
    FloatInput,
    HandleInput,
    IntInput,
    MessageTextInput,
    Output,
    QueryInput,
    SecretStrInput,
    StrInput,
)
from lfx.schema.data import Data
from lfx.schema.dataframe import DataFrame


_STOPWORDS = {
    "aber",
    "als",
    "auch",
    "auf",
    "aus",
    "bei",
    "das",
    "dem",
    "den",
    "der",
    "des",
    "die",
    "ein",
    "eine",
    "einer",
    "eines",
    "fuer",
    "für",
    "hat",
    "ich",
    "ihre",
    "im",
    "in",
    "ist",
    "mit",
    "muss",
    "müssen",
    "nach",
    "oder",
    "soll",
    "sollte",
    "sollten",
    "sich",
    "sind",
    "und",
    "vom",
    "von",
    "was",
    "werden",
    "welche",
    "welcher",
    "welches",
    "wie",
    "wir",
    "zwischen",
    "zum",
    "zur",
}

_COMPARISON_MARKERS = {
    "unterschied",
    "vergleich",
    "gegenueber",
    "gegenüber",
    "versus",
    "vs",
}


class AdaptiveQdrantRetrieverComponent(LCVectorStoreComponent):
    display_name = "Adaptive Qdrant Retriever"
    description = (
        "Search the BSI IT-Grundschutz corpus for relevant passages about requirements, measures, roles, "
        "definitions, and comparisons. Use this tool when you need grounded source excerpts and citation "
        "metadata before answering. Always pass a non-empty `query` argument. Never call this tool with "
        "empty `{}`. Example: {\"query\":\"Passwortschutz\"}."
    )
    icon = "Qdrant"
    name = "AdaptiveQdrantRetriever"

    inputs = [
        StrInput(name="collection_name", display_name="Collection Name", required=True),
        StrInput(
            name="host",
            display_name="Host",
            value="qdrant",
            info="Inside Docker Compose, use the service DNS name `qdrant`. Use `localhost` only when Langflow runs directly on the host.",
            advanced=True,
        ),
        IntInput(name="port", display_name="Port", value=6333, advanced=True),
        IntInput(name="grpc_port", display_name="gRPC Port", value=6334, advanced=True),
        SecretStrInput(name="api_key", display_name="Qdrant API Key", advanced=True),
        StrInput(name="prefix", display_name="Prefix", advanced=True),
        IntInput(name="timeout", display_name="Timeout", advanced=True),
        StrInput(name="path", display_name="Path", advanced=True),
        StrInput(name="url", display_name="URL", advanced=True),
        DropdownInput(
            name="distance_func",
            display_name="Distance Function",
            options=["Cosine", "Euclidean", "Dot Product"],
            value="Cosine",
            advanced=True,
        ),
        StrInput(
            name="content_payload_key",
            display_name="Content Payload Key",
            value="text",
            info="This collection stores chunk text under `text`.",
            advanced=True,
        ),
        StrInput(
            name="metadata_payload_key",
            display_name="Metadata Payload Key",
            value="metadata",
            info="Optional nested metadata key. Flat payload fields are merged automatically.",
            advanced=True,
        ),
        HandleInput(
            name="ingest_data",
            display_name="Ingest Data",
            input_types=["Data", "DataFrame"],
            is_list=True,
        ),
        QueryInput(
            name="query",
            display_name="Query",
            info=(
                "Preferred tool argument. Always pass the user's question or a focused search phrase when the "
                "agent calls this tool."
            ),
            placeholder="e.g. Kapitel 1.3 IT-Grundschutz",
            required=False,
            tool_mode=True,
        ),
        MessageTextInput(
            name="fallback_query",
            display_name="Fallback Query",
            info="Optional fallback question from Chat Input or Agent Input. Used when the tool is triggered without a `query` argument.",
            required=False,
            tool_mode=False,
            advanced=True,
        ),
        QueryInput(
            name="search_query",
            display_name="Search Query",
            info="Backward-compatible alias for direct flows that already use `search_query`.",
            placeholder="e.g. Kapitel 1.3 IT-Grundschutz",
            required=False,
            tool_mode=False,
            advanced=True,
        ),
        BoolInput(
            name="should_cache_vector_store",
            display_name="Cache Vector Store",
            value=True,
            advanced=True,
            info=(
                "If True, the vector store will be cached for the current build of the component. "
                "This is useful for components with multiple output methods."
            ),
        ),
        HandleInput(name="embedding", display_name="Embedding", input_types=["Embeddings"]),
        HandleInput(
            name="judge_llm",
            display_name="Judge LLM",
            input_types=["LanguageModel"],
            info="Optional LLM that decides whether the retrieved snippets are sufficient to answer the question.",
            advanced=True,
        ),
        HandleInput(
            name="rewrite_llm",
            display_name="Rewrite LLM",
            input_types=["LanguageModel"],
            info="Optional LLM used to rewrite the search query after an insufficient retrieval round. Falls back to Judge LLM.",
            advanced=True,
        ),
        HandleInput(
            name="rerank_llm",
            display_name="Rerank LLM",
            input_types=["LanguageModel"],
            info="Optional LLM used to rerank retrieved snippets before judging. Falls back to Judge LLM.",
            advanced=True,
        ),
        StrInput(
            name="top_k_schedule",
            display_name="Top-K Schedule",
            value="3,5,8,10",
            info="Comma-separated top-k values to try in order.",
        ),
        FloatInput(
            name="min_top_score",
            display_name="Min Top Score",
            value=0.35,
            info="If the best hit reaches this score, the attempt can be accepted.",
            advanced=True,
        ),
        FloatInput(
            name="min_token_overlap_ratio",
            display_name="Min Token Overlap Ratio",
            value=0.2,
            info="Minimum query-token overlap ratio to treat a hit as on-topic.",
            advanced=True,
        ),
        IntInput(
            name="min_matching_hits",
            display_name="Min Matching Hits",
            value=1,
            info="Minimum number of on-topic hits needed to accept an attempt.",
            advanced=True,
        ),
        FloatInput(
            name="score_threshold",
            display_name="Score Threshold",
            value=0.0,
            info="Optional Qdrant score threshold. Set to 0 to disable.",
            advanced=True,
        ),
        BoolInput(
            name="return_best_effort_results",
            display_name="Return Best Effort Results",
            value=True,
            info="If False, return no hits when every attempt looks weak.",
            advanced=True,
        ),
        BoolInput(
            name="rewrite_on_insufficient",
            display_name="Rewrite On Insufficient",
            value=True,
            info="If retrieval stays insufficient after a full top-k schedule, rewrite the query and retry.",
            advanced=True,
        ),
        BoolInput(
            name="rerank_with_llm",
            display_name="Rerank With LLM",
            value=False,
            info="Use an LLM to reorder retrieved snippets before sufficiency checks.",
            advanced=True,
        ),
        BoolInput(
            name="use_local_bm25",
            display_name="Use Local BM25",
            value=False,
            info="Build a cached lexical index from the collection payloads and fuse BM25 results with dense retrieval.",
            advanced=True,
        ),
        IntInput(
            name="max_query_rewrites",
            display_name="Max Query Rewrites",
            value=2,
            info="Maximum number of rewritten queries to try after the original query.",
            advanced=True,
        ),
        IntInput(
            name="judge_max_documents",
            display_name="Judge Max Documents",
            value=5,
            info="Maximum number of retrieved snippets passed to the Judge LLM per attempt.",
            advanced=True,
        ),
        IntInput(
            name="judge_max_chars_per_document",
            display_name="Judge Max Chars Per Document",
            value=700,
            info="Maximum number of characters from each snippet passed to the Judge LLM.",
            advanced=True,
        ),
        IntInput(
            name="rewrite_max_documents",
            display_name="Rewrite Max Documents",
            value=5,
            info="Maximum number of retrieved snippets passed to the Rewrite LLM per rewrite round.",
            advanced=True,
        ),
        IntInput(
            name="rewrite_max_chars_per_document",
            display_name="Rewrite Max Chars Per Document",
            value=500,
            info="Maximum number of characters from each snippet passed to the Rewrite LLM.",
            advanced=True,
        ),
        IntInput(
            name="rerank_max_documents",
            display_name="Rerank Max Documents",
            value=8,
            info="Maximum number of retrieved snippets passed to the Rerank LLM per attempt.",
            advanced=True,
        ),
        IntInput(
            name="rerank_max_chars_per_document",
            display_name="Rerank Max Chars Per Document",
            value=500,
            info="Maximum number of characters from each snippet passed to the Rerank LLM.",
            advanced=True,
        ),
        IntInput(
            name="hybrid_candidate_limit",
            display_name="Hybrid Candidate Limit",
            value=12,
            info="Number of dense and BM25 candidates gathered before fusion.",
            advanced=True,
        ),
        IntInput(
            name="rrf_k",
            display_name="RRF K",
            value=60,
            info="Reciprocal Rank Fusion constant used to combine dense and BM25 candidates.",
            advanced=True,
        ),
    ]

    outputs = [
        Output(
            display_name="Search Corpus",
            name="search_corpus",
            method="search_corpus",
            info=(
                "Search the BSI IT-Grundschutz corpus and return structured results with retrieved snippets, "
                "source documents, and retrieval metadata."
            ),
            tool_mode=True,
        ),
        Output(display_name="Search Results", name="search_results", method="search_documents", tool_mode=False),
        Output(display_name="DataFrame", name="dataframe", method="as_dataframe", tool_mode=False),
        Output(display_name="Attempt Debug", name="attempt_debug", method="attempt_debug", tool_mode=False),
    ]

    _cached_search_results: list[Data] | None = None
    _cached_debug_payload: dict[str, Any] | None = None
    _bm25_corpus_cache: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}

    def _resolved_query(self) -> str:
        return str(
            getattr(self, "query", "")
            or getattr(self, "fallback_query", "")
            or getattr(self, "search_query", "")
            or ""
        ).strip()

    def _server_kwargs(self) -> dict[str, Any]:
        url = (self.url or "").strip()
        path = (self.path or "").strip()
        timeout = int(self.timeout) if self.timeout else None

        base_kwargs = {
            "api_key": self.api_key,
            "prefix": self.prefix or None,
            "timeout": timeout,
        }

        # QdrantClient accepts either a URL/path based config or a host/port based config.
        # Mixing both is ambiguous and in Docker often falls back to localhost unexpectedly.
        if url:
            base_kwargs["url"] = url
            return {key: value for key, value in base_kwargs.items() if value is not None}

        if path:
            base_kwargs["path"] = path
            return {key: value for key, value in base_kwargs.items() if value is not None}

        base_kwargs["host"] = self.host or None
        base_kwargs["port"] = int(self.port)
        base_kwargs["grpc_port"] = int(self.grpc_port)
        return {key: value for key, value in base_kwargs.items() if value is not None}

    @check_cached_vector_store
    def build_vector_store(self) -> QdrantClient:
        server_kwargs = self._server_kwargs()

        if not isinstance(self.embedding, Embeddings):
            msg = "Invalid embedding object"
            raise TypeError(msg)

        return QdrantClient(**server_kwargs)

    def _schedule(self) -> list[int]:
        values: list[int] = []
        for part in str(self.top_k_schedule or "").split(","):
            text = part.strip()
            if not text:
                continue
            if text.isdigit():
                value = int(text)
                if value > 0 and value not in values:
                    values.append(value)
        return values or [3, 5, 8, 10]

    def _query_terms(self, text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"\w+", (text or "").lower(), flags=re.UNICODE)
            if len(token) >= 4 and token not in _STOPWORDS and not token.isdigit()
        }

    def _lexical_tokens(self, text: str) -> list[str]:
        return [
            token
            for token in re.findall(r"\w+", (text or "").lower(), flags=re.UNICODE)
            if len(token) >= 2 and token not in _STOPWORDS and not token.isdigit()
        ]

    def _doc_terms(self, page_content: str, metadata: dict[str, Any]) -> set[str]:
        joined = " ".join(
            str(value)
            for value in [
                page_content,
                metadata.get("file"),
                metadata.get("section_title"),
                metadata.get("title"),
                metadata.get("document"),
                metadata.get("source"),
            ]
            if value
        )
        return self._query_terms(joined)

    def _split_payload(self, payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        text_key = (self.content_payload_key or "text").strip() or "text"
        metadata_key = (self.metadata_payload_key or "").strip()

        page_content = payload.get(text_key)
        if page_content is None and text_key != "text":
            page_content = payload.get("text")

        metadata: dict[str, Any] = {}
        nested_metadata = payload.get(metadata_key) if metadata_key else None
        if isinstance(nested_metadata, dict):
            metadata.update(nested_metadata)

        for key, value in payload.items():
            if key == text_key or (metadata_key and key == metadata_key):
                continue
            metadata.setdefault(key, value)

        if isinstance(page_content, str):
            cleaned_content = page_content.strip()
        else:
            cleaned_content = ""

        return cleaned_content, metadata

    def _to_data(
        self,
        page_content: str,
        metadata: dict[str, Any],
        score: float,
        overlap_ratio: float,
        *,
        point_id: Any = None,
        retrieval_source: str = "dense",
    ) -> Data:
        payload = dict(metadata)
        payload["text"] = page_content
        payload["retrieval_score"] = float(score)
        payload["token_overlap_ratio"] = float(overlap_ratio)
        payload["retrieval_sources"] = [retrieval_source]
        if point_id is not None:
            payload["_retrieval_id"] = str(point_id)
        if retrieval_source == "dense":
            payload["dense_retrieval_score"] = float(score)
        elif retrieval_source == "bm25":
            payload["bm25_score"] = float(score)
        return Data(data=payload, text_key="text")

    def _result_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for item in self.search_documents():
            if isinstance(item, Data) and isinstance(item.data, dict):
                rows.append(dict(item.data))
            elif isinstance(item, dict):
                rows.append(dict(item))
        return rows

    def _truncate_text(self, value: str, max_chars: int) -> str:
        text = " ".join((value or "").split()).strip()
        if max_chars <= 0 or len(text) <= max_chars:
            return text
        return text[: max_chars - 3].rstrip() + "..."

    def _clean_text(self, value: str) -> str:
        return " ".join((value or "").split()).strip()

    def _metadata_from_row(self, row: dict[str, Any]) -> dict[str, Any]:
        metadata = row.get("metadata")
        return metadata if isinstance(metadata, dict) else {}

    def _coerce_int(self, value: Any) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            match = re.search(r"\d+", value.strip())
            if match:
                return int(match.group(0))
        return None

    def _resolve_text(self, row: dict[str, Any]) -> str:
        for key in ("text", "page_content", "content", "chunk", "body"):
            text = self._clean_text(str(row.get(key) or ""))
            if text:
                return text
        metadata = self._metadata_from_row(row)
        for key in ("text", "page_content", "content"):
            text = self._clean_text(str(metadata.get(key) or ""))
            if text:
                return text
        return ""

    def _resolve_file(self, row: dict[str, Any]) -> str:
        metadata = self._metadata_from_row(row)
        for container in (row, metadata):
            for key in ("file", "document", "title", "source"):
                value = self._clean_text(str(container.get(key) or ""))
                if value:
                    return value
        source = row.get("source")
        if isinstance(source, dict):
            for key in ("file", "document", "title", "source"):
                value = self._clean_text(str(source.get(key) or ""))
                if value:
                    return value
        return ""

    def _resolve_page(self, row: dict[str, Any]) -> int | None:
        metadata = self._metadata_from_row(row)
        for container in (row, metadata):
            for key in ("page_start", "page", "page_number"):
                value = self._coerce_int(container.get(key))
                if value is not None:
                    return value
        return None

    def _resolve_page_end(self, row: dict[str, Any]) -> int | None:
        metadata = self._metadata_from_row(row)
        for container in (row, metadata):
            value = self._coerce_int(container.get("page_end"))
            if value is not None:
                return value
        return None

    def _resolve_section(self, row: dict[str, Any]) -> str:
        metadata = self._metadata_from_row(row)
        for container in (row, metadata):
            for key in ("section_title", "title"):
                value = self._clean_text(str(container.get(key) or ""))
                if value:
                    return value
        return ""

    def _source_documents(self) -> list[dict[str, Any]]:
        docs: list[dict[str, Any]] = []
        for row in self._result_rows():
            page_content = self._resolve_text(row)
            if not page_content:
                continue

            metadata: dict[str, Any] = {}
            file_name = self._resolve_file(row)
            page = self._resolve_page(row)
            page_end = self._resolve_page_end(row)
            section_title = self._resolve_section(row)

            if file_name:
                metadata["file"] = file_name
            if page is not None:
                metadata["page"] = page
            if page_end is not None:
                metadata["page_end"] = page_end
            if section_title:
                metadata["section_title"] = section_title

            docs.append(
                {
                    "page_content": page_content,
                    "metadata": metadata,
                }
            )
        return docs

    def _build_context_string(self) -> str:
        blocks: list[str] = []
        for index, doc in enumerate(self._source_documents(), start=1):
            metadata = doc.get("metadata") or {}
            lines = [f"[Quelle {index}]"]
            file_name = (metadata.get("file") or "").strip()
            if file_name:
                lines.append(f"file: {file_name}")
            page = metadata.get("page")
            if page is not None:
                lines.append(f"page: {page}")
            page_end = metadata.get("page_end")
            if page_end is not None:
                lines.append(f"page_end: {page_end}")
            section_title = (metadata.get("section_title") or "").strip()
            if section_title:
                lines.append(f"section_title: {section_title}")
            lines.append(f"page_content: {doc.get('page_content', '')}")
            blocks.append("\n".join(lines))
        return "\n\n---\n\n".join(blocks)

    def _normalize_query(self, value: str) -> str:
        return " ".join((value or "").lower().split()).strip()

    def _comparison_terms(self, text: str) -> tuple[set[str], set[str]]:
        normalized = self._normalize_query(text).replace("-", " ")
        if not normalized:
            return set(), set()

        query_terms = self._query_terms(normalized)
        if not (_COMPARISON_MARKERS & query_terms):
            return set(), set()

        patterns = (
            r"\bzwischen\s+(.+?)\s+und\s+(.+?)(?:[?!.]|$)",
            r"\b(.+?)\s+(?:versus|vs\.?)\s+(.+?)(?:[?!.]|$)",
        )

        for pattern in patterns:
            match = re.search(pattern, normalized, flags=re.IGNORECASE)
            if not match:
                continue

            left_terms = self._query_terms(match.group(1))
            right_terms = self._query_terms(match.group(2))
            if left_terms and right_terms:
                return left_terms, right_terms

        return set(), set()

    def _bm25_cache_key(self) -> tuple[str, str, str, str, str]:
        return (
            str((self.url or "").strip() or (self.host or "").strip() or ""),
            str(int(self.port or 6333)),
            str(self.collection_name or "").strip(),
            str(self.content_payload_key or "text").strip() or "text",
            str(self.metadata_payload_key or "").strip(),
        )

    def _bm25_document_text(self, page_content: str, metadata: dict[str, Any]) -> str:
        return " ".join(
            str(value)
            for value in [
                page_content,
                metadata.get("section_title"),
                metadata.get("title"),
                metadata.get("file"),
                metadata.get("document"),
                metadata.get("source"),
            ]
            if value
        )

    def _scroll_points(self, vector_store: QdrantClient, limit: int = 256) -> list[Any]:
        if not hasattr(vector_store, "scroll"):
            msg = "QdrantClient does not support scroll() in this runtime."
            raise AttributeError(msg)

        points: list[Any] = []
        offset: Any = None
        seen_offsets: set[str] = set()

        while True:
            response = vector_store.scroll(
                collection_name=self.collection_name,
                limit=limit,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            if isinstance(response, tuple):
                batch, next_offset = response
            else:
                batch = getattr(response, "points", None) or getattr(response, "result", None) or []
                next_offset = getattr(response, "next_page_offset", None)
                if next_offset is None:
                    next_offset = getattr(response, "next_offset", None)

            batch_list = list(batch or [])
            if not batch_list:
                break

            points.extend(batch_list)
            if next_offset is None:
                break

            offset_key = repr(next_offset)
            if offset_key in seen_offsets:
                break
            seen_offsets.add(offset_key)
            offset = next_offset

        return points

    def _load_bm25_corpus(self, vector_store: QdrantClient) -> dict[str, Any]:
        cache_key = self._bm25_cache_key()
        cached = self._bm25_corpus_cache.get(cache_key)
        if cached is not None:
            return cached

        docs: list[dict[str, Any]] = []
        doc_freqs: Counter[str] = Counter()
        total_doc_len = 0

        for index, point in enumerate(self._scroll_points(vector_store), start=1):
            raw_payload = dict(getattr(point, "payload", {}) or {})
            page_content, metadata = self._split_payload(raw_payload)
            if not page_content:
                continue

            doc_text = self._bm25_document_text(page_content, metadata)
            tokens = self._lexical_tokens(doc_text)
            if not tokens:
                continue

            term_freqs = Counter(tokens)
            doc_freqs.update(term_freqs.keys())
            total_doc_len += len(tokens)
            docs.append(
                {
                    "point_id": str(getattr(point, "id", None) or raw_payload.get("_id") or index),
                    "page_content": page_content,
                    "metadata": metadata,
                    "term_freqs": dict(term_freqs),
                    "doc_len": len(tokens),
                }
            )

        total_docs = len(docs)
        corpus = {
            "docs": docs,
            "doc_freqs": dict(doc_freqs),
            "total_docs": total_docs,
            "avg_doc_len": (total_doc_len / total_docs) if total_docs else 0.0,
        }
        self._bm25_corpus_cache[cache_key] = corpus
        return corpus

    def _bm25_results(self, vector_store: QdrantClient, query: str, limit: int) -> list[Data]:
        corpus = self._load_bm25_corpus(vector_store)
        docs = list(corpus.get("docs") or [])
        if not docs:
            return []

        query_tokens = self._lexical_tokens(query)
        if not query_tokens:
            return []

        doc_freqs = dict(corpus.get("doc_freqs") or {})
        total_docs = int(corpus.get("total_docs") or 0)
        avg_doc_len = float(corpus.get("avg_doc_len") or 0.0) or 1.0
        if total_docs <= 0:
            return []

        k1 = 1.5
        b = 0.75
        query_freqs = Counter(query_tokens)
        scored_docs: list[tuple[float, dict[str, Any]]] = []

        for doc in docs:
            term_freqs = dict(doc.get("term_freqs") or {})
            doc_len = max(1, int(doc.get("doc_len") or 0))
            score = 0.0

            for token, query_weight in query_freqs.items():
                tf = int(term_freqs.get(token) or 0)
                df = int(doc_freqs.get(token) or 0)
                if tf <= 0 or df <= 0:
                    continue

                idf = math.log(1.0 + (total_docs - df + 0.5) / (df + 0.5))
                denominator = tf + k1 * (1.0 - b + b * (doc_len / avg_doc_len))
                score += query_weight * idf * ((tf * (k1 + 1.0)) / denominator)

            if score > 0.0:
                scored_docs.append((score, doc))

        scored_docs.sort(key=lambda item: item[0], reverse=True)
        results: list[Data] = []
        for score, doc in scored_docs[: max(1, limit)]:
            results.append(
                self._to_data(
                    str(doc.get("page_content") or ""),
                    dict(doc.get("metadata") or {}),
                    score,
                    0.0,
                    point_id=doc.get("point_id"),
                    retrieval_source="bm25",
                )
            )
        return results

    def _result_identifier(self, result: Data, fallback: str) -> str:
        row = dict(result.data or {})
        for key in ("_retrieval_id", "_id", "id"):
            value = row.get(key)
            if value not in (None, ""):
                return str(value)

        fallback_parts = [
            self._clean_text(str(row.get("file") or "")),
            str(row.get("page_start") or row.get("page") or ""),
            self._clean_text(str(row.get("section_title") or row.get("title") or "")),
            self._truncate_text(str(row.get("text") or ""), 120),
        ]
        derived = "|".join(part for part in fallback_parts if part)
        return derived or fallback

    def _merge_result_payload(self, target: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
        merged = dict(target)

        for key, value in source.items():
            if key == "retrieval_sources":
                existing_sources = set(merged.get("retrieval_sources") or [])
                existing_sources.update(value or [])
                merged["retrieval_sources"] = sorted(existing_sources)
                continue

            if key in {"dense_retrieval_score", "bm25_score", "rrf_score"}:
                if value is not None:
                    merged[key] = float(value)
                continue

            if key in {"retrieval_rank_dense", "retrieval_rank_bm25"}:
                if value is not None:
                    merged[key] = value
                continue

            if key not in merged or merged.get(key) in (None, "", []):
                merged[key] = value

        return merged

    def _fuse_dense_and_bm25(
        self,
        dense_results: list[Data],
        bm25_results: list[Data],
        top_k: int,
    ) -> tuple[list[Data], dict[str, Any]]:
        bm25_enabled = bool(getattr(self, "use_local_bm25", False))
        if not bm25_enabled:
            return dense_results[:top_k], {"enabled": False, "used": False, "reason": "bm25_disabled"}

        if not bm25_results:
            return dense_results[:top_k], {
                "enabled": True,
                "used": False,
                "reason": "no_bm25_hits",
                "dense_candidates": len(dense_results),
                "bm25_candidates": 0,
            }

        rrf_k = max(1, int(self.rrf_k or 60))
        combined: dict[str, dict[str, Any]] = {}

        def register(result: Data, source: str, rank: int) -> None:
            key = self._result_identifier(result, f"{source}-{rank}")
            entry = combined.get(key)
            source_payload = dict(result.data or {})
            if entry is None:
                entry = {
                    "result": result,
                    "rrf_score": 0.0,
                    "dense_rank": None,
                    "bm25_rank": None,
                }
                combined[key] = entry
            else:
                merged_payload = self._merge_result_payload(dict(entry["result"].data or {}), source_payload)
                entry["result"] = Data(data=merged_payload, text_key="text")

            entry["rrf_score"] += 1.0 / (rrf_k + rank)
            if source == "dense":
                entry["dense_rank"] = rank
            elif source == "bm25":
                entry["bm25_rank"] = rank

        for rank, result in enumerate(dense_results, start=1):
            register(result, "dense", rank)
        for rank, result in enumerate(bm25_results, start=1):
            register(result, "bm25", rank)

        ordered_entries = sorted(combined.values(), key=lambda item: item["rrf_score"], reverse=True)
        fused_results: list[Data] = []
        for entry in ordered_entries[:top_k]:
            payload = dict(entry["result"].data or {})
            payload["rrf_score"] = float(entry["rrf_score"])
            payload["retrieval_rank_dense"] = entry["dense_rank"]
            payload["retrieval_rank_bm25"] = entry["bm25_rank"]
            payload["retrieval_sources"] = sorted(set(payload.get("retrieval_sources") or []))

            if payload.get("dense_retrieval_score") is not None:
                payload["retrieval_score"] = float(payload["dense_retrieval_score"])
            elif payload.get("bm25_score") is not None:
                payload["retrieval_score"] = float(payload["bm25_score"])

            fused_results.append(Data(data=payload, text_key="text"))

        return fused_results, {
            "enabled": True,
            "used": True,
            "reason": "rrf_fusion",
            "dense_candidates": len(dense_results),
            "bm25_candidates": len(bm25_results),
            "rrf_k": rrf_k,
        }

    def _llm_response_text(self, value: Any) -> str:
        if isinstance(value, str):
            return value.strip()

        if isinstance(value, BaseMessage):
            content = value.content
        else:
            content = getattr(value, "content", value)

        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item.strip())
                    continue
                if isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                    if isinstance(text, str):
                        parts.append(text.strip())
            return "\n".join(part for part in parts if part).strip()

        return str(content).strip()

    def _judge_prompt(self, query: str, results: list[Data], top_k: int) -> str:
        max_docs = max(1, int(self.judge_max_documents or 5))
        max_chars = max(120, int(self.judge_max_chars_per_document or 700))
        blocks: list[str] = []

        for index, result in enumerate(results[:max_docs], start=1):
            row = dict(result.data or {})
            file_name = row.get("file") or row.get("source") or ""
            section_title = row.get("section_title") or row.get("title") or ""
            page_start = row.get("page_start") or row.get("page") or ""
            page_end = row.get("page_end") or ""
            score = row.get("retrieval_score")
            snippet = self._truncate_text(str(row.get("text") or ""), max_chars)

            header_parts = [f"[Snippet {index}]"]
            if score is not None:
                try:
                    header_parts.append(f"score={float(score):.3f}")
                except (TypeError, ValueError):
                    pass

            meta_lines = []
            if file_name:
                meta_lines.append(f"file: {file_name}")
            if section_title:
                meta_lines.append(f"section_title: {section_title}")
            if page_start != "":
                meta_lines.append(f"page_start: {page_start}")
            if page_end != "":
                meta_lines.append(f"page_end: {page_end}")

            block_lines = [" ".join(header_parts)]
            block_lines.extend(meta_lines)
            block_lines.append(f"text: {snippet}")
            blocks.append("\n".join(block_lines))

        snippets_text = "\n\n".join(blocks)
        return (
            "You are judging retrieval quality for a RAG system.\n"
            "Decide whether the retrieved snippets are sufficient to answer the user question accurately and specifically.\n"
            "Mark INSUFFICIENT if the snippets are only loosely related, discuss another topic, or miss the requested requirement.\n"
            "Return JSON only in this exact shape: "
            '{"decision":"SUFFICIENT"|"INSUFFICIENT","reason":"short reason"}'
            "\n\n"
            f"Question:\n{query}\n\n"
            f"Attempt top_k: {top_k}\n\n"
            f"Retrieved snippets:\n{snippets_text}\n"
        )

    def _parse_judge_response(self, text: str) -> tuple[bool | None, str]:
        cleaned = (text or "").strip()
        if not cleaned:
            return None, ""

        decision_match = re.search(r'"decision"\s*:\s*"(SUFFICIENT|INSUFFICIENT)"', cleaned, flags=re.IGNORECASE)
        if decision_match:
            decision = decision_match.group(1).upper() == "SUFFICIENT"
        elif re.search(r"\bINSUFFICIENT\b", cleaned, flags=re.IGNORECASE):
            decision = False
        elif re.search(r"\bSUFFICIENT\b", cleaned, flags=re.IGNORECASE):
            decision = True
        else:
            decision = None

        reason_match = re.search(r'"reason"\s*:\s*"([^"]+)"', cleaned, flags=re.IGNORECASE)
        if reason_match:
            reason = reason_match.group(1).strip()
        else:
            lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
            if lines and re.fullmatch(r"(SUFFICIENT|INSUFFICIENT)", lines[0], flags=re.IGNORECASE):
                lines = lines[1:]
            reason = lines[0] if lines else ""

        return decision, self._truncate_text(reason, 240)

    def _rerank_prompt(self, query: str, results: list[Data]) -> str:
        max_docs = max(2, int(self.rerank_max_documents or 8))
        max_chars = max(120, int(self.rerank_max_chars_per_document or 500))
        blocks: list[str] = []

        for index, result in enumerate(results[:max_docs], start=1):
            row = dict(result.data or {})
            file_name = row.get("file") or row.get("source") or ""
            section_title = row.get("section_title") or row.get("title") or ""
            page_start = row.get("page_start") or row.get("page") or ""
            score = row.get("retrieval_score")
            snippet = self._truncate_text(str(row.get("text") or ""), max_chars)

            header_parts = [f"[Snippet {index}]"]
            if score is not None:
                try:
                    header_parts.append(f"dense_score={float(score):.3f}")
                except (TypeError, ValueError):
                    pass

            block_lines = [" ".join(header_parts)]
            if file_name:
                block_lines.append(f"file: {file_name}")
            if section_title:
                block_lines.append(f"section_title: {section_title}")
            if page_start != "":
                block_lines.append(f"page_start: {page_start}")
            block_lines.append(f"text: {snippet}")
            blocks.append("\n".join(block_lines))

        snippets_text = "\n\n".join(blocks)
        return (
            "You rerank retrieval candidates for a RAG system.\n"
            "Order the snippets from most useful to least useful for answering the user question.\n"
            "Prefer direct definitions, explicit comparisons, and snippets that answer the exact question.\n"
            "Penalize adjacent topics and snippets that only mention one side of a comparison.\n"
            'Return JSON only in this exact shape: {"ranking":[1,2,3],"reason":"short reason"}'
            "\n\n"
            f"Question:\n{query}\n\n"
            f"Retrieved snippets:\n{snippets_text}\n"
        )

    def _parse_rerank_response(self, text: str, total_docs: int) -> tuple[list[int], str]:
        cleaned = (text or "").strip()
        if not cleaned:
            return [], ""

        ranking: list[int] = []
        reason = ""
        json_match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if json_match:
            try:
                payload = json.loads(json_match.group(0))
            except json.JSONDecodeError:
                payload = {}
            if isinstance(payload, dict):
                raw_ranking = payload.get("ranking")
                raw_reason = payload.get("reason")
                if isinstance(raw_ranking, list):
                    seen: set[int] = set()
                    for item in raw_ranking:
                        try:
                            index = int(item)
                        except (TypeError, ValueError):
                            continue
                        if 1 <= index <= total_docs and index not in seen:
                            ranking.append(index)
                            seen.add(index)
                if isinstance(raw_reason, str):
                    reason = raw_reason.strip()

        if not ranking:
            match = re.search(r'"ranking"\s*:\s*\[([^\]]+)\]', cleaned, flags=re.IGNORECASE)
            if match:
                seen = set()
                for item in re.findall(r"\d+", match.group(1)):
                    index = int(item)
                    if 1 <= index <= total_docs and index not in seen:
                        ranking.append(index)
                        seen.add(index)

        if not reason:
            reason_match = re.search(r'"reason"\s*:\s*"([^"]+)"', cleaned, flags=re.IGNORECASE)
            if reason_match:
                reason = reason_match.group(1).strip()

        return ranking, self._truncate_text(reason, 240)

    def _judge_attempt(self, query: str, results: list[Data], top_k: int) -> tuple[bool | None, str, str]:
        judge_llm = getattr(self, "judge_llm", None)
        if judge_llm is None or not results:
            return None, "", ""

        try:
            response = judge_llm.invoke(
                self._judge_prompt(query, results, top_k),
                config={"callbacks": self.get_langchain_callbacks()},
            )
        except Exception as exc:  # noqa: BLE001
            message = f"Judge LLM error: {exc!s}"
            return None, self._truncate_text(message, 240), message

        raw_text = self._llm_response_text(response)
        decision, reason = self._parse_judge_response(raw_text)
        return decision, reason, raw_text

    def _rerank_results(self, query: str, results: list[Data]) -> tuple[list[Data], dict[str, Any]]:
        rerank_enabled = bool(getattr(self, "rerank_with_llm", False))
        rerank_llm = getattr(self, "rerank_llm", None) or getattr(self, "judge_llm", None)
        if not rerank_enabled:
            return results, {"enabled": False, "used": False, "reason": "rerank_disabled"}
        if rerank_llm is None:
            return results, {"enabled": True, "used": False, "reason": "no_rerank_llm"}
        if len(results) < 2:
            return results, {"enabled": True, "used": False, "reason": "not_enough_results"}

        max_docs = max(2, int(self.rerank_max_documents or 8))
        subset = list(results[:max_docs])
        tail = list(results[max_docs:])

        try:
            response = rerank_llm.invoke(
                self._rerank_prompt(query, subset),
                config={"callbacks": self.get_langchain_callbacks()},
            )
        except Exception as exc:  # noqa: BLE001
            message = f"Rerank LLM error: {exc!s}"
            return results, {
                "enabled": True,
                "used": True,
                "reason": "rerank_error",
                "raw_response": message,
            }

        raw_text = self._llm_response_text(response)
        ranking, reason = self._parse_rerank_response(raw_text, len(subset))
        if not ranking:
            return results, {
                "enabled": True,
                "used": True,
                "reason": "rerank_unusable",
                "raw_response": self._truncate_text(raw_text, 400),
            }

        ordered_subset: list[Data] = []
        seen_positions: set[int] = set()
        for index in ranking:
            position = index - 1
            if position in seen_positions:
                continue
            ordered_subset.append(subset[position])
            seen_positions.add(position)
        for position, result in enumerate(subset):
            if position not in seen_positions:
                ordered_subset.append(result)

        reranked_results = ordered_subset + tail
        for position, result in enumerate(reranked_results, start=1):
            if isinstance(result.data, dict):
                result.data["llm_rerank_position"] = position

        return reranked_results, {
            "enabled": True,
            "used": True,
            "reason": "reranked",
            "ranking": ranking,
            "rerank_reason": reason,
            "raw_response": self._truncate_text(raw_text, 400),
        }

    def _rewrite_prompt(
        self,
        original_query: str,
        current_query: str,
        attempts: list[dict[str, Any]],
        results: list[Data],
    ) -> str:
        max_docs = max(1, int(self.rewrite_max_documents or 5))
        max_chars = max(120, int(self.rewrite_max_chars_per_document or 500))
        last_attempt = attempts[-1] if attempts else {}
        blocks: list[str] = []

        for index, result in enumerate(results[:max_docs], start=1):
            row = dict(result.data or {})
            file_name = row.get("file") or row.get("source") or ""
            section_title = row.get("section_title") or row.get("title") or ""
            score = row.get("retrieval_score")
            snippet = self._truncate_text(str(row.get("text") or ""), max_chars)

            header_parts = [f"[Snippet {index}]"]
            if score is not None:
                try:
                    header_parts.append(f"score={float(score):.3f}")
                except (TypeError, ValueError):
                    pass

            block_lines = [" ".join(header_parts)]
            if file_name:
                block_lines.append(f"file: {file_name}")
            if section_title:
                block_lines.append(f"section_title: {section_title}")
            block_lines.append(f"text: {snippet}")
            blocks.append("\n".join(block_lines))

        snippets_text = "\n\n".join(blocks) if blocks else "No usable snippets were retrieved."
        judge_reason = self._truncate_text(str(last_attempt.get("judge_reason") or ""), 240)
        return (
            "You rewrite search queries for semantic retrieval over German BSI IT-Grundschutz and related security documents.\n"
            "Keep the original user intent, but rewrite the query so it better matches likely document wording.\n"
            "Prefer concise German noun phrases and canonical technical terms.\n"
            "Do not answer the question.\n"
            'Return JSON only in this exact shape: {"query":"rewritten search query"}'
            "\n\n"
            f"Original user question:\n{original_query}\n\n"
            f"Current failed search query:\n{current_query}\n\n"
            f"Last judge reason:\n{judge_reason or 'No judge reason available.'}\n\n"
            f"Retrieved snippets:\n{snippets_text}\n"
        )

    def _parse_rewrite_response(self, text: str, current_query: str) -> str:
        cleaned = (text or "").strip()
        if not cleaned:
            return ""

        match = re.search(r'"query"\s*:\s*"([^"]+)"', cleaned, flags=re.IGNORECASE)
        if match:
            candidate = match.group(1).strip()
        else:
            lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
            candidate = lines[0] if lines else ""

        candidate = candidate.strip().strip("`").strip("\"'")
        candidate = self._truncate_text(candidate, 300)
        if not candidate:
            return ""
        if self._normalize_query(candidate) == self._normalize_query(current_query):
            return ""
        return candidate

    def _rewrite_query(
        self,
        original_query: str,
        current_query: str,
        attempts: list[dict[str, Any]],
        results: list[Data],
    ) -> tuple[str, dict[str, Any]]:
        rewrite_enabled = bool(self.rewrite_on_insufficient)
        rewrite_llm = getattr(self, "rewrite_llm", None) or getattr(self, "judge_llm", None)
        if not rewrite_enabled:
            return "", {"enabled": False, "used": False, "reason": "rewrite_disabled"}
        if rewrite_llm is None:
            return "", {"enabled": True, "used": False, "reason": "no_rewrite_llm"}

        try:
            response = rewrite_llm.invoke(
                self._rewrite_prompt(original_query, current_query, attempts, results),
                config={"callbacks": self.get_langchain_callbacks()},
            )
        except Exception as exc:  # noqa: BLE001
            message = f"Rewrite LLM error: {exc!s}"
            return "", {
                "enabled": True,
                "used": True,
                "reason": "rewrite_error",
                "raw_response": message,
            }

        raw_text = self._llm_response_text(response)
        rewritten_query = self._parse_rewrite_response(raw_text, current_query)
        return rewritten_query, {
            "enabled": True,
            "used": True,
            "reason": "rewritten" if rewritten_query else "rewrite_unusable",
            "raw_response": self._truncate_text(raw_text, 400),
            "rewritten_query": rewritten_query,
        }

    def _query_scored_points(
        self,
        vector_store: QdrantClient,
        query_vector: list[float],
        top_k: int,
        score_threshold: float | None,
    ) -> list[Any]:
        common_kwargs = {
            "collection_name": self.collection_name,
            "limit": top_k,
            "with_payload": True,
            "with_vectors": False,
            "score_threshold": score_threshold,
        }

        if hasattr(vector_store, "query_points"):
            response = vector_store.query_points(
                query=query_vector,
                **common_kwargs,
            )
            return list(getattr(response, "points", []) or [])

        if hasattr(vector_store, "search"):
            return list(
                vector_store.search(
                    query_vector=query_vector,
                    **common_kwargs,
                )
            )

        msg = "QdrantClient does not support query_points() or search() in this runtime."
        raise AttributeError(msg)

    def _search_attempt(
        self,
        vector_store: QdrantClient,
        original_query: str,
        search_query: str,
        query_vector: list[float],
        top_k: int,
    ) -> tuple[list[Data], dict[str, Any]]:
        score_threshold = float(self.score_threshold or 0.0) or None
        bm25_enabled = bool(getattr(self, "use_local_bm25", False))
        candidate_limit = max(top_k, int(self.hybrid_candidate_limit or top_k)) if bm25_enabled else top_k
        points = self._query_scored_points(vector_store, query_vector, candidate_limit, score_threshold)

        dense_results: list[Data] = []
        dense_scores: list[float] = []

        for point in points:
            raw_payload = dict(getattr(point, "payload", {}) or {})
            page_content, metadata = self._split_payload(raw_payload)
            if not page_content:
                continue

            score = float(getattr(point, "score", 0.0) or 0.0)
            dense_scores.append(score)
            dense_results.append(
                self._to_data(
                    page_content,
                    metadata,
                    score,
                    0.0,
                    point_id=getattr(point, "id", None),
                    retrieval_source="dense",
                )
            )

        bm25_results = self._bm25_results(vector_store, search_query, candidate_limit) if bm25_enabled else []
        results, fusion_info = self._fuse_dense_and_bm25(dense_results, bm25_results, top_k)

        query_terms = self._query_terms(original_query)
        comparison_left_terms, comparison_right_terms = self._comparison_terms(original_query)
        comparison_query = bool(comparison_left_terms and comparison_right_terms)
        overlaps: list[float] = []
        matching_hits = 0
        comparison_left_covered = False
        comparison_right_covered = False

        for result in results:
            row = dict(result.data or {})
            page_content = str(row.get("text") or "")
            metadata = {
                key: value
                for key, value in row.items()
                if key
                not in {
                    "text",
                    "retrieval_score",
                    "dense_retrieval_score",
                    "bm25_score",
                    "token_overlap_ratio",
                    "_retrieval_id",
                    "retrieval_sources",
                    "rrf_score",
                    "retrieval_rank_dense",
                    "retrieval_rank_bm25",
                }
            }

            doc_terms = self._doc_terms(page_content, metadata)
            overlap_ratio = (len(query_terms & doc_terms) / len(query_terms)) if query_terms else 0.0
            if overlap_ratio >= float(self.min_token_overlap_ratio or 0.0):
                matching_hits += 1
            if comparison_query:
                comparison_left_covered = comparison_left_covered or bool(comparison_left_terms & doc_terms)
                comparison_right_covered = comparison_right_covered or bool(comparison_right_terms & doc_terms)
            row["token_overlap_ratio"] = float(overlap_ratio)
            result.data = row
            overlaps.append(float(overlap_ratio))

        top_score = max(dense_scores) if dense_scores else 0.0
        avg_score = mean(dense_scores) if dense_scores else 0.0
        best_overlap = max(overlaps) if overlaps else 0.0
        bm25_scores = [
            float((result.data or {}).get("bm25_score") or 0.0)
            for result in bm25_results
            if isinstance(result.data, dict)
        ]
        bm25_top_score = max(bm25_scores) if bm25_scores else 0.0
        fused_scores = [
            float((result.data or {}).get("rrf_score") or (result.data or {}).get("retrieval_score") or 0.0)
            for result in results
            if isinstance(result.data, dict)
        ]
        fused_top_score = max(fused_scores) if fused_scores else 0.0

        min_overlap_ratio = float(self.min_token_overlap_ratio or 0.0)
        min_hits = int(self.min_matching_hits or 1)
        score_gate_passed = top_score >= float(self.min_top_score or 0.0)
        overlap_gate_passed = matching_hits >= min_hits and best_overlap >= min_overlap_ratio
        comparison_gate_passed = (
            (comparison_left_covered and comparison_right_covered) if comparison_query else True
        )
        retrieval_gate_passed = score_gate_passed or (bm25_enabled and bool(bm25_results))
        results, rerank_info = self._rerank_results(original_query, results)

        if query_terms:
            heuristic_sufficient = retrieval_gate_passed and overlap_gate_passed and comparison_gate_passed
        else:
            heuristic_sufficient = retrieval_gate_passed and comparison_gate_passed

        judge_configured = getattr(self, "judge_llm", None) is not None
        judge_decision, judge_reason, judge_raw_response = self._judge_attempt(original_query, results, top_k)
        if judge_decision is None:
            sufficient = bool(results) and heuristic_sufficient
            decision_source = "heuristic_fallback" if judge_raw_response else "heuristic"
        else:
            sufficient = bool(results) and judge_decision
            decision_source = "judge_llm"

        attempt = {
            "query": original_query,
            "search_query": search_query,
            "top_k": top_k,
            "hits": len(results),
            "top_score": round(top_score, 6),
            "avg_score": round(avg_score, 6),
            "fused_top_score": round(fused_top_score, 6),
            "bm25_top_score": round(bm25_top_score, 6),
            "best_overlap_ratio": round(best_overlap, 6),
            "matching_hits": matching_hits,
            "bm25_enabled": bm25_enabled,
            "bm25_candidate_hits": len(bm25_results),
            "dense_candidate_hits": len(dense_results),
            "score_gate_passed": score_gate_passed,
            "retrieval_gate_passed": retrieval_gate_passed,
            "overlap_gate_passed": overlap_gate_passed,
            "comparison_query": comparison_query,
            "comparison_left_terms": sorted(comparison_left_terms),
            "comparison_right_terms": sorted(comparison_right_terms),
            "comparison_gate_passed": comparison_gate_passed,
            "fusion_enabled": fusion_info.get("enabled", False),
            "fusion_used": fusion_info.get("used", False),
            "fusion_status": fusion_info.get("reason", ""),
            "fusion_dense_candidates": fusion_info.get("dense_candidates", len(dense_results)),
            "fusion_bm25_candidates": fusion_info.get("bm25_candidates", len(bm25_results)),
            "fusion_rrf_k": fusion_info.get("rrf_k"),
            "rerank_enabled": rerank_info.get("enabled", False),
            "rerank_used": rerank_info.get("used", False),
            "rerank_status": rerank_info.get("reason", ""),
            "rerank_reason": rerank_info.get("rerank_reason", ""),
            "rerank_ranking": rerank_info.get("ranking", []),
            "rerank_raw_response": rerank_info.get("raw_response", ""),
            "heuristic_sufficient": bool(results) and heuristic_sufficient,
            "query_terms": sorted(query_terms),
            "judge_used": judge_configured and bool(results),
            "judge_decision": (
                "SUFFICIENT"
                if judge_decision is True
                else "INSUFFICIENT"
                if judge_decision is False
                else "UNPARSEABLE"
                if judge_raw_response
                else "NOT_USED"
            ),
            "judge_reason": judge_reason,
            "judge_raw_response": self._truncate_text(judge_raw_response, 400) if judge_raw_response else "",
            "decision_source": decision_source,
            "sufficient": sufficient,
        }
        return results, attempt

    def _run_plan(self) -> tuple[list[Data], dict[str, Any]]:
        if self._cached_search_results is not None and self._cached_debug_payload is not None:
            return self._cached_search_results, self._cached_debug_payload

        query = self._resolved_query()
        if not query:
            msg = (
                "Missing search query. Call `search_corpus` with a non-empty JSON argument like "
                '{"query":"Passwortschutz"} or connect Chat Input/Agent Input to `fallback_query`.'
            )
            raise ValueError(msg)

        vector_store = self.build_vector_store()
        attempts: list[dict[str, Any]] = []
        rounds: list[dict[str, Any]] = []
        rewrites: list[dict[str, Any]] = []
        selected_results: list[Data] = []
        selected_top_k: int | None = None
        selected_query = query
        selected_sufficient = False
        last_results: list[Data] = []
        stop_reason = "max_retries_exhausted"
        current_query = query
        seen_queries = {self._normalize_query(query)}
        max_rewrites = max(0, int(self.max_query_rewrites or 0))
        total_rounds = 1 + max_rewrites if bool(self.rewrite_on_insufficient) else 1

        for round_index in range(total_rounds):
            query_vector = self.embedding.embed_query(current_query)
            round_attempts: list[dict[str, Any]] = []

            for top_k in self._schedule():
                results, attempt = self._search_attempt(
                    vector_store,
                    query,
                    current_query,
                    query_vector,
                    top_k,
                )
                attempt["round"] = round_index + 1
                round_attempts.append(attempt)
                attempts.append(attempt)
                last_results = results
                if attempt["sufficient"]:
                    selected_results = results
                    selected_top_k = top_k
                    selected_query = current_query
                    selected_sufficient = True
                    stop_reason = "sufficient_results"
                    break

            rounds.append({"round": round_index + 1, "query": current_query, "attempts": round_attempts})
            if selected_sufficient:
                break

            if round_index == total_rounds - 1:
                stop_reason = "max_rewrites_exhausted" if bool(self.rewrite_on_insufficient) else "insufficient_results"
                break

            rewritten_query, rewrite_info = self._rewrite_query(query, current_query, round_attempts, last_results)
            rewrite_info["round"] = round_index + 1
            rewrite_info["from_query"] = current_query
            rewrites.append(rewrite_info)

            if not rewritten_query:
                stop_reason = rewrite_info.get("reason", "rewrite_unusable")
                break

            normalized_query = self._normalize_query(rewritten_query)
            if normalized_query in seen_queries:
                rewrite_info["reason"] = "rewrite_repeated_query"
                stop_reason = "rewrite_repeated_query"
                break

            seen_queries.add(normalized_query)
            current_query = rewritten_query

        if not selected_sufficient and bool(self.return_best_effort_results):
            selected_results = last_results
            selected_top_k = attempts[-1]["top_k"] if attempts else None
            selected_query = attempts[-1]["search_query"] if attempts else query

        payload = {
            "query": query,
            "selected_query": selected_query,
            "selected_top_k": selected_top_k,
            "sufficient": selected_sufficient,
            "returned_results": len(selected_results),
            "rewrite_enabled": bool(self.rewrite_on_insufficient),
            "max_query_rewrites": max_rewrites,
            "rewrite_count": len(rewrites),
            "rewrites": rewrites,
            "rounds": rounds,
            "stop_reason": stop_reason,
            "attempts": attempts,
        }

        self._cached_search_results = selected_results
        self._cached_debug_payload = payload
        self.status = payload
        return selected_results, payload

    def search_documents(self) -> list[Data]:
        results, _ = self._run_plan()
        return results

    def as_dataframe(self) -> DataFrame:
        return DataFrame(self.search_documents())

    def attempt_debug(self) -> Data:
        _, payload = self._run_plan()
        return Data(data=payload)

    def search_corpus(self) -> Data:
        _, debug_payload = self._run_plan()
        source_documents = self._source_documents()
        context = self._build_context_string()
        active_query = self._resolved_query()

        payload: dict[str, Any] = {
            "query": debug_payload.get("query", active_query),
            "selected_query": debug_payload.get("selected_query", active_query),
            "sufficient": bool(debug_payload.get("sufficient", False)),
            "returned_results": len(source_documents),
            "context": context,
            "source_documents": source_documents,
        }

        for key in ("stop_reason", "rewrite_count", "rewrite_enabled"):
            if key in debug_payload:
                payload[key] = debug_payload[key]

        self.status = payload
        return Data(data=payload)
