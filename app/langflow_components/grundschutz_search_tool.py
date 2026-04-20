from __future__ import annotations

import importlib.util
from pathlib import Path
import re
from typing import Any

from lfx.schema.data import Data
from lfx.template.field.base import Output


def _load_adaptive_retriever_component():
    try:
        from adaptive_qdrant_retriever import AdaptiveQdrantRetrieverComponent

        return AdaptiveQdrantRetrieverComponent
    except ModuleNotFoundError:
        module_path = Path(__file__).with_name("adaptive_qdrant_retriever.py")
        spec = importlib.util.spec_from_file_location("adaptive_qdrant_retriever", module_path)
        if spec is None or spec.loader is None:
            msg = f"Unable to load adaptive_qdrant_retriever from {module_path}"
            raise ImportError(msg)

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.AdaptiveQdrantRetrieverComponent


AdaptiveQdrantRetrieverComponent = _load_adaptive_retriever_component()


class GrundschutzSearchToolComponent(AdaptiveQdrantRetrieverComponent):
    display_name = "Grundschutz Search Tool"
    description = (
        "Search the BSI IT-Grundschutz corpus with adaptive Qdrant retrieval and return "
        "structured search results plus citation-ready source_documents."
    )
    icon = "Search"
    name = "GrundschutzSearchTool"

    outputs = [
        Output(
            display_name="Tool Payload",
            name="tool_payload",
            method="search_corpus",
            info="Structured retrieval payload for agent tool use.",
            tool_mode=True,
        ),
        Output(
            display_name="Search Results",
            name="search_results",
            method="search_documents",
            tool_mode=False,
        ),
        Output(
            display_name="Source Documents",
            name="source_documents",
            method="build_source_documents",
            tool_mode=False,
        ),
        Output(
            display_name="Attempt Debug",
            name="attempt_debug",
            method="attempt_debug",
            tool_mode=False,
        ),
    ]

    def _result_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for item in self.search_documents():
            if isinstance(item, Data) and isinstance(item.data, dict):
                rows.append(dict(item.data))
            elif isinstance(item, dict):
                rows.append(dict(item))
        return rows

    def _clean_text(self, value: Any) -> str:
        if not isinstance(value, str):
            return ""
        return " ".join(value.split()).strip()

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
            text = self._clean_text(row.get(key))
            if text:
                return text
        metadata = self._metadata_from_row(row)
        for key in ("text", "page_content", "content"):
            text = self._clean_text(metadata.get(key))
            if text:
                return text
        return ""

    def _resolve_file(self, row: dict[str, Any]) -> str:
        metadata = self._metadata_from_row(row)
        for container in (row, metadata):
            for key in ("file", "document", "title", "source"):
                value = self._clean_text(container.get(key))
                if value:
                    return value
        source = row.get("source")
        if isinstance(source, dict):
            for key in ("file", "document", "title", "source"):
                value = self._clean_text(source.get(key))
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
            for key in ("page_end",):
                value = self._coerce_int(container.get(key))
                if value is not None:
                    return value
        return None

    def _resolve_section(self, row: dict[str, Any]) -> str:
        metadata = self._metadata_from_row(row)
        for container in (row, metadata):
            for key in ("section_title", "title"):
                value = self._clean_text(container.get(key))
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

    def build_source_documents(self) -> Data:
        payload = {"source_documents": self._source_documents()}
        self.status = payload
        return Data(data=payload)

    def search_corpus(self) -> Data:
        _, debug_payload = self._run_plan()
        search_results = self._result_rows()
        source_documents = self._source_documents()

        payload: dict[str, Any] = {
            "query": debug_payload.get("query", self.search_query),
            "selected_query": debug_payload.get("selected_query", self.search_query),
            "selected_top_k": debug_payload.get("selected_top_k"),
            "sufficient": bool(debug_payload.get("sufficient", False)),
            "returned_results": len(search_results),
            "search_results": search_results,
            "source_documents": source_documents,
        }

        for key in ("stop_reason", "rewrite_count", "rewrite_enabled"):
            if key in debug_payload:
                payload[key] = debug_payload[key]

        self.status = payload
        return Data(data=payload)
