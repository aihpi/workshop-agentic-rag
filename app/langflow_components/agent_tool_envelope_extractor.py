from __future__ import annotations

import json
import re
from typing import Any

from lfx.custom.custom_component.component import Component
from lfx.inputs.inputs import HandleInput
from lfx.schema.data import Data
from lfx.schema.message import Message
from lfx.template.field.base import Output


class AgentToolEnvelopeExtractor(Component):
    display_name = "Agent Tool Envelope Extractor"
    description = "Extract the latest retrieval tool payload from an Agent message and expose answer_text plus source_documents."
    documentation = ""
    icon = "PackageSearch"
    name = "AgentToolEnvelopeExtractor"

    inputs = [
        HandleInput(
            name="agent_response",
            display_name="Agent Response",
            input_types=["Message", "Data"],
            info="Agent.response output that may contain tool payloads in content blocks.",
            required=True,
        ),
    ]

    outputs = [
        Output(
            display_name="Envelope",
            name="envelope",
            method="envelope_response",
            tool_mode=False,
        ),
    ]

    def _clean_text(self, value: Any) -> str:
        if not isinstance(value, str):
            return ""
        return value.strip()

    def _coerce_payload(self, value: Any) -> Any:
        if isinstance(value, Data):
            return value.data
        if isinstance(value, dict | list):
            return value
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return ""
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                match = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
                if match:
                    try:
                        return json.loads(match.group(1))
                    except json.JSONDecodeError:
                        return text
                return text
        return value

    def _answer_text(self) -> str:
        value = self.agent_response

        if isinstance(value, Message):
            return self._clean_text(value.text)

        if isinstance(value, Data) and isinstance(value.data, dict):
            for key in ("answer_text", "text", "message", "result", "output_text"):
                candidate = value.data.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    return candidate.strip()

        if isinstance(value, dict):
            for key in ("answer_text", "text", "message", "result", "output_text"):
                candidate = value.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    return candidate.strip()

        return self._clean_text(value)

    def _tool_payloads(self) -> list[Any]:
        value = self.agent_response
        if not isinstance(value, Message):
            return []

        payloads: list[Any] = []
        for block in value.content_blocks or []:
            for content in getattr(block, "contents", []) or []:
                if getattr(content, "type", None) != "tool_use":
                    continue
                payload = self._coerce_payload(getattr(content, "output", None))
                if isinstance(payload, dict | list):
                    payloads.append(payload)
        return payloads

    def _normalize_source_documents(self, value: Any) -> list[dict[str, Any]]:
        payload = self._coerce_payload(value)
        if isinstance(payload, dict):
            payload = payload.get("source_documents", payload)

        if not isinstance(payload, list):
            return []

        docs: list[dict[str, Any]] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            page_content = self._clean_text(item.get("page_content")) or self._clean_text(item.get("text"))
            if not page_content:
                continue
            metadata = item.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
            docs.append(
                {
                    "page_content": page_content,
                    "metadata": metadata,
                }
            )
        return docs

    def _normalize_search_results(self, value: Any) -> list[dict[str, Any]]:
        payload = self._coerce_payload(value)
        if isinstance(payload, dict):
            payload = payload.get("search_results", payload)
        if not isinstance(payload, list):
            return []
        normalized: list[dict[str, Any]] = []
        for item in payload:
            if isinstance(item, Data) and isinstance(item.data, dict):
                normalized.append(dict(item.data))
            elif isinstance(item, dict):
                if isinstance(item.get("data"), dict):
                    normalized.append(dict(item["data"]))
                else:
                    normalized.append(dict(item))
        return normalized

    def _derive_source_documents(self, search_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        docs: list[dict[str, Any]] = []
        for row in search_results:
            page_content = self._clean_text(row.get("text")) or self._clean_text(row.get("page_content"))
            if not page_content:
                continue

            metadata: dict[str, Any] = {}
            for key in ("file", "page", "page_end", "section_title"):
                if row.get(key) not in (None, "", []):
                    metadata[key] = row[key]

            docs.append(
                {
                    "page_content": page_content,
                    "metadata": metadata,
                }
            )
        return docs

    def _latest_tool_payload(self) -> Any:
        payloads = self._tool_payloads()
        if not payloads:
            return {}

        for payload in reversed(payloads):
            if isinstance(payload, dict) and self._normalize_source_documents(payload.get("source_documents")):
                return payload
        for payload in reversed(payloads):
            if self._normalize_search_results(payload):
                return payload
        return payloads[-1]

    def envelope_response(self) -> Data:
        value = self.agent_response

        if isinstance(value, Data) and isinstance(value.data, dict):
            source_documents = self._normalize_source_documents(value.data.get("source_documents"))
            search_results = self._normalize_search_results(value.data.get("search_results"))
            payload = {
                "answer_text": self._answer_text(),
                "source_documents": source_documents or self._derive_source_documents(search_results),
            }
            if search_results:
                payload["search_results"] = search_results
            self.status = payload
            return Data(data=payload)

        tool_payload = self._latest_tool_payload()
        source_documents = self._normalize_source_documents(tool_payload)
        search_results = self._normalize_search_results(tool_payload)
        if not source_documents and search_results:
            source_documents = self._derive_source_documents(search_results)

        payload: dict[str, Any] = {
            "answer_text": self._answer_text(),
            "source_documents": source_documents,
        }
        if search_results:
            payload["search_results"] = search_results

        if isinstance(tool_payload, dict):
            for key in ("query", "selected_query", "selected_top_k", "sufficient", "returned_results", "stop_reason"):
                if key in tool_payload:
                    payload[key] = tool_payload[key]

        self.status = payload
        return Data(data=payload)
