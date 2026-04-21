from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from lfx.custom.custom_component.component import Component
from lfx.inputs.inputs import HandleInput
from lfx.schema.data import Data
from lfx.schema.dataframe import DataFrame
from lfx.schema.message import Message
from lfx.template.field.base import Output


class AnswerEnvelopeBuilder(Component):
    display_name = "Answer Envelope Builder"
    description = "Bundle answer text and source_documents into one API payload for Chainlit."
    documentation = ""
    icon = "PackageOpen"
    name = "AnswerEnvelopeBuilder"

    inputs = [
        HandleInput(
            name="answer_input",
            display_name="Answer",
            input_types=["Message", "Data", "DataFrame"],
            info="Final answer text or message produced by the answer path.",
            required=True,
        ),
        HandleInput(
            name="source_documents_input",
            display_name="Source Documents",
            input_types=["Data", "DataFrame"],
            info="Output from SourceDocumentsBuilder.source_documents.",
            required=False,
        ),
    ]

    outputs = [
        Output(
            display_name="Envelope",
            name="envelope",
            info="API payload with answer_text and source_documents.",
            method="build_envelope",
        ),
    ]

    def _clean_text(self, value: Any) -> str:
        if not isinstance(value, str):
            return ""
        return value.strip()

    def _answer_text(self) -> str:
        value = self.answer_input

        if isinstance(value, Message):
            return self._clean_text(value.text)

        if isinstance(value, Data):
            payload = value.data
            if isinstance(payload, dict):
                for key in ("answer_text", "text", "message", "result", "output_text"):
                    candidate = payload.get(key)
                    if isinstance(candidate, str) and candidate.strip():
                        return candidate.strip()
                    if isinstance(candidate, dict):
                        nested_text = candidate.get("text")
                        if isinstance(nested_text, str) and nested_text.strip():
                            return nested_text.strip()
            return ""

        if isinstance(value, DataFrame):
            rows = value.to_dict(orient="records")
            if rows:
                first_row = rows[0]
                if isinstance(first_row, dict):
                    for key in ("answer_text", "text", "message", "result", "output_text"):
                        candidate = first_row.get(key)
                        if isinstance(candidate, str) and candidate.strip():
                            return candidate.strip()
            return ""

        return self._clean_text(value)

    def _normalize_source_documents(self, value: Any) -> list[dict[str, Any]]:
        if isinstance(value, Data):
            payload = value.data
            if isinstance(payload, dict):
                raw_docs = payload.get("source_documents")
                if isinstance(raw_docs, list):
                    return [item for item in raw_docs if isinstance(item, dict)]
            return []

        if isinstance(value, DataFrame):
            docs: list[dict[str, Any]] = []
            for row in value.to_dict(orient="records"):
                if not isinstance(row, dict):
                    continue
                page_content = self._clean_text(row.get("page_content")) or self._clean_text(row.get("text"))
                if not page_content:
                    continue
                metadata = row.get("metadata")
                if not isinstance(metadata, dict):
                    metadata = {}
                docs.append({"page_content": page_content, "metadata": metadata})
            return docs

        if isinstance(value, Iterable) and not isinstance(value, str | bytes | dict):
            return [item for item in value if isinstance(item, dict)]

        return []

    def build_envelope(self) -> Data:
        payload = {
            "answer_text": self._answer_text(),
            "source_documents": self._normalize_source_documents(self.source_documents_input),
        }
        data = Data(data=payload)
        self.status = payload
        return data
