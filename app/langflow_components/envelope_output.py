from __future__ import annotations

from collections.abc import Generator, Iterable
from typing import Any

import orjson
from fastapi.encoders import jsonable_encoder

from lfx.base.io.chat import ChatComponent
from lfx.helpers.data import safe_convert
from lfx.inputs.inputs import BoolInput, DropdownInput, HandleInput, MessageTextInput
from lfx.schema.data import Data
from lfx.schema.dataframe import DataFrame
from lfx.schema.message import Message
from lfx.schema.properties import Source
from lfx.template.field.base import Output
from lfx.utils.constants import (
    MESSAGE_SENDER_AI,
    MESSAGE_SENDER_NAME_AI,
    MESSAGE_SENDER_USER,
)


class EnvelopeOutputComponent(ChatComponent):
    display_name = "Envelope Output"
    description = "Send the answer as a chat message and expose answer_text plus source_documents via API."
    documentation = ""
    icon = "PackageOpen"
    name = "EnvelopeOutput"
    minimized = True

    inputs = [
        HandleInput(
            name="answer_input",
            display_name="Answer",
            info="Final answer text or message produced by the answer path.",
            input_types=["Data", "DataFrame", "Message"],
            required=True,
        ),
        HandleInput(
            name="source_documents_input",
            display_name="Source Documents",
            input_types=["Data", "DataFrame"],
            info="Output from SourceDocumentsBuilder.source_documents.",
            required=False,
        ),
        BoolInput(
            name="should_store_message",
            display_name="Store Messages",
            info="Store the message in the history.",
            value=True,
            advanced=True,
        ),
        DropdownInput(
            name="sender",
            display_name="Sender Type",
            options=[MESSAGE_SENDER_AI, MESSAGE_SENDER_USER],
            value=MESSAGE_SENDER_AI,
            advanced=True,
            info="Type of sender.",
        ),
        MessageTextInput(
            name="sender_name",
            display_name="Sender Name",
            info="Name of the sender.",
            value=MESSAGE_SENDER_NAME_AI,
            advanced=True,
        ),
        MessageTextInput(
            name="session_id",
            display_name="Session ID",
            info="The session ID of the chat. If empty, the current session ID parameter will be used.",
            advanced=True,
        ),
        MessageTextInput(
            name="context_id",
            display_name="Context ID",
            info="The context ID of the chat. Adds an extra layer to the local memory.",
            value="",
            advanced=True,
        ),
        MessageTextInput(
            name="data_template",
            display_name="Data Template",
            value="{text}",
            advanced=True,
            info="Template to convert Data to Text. If left empty, it will be dynamically set to the Data's text key.",
        ),
        BoolInput(
            name="clean_data",
            display_name="Basic Clean Data",
            value=True,
            advanced=True,
            info="Whether to clean data before converting to string.",
        ),
    ]

    outputs = [
        Output(
            display_name="Output Message",
            name="message",
            method="message_response",
        ),
        Output(
            display_name="Envelope",
            name="envelope",
            info="API payload with answer_text and source_documents.",
            method="envelope_response",
        ),
    ]

    def _build_source(self, id_: str | None, display_name: str | None, source: str | None) -> Source:
        source_dict = {}
        if id_:
            source_dict["id"] = id_
        if display_name:
            source_dict["display_name"] = display_name
        if source:
            if hasattr(source, "model_name"):
                source_dict["source"] = source.model_name
            elif hasattr(source, "model"):
                source_dict["source"] = str(source.model)
            else:
                source_dict["source"] = str(source)
        return Source(**source_dict)

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

    async def message_response(self) -> Message:
        text = self.convert_to_string()

        source, _, display_name, source_id = self.get_properties_from_source_component()

        if isinstance(self.answer_input, Message) and not self.is_connected_to_chat_input():
            message = self.answer_input
            message.text = text
            existing_session_id = message.session_id
        else:
            message = Message(text=text)
            existing_session_id = None

        message.sender = self.sender
        message.sender_name = self.sender_name
        message.session_id = (
            self.session_id or existing_session_id or (self.graph.session_id if hasattr(self, "graph") else None) or ""
        )
        message.context_id = self.context_id
        message.flow_id = self.graph.flow_id if hasattr(self, "graph") else None
        message.properties.source = self._build_source(source_id, display_name, source)

        if message.session_id and self.should_store_message:
            stored_message = await self.send_message(message)
            self.message.value = stored_message
            message = stored_message

        self.status = message
        return message

    def _serialize_data(self, data: Data) -> str:
        serializable_data = jsonable_encoder(data.data)
        json_bytes = orjson.dumps(serializable_data, option=orjson.OPT_INDENT_2)
        return "```json\n" + json_bytes.decode("utf-8") + "\n```"

    def _validate_answer_input(self) -> None:
        value = self.answer_input
        if value is None:
            msg = "Input data cannot be None"
            raise ValueError(msg)
        if isinstance(value, list) and not all(isinstance(item, Message | Data | DataFrame | str) for item in value):
            invalid_types = [
                type(item).__name__
                for item in value
                if not isinstance(item, Message | Data | DataFrame | str)
            ]
            msg = f"Expected Data or DataFrame or Message or str, got {invalid_types}"
            raise TypeError(msg)
        if not isinstance(value, Message | Data | DataFrame | str | list | Generator | type(None)):
            type_name = type(value).__name__
            msg = f"Expected Data or DataFrame or Message or str, Generator or None, got {type_name}"
            raise TypeError(msg)

    def convert_to_string(self) -> str | Generator[Any, None, None]:
        self._validate_answer_input()
        value = self.answer_input
        if isinstance(value, list):
            clean_data: bool = getattr(self, "clean_data", False)
            return "\n".join([safe_convert(item, clean_data=clean_data) for item in value])
        if isinstance(value, Generator):
            return value
        return safe_convert(value)

    def envelope_response(self) -> Data:
        payload = {
            "answer_text": self._answer_text(),
            "source_documents": self._normalize_source_documents(self.source_documents_input),
        }
        data = Data(data=payload)
        self.status = payload
        return data
