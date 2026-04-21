from lfx.custom.custom_component.component import Component
from lfx.inputs.inputs import HandleInput, MessageTextInput
from lfx.schema.data import Data
from lfx.schema.dataframe import DataFrame
from lfx.schema.message import Message
from lfx.template.field.base import Output


class SourceDocumentsBuilder(Component):
    display_name = "Source Documents Builder"
    description = "Build deterministic source_documents payloads and numbered context from retrieval rows."
    documentation = ""
    icon = "FileJson"
    name = "SourceDocumentsBuilder"

    inputs = [
        HandleInput(
            name="input_data",
            display_name="Data or DataFrame",
            input_types=["DataFrame", "Data"],
            info="Retriever output containing text and source metadata.",
            required=True,
        ),
        MessageTextInput(
            name="text_key",
            display_name="Text Field",
            value="text",
            info="Primary field that contains the retrieved chunk text.",
            advanced=True,
        ),
        MessageTextInput(
            name="file_key",
            display_name="File Field",
            value="file",
            info="Primary field that contains the PDF file name.",
            advanced=True,
        ),
        MessageTextInput(
            name="page_key",
            display_name="Page Field",
            value="page_start",
            info="Primary field that contains the start page.",
            advanced=True,
        ),
        MessageTextInput(
            name="page_end_key",
            display_name="Page End Field",
            value="page_end",
            info="Primary field that contains the end page.",
            advanced=True,
        ),
        MessageTextInput(
            name="section_key",
            display_name="Section Field",
            value="section_title",
            info="Primary field that contains the section title.",
            advanced=True,
        ),
        MessageTextInput(
            name="separator",
            display_name="Context Separator",
            value="\n\n---\n\n",
            info="Separator used between numbered source blocks in the prompt context.",
            advanced=True,
        ),
    ]

    outputs = [
        Output(
            display_name="Context Message",
            name="context_message",
            info="Numbered context blocks for the answer prompt.",
            method="build_context_message",
        ),
        Output(
            display_name="Source Documents",
            name="source_documents",
            info="Deterministic source_documents payload for API consumers.",
            method="build_source_documents",
        ),
    ]

    def _as_rows(self) -> list[dict]:
        value = self.input_data

        if isinstance(value, DataFrame):
            return [row.to_dict() for _, row in value.iterrows()]

        if isinstance(value, Data):
            return self._rows_from_payload(value.data)

        if isinstance(value, list):
            rows: list[dict] = []
            for item in value:
                if isinstance(item, Data):
                    rows.extend(self._rows_from_payload(item.data))
                elif isinstance(item, dict):
                    rows.append(item)
            return rows

        if isinstance(value, dict):
            payload = value.get("data", value)
            return self._rows_from_payload(payload)

        msg = f"Unsupported input type: {type(value)}. Expected DataFrame or Data."
        raise ValueError(msg)

    def _rows_from_payload(self, payload) -> list[dict]:
        if isinstance(payload, dict):
            if isinstance(payload.get("results"), list):
                return [item for item in payload["results"] if isinstance(item, dict)]
            return [payload]
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        return []

    def _clean_text(self, value) -> str:
        if not isinstance(value, str):
            return ""
        return " ".join(value.split()).strip()

    def _row_metadata(self, row: dict) -> dict:
        metadata = row.get("metadata")
        return metadata if isinstance(metadata, dict) else {}

    def _coerce_int(self, value):
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            text = value.strip()
            if text.isdigit():
                return int(text)
        return None

    def _resolve_file(self, row: dict) -> str:
        file_name = self._clean_text(row.get(self.file_key))
        if file_name:
            return file_name

        metadata = self._row_metadata(row)
        file_name = self._clean_text(metadata.get(self.file_key))
        if file_name:
            return file_name
        for key in ("file", "document", "title", "source"):
            file_name = self._clean_text(metadata.get(key))
            if file_name:
                return file_name

        source = row.get("source")
        if isinstance(source, dict):
            for key in ("file", "document", "title", "source"):
                file_name = self._clean_text(source.get(key))
                if file_name:
                    return file_name

        for key in ("document", "title", "source"):
            file_name = self._clean_text(row.get(key))
            if file_name:
                return file_name
        return ""

    def _resolve_page(self, row: dict):
        page = self._coerce_int(row.get(self.page_key))
        if page is not None:
            return page

        metadata = self._row_metadata(row)
        page = self._coerce_int(metadata.get(self.page_key))
        if page is not None:
            return page

        page_value = row.get("page")
        if isinstance(page_value, dict):
            page = self._coerce_int(page_value.get("start"))
            if page is not None:
                return page

        metadata_page = metadata.get("page")
        if isinstance(metadata_page, dict):
            page = self._coerce_int(metadata_page.get("start"))
            if page is not None:
                return page

        for key in ("page", "page_number"):
            page = self._coerce_int(row.get(key))
            if page is not None:
                return page
            page = self._coerce_int(metadata.get(key))
            if page is not None:
                return page
        return None

    def _resolve_page_end(self, row: dict):
        page_end = self._coerce_int(row.get(self.page_end_key))
        if page_end is not None:
            return page_end

        metadata = self._row_metadata(row)
        page_end = self._coerce_int(metadata.get(self.page_end_key))
        if page_end is not None:
            return page_end

        page_value = row.get("page")
        if isinstance(page_value, dict):
            page_end = self._coerce_int(page_value.get("end"))
            if page_end is not None:
                return page_end

        metadata_page = metadata.get("page")
        if isinstance(metadata_page, dict):
            page_end = self._coerce_int(metadata_page.get("end"))
            if page_end is not None:
                return page_end
        return None

    def _resolve_section(self, row: dict) -> str:
        section = self._clean_text(row.get(self.section_key))
        if section:
            return section
        metadata = self._row_metadata(row)
        section = self._clean_text(metadata.get(self.section_key))
        if section:
            return section
        section = self._clean_text(metadata.get("title"))
        if section:
            return section
        return self._clean_text(row.get("title"))

    def _resolve_text(self, row: dict) -> str:
        for key in (self.text_key, "page_content", "content", "chunk", "body"):
            text = self._clean_text(row.get(key))
            if text:
                return text
        return ""

    def _source_documents(self) -> list[dict]:
        docs: list[dict] = []
        for row in self._as_rows():
            page_content = self._resolve_text(row)
            if not page_content:
                continue

            metadata = {}
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

    def build_context_message(self) -> Message:
        blocks: list[str] = []
        for index, doc in enumerate(self._source_documents(), start=1):
            metadata = doc.get("metadata") or {}
            lines = [f"[Quelle {index}]"]

            file_name = self._clean_text(metadata.get("file"))
            if file_name:
                lines.append(f"file: {file_name}")

            page = metadata.get("page")
            if page is not None:
                lines.append(f"page: {page}")

            page_end = metadata.get("page_end")
            if page_end is not None:
                lines.append(f"page_end: {page_end}")

            section_title = self._clean_text(metadata.get("section_title"))
            if section_title:
                lines.append(f"section_title: {section_title}")

            lines.append(f"page_content: {doc.get('page_content', '')}")
            blocks.append("\n".join(lines))

        text = self.separator.join(blocks)
        message = Message(text=text)
        self.status = message
        return message

    def build_source_documents(self) -> Data:
        payload = {"source_documents": self._source_documents()}
        data = Data(data=payload)
        self.status = payload
        return data
