from __future__ import annotations

import asyncio
import json
from typing import Any
from urllib import error, request

from core.settings import LANGFLOW_API_KEY, LANGFLOW_BASE_URL, LANGFLOW_FLOW_ID, LANGFLOW_OUTPUT_COMPONENT


class LangflowError(RuntimeError):
    pass


_HEADER_CHAR_REPLACEMENTS = str.maketrans(
    {
        "\u00a0": " ",
        "\u2009": " ",
        "\u202f": " ",
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2026": "...",
    }
)


def _clean_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


def _sanitize_header_value(value: Any) -> str:
    text = _clean_text(value)
    if not text:
        return ""

    # HTTP header values must be single-line latin-1. Preserve as much signal as
    # possible while avoiding request crashes on German typography and prompt formatting.
    text = text.translate(_HEADER_CHAR_REPLACEMENTS)
    text = text.replace("\r", " ").replace("\n", " ")
    text = " ".join(text.split())
    try:
        text.encode("latin-1")
    except UnicodeEncodeError:
        text = text.encode("latin-1", errors="replace").decode("latin-1")
    return text


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned.isdigit():
            return int(cleaned)
    return None


def _coerce_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    cleaned = value.strip()
    if not cleaned or cleaned[0] not in "[{":
        return value
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return value


def _iter_dicts(value: Any):
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from _iter_dicts(item)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_dicts(item)


def _extract_answer_text(payload: dict[str, Any]) -> str:
    candidates: list[tuple[int, str]] = []
    for node in _iter_dicts(payload):
        answer_text = _clean_text(node.get("answer_text"))
        if answer_text:
            candidates.append((5, answer_text))

        message = node.get("message")
        if isinstance(message, dict):
            text = _clean_text(message.get("text"))
            if text:
                candidates.append((4, text))

        result_text = _clean_text(node.get("result"))
        if result_text:
            priority = 3 if (len(result_text) > 40 or "\n" in result_text) else 2
            candidates.append((priority, result_text))

        output_text = _clean_text(node.get("output_text"))
        if output_text:
            candidates.append((3, output_text))

    if not candidates:
        return ""
    candidates.sort(key=lambda item: (item[0], len(item[1])), reverse=True)
    return candidates[0][1]


def _normalize_citation(item: Any, fallback_number: int) -> dict[str, Any] | None:
    value = _coerce_json(item)
    if not isinstance(value, dict):
        return None

    citation_number = _coerce_int(value.get("citation_number"))
    if citation_number is None or citation_number < 1:
        citation_number = fallback_number

    citation: dict[str, Any] = {"citation_number": citation_number}
    for field in ("file", "section_title", "evidence"):
        text = _clean_text(value.get(field))
        if text:
            citation[field] = text

    page_start = _coerce_int(value.get("page_start"))
    page_end = _coerce_int(value.get("page_end"))
    if page_start is not None:
        citation["page_start"] = page_start
    if page_end is not None:
        citation["page_end"] = page_end
    return citation


def _citation_from_source_document(item: Any, fallback_number: int) -> dict[str, Any] | None:
    value = _coerce_json(item)
    if not isinstance(value, dict):
        return None

    metadata = value.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    source = metadata.get("source")
    source_file = ""
    if isinstance(source, dict):
        source_file = _clean_text(source.get("file"))
    if not source_file:
        for key in ("file", "document", "title", "source"):
            candidate = metadata.get(key)
            if isinstance(candidate, str) and candidate.strip():
                source_file = candidate.strip()
                break

    page_start = (
        _coerce_int(metadata.get("page_start"))
        or _coerce_int(metadata.get("page"))
        or _coerce_int(metadata.get("page_number"))
    )
    page_end = _coerce_int(metadata.get("page_end"))
    section_title = _clean_text(metadata.get("section_title")) or _clean_text(metadata.get("title"))
    evidence = _clean_text(value.get("page_content")) or _clean_text(value.get("text"))

    citation: dict[str, Any] = {"citation_number": fallback_number}
    if source_file:
        citation["file"] = source_file
    if page_start is not None:
        citation["page_start"] = page_start
    if page_end is not None:
        citation["page_end"] = page_end
    if section_title:
        citation["section_title"] = section_title
    if evidence:
        citation["evidence"] = evidence
    return citation if len(citation) > 1 else None


def _extract_citations(payload: dict[str, Any]) -> list[dict[str, Any]]:
    citations: list[dict[str, Any]] = []
    for node in _iter_dicts(payload):
        for key in ("citations", "references"):
            raw_value = _coerce_json(node.get(key))
            if not isinstance(raw_value, list):
                continue
            for index, item in enumerate(raw_value, start=1):
                citation = _normalize_citation(item, index)
                if citation:
                    citations.append(citation)

        raw_source_docs = _coerce_json(node.get("source_documents"))
        if isinstance(raw_source_docs, list):
            for index, item in enumerate(raw_source_docs, start=1):
                citation = _citation_from_source_document(item, index)
                if citation:
                    citations.append(citation)

    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for citation in citations:
        key = json.dumps(citation, ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(citation)
    deduped.sort(key=lambda item: int(item.get("citation_number") or 0))
    return deduped


def _run_langflow_sync(
    *,
    input_value: str,
    session_id: str,
    global_vars: dict[str, str],
) -> dict[str, Any]:
    if not LANGFLOW_FLOW_ID or not LANGFLOW_FLOW_ID.strip():
        raise LangflowError("LANGFLOW_FLOW_ID is not configured")

    url = f"{LANGFLOW_BASE_URL}/api/v1/run/{LANGFLOW_FLOW_ID.strip()}"
    payload = {
        "input_value": input_value,
        "session_id": session_id,
        "input_type": "chat",
        "output_type": "any",
    }
    if LANGFLOW_OUTPUT_COMPONENT and LANGFLOW_OUTPUT_COMPONENT.strip():
        payload["output_component"] = LANGFLOW_OUTPUT_COMPONENT.strip()
    headers = {"Content-Type": "application/json", "accept": "application/json"}
    if LANGFLOW_API_KEY:
        headers["x-api-key"] = LANGFLOW_API_KEY
    for key, value in global_vars.items():
        safe_value = _sanitize_header_value(value)
        if not safe_value:
            continue
        headers[f"X-LANGFLOW-GLOBAL-VAR-{key}"] = safe_value

    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(url, data=body, headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise LangflowError(f"Langflow HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise LangflowError(f"Langflow connection failed: {exc.reason}") from exc
    except TimeoutError as exc:
        raise LangflowError("Langflow request timed out") from exc
    except UnicodeEncodeError as exc:
        raise LangflowError("Langflow request headers could not be encoded safely") from exc

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LangflowError("Langflow returned invalid JSON") from exc

    answer_text = _extract_answer_text(payload)
    citations = _extract_citations(payload)
    return {
        "answer_text": answer_text,
        "citations": citations,
        "raw_response": payload,
    }


async def run_langflow(
    *,
    input_value: str,
    session_id: str,
    global_vars: dict[str, str],
) -> dict[str, Any]:
    return await asyncio.to_thread(
        _run_langflow_sync,
        input_value=input_value,
        session_id=session_id,
        global_vars=global_vars,
    )
