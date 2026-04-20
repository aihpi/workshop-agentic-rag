from __future__ import annotations

from typing import Any

import litellm

from settings import CHAT_MODEL, EMBED_MODEL, LITELLM_API_KEY, LITELLM_BASE_URL


def _client_args() -> dict[str, Any]:
    args: dict[str, Any] = {}
    if LITELLM_BASE_URL:
        args["api_base"] = LITELLM_BASE_URL
    if LITELLM_API_KEY:
        args["api_key"] = LITELLM_API_KEY
    return args


async def chat(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | None = "auto",
    model: str | None = None,
):
    payload: dict[str, Any] = {
        "model": model or CHAT_MODEL,
        "messages": messages,
        **_client_args(),
    }
    if tools:
        payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice
    return await litellm.acompletion(**payload)


async def stream_chat(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, tool_choice: str | None = None):
    payload: dict[str, Any] = {
        "model": CHAT_MODEL,
        "messages": messages,
        "stream": True,
        **_client_args(),
    }
    if tools:
        payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice
    return await litellm.acompletion(**payload)


async def embed(texts: list[str]) -> list[list[float]]:
    response = await litellm.aembedding(
        model=EMBED_MODEL,
        input=texts,
        encoding_format="float",
        **_client_args(),
    )
    data = sorted(response.data, key=lambda item: item["index"])
    return [item["embedding"] for item in data]


def message_to_dict(message: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "role": message.role,
        "content": message.content,
    }
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        data["tool_calls"] = [
            {
                "id": tc.id,
                "type": tc.type,
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in tool_calls
        ]
    return data
