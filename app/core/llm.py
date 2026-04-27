from __future__ import annotations

import logging
import time
from typing import Any

import httpx
import litellm

from core.settings import (
    CHAT_MODEL,
    EMBED_MODEL,
    FALLBACK_CHAT_MODEL,
    LITELLM_API_KEY,
    LITELLM_BASE_URL,
)

logger = logging.getLogger(__name__)


def _client_args() -> dict[str, Any]:
    args: dict[str, Any] = {}
    if LITELLM_BASE_URL:
        args["api_base"] = LITELLM_BASE_URL
    if LITELLM_API_KEY:
        args["api_key"] = LITELLM_API_KEY
    return args


def _fallback_args(primary_model: str) -> dict[str, Any]:
    """LiteLLM's native fallback support — if set, `litellm.acompletion()`
    automatically retries with FALLBACK_CHAT_MODEL on provider errors
    (5xx, timeouts, connection refused). No manual try/except needed.
    Skipped if the fallback is unset or identical to the primary."""
    if not FALLBACK_CHAT_MODEL or FALLBACK_CHAT_MODEL == primary_model:
        return {}
    return {"fallbacks": [FALLBACK_CHAT_MODEL]}


# ---------------------------------------------------------------------------
# Pre-flight model availability check
# ---------------------------------------------------------------------------
#
# Without this, when CHAT_MODEL is configured but the underlying vLLM pod is
# down, every chat turn pays ~15s waiting for LiteLLM SDK retries before
# the `fallbacks=` kicks in. Gateway's /v1/models lists models whose backend
# is registered and reachable; missing from that list ⇒ skip primary, go
# straight to fallback. ~50ms once per cache window, then zero overhead.

_MODEL_AVAIL_TTL_S = 30.0
_model_avail_cache: dict[str, tuple[float, set[str]]] = {}


def _bare_model_name(model: str) -> str:
    """Strip LiteLLM's provider prefix ('openai/foo' → 'foo'). The gateway's
    /v1/models lists ids without the provider hint."""
    return model.split("/", 1)[1] if "/" in model else model


async def _list_available_models(base_url: str) -> set[str] | None:
    """Return the set of model ids the LiteLLM gateway claims to serve.

    None on any failure (timeout, non-2xx, malformed JSON) — callers treat
    that as 'health unknown, behave as if primary is up' so a flaky probe
    never makes things *worse* than the no-precheck baseline.
    """
    now = time.monotonic()
    cached = _model_avail_cache.get(base_url)
    if cached and cached[0] > now:
        return cached[1]

    headers: dict[str, str] = {}
    if LITELLM_API_KEY:
        headers["Authorization"] = f"Bearer {LITELLM_API_KEY}"
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{base_url}/v1/models", headers=headers)
    except httpx.HTTPError as exc:
        logger.warning("llm.health_probe_failed reason=%s", type(exc).__name__)
        return None
    if resp.status_code != 200:
        logger.warning("llm.health_probe_status status=%d", resp.status_code)
        return None
    try:
        data = resp.json()
    except ValueError:
        return None
    ids = {
        m["id"]
        for m in (data.get("data") or [])
        if isinstance(m, dict) and isinstance(m.get("id"), str)
    }
    _model_avail_cache[base_url] = (now + _MODEL_AVAIL_TTL_S, ids)
    return ids


async def _resolve_model(primary: str) -> str:
    """If the gateway reports the primary model as unreachable but the
    fallback as reachable, return the fallback so we skip the retry
    pingpong. Otherwise return the primary unchanged.
    """
    if not LITELLM_BASE_URL or not FALLBACK_CHAT_MODEL:
        return primary
    if FALLBACK_CHAT_MODEL == primary:
        return primary
    available = await _list_available_models(LITELLM_BASE_URL)
    if available is None:
        # Probe failed — fail open, let LiteLLM's `fallbacks=` handle real errors.
        return primary
    bare_primary = _bare_model_name(primary)
    if bare_primary in available:
        return primary
    bare_fallback = _bare_model_name(FALLBACK_CHAT_MODEL)
    if bare_fallback in available:
        logger.info(
            "llm.preflight_swap primary=%s fallback=%s reason=primary_not_in_models",
            primary, FALLBACK_CHAT_MODEL,
        )
        return FALLBACK_CHAT_MODEL
    return primary


async def chat(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | None = "auto",
    model: str | None = None,
):
    primary = await _resolve_model(model or CHAT_MODEL)
    payload: dict[str, Any] = {
        "model": primary,
        "messages": messages,
        **_client_args(),
        **_fallback_args(primary),
    }
    if tools:
        payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice
    return await litellm.acompletion(**payload)


async def stream_chat(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, tool_choice: str | None = None):
    primary = await _resolve_model(CHAT_MODEL)
    payload: dict[str, Any] = {
        "model": primary,
        "messages": messages,
        "stream": True,
        **_client_args(),
        **_fallback_args(primary),
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
