"""The Anthropic adapter — the only file that imports the SDK.

Translates neutral types to Claude's content-blocks format, calls
`messages.create`, translates the response back, and computes cost from a
model-specific price table.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from anthropic import Anthropic

from dobby.config import ANTHROPIC_API_KEY, LLM_MODEL
from dobby.llm.types import (
    LLMResponse,
    Message,
    Stop,
    ToolCall,
    ToolResult,
    ToolSchema,
    Usage,
)

logger = logging.getLogger(__name__)

# US$ per million tokens, lifted from PRD §Token cost and usage tracking.
PRICES: dict[str, dict[str, float]] = {
    "claude-haiku-4-5-20251001": {
        "input": 0.10,
        "output": 0.50,
        "cache_read": 0.01,
        "cache_write": 0.125,
    },
}

MAX_TOKENS = 1024


class AnthropicClient:
    """Satisfies the LLMClient protocol by wrapping anthropic.Anthropic.

    Note on temperature: Anthropic's v2 Messages API (SDK ≥ 2.x, Claude 4.x
    models) removed the `temperature` parameter. The constructor accepts it
    for signature uniformity with other adapters (e.g. Ollama), but ignores
    it. For evals we rely on multi-run averaging to control variance.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
    ) -> None:
        self._client = Anthropic(api_key=api_key or ANTHROPIC_API_KEY)
        self._model = model or LLM_MODEL

    def complete(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSchema],
    ) -> LLMResponse:
        claude_messages = _to_claude_messages(messages)
        claude_tools = _to_claude_tools(tools)

        start = time.monotonic()
        response = self._client.messages.create(
            model=self._model,
            system=system,
            messages=claude_messages,
            tools=claude_tools,
            max_tokens=MAX_TOKENS,
        )
        latency_ms = int((time.monotonic() - start) * 1000)

        return _from_claude_response(response, self._model, latency_ms)


def _to_claude_messages(messages: list[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.tool_results:
            out.append(
                {
                    "role": "user",
                    "content": [_tool_result_block(tr) for tr in m.tool_results],
                }
            )
        elif m.tool_calls:
            content: list[dict[str, Any]] = []
            if m.text:
                content.append({"type": "text", "text": m.text})
            for tc in m.tool_calls:
                content.append(
                    {
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.input,
                    }
                )
            out.append({"role": m.role, "content": content})
        else:
            out.append({"role": m.role, "content": m.text or ""})
    return out


def _tool_result_block(tr: ToolResult) -> dict[str, Any]:
    block: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": tr.id,
        "content": tr.content,
    }
    if tr.is_error:
        block["is_error"] = True
    return block


def _to_claude_tools(tools: list[ToolSchema]) -> list[dict[str, Any]]:
    return [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": t.input_schema,
        }
        for t in tools
    ]


def _from_claude_response(response: Any, model: str, latency_ms: int) -> LLMResponse:
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for block in response.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append(
                ToolCall(id=block.id, name=block.name, input=dict(block.input))
            )

    stop: Stop = "tool" if response.stop_reason == "tool_use" else "end"
    usage = _usage_from_response(response, model, latency_ms)
    _log_call(usage, model, stop)
    return LLMResponse(
        text="".join(text_parts),
        tool_calls=tool_calls,
        stop=stop,
        usage=usage,
    )


def _usage_from_response(response: Any, model: str, latency_ms: int) -> Usage:
    u = response.usage
    cache_read = getattr(u, "cache_read_input_tokens", 0) or 0
    cache_write = getattr(u, "cache_creation_input_tokens", 0) or 0
    cost = _compute_cost(model, u.input_tokens, u.output_tokens, cache_read, cache_write)
    return Usage(
        input_tokens=u.input_tokens,
        output_tokens=u.output_tokens,
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
        cost_usd=cost,
        latency_ms=latency_ms,
    )


def _compute_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read: int,
    cache_write: int,
) -> float:
    prices = PRICES.get(model)
    if prices is None:
        logger.warning("no price table for model %s, logging cost=$0", model)
        return 0.0
    return (
        prices["input"] * input_tokens
        + prices["output"] * output_tokens
        + prices["cache_read"] * cache_read
        + prices["cache_write"] * cache_write
    ) / 1_000_000


def _log_call(usage: Usage, model: str, stop: Stop) -> None:
    logger.info(
        "llm_call model=%s in=%d out=%d cache_r=%d cache_w=%d cost=$%.6f latency=%dms stop=%s",
        model,
        usage.input_tokens,
        usage.output_tokens,
        usage.cache_read_tokens,
        usage.cache_write_tokens,
        usage.cost_usd,
        usage.latency_ms,
        stop,
    )
