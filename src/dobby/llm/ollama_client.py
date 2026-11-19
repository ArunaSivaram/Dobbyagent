"""Ollama adapter — talks to the OpenAI-compatible endpoint on localhost:11434.

Only this file knows about OpenAI's content format. The agent and the evals
see the same neutral LLMClient contract as the Anthropic adapter.

No new dependencies: stdlib `urllib.request` for a single JSON POST. Cost is
always $0 because the model runs on the user's Mac.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any

from dobby.config import OLLAMA_BASE_URL, OLLAMA_MODEL
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

HTTP_TIMEOUT_S = 120  # local inference; may take a while on first call


class OllamaClient:
    """Satisfies the LLMClient protocol by POSTing to /v1/chat/completions."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
    ) -> None:
        self._base_url = (base_url or OLLAMA_BASE_URL).rstrip("/")
        self._model = model or OLLAMA_MODEL
        self._temperature = temperature

    def complete(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSchema],
    ) -> LLMResponse:
        body: dict[str, Any] = {
            "model": self._model,
            "messages": _to_openai_messages(system, messages),
            "tools": _to_openai_tools(tools),
        }
        if self._temperature is not None:
            body["temperature"] = self._temperature

        start = time.monotonic()
        response_json = self._post(f"{self._base_url}/chat/completions", body)
        latency_ms = int((time.monotonic() - start) * 1000)

        return _from_openai_response(response_json, self._model, latency_ms)

    def _post(self, url: str, body: dict[str, Any]) -> dict[str, Any]:
        """Overridable hook for tests. Returns the parsed JSON response."""
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            raise RuntimeError(
                f"Ollama returned HTTP {e.code} at {url}: {e.read().decode()!r}"
            ) from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Could not reach Ollama at {url}. Is the app running? ({e.reason})"
            ) from e


# --- conversion: neutral → OpenAI ----------------------------------------

def _to_openai_messages(system: str, messages: list[Message]) -> list[dict[str, Any]]:
    """Flatten neutral messages into OpenAI's one-dict-per-turn format."""
    out: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for m in messages:
        if m.tool_results:
            # OpenAI requires one "tool" message per result, not a single
            # user turn with multiple content blocks (that's Anthropic's shape).
            for tr in m.tool_results:
                out.append(_openai_tool_result(tr))
            continue
        if m.tool_calls:
            out.append(_openai_assistant_with_tool_calls(m))
            continue
        out.append({"role": m.role, "content": m.text or ""})
    return out


def _openai_tool_result(tr: ToolResult) -> dict[str, Any]:
    content = tr.content
    if tr.is_error:
        # Flag errors inline; OpenAI's "tool" role has no `is_error` field.
        content = f"[ERROR] {content}"
    return {"role": "tool", "tool_call_id": tr.id, "content": content}


def _openai_assistant_with_tool_calls(m: Message) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": m.text or "",
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.input),  # OpenAI wants a JSON string here
                },
            }
            for tc in m.tool_calls
        ],
    }


def _to_openai_tools(tools: list[ToolSchema]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema,
            },
        }
        for t in tools
    ]


# --- conversion: OpenAI → neutral ----------------------------------------

def _from_openai_response(
    response: dict[str, Any], model: str, latency_ms: int
) -> LLMResponse:
    choices = response.get("choices") or []
    if not choices:
        raise RuntimeError(f"Ollama returned no choices: {response!r}")
    choice = choices[0]
    msg = choice.get("message") or {}

    text = msg.get("content") or ""
    tool_calls = [_tool_call_from(tc) for tc in (msg.get("tool_calls") or [])]

    finish = choice.get("finish_reason", "stop")
    stop: Stop = "tool" if finish == "tool_calls" else "end"

    # Some small models emit tool_calls alongside finish_reason="stop". Trust
    # the presence of tool_calls — the agent needs to execute them either way.
    if tool_calls and stop == "end":
        stop = "tool"

    usage = _usage_from(response, latency_ms)
    _log_call(usage, model, stop)
    return LLMResponse(text=text, tool_calls=tool_calls, stop=stop, usage=usage)


def _tool_call_from(raw: dict[str, Any]) -> ToolCall:
    fn = raw.get("function", {})
    args_raw = fn.get("arguments", "{}")
    # OpenAI spec says `arguments` is a JSON string, but some models emit a dict.
    if isinstance(args_raw, str):
        try:
            args = json.loads(args_raw) if args_raw else {}
        except json.JSONDecodeError:
            # Pass the broken payload through; the agent surfaces errors as
            # ToolResult(is_error=True) when the tool fails to execute.
            args = {"_raw": args_raw}
    else:
        args = dict(args_raw)
    return ToolCall(id=raw.get("id", ""), name=fn.get("name", ""), input=args)


def _usage_from(response: dict[str, Any], latency_ms: int) -> Usage:
    u = response.get("usage") or {}
    return Usage(
        input_tokens=int(u.get("prompt_tokens", 0)),
        output_tokens=int(u.get("completion_tokens", 0)),
        cache_read_tokens=0,
        cache_write_tokens=0,
        cost_usd=0.0,  # local inference — free
        latency_ms=latency_ms,
    )


def _log_call(usage: Usage, model: str, stop: Stop) -> None:
    logger.info(
        "llm_call model=%s in=%d out=%d cost=$0 latency=%dms stop=%s (ollama)",
        model,
        usage.input_tokens,
        usage.output_tokens,
        usage.latency_ms,
        stop,
    )
