"""Neutral types shared by the agent, the LLM adapters, and the tool side.

Both sides speak this vocabulary; neither side imports the other.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant"]
Stop = Literal["end", "tool"]


@dataclass(frozen=True)
class ToolCall:
    """A model's request to run one tool. `id` pairs the call to its result."""

    id: str
    name: str
    input: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    """The outcome of running one tool. `id` matches the originating ToolCall."""

    id: str
    content: str
    is_error: bool = False


@dataclass(frozen=True)
class ToolSchema:
    """Neutral description of one tool. Adapters translate it to each provider's format."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class Message:
    """One turn in the conversation.

    - A user text turn uses `text`.
    - An assistant turn may have `text` and/or `tool_calls`.
    - A user tool-results turn uses `tool_results` (one per preceding ToolCall).
    """

    role: Role
    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)


@dataclass(frozen=True)
class Usage:
    """Token counts and dollar cost for one model call."""

    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    cost_usd: float
    latency_ms: int


@dataclass(frozen=True)
class LLMResponse:
    """One model turn, as the agent sees it.

    `stop == "tool"` means the agent must run the tool_calls and loop.
    `stop == "end"` means this is the final answer.
    """

    text: str
    tool_calls: list[ToolCall]
    stop: Stop
    usage: Usage


@dataclass(frozen=True)
class TranscribeResult:
    """Outcome of asking the LLM to transcribe an image or PDF to plain text.

    `is_readable=False` means the model couldn't extract event-relevant text —
    the caller should ask the user for a clearer version rather than running
    the agent on an empty transcript.
    """

    is_readable: bool
    transcript: str
    reason: str
    usage: Usage
