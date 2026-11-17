"""The contract every LLM adapter must satisfy. The agent only sees this."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from dobby.llm.types import LLMResponse, Message, ToolSchema


@runtime_checkable
class LLMClient(Protocol):
    """Any class with a matching `complete` method is an LLMClient."""

    def complete(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSchema],
    ) -> LLMResponse: ...
