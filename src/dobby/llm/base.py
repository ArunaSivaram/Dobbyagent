"""The contract every LLM adapter must satisfy. The agent only sees this."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from dobby.llm.types import LLMResponse, Message, ToolSchema, TranscribeResult


@runtime_checkable
class LLMClient(Protocol):
    """Any class with a matching `complete` method is an LLMClient."""

    def complete(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSchema],
    ) -> LLMResponse: ...


@runtime_checkable
class Transcriber(Protocol):
    """A capability — turn image or PDF bytes into plain text we can feed the agent.

    Kept separate from LLMClient because not every backend can do this
    (e.g. OllamaClient stays text-only).
    """

    def transcribe(self, data: bytes, mime_type: str) -> TranscribeResult: ...
