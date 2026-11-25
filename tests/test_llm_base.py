from dataclasses import FrozenInstanceError

import pytest

from dobby.llm.base import LLMClient, Transcriber
from dobby.llm.types import LLMResponse, Message, ToolSchema, TranscribeResult, Usage


def test_tool_schema_constructs():
    schema = ToolSchema(
        name="list_events",
        description="Check calendar between two dates.",
        input_schema={"type": "object", "properties": {}, "required": []},
    )
    assert schema.name == "list_events"
    assert schema.description.startswith("Check")


def test_tool_schema_is_frozen():
    schema = ToolSchema(name="x", description="y", input_schema={})
    with pytest.raises(FrozenInstanceError):
        schema.name = "z"


def test_class_with_complete_satisfies_llm_client_protocol():
    class ScriptedClient:
        def complete(
            self,
            system: str,
            messages: list[Message],
            tools: list[ToolSchema],
        ) -> LLMResponse:
            return LLMResponse(
                text="done",
                tool_calls=[],
                stop="end",
                usage=Usage(
                    input_tokens=0,
                    output_tokens=0,
                    cache_read_tokens=0,
                    cache_write_tokens=0,
                    cost_usd=0.0,
                    latency_ms=0,
                ),
            )

    assert isinstance(ScriptedClient(), LLMClient)


def test_class_without_complete_does_not_satisfy_protocol():
    class NotAClient:
        def something_else(self) -> None:
            pass

    assert not isinstance(NotAClient(), LLMClient)


def test_class_with_transcribe_satisfies_transcriber_protocol():
    class ScriptedTranscriber:
        def transcribe(self, data: bytes, mime_type: str) -> TranscribeResult:
            return TranscribeResult(
                is_readable=True,
                transcript="hi",
                reason="",
                usage=Usage(
                    input_tokens=0, output_tokens=0,
                    cache_read_tokens=0, cache_write_tokens=0,
                    cost_usd=0.0, latency_ms=0,
                ),
            )

    assert isinstance(ScriptedTranscriber(), Transcriber)


def test_llm_client_without_transcribe_does_not_satisfy_transcriber():
    """Keeping Transcriber separate from LLMClient — this is the reason."""
    class TextOnlyClient:
        def complete(
            self,
            system: str,
            messages: list[Message],
            tools: list[ToolSchema],
        ) -> LLMResponse:
            raise NotImplementedError

    client = TextOnlyClient()
    assert isinstance(client, LLMClient)
    assert not isinstance(client, Transcriber)
