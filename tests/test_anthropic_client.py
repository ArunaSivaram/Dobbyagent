"""Tests for the Anthropic adapter — no network.

We test the pure conversion helpers and the cost math; the SDK is never called.
"""
from __future__ import annotations

import base64
from types import SimpleNamespace

import pytest

from dobby.llm.anthropic_client import (
    AnthropicClient,
    _compute_cost,
    _from_claude_response,
    _parse_transcribe_response,
    _to_claude_messages,
    _to_claude_tools,
    _to_file_block,
)
from dobby.llm.types import Message, ToolCall, ToolResult, ToolSchema, Usage


def test_user_text_message_converts_to_plain_content():
    messages = [Message(role="user", text="Sports Day on 14 Nov")]
    claude = _to_claude_messages(messages)
    assert claude == [{"role": "user", "content": "Sports Day on 14 Nov"}]


def test_assistant_tool_call_message_converts_to_content_blocks():
    messages = [
        Message(
            role="assistant",
            text="checking the calendar",
            tool_calls=[
                ToolCall(
                    id="tc_1",
                    name="list_events",
                    input={"start_date": "2026-11-01", "end_date": "2026-11-30"},
                )
            ],
        )
    ]
    claude = _to_claude_messages(messages)
    assert claude == [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "checking the calendar"},
                {
                    "type": "tool_use",
                    "id": "tc_1",
                    "name": "list_events",
                    "input": {"start_date": "2026-11-01", "end_date": "2026-11-30"},
                },
            ],
        }
    ]


def test_user_tool_results_message_converts_to_tool_result_blocks():
    messages = [
        Message(
            role="user",
            tool_results=[
                ToolResult(id="tc_1", content="[]"),
                ToolResult(id="tc_2", content="ValueError: bad date", is_error=True),
            ],
        )
    ]
    claude = _to_claude_messages(messages)
    assert claude == [
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "tc_1", "content": "[]"},
                {
                    "type": "tool_result",
                    "tool_use_id": "tc_2",
                    "content": "ValueError: bad date",
                    "is_error": True,
                },
            ],
        }
    ]


def test_tool_schemas_convert_to_claude_shape():
    tools = [
        ToolSchema(
            name="list_events",
            description="list",
            input_schema={"type": "object", "properties": {}, "required": []},
        )
    ]
    assert _to_claude_tools(tools) == [
        {
            "name": "list_events",
            "description": "list",
            "input_schema": {"type": "object", "properties": {}, "required": []},
        }
    ]


def test_claude_response_with_tool_use_converts_to_llm_response():
    response = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="let me check"),
            SimpleNamespace(
                type="tool_use",
                id="tc_1",
                name="list_events",
                input={"start_date": "2026-11-01", "end_date": "2026-11-30"},
            ),
        ],
        stop_reason="tool_use",
        usage=SimpleNamespace(
            input_tokens=1000,
            output_tokens=500,
            cache_read_input_tokens=200,
            cache_creation_input_tokens=100,
        ),
    )
    out = _from_claude_response(response, "claude-haiku-4-5-20251001", latency_ms=321)

    assert out.text == "let me check"
    assert out.stop == "tool"
    assert len(out.tool_calls) == 1
    tc = out.tool_calls[0]
    assert tc.id == "tc_1"
    assert tc.name == "list_events"
    assert tc.input == {"start_date": "2026-11-01", "end_date": "2026-11-30"}
    assert out.usage.input_tokens == 1000
    assert out.usage.output_tokens == 500
    assert out.usage.cache_read_tokens == 200
    assert out.usage.cache_write_tokens == 100
    assert out.usage.latency_ms == 321


def test_claude_response_with_end_turn_converts_to_stop_end():
    response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="all done")],
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )
    out = _from_claude_response(response, "claude-haiku-4-5-20251001", latency_ms=50)
    assert out.stop == "end"
    assert out.text == "all done"
    assert out.tool_calls == []
    # Cache fields absent in the SDK response should default to 0.
    assert out.usage.cache_read_tokens == 0
    assert out.usage.cache_write_tokens == 0


def test_cost_math_matches_hand_calculation():
    """(d) Haiku 4.5 priced per PRD: 0.10 / 0.50 / 0.01 / 0.125 per million tokens.

    Hand calc: (0.10*1000 + 0.50*500 + 0.01*200 + 0.125*100) / 1_000_000
             = (100 + 250 + 2 + 12.5) / 1_000_000
             = 0.0003645
    """
    cost = _compute_cost(
        model="claude-haiku-4-5-20251001",
        input_tokens=1000,
        output_tokens=500,
        cache_read=200,
        cache_write=100,
    )
    assert cost == 364.5 / 1_000_000


def test_cost_math_unknown_model_returns_zero():
    cost = _compute_cost(
        model="claude-unknown",
        input_tokens=1000,
        output_tokens=500,
        cache_read=0,
        cache_write=0,
    )
    assert cost == 0.0


# --- transcribe ----------------------------------------------------------


def _usage(cost: float = 0.0) -> Usage:
    return Usage(
        input_tokens=0, output_tokens=0,
        cache_read_tokens=0, cache_write_tokens=0,
        cost_usd=cost, latency_ms=0,
    )


def test_file_block_builds_image_block_for_jpeg():
    block = _to_file_block(b"\xff\xd8\xff", "image/jpeg")
    assert block["type"] == "image"
    assert block["source"]["media_type"] == "image/jpeg"
    assert block["source"]["data"] == base64.b64encode(b"\xff\xd8\xff").decode("ascii")


def test_file_block_builds_document_block_for_pdf():
    block = _to_file_block(b"%PDF-1.4", "application/pdf")
    assert block["type"] == "document"
    assert block["source"]["media_type"] == "application/pdf"
    assert block["source"]["data"] == base64.b64encode(b"%PDF-1.4").decode("ascii")


def test_file_block_rejects_unsupported_mime():
    with pytest.raises(ValueError, match="unsupported mime type"):
        _to_file_block(b"...", "application/zip")


def test_parse_transcribe_response_with_readable_json():
    raw = '{"is_readable": true, "transcript": "PTA meeting 10 Nov 7pm", "reason": ""}'
    result = _parse_transcribe_response(raw, _usage())
    assert result.is_readable is True
    assert result.transcript == "PTA meeting 10 Nov 7pm"
    assert result.reason == ""


def test_parse_transcribe_response_with_unreadable_json():
    raw = '{"is_readable": false, "transcript": "", "reason": "image too blurry"}'
    result = _parse_transcribe_response(raw, _usage())
    assert result.is_readable is False
    assert result.transcript == ""
    assert result.reason == "image too blurry"


def test_parse_transcribe_response_strips_code_fences():
    raw = '```json\n{"is_readable": true, "transcript": "hi", "reason": ""}\n```'
    result = _parse_transcribe_response(raw, _usage())
    assert result.is_readable is True
    assert result.transcript == "hi"


def test_parse_transcribe_response_falls_back_when_not_json():
    raw = "Sure! Here's the transcript: PTA meeting on 10 Nov..."
    result = _parse_transcribe_response(raw, _usage())
    assert result.is_readable is False
    assert "JSON" in result.reason


def test_transcribe_sends_image_block_and_returns_parsed_result():
    """Integration-ish: AnthropicClient.transcribe with a stub Anthropic SDK."""
    captured: dict[str, object] = {}

    class _StubMessages:
        def create(self, **kwargs: object) -> SimpleNamespace:
            captured.update(kwargs)
            return SimpleNamespace(
                content=[SimpleNamespace(
                    type="text",
                    text='{"is_readable": true, "transcript": "hello world", "reason": ""}',
                )],
                usage=SimpleNamespace(input_tokens=20, output_tokens=5),
            )

    client = AnthropicClient(api_key="unused", model="claude-haiku-4-5-20251001")
    client._client = SimpleNamespace(messages=_StubMessages())

    result = client.transcribe(b"\xff\xd8\xff", "image/jpeg")

    assert result.is_readable is True
    assert result.transcript == "hello world"
    assert result.usage.input_tokens == 20
    assert result.usage.output_tokens == 5
    # The outgoing request included our image block as the first content item.
    user_content = captured["messages"][0]["content"]
    assert user_content[0]["type"] == "image"
    assert user_content[0]["source"]["media_type"] == "image/jpeg"
