"""Tests for the Anthropic adapter — no network.

We test the pure conversion helpers and the cost math; the SDK is never called.
"""
from __future__ import annotations

from types import SimpleNamespace

from dobby.llm.anthropic_client import (
    _compute_cost,
    _from_claude_response,
    _to_claude_messages,
    _to_claude_tools,
)
from dobby.llm.types import Message, ToolCall, ToolResult, ToolSchema


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
