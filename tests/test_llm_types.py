from dataclasses import FrozenInstanceError

import pytest

from dobby.llm.types import LLMResponse, Message, ToolCall, ToolResult, Usage


def test_tool_call_is_frozen():
    tc = ToolCall(id="t1", name="create_event", input={"title": "Sports Day"})
    with pytest.raises(FrozenInstanceError):
        tc.id = "t2"


def test_tool_result_defaults_is_error_false():
    tr = ToolResult(id="t1", content="ok")
    assert tr.is_error is False


def test_message_defaults_have_empty_lists():
    m = Message(role="user", text="hi")
    assert m.tool_calls == []
    assert m.tool_results == []


def test_llm_response_roundtrip_fields():
    usage = Usage(
        input_tokens=100,
        output_tokens=50,
        cache_read_tokens=0,
        cache_write_tokens=0,
        cost_usd=0.0001,
        latency_ms=320,
    )
    resp = LLMResponse(text="done", tool_calls=[], stop="end", usage=usage)
    assert resp.stop == "end"
    assert resp.usage.cost_usd == 0.0001
