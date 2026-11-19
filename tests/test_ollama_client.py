"""Tests for the Ollama adapter — no network.

We test the pure conversion helpers and the full `complete` call via a
monkey-patched `_post` that returns pre-recorded JSON.
"""
from __future__ import annotations

import json

from dobby.llm.ollama_client import (
    OllamaClient,
    _from_openai_response,
    _to_openai_messages,
    _to_openai_tools,
)
from dobby.llm.types import Message, ToolCall, ToolResult, ToolSchema

# --- neutral → OpenAI -----------------------------------------------------

def test_system_prompt_is_prepended_as_message():
    out = _to_openai_messages("You are Dobby.", [])
    assert out == [{"role": "system", "content": "You are Dobby."}]


def test_user_text_becomes_user_message():
    msgs = [Message(role="user", text="Sports day on 14 Nov")]
    out = _to_openai_messages("sys", msgs)
    assert out[1] == {"role": "user", "content": "Sports day on 14 Nov"}


def test_assistant_tool_calls_stringify_arguments_as_json():
    """OpenAI's tool_call.function.arguments is a JSON STRING, not a dict."""
    msgs = [
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
    out = _to_openai_messages("sys", msgs)
    assert out[1]["role"] == "assistant"
    assert out[1]["content"] == "checking the calendar"
    assert len(out[1]["tool_calls"]) == 1
    tc = out[1]["tool_calls"][0]
    assert tc["id"] == "tc_1"
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "list_events"
    # Arguments must be a string; json.loads gets us back to the dict.
    args = json.loads(tc["function"]["arguments"])
    assert args == {"start_date": "2026-11-01", "end_date": "2026-11-30"}


def test_tool_results_become_one_tool_message_each():
    """Unlike Anthropic (one user turn with blocks), OpenAI needs one 'tool'
    message per tool_call result."""
    msgs = [
        Message(
            role="user",
            tool_results=[
                ToolResult(id="tc_1", content="[]"),
                ToolResult(id="tc_2", content="ValueError: bad date", is_error=True),
            ],
        )
    ]
    out = _to_openai_messages("sys", msgs)
    # Skip system, we expect TWO tool messages.
    assert out[1] == {"role": "tool", "tool_call_id": "tc_1", "content": "[]"}
    # is_error is flagged inline since OpenAI has no is_error field.
    assert out[2]["role"] == "tool"
    assert out[2]["tool_call_id"] == "tc_2"
    assert out[2]["content"].startswith("[ERROR]")


def test_tool_schemas_convert_to_openai_function_shape():
    tools = [
        ToolSchema(
            name="list_events",
            description="List events.",
            input_schema={"type": "object", "properties": {}, "required": []},
        )
    ]
    assert _to_openai_tools(tools) == [
        {
            "type": "function",
            "function": {
                "name": "list_events",
                "description": "List events.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }
    ]


# --- OpenAI → neutral -----------------------------------------------------

def test_text_only_response_converts_to_stop_end():
    raw = {
        "choices": [{"message": {"content": "all done"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 50, "completion_tokens": 10},
    }
    out = _from_openai_response(raw, "qwen3:8b", latency_ms=80)
    assert out.text == "all done"
    assert out.stop == "end"
    assert out.tool_calls == []
    assert out.usage.input_tokens == 50
    assert out.usage.output_tokens == 10
    assert out.usage.cost_usd == 0.0  # local inference is free
    assert out.usage.latency_ms == 80


def test_tool_calls_response_converts_to_stop_tool_and_parses_args():
    raw = {
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_xyz",
                            "type": "function",
                            "function": {
                                "name": "list_events",
                                "arguments": '{"start_date":"2026-11-01","end_date":"2026-11-30"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20},
    }
    out = _from_openai_response(raw, "qwen3:8b", latency_ms=200)
    assert out.stop == "tool"
    assert len(out.tool_calls) == 1
    tc = out.tool_calls[0]
    assert tc.id == "call_xyz"
    assert tc.name == "list_events"
    assert tc.input == {"start_date": "2026-11-01", "end_date": "2026-11-30"}


def test_tool_calls_with_finish_stop_still_treated_as_tool():
    """Small models sometimes emit tool_calls alongside finish_reason='stop'.
    Presence of tool_calls wins — the agent needs to execute them either way."""
    raw = {
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "x",
                            "type": "function",
                            "function": {"name": "list_events", "arguments": "{}"},
                        }
                    ],
                },
                "finish_reason": "stop",  # buggy model: should have been "tool_calls"
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 5},
    }
    out = _from_openai_response(raw, "qwen3:8b", latency_ms=50)
    assert out.stop == "tool"
    assert len(out.tool_calls) == 1


def test_tool_call_arguments_as_dict_also_handled():
    """Some Ollama models skip the JSON-stringification and emit a dict."""
    raw = {
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "x",
                            "type": "function",
                            "function": {
                                "name": "list_events",
                                "arguments": {"start_date": "2026-11-01", "end_date": "2026-11-30"},
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 5},
    }
    out = _from_openai_response(raw, "qwen3:8b", latency_ms=50)
    assert out.tool_calls[0].input == {"start_date": "2026-11-01", "end_date": "2026-11-30"}


def test_malformed_arguments_do_not_crash():
    """If a model emits garbage in `arguments`, we keep it as _raw so the
    agent surfaces the eventual tool failure via is_error — not a crash here."""
    raw = {
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "x",
                            "type": "function",
                            "function": {"name": "list_events", "arguments": "not json"},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }
    out = _from_openai_response(raw, "qwen3:8b", latency_ms=1)
    assert out.tool_calls[0].input == {"_raw": "not json"}


def test_usage_cost_is_always_zero_even_with_large_token_counts():
    raw = {
        "choices": [{"message": {"content": "x"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 100_000, "completion_tokens": 50_000},
    }
    out = _from_openai_response(raw, "qwen3:8b", latency_ms=1)
    assert out.usage.cost_usd == 0.0


# --- full `complete` via monkey-patched _post ----------------------------

def test_complete_full_round_trip_with_fake_transport(monkeypatch):
    """Mock _post so no HTTP is made; verify the request body and response."""
    captured: dict[str, object] = {}

    def fake_post(self, url, body):
        captured["url"] = url
        captured["body"] = body
        return {
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3},
        }

    monkeypatch.setattr(OllamaClient, "_post", fake_post)
    client = OllamaClient(
        base_url="http://fake:11434/v1", model="qwen3:8b", temperature=0.0
    )
    response = client.complete(
        system="sys",
        messages=[Message(role="user", text="hello")],
        tools=[ToolSchema(name="list_events", description="x", input_schema={})],
    )

    assert response.text == "hi"
    assert response.stop == "end"
    assert response.usage.cost_usd == 0.0
    assert captured["url"] == "http://fake:11434/v1/chat/completions"
    body = captured["body"]
    assert body["model"] == "qwen3:8b"
    assert body["temperature"] == 0.0
    assert body["messages"][0] == {"role": "system", "content": "sys"}
    assert body["messages"][1] == {"role": "user", "content": "hello"}
    assert body["tools"][0]["type"] == "function"


def test_complete_omits_temperature_when_none():
    captured: dict[str, object] = {}

    def fake_post(self, url, body):
        captured["body"] = body
        return {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

    client = OllamaClient(base_url="http://fake/v1", model="qwen3:8b")
    client._post = fake_post.__get__(client, OllamaClient)  # type: ignore[attr-defined]
    client.complete("sys", [], [])

    assert "temperature" not in captured["body"]
