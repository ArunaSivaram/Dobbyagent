"""Agent loop tests. No network — a scripted LLM client drives everything."""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from datetime import date

from dobby.agent import MAX_STEPS, RunResult, run_agent
from dobby.calendar.base import Event
from dobby.calendar.fake import InMemoryCalendar
from dobby.llm.types import LLMResponse, Message, ToolCall, ToolSchema, Usage

TODAY = date(2026, 10, 29)


def _usage(input_tokens: int = 100, output_tokens: int = 50, cost: float = 0.0001) -> Usage:
    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=0,
        cache_write_tokens=0,
        cost_usd=cost,
        latency_ms=100,
    )


@dataclass
class FakeLLMClient:
    """Pops a pre-scripted response on each call and records what it saw."""

    responses: list[LLMResponse]
    calls: list[tuple[str, list[Message], list[ToolSchema]]] = field(default_factory=list)

    def complete(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSchema],
    ) -> LLMResponse:
        self.calls.append((system, copy.deepcopy(messages), tools))
        return self.responses.pop(0)


def test_no_dates_message_creates_nothing():
    """(e) A date-free message should produce zero create_event calls and one model call."""
    client = FakeLLMClient(
        responses=[
            LLMResponse(
                text="No events to add.",
                tool_calls=[],
                stop="end",
                usage=_usage(),
            )
        ]
    )
    cal = InMemoryCalendar()
    result = run_agent("Thanks for coming yesterday!", cal, TODAY, client)

    assert result.created_events == []
    assert result.reply == "No events to add."
    assert len(result.usages) == 1
    assert cal.list_events(date(2026, 1, 1), date(2027, 12, 31)) == []


def test_tool_result_reaches_next_call_matched_by_id():
    """(a) Each ToolResult id must match the ToolCall id from the previous turn."""
    tc_id = "call_abc123"
    client = FakeLLMClient(
        responses=[
            LLMResponse(
                text="",
                tool_calls=[
                    ToolCall(
                        id=tc_id,
                        name="list_events",
                        input={"start_date": "2026-11-01", "end_date": "2026-11-30"},
                    )
                ],
                stop="tool",
                usage=_usage(),
            ),
            LLMResponse(text="done", tool_calls=[], stop="end", usage=_usage()),
        ]
    )
    cal = InMemoryCalendar(seed=[Event(title="Existing", date=date(2026, 11, 10))])
    result = run_agent("anything", cal, TODAY, client)

    # The second call must have seen a user turn with a matching tool_result.
    _, second_call_messages, _ = client.calls[1]
    last_turn = second_call_messages[-1]
    assert last_turn.role == "user"
    assert len(last_turn.tool_results) == 1
    tr = last_turn.tool_results[0]
    assert tr.id == tc_id
    assert tr.is_error is False
    payload = json.loads(tr.content)
    assert payload[0]["title"] == "Existing"
    assert result.reply == "done"


def test_tool_error_returns_is_error_true():
    """(c) Bad tool input must surface to the model as ToolResult(is_error=True)."""
    client = FakeLLMClient(
        responses=[
            LLMResponse(
                text="",
                tool_calls=[
                    ToolCall(
                        id="t1",
                        name="list_events",
                        input={"start_date": "not-a-date", "end_date": "2026-11-30"},
                    )
                ],
                stop="tool",
                usage=_usage(),
            ),
            LLMResponse(text="failed", tool_calls=[], stop="end", usage=_usage()),
        ]
    )
    result = run_agent("x", InMemoryCalendar(), TODAY, client)

    _, second_call_messages, _ = client.calls[1]
    tr = second_call_messages[-1].tool_results[0]
    assert tr.is_error is True
    assert "ValueError" in tr.content
    assert isinstance(result, RunResult)


def test_max_steps_stops_the_loop():
    """(b) After MAX_STEPS calls the loop must stop even if the model keeps asking for tools."""
    looping_response = LLMResponse(
        text="",
        tool_calls=[
            ToolCall(
                id="loop",
                name="list_events",
                input={"start_date": "2026-11-01", "end_date": "2026-11-02"},
            )
        ],
        stop="tool",
        usage=_usage(),
    )
    client = FakeLLMClient(responses=[looping_response] * MAX_STEPS)
    result = run_agent("x", InMemoryCalendar(), TODAY, client)

    assert len(client.calls) == MAX_STEPS
    assert len(result.usages) == MAX_STEPS
    assert "stopped" in result.reply.lower()


def test_create_event_is_called_and_recorded():
    """A full create flow: list → create → end. Verifies created_events is populated."""
    client = FakeLLMClient(
        responses=[
            LLMResponse(
                text="",
                tool_calls=[
                    ToolCall(
                        id="t1",
                        name="list_events",
                        input={"start_date": "2026-11-01", "end_date": "2026-11-30"},
                    )
                ],
                stop="tool",
                usage=_usage(),
            ),
            LLMResponse(
                text="",
                tool_calls=[
                    ToolCall(
                        id="t2",
                        name="create_event",
                        input={
                            "title": "Sports Day",
                            "date": "2026-11-14",
                            "start_time": "08:30",
                            "end_time": "12:00",
                            "location": "School field",
                        },
                    )
                ],
                stop="tool",
                usage=_usage(),
            ),
            LLMResponse(text="Added Sports Day.", tool_calls=[], stop="end", usage=_usage()),
        ]
    )
    cal = InMemoryCalendar()
    result = run_agent("Sports Day on Fri 14 Nov 8.30am to 12pm", cal, TODAY, client)

    assert len(result.created_events) == 1
    event = result.created_events[0]
    assert event.title == "Sports Day"
    assert event.date == date(2026, 11, 14)
    assert event.location == "School field"
    assert len(result.usages) == 3
    assert result.reply == "Added Sports Day."


def test_total_cost_sums_usages():
    """(d) total_cost_usd should equal the sum of per-call usages."""
    result = RunResult(
        reply="x",
        created_events=[],
        usages=[
            Usage(1, 1, 0, 0, cost_usd=0.000123, latency_ms=10),
            Usage(1, 1, 0, 0, cost_usd=0.000456, latency_ms=10),
            Usage(1, 1, 0, 0, cost_usd=0.000789, latency_ms=10),
        ],
    )
    assert result.total_cost_usd == 0.000123 + 0.000456 + 0.000789


def test_system_prompt_mentions_today_and_treats_message_as_data():
    """The system prompt must inject today's date + weekday and the data-not-instructions rule."""
    client = FakeLLMClient(
        responses=[LLMResponse(text="ok", tool_calls=[], stop="end", usage=_usage())]
    )
    run_agent("hi", InMemoryCalendar(), TODAY, client)

    system, _, _ = client.calls[0]
    assert "2026-10-29" in system
    assert "Thursday" in system  # TODAY is a Thursday
    assert "data" in system.lower()
