"""Tests for the pure Telegram handlers. No bot, no network — fakes only."""
from __future__ import annotations

from datetime import date, time

from dobby.calendar.fake import InMemoryCalendar
from dobby.llm.types import LLMResponse, ToolCall, Usage
from dobby.telegram.handlers import handle_text, handle_undo, handle_upcoming

TODAY = date(2026, 11, 20)


# ---------- handle_upcoming ----------


def test_upcoming_empty_calendar_mentions_window():
    cal = InMemoryCalendar()
    reply = handle_upcoming(cal, TODAY)
    assert "7 days" in reply
    assert "No events" in reply


def test_upcoming_lists_events_in_date_order():
    cal = InMemoryCalendar()
    cal.create_event(title="Soccer", date=date(2026, 11, 22), start_time=time(10, 0))
    cal.create_event(title="School play", date=date(2026, 11, 25), start_time=time(18, 0))
    cal.create_event(title="DEADLINE: form", date=date(2026, 11, 21))
    reply = handle_upcoming(cal, TODAY)
    lines = reply.splitlines()
    assert lines[0].startswith("Next 7 days")
    deadline_idx = next(i for i, l in enumerate(lines) if "DEADLINE: form" in l)
    soccer_idx = next(i for i, l in enumerate(lines) if "Soccer" in l)
    play_idx = next(i for i, l in enumerate(lines) if "School play" in l)
    assert deadline_idx < soccer_idx < play_idx


def test_upcoming_filters_out_events_past_window():
    cal = InMemoryCalendar()
    cal.create_event(title="Near", date=date(2026, 11, 22))
    cal.create_event(title="Far", date=date(2026, 12, 20))  # 30 days away
    reply = handle_upcoming(cal, TODAY)
    assert "Near" in reply
    assert "Far" not in reply


def test_upcoming_formats_timed_event_with_time():
    cal = InMemoryCalendar()
    cal.create_event(title="Soccer", date=date(2026, 11, 22), start_time=time(10, 30))
    reply = handle_upcoming(cal, TODAY)
    assert "10:30" in reply
    assert "Soccer" in reply


def test_upcoming_includes_location_when_set():
    cal = InMemoryCalendar()
    cal.create_event(title="Play", date=date(2026, 11, 22), location="Main Hall")
    reply = handle_upcoming(cal, TODAY)
    assert "Main Hall" in reply


# ---------- handle_undo ----------


def test_undo_empty_state_is_a_no_op():
    cal = InMemoryCalendar()
    reply = handle_undo(cal, [])
    assert reply == "Nothing to undo."


def test_undo_deletes_and_reports_count():
    cal = InMemoryCalendar()
    e1 = cal.create_event(title="A", date=date(2026, 11, 22))
    e2 = cal.create_event(title="B", date=date(2026, 11, 23))
    assert e1.id is not None and e2.id is not None

    reply = handle_undo(cal, [e1.id, e2.id])
    assert "2" in reply
    assert cal.list_events(date(2026, 1, 1), date(2027, 1, 1)) == []


def test_undo_surfaces_partial_failure():
    class HalfBrokenCalendar(InMemoryCalendar):
        def delete_event(self, event_id: str) -> None:
            if event_id == "bad":
                raise RuntimeError("gone")
            super().delete_event(event_id)

    cal = HalfBrokenCalendar()
    e1 = cal.create_event(title="A", date=date(2026, 11, 22))
    assert e1.id is not None

    reply = handle_undo(cal, [e1.id, "bad"])
    assert "1" in reply
    assert "failed" in reply.lower()


# ---------- handle_text ----------


def _zero_usage() -> Usage:
    return Usage(
        input_tokens=0,
        output_tokens=0,
        cache_read_tokens=0,
        cache_write_tokens=0,
        cost_usd=0.0,
        latency_ms=0,
    )


class _ScriptedClient:
    """LLM client that returns a pre-scripted sequence of responses."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)

    def complete(self, system, messages, tools):
        return self._responses.pop(0)


def test_handle_text_returns_reply_and_created_event_ids():
    create_call = ToolCall(
        id="tc1",
        name="create_event",
        input={"title": "Soccer", "date": "2026-11-22", "start_time": "10:00"},
    )
    list_call = ToolCall(
        id="tc0",
        name="list_events",
        input={"start_date": "2026-11-20", "end_date": "2026-11-30"},
    )
    responses = [
        LLMResponse(text="", stop="tool", tool_calls=[list_call], usage=_zero_usage()),
        LLMResponse(text="", stop="tool", tool_calls=[create_call], usage=_zero_usage()),
        LLMResponse(text="Added Soccer on Sat Nov 22 at 10:00.", stop="end", tool_calls=[], usage=_zero_usage()),
    ]
    cal = InMemoryCalendar()

    reply, ids = handle_text("soccer saturday 10am", cal, TODAY, _ScriptedClient(responses))

    assert "Soccer" in reply
    assert len(ids) == 1
    assert ids[0]
    assert cal.list_events(date(2026, 11, 22), date(2026, 11, 22))[0].title == "Soccer"


def test_handle_text_with_no_creates_returns_empty_ids():
    responses = [
        LLMResponse(
            text="No dates mentioned, nothing to add.", stop="end", tool_calls=[], usage=_zero_usage()
        ),
    ]
    cal = InMemoryCalendar()

    reply, ids = handle_text("just saying hi", cal, TODAY, _ScriptedClient(responses))

    assert "nothing to add" in reply.lower()
    assert ids == []
