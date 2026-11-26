"""Tests for the pure Telegram handlers. No bot, no network — fakes only."""
from __future__ import annotations

from datetime import date, time
from io import BytesIO

from pypdf import PdfWriter

from dobby.calendar.base import Event
from dobby.calendar.fake import InMemoryCalendar
from dobby.llm.types import LLMResponse, ToolCall, TranscribeResult, Usage
from dobby.telegram.handlers import (
    MAX_IMAGE_BYTES,
    commit_pending,
    handle_undo,
    handle_upcoming,
    plan_from_file,
    plan_from_text,
)
from dobby.telegram.session import PendingEvent

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


# ---------- plan_from_text + commit_pending ----------


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


def test_plan_from_text_returns_pending_events_without_writing_to_calendar():
    create_call = ToolCall(
        id="tc1",
        name="create_event",
        input={"title": "Soccer", "date": "2026-11-22", "start_time": "10:00"},
    )
    responses = [
        LLMResponse(text="", stop="tool", tool_calls=[create_call], usage=_zero_usage()),
        LLMResponse(text="Added.", stop="end", tool_calls=[], usage=_zero_usage()),
    ]
    cal = InMemoryCalendar()

    pending = plan_from_text("soccer saturday 10am", cal, TODAY, _ScriptedClient(responses))

    assert len(pending) == 1
    assert pending[0].draft.title == "Soccer"
    # Crucially: the real calendar is still empty — nothing was committed.
    assert cal.list_events(date(2026, 1, 1), date(2027, 1, 1)) == []


def test_plan_from_text_with_no_events_returns_empty_list():
    responses = [
        LLMResponse(
            text="No dates mentioned.", stop="end", tool_calls=[], usage=_zero_usage()
        ),
    ]
    cal = InMemoryCalendar()

    pending = plan_from_text("just saying hi", cal, TODAY, _ScriptedClient(responses))

    assert pending == []


def test_plan_from_text_passes_real_calendar_to_agent_for_list_events():
    """Duplicate-check reads must hit the real calendar, not an empty staging."""
    captured_tool_results: list[str] = []

    class _RecordingClient:
        def __init__(self, responses: list[LLMResponse]) -> None:
            self._responses = list(responses)

        def complete(self, system, messages, tools):
            # Capture what tool_results came back on each turn.
            for m in messages:
                for tr in m.tool_results:
                    captured_tool_results.append(tr.content)
            return self._responses.pop(0)

    list_call = ToolCall(
        id="tc0", name="list_events",
        input={"start_date": "2026-11-01", "end_date": "2026-11-30"},
    )
    responses = [
        LLMResponse(text="", stop="tool", tool_calls=[list_call], usage=_zero_usage()),
        LLMResponse(text="Already there.", stop="end", tool_calls=[], usage=_zero_usage()),
    ]
    cal = InMemoryCalendar()
    cal.create_event(title="Already booked", date=date(2026, 11, 22))

    plan_from_text("does soccer exist?", cal, TODAY, _RecordingClient(responses))

    # The agent's list_events tool call should have seen the real calendar.
    assert any("Already booked" in r for r in captured_tool_results)


def test_commit_pending_prepends_category_and_resolves_duration():
    draft = _draft_event(title="Weekly sync", start_time=time(10, 0))
    pending = PendingEvent(
        draft=draft, category="Professional", event_type="Meeting", duration_hours=2,
    )
    cal = InMemoryCalendar()

    event_id = commit_pending(pending, cal)

    assert event_id
    [created] = cal.list_events(draft.date, draft.date)
    assert created.title == "Professional : Weekly sync"
    assert created.start_time == time(10, 0)
    assert created.end_time == time(12, 0)  # 10:00 + 2h


def test_commit_pending_defaults_to_9am_when_draft_has_no_time():
    """Per Module 5 spec: no time mentioned → commit at 9 AM with duration."""
    draft = _draft_event(title="PTA meeting", start_time=None)
    pending = PendingEvent(draft=draft, category="Family", duration_hours=1)
    cal = InMemoryCalendar()

    commit_pending(pending, cal)

    [created] = cal.list_events(draft.date, draft.date)
    assert created.title == "Family : PTA meeting"
    assert created.start_time == time(9, 0)
    assert created.end_time == time(10, 0)


def test_commit_pending_keeps_multi_day_events_all_day():
    """Multi-day drafts stay all-day; duration_hours is ignored."""
    draft = Event(
        title="Camp", date=date(2026, 11, 16), end_date=date(2026, 11, 18),
        start_time=None,
    )
    pending = PendingEvent(draft=draft, category="Family", duration_hours=5)
    cal = InMemoryCalendar()

    commit_pending(pending, cal)

    [created] = cal.list_events(draft.date, draft.date)
    assert created.start_time is None
    assert created.end_time is None
    assert created.end_date == date(2026, 11, 18)


def _draft_event(title: str, start_time: time | None) -> Event:
    return Event(title=title, date=date(2026, 11, 22), start_time=start_time)


# ---------- plan_from_file ----------


class _ScriptedTranscriber:
    """Transcriber that returns a pre-configured result without touching the network."""

    def __init__(self, result: TranscribeResult) -> None:
        self._result = result

    def transcribe(self, data: bytes, mime_type: str) -> TranscribeResult:
        return self._result


def _make_pdf(n_pages: int) -> bytes:
    writer = PdfWriter()
    for _ in range(n_pages):
        writer.add_blank_page(width=100, height=100)
    buf = BytesIO()
    writer.write(buf)
    return buf.getvalue()


def _readable(transcript: str) -> TranscribeResult:
    return TranscribeResult(
        is_readable=True, transcript=transcript, reason="",
        usage=_zero_usage(),
    )


def _unreadable(reason: str) -> TranscribeResult:
    return TranscribeResult(
        is_readable=False, transcript="", reason=reason,
        usage=_zero_usage(),
    )


def test_plan_from_file_rejects_unsupported_mime():
    pending, error = plan_from_file(
        data=b"...", mime_type="application/zip",
        calendar=InMemoryCalendar(), today=TODAY,
        client=_ScriptedClient([]), transcriber=_ScriptedTranscriber(_readable("x")),
    )
    assert pending is None
    assert error is not None and "only read images and PDFs" in error


def test_plan_from_file_rejects_oversized_image():
    too_big = b"\x00" * (MAX_IMAGE_BYTES + 1)
    pending, error = plan_from_file(
        data=too_big, mime_type="image/jpeg",
        calendar=InMemoryCalendar(), today=TODAY,
        client=_ScriptedClient([]), transcriber=_ScriptedTranscriber(_readable("x")),
    )
    assert pending is None
    assert error is not None and "MB" in error and "under" in error


def test_plan_from_file_rejects_pdf_with_too_many_pages():
    pdf = _make_pdf(n_pages=6)
    pending, error = plan_from_file(
        data=pdf, mime_type="application/pdf",
        calendar=InMemoryCalendar(), today=TODAY,
        client=_ScriptedClient([]), transcriber=_ScriptedTranscriber(_readable("x")),
    )
    assert pending is None
    assert error is not None and "6 pages" in error


def test_plan_from_file_rejects_corrupted_pdf():
    pending, error = plan_from_file(
        data=b"not actually a pdf", mime_type="application/pdf",
        calendar=InMemoryCalendar(), today=TODAY,
        client=_ScriptedClient([]), transcriber=_ScriptedTranscriber(_readable("x")),
    )
    assert pending is None
    assert error is not None and (
        "couldn't open" in error.lower() or "corrupted" in error.lower()
    )


def test_plan_from_file_returns_error_when_transcriber_says_no():
    pdf = _make_pdf(n_pages=1)
    pending, error = plan_from_file(
        data=pdf, mime_type="application/pdf",
        calendar=InMemoryCalendar(), today=TODAY,
        client=_ScriptedClient([]),
        transcriber=_ScriptedTranscriber(_unreadable("page is blank")),
    )
    assert pending is None
    assert error is not None and "blank" in error.lower()


def test_plan_from_file_returns_pending_events_without_writing():
    pdf = _make_pdf(n_pages=1)
    transcript = "Concert at Melody Hall on Saturday 14 March 2027 at 8pm."
    create_call = ToolCall(
        id="tc1", name="create_event",
        input={"title": "Concert", "date": "2027-03-14", "start_time": "20:00"},
    )
    responses = [
        LLMResponse(text="", stop="tool", tool_calls=[create_call], usage=_zero_usage()),
        LLMResponse(text="Done.", stop="end", tool_calls=[], usage=_zero_usage()),
    ]
    cal = InMemoryCalendar()

    pending, error = plan_from_file(
        data=pdf, mime_type="application/pdf",
        calendar=cal, today=TODAY,
        client=_ScriptedClient(responses),
        transcriber=_ScriptedTranscriber(_readable(transcript)),
    )

    assert error is None
    assert pending is not None and len(pending) == 1
    assert pending[0].draft.title == "Concert"
    # Real calendar untouched until commit_pending is called.
    assert cal.list_events(date(2027, 3, 14), date(2027, 3, 14)) == []
