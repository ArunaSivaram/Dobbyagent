"""Pure tests for the Module 5 render helpers."""
from __future__ import annotations

from datetime import date, time

from dobby.calendar.base import Event
from dobby.telegram.render import (
    render_done,
    render_keyboard,
    render_proposal,
)
from dobby.telegram.session import PendingEvent


def _draft(**overrides) -> Event:
    base = {"title": "Sports Day", "date": date(2026, 11, 22), "start_time": time(9, 0)}
    base.update(overrides)
    return Event(**base)


def _all_labels(keyboard) -> list[str]:
    return [label for row in keyboard for label, _ in row]


def _all_callbacks(keyboard) -> list[str]:
    return [cb for row in keyboard for _, cb in row]


# ---------- render_proposal ----------


def test_proposal_shows_position_and_total():
    p = PendingEvent(draft=_draft())
    text = render_proposal(p, position=2, total=3)
    assert "2 of 3" in text


def test_proposal_includes_category_type_and_title():
    p = PendingEvent(
        draft=_draft(title="Weekly sync"),
        category="Professional",
        event_type="Meeting",
    )
    text = render_proposal(p, 1, 1)
    assert "Professional" in text
    assert "Meeting" in text
    assert "Weekly sync" in text


def test_proposal_shows_time_range_and_duration_for_timed_event():
    p = PendingEvent(draft=_draft(start_time=time(10, 0)), duration_hours=2)
    text = render_proposal(p, 1, 1)
    assert "10:00" in text
    assert "12:00" in text
    assert "(2h)" in text


def test_proposal_defaults_to_9am_for_single_day_without_time():
    """Per Module 5 spec: no time mentioned → show 9 AM + 1h default."""
    p = PendingEvent(draft=_draft(start_time=None))
    text = render_proposal(p, 1, 1)
    assert "09:00" in text
    assert "All day" not in text


def test_proposal_marks_all_day_for_multi_day_draft():
    draft = _draft(date=date(2026, 11, 16), end_date=date(2026, 11, 18), start_time=None)
    p = PendingEvent(draft=draft)
    text = render_proposal(p, 1, 1)
    assert "16 Nov 2026" in text
    assert "18 Nov 2026" in text
    assert "All day" in text


# ---------- render_keyboard ----------


def test_keyboard_has_category_type_and_confirm_rows_at_minimum():
    kb = render_keyboard(PendingEvent(draft=_draft()))
    cbs = _all_callbacks(kb)
    assert "cat:Family" in cbs
    assert "cat:Professional" in cbs
    assert "type:Event" in cbs
    assert "type:Meeting" in cbs
    assert "confirm" in cbs
    assert "cancel" in cbs


def test_keyboard_ticks_active_category():
    p = PendingEvent(draft=_draft(), category="Professional")
    labels = _all_labels(render_keyboard(p))
    assert any("✓ Professional" in lbl for lbl in labels)
    assert not any(lbl == "✓ Family" for lbl in labels)


def test_keyboard_ticks_active_event_type():
    p = PendingEvent(draft=_draft(), event_type="Meeting")
    labels = _all_labels(render_keyboard(p))
    assert any("✓ Meeting" in lbl for lbl in labels)
    assert not any(lbl == "✓ Event" for lbl in labels)


def test_keyboard_shows_duration_row_even_when_no_time_mentioned():
    """Single-day no-time defaults to 9 AM → duration row should appear."""
    kb = render_keyboard(PendingEvent(draft=_draft(start_time=None)))
    assert "dur:+1" in _all_callbacks(kb)
    assert "dur:-1" in _all_callbacks(kb)


def test_keyboard_hides_duration_row_only_for_multi_day_all_day():
    draft = _draft(date=date(2026, 11, 16), end_date=date(2026, 11, 18), start_time=None)
    kb = render_keyboard(PendingEvent(draft=draft))
    assert "dur:+1" not in _all_callbacks(kb)


def test_keyboard_duration_label_shows_current_hours():
    p = PendingEvent(draft=_draft(start_time=time(9, 0)), duration_hours=4)
    labels = _all_labels(render_keyboard(p))
    assert "4h" in labels


# ---------- render_done ----------


def test_done_summary_full_success():
    assert "3" in render_done(committed_count=3, total=3)


def test_done_summary_partial_cancellation():
    msg = render_done(committed_count=1, total=3)
    assert "1" in msg
    assert "3" in msg
    assert "cancel" in msg.lower()


def test_done_summary_nothing_scheduled():
    msg = render_done(committed_count=0, total=0)
    assert "nothing" in msg.lower()
