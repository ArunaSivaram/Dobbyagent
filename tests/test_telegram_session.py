"""Pure tests for the Module 5 session/state types."""
from __future__ import annotations

from datetime import date, time

import pytest

from dobby.calendar.base import Event
from dobby.telegram.session import (
    DEFAULT_CATEGORY,
    DEFAULT_DURATION_HOURS,
    DEFAULT_EVENT_TYPE,
    MAX_DURATION_HOURS,
    MIN_DURATION_HOURS,
    ChatSession,
    PendingEvent,
)


def _draft(**overrides) -> Event:
    base = {"title": "Sports Day", "date": date(2026, 11, 22), "start_time": time(9, 0)}
    base.update(overrides)
    return Event(**base)


# ---------- PendingEvent ----------


def test_pending_event_defaults_match_module_5_spec():
    p = PendingEvent(draft=_draft())
    assert p.category == DEFAULT_CATEGORY == "Family"
    assert p.event_type == DEFAULT_EVENT_TYPE == "Event"
    assert p.duration_hours == DEFAULT_DURATION_HOURS == 1


def test_with_category_returns_new_instance():
    p = PendingEvent(draft=_draft())
    p2 = p.with_category("Professional")
    assert p.category == "Family"  # original untouched (frozen)
    assert p2.category == "Professional"


def test_with_event_type_returns_new_instance():
    p = PendingEvent(draft=_draft())
    p2 = p.with_event_type("Meeting")
    assert p2.event_type == "Meeting"


def test_with_duration_delta_clamps_to_bounds():
    p = PendingEvent(draft=_draft(), duration_hours=1)
    assert p.with_duration_delta(-1).duration_hours == MIN_DURATION_HOURS
    assert p.with_duration_delta(+100).duration_hours == MAX_DURATION_HOURS


def test_final_title_prepends_category_and_space_colon_space():
    p = PendingEvent(draft=_draft(title="Daughter's sports day"), category="Family")
    assert p.final_title() == "Family : Daughter's sports day"


def test_resolved_times_computes_end_from_duration_when_timed():
    p = PendingEvent(
        draft=_draft(start_time=time(10, 0)), duration_hours=2,
    )
    start, end = p.resolved_times()
    assert start == time(10, 0)
    assert end == time(12, 0)


def test_resolved_times_defaults_to_9am_when_no_start_time_single_day():
    """Per Module 5 spec: no time mentioned → default to 9 AM for 1h."""
    p = PendingEvent(draft=_draft(start_time=None), duration_hours=1)
    start, end = p.resolved_times()
    assert start == time(9, 0)
    assert end == time(10, 0)


def test_resolved_times_returns_none_only_for_multi_day_all_day_drafts():
    draft = _draft(
        start_time=None, date=date(2026, 11, 16), end_date=date(2026, 11, 18),
    )
    p = PendingEvent(draft=draft, duration_hours=3)  # duration ignored
    assert p.resolved_times() == (None, None)


def test_is_timed_true_for_single_day_without_time():
    """The 'no time mentioned' case defaults to 9 AM, so it IS timed."""
    p = PendingEvent(draft=_draft(start_time=None))
    assert p.is_timed() is True


def test_is_timed_false_only_for_multi_day_all_day():
    draft = _draft(
        start_time=None, date=date(2026, 11, 16), end_date=date(2026, 11, 18),
    )
    p = PendingEvent(draft=draft)
    assert p.is_timed() is False


def test_resolved_times_overrides_draft_end_time_with_user_duration():
    """If the agent drafted 9-10 but user picks 2h, final is 9-11."""
    draft = _draft(start_time=time(9, 0), end_time=time(10, 0))
    p = PendingEvent(draft=draft, duration_hours=2)
    start, end = p.resolved_times()
    assert start == time(9, 0)
    assert end == time(11, 0)


# ---------- ChatSession ----------


def test_empty_session_is_done():
    s = ChatSession(chat_id=1)
    assert s.is_done
    assert s.total == 0


def test_current_raises_when_queue_empty():
    s = ChatSession(chat_id=1)
    with pytest.raises(IndexError):
        s.current()


def test_advance_walks_through_queue():
    p1 = PendingEvent(draft=_draft(title="A"))
    p2 = PendingEvent(draft=_draft(title="B"))
    s = ChatSession(chat_id=1, queue=[p1, p2])

    assert s.current().draft.title == "A"
    assert not s.is_done
    s.advance()
    assert s.current().draft.title == "B"
    s.advance()
    assert s.is_done


def test_replace_current_mutates_queue_at_current_idx():
    p1 = PendingEvent(draft=_draft(title="A"))
    s = ChatSession(chat_id=1, queue=[p1])
    s.replace_current(p1.with_category("Professional"))
    assert s.current().category == "Professional"


def test_record_commit_accumulates_ids():
    s = ChatSession(chat_id=1)
    s.record_commit("ev1")
    s.record_commit("ev2")
    assert s.committed_ids == ["ev1", "ev2"]
