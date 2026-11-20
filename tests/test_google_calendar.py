"""Tests for the Google Calendar adapter. No network — a fake service captures calls."""
from __future__ import annotations

from datetime import date, time
from typing import Any

from dobby.calendar.base import Event
from dobby.calendar.google import GoogleCalendar, _from_google, _to_google


class _FakeExecute:
    def __init__(self, payload: Any) -> None:
        self._payload = payload

    def execute(self) -> Any:
        return self._payload


class _FakeEventsResource:
    def __init__(self, list_payload: dict[str, Any]) -> None:
        self.list_payload = list_payload
        self.list_kwargs: dict[str, Any] | None = None
        self.insert_kwargs: dict[str, Any] | None = None
        self.delete_kwargs: dict[str, Any] | None = None

    def list(self, **kwargs: Any) -> _FakeExecute:
        self.list_kwargs = kwargs
        return _FakeExecute(self.list_payload)

    def insert(self, **kwargs: Any) -> _FakeExecute:
        self.insert_kwargs = kwargs
        return _FakeExecute({"id": "created-id"})

    def delete(self, **kwargs: Any) -> _FakeExecute:
        self.delete_kwargs = kwargs
        return _FakeExecute(None)


class _FakeService:
    def __init__(self, list_payload: dict[str, Any] | None = None) -> None:
        self._events_resource = _FakeEventsResource(list_payload or {"items": []})

    def events(self) -> _FakeEventsResource:
        return self._events_resource


# ---------- _to_google ----------


def test_to_google_tags_event_with_dobby_marker():
    event = Event(title="Sports Day", date=date(2026, 11, 14))
    body = _to_google(event)
    assert body["extendedProperties"]["private"]["dobby"] == "1"


def test_to_google_all_day_uses_date_and_exclusive_end():
    event = Event(title="Sports Day", date=date(2026, 11, 14))
    body = _to_google(event)
    assert body["start"] == {"date": "2026-11-14"}
    assert body["end"] == {"date": "2026-11-15"}


def test_to_google_multi_day_all_day_end_is_exclusive():
    event = Event(title="Camp", date=date(2026, 12, 1), end_date=date(2026, 12, 5))
    body = _to_google(event)
    assert body["start"] == {"date": "2026-12-01"}
    assert body["end"] == {"date": "2026-12-06"}


def test_to_google_timed_event_uses_datetime_with_timezone():
    event = Event(
        title="Soccer",
        date=date(2026, 11, 21),
        start_time=time(10, 0),
        end_time=time(11, 30),
    )
    body = _to_google(event)
    assert body["start"]["dateTime"].startswith("2026-11-21T10:00")
    assert body["end"]["dateTime"].startswith("2026-11-21T11:30")
    assert "timeZone" in body["start"]
    assert "timeZone" in body["end"]


def test_to_google_timed_without_end_time_defaults_to_one_hour():
    event = Event(title="Meeting", date=date(2026, 11, 21), start_time=time(10, 0))
    body = _to_google(event)
    assert body["end"]["dateTime"].startswith("2026-11-21T11:00")


def test_to_google_includes_location_and_notes_when_set():
    event = Event(
        title="Camp",
        date=date(2026, 12, 1),
        location="School field",
        notes="Bring sunscreen",
    )
    body = _to_google(event)
    assert body["location"] == "School field"
    assert body["description"] == "Bring sunscreen"


def test_to_google_omits_optional_fields_when_unset():
    event = Event(title="Sports Day", date=date(2026, 11, 14))
    body = _to_google(event)
    assert "location" not in body
    assert "description" not in body


# ---------- _from_google ----------


def test_from_google_parses_all_day_event():
    item = {
        "summary": "School holiday",
        "start": {"date": "2026-12-20"},
        "end": {"date": "2026-12-21"},
    }
    event = _from_google(item)
    assert event.title == "School holiday"
    assert event.date == date(2026, 12, 20)
    assert event.start_time is None
    assert event.end_date is None


def test_from_google_parses_multi_day_all_day_event():
    item = {
        "summary": "Camp",
        "start": {"date": "2026-12-01"},
        "end": {"date": "2026-12-06"},
    }
    event = _from_google(item)
    assert event.date == date(2026, 12, 1)
    assert event.end_date == date(2026, 12, 5)


def test_from_google_parses_timed_event():
    item = {
        "summary": "Soccer",
        "start": {"dateTime": "2026-11-21T10:00:00+08:00"},
        "end": {"dateTime": "2026-11-21T11:30:00+08:00"},
        "location": "Field 3",
    }
    event = _from_google(item)
    assert event.title == "Soccer"
    assert event.date == date(2026, 11, 21)
    assert event.start_time == time(10, 0)
    assert event.end_time == time(11, 30)
    assert event.location == "Field 3"


# ---------- integration: GoogleCalendar through the fake service ----------


def test_create_event_calls_insert_with_tagged_body():
    service = _FakeService()
    cal = GoogleCalendar(service=service, calendar_id="primary")

    cal.create_event(title="Sports Day", date=date(2026, 11, 14))

    insert_kwargs = service._events_resource.insert_kwargs
    assert insert_kwargs is not None
    assert insert_kwargs["calendarId"] == "primary"
    body = insert_kwargs["body"]
    assert body["summary"] == "Sports Day"
    assert body["extendedProperties"]["private"]["dobby"] == "1"


def test_create_event_captures_server_assigned_id():
    service = _FakeService()
    cal = GoogleCalendar(service=service, calendar_id="primary")

    event = cal.create_event(title="Sports Day", date=date(2026, 11, 14))

    assert event.id == "created-id"


def test_delete_event_calls_google_delete():
    service = _FakeService()
    cal = GoogleCalendar(service=service, calendar_id="primary")

    cal.delete_event("abc-123")

    delete_kwargs = service._events_resource.delete_kwargs
    assert delete_kwargs is not None
    assert delete_kwargs["calendarId"] == "primary"
    assert delete_kwargs["eventId"] == "abc-123"


def test_list_events_passes_window_and_parses_items():
    payload = {
        "items": [
            {
                "summary": "Existing meeting",
                "start": {"dateTime": "2026-11-15T09:00:00+08:00"},
                "end": {"dateTime": "2026-11-15T10:00:00+08:00"},
            },
            {
                "summary": "Public holiday",
                "start": {"date": "2026-11-16"},
                "end": {"date": "2026-11-17"},
            },
        ]
    }
    service = _FakeService(list_payload=payload)
    cal = GoogleCalendar(service=service, calendar_id="primary")

    events = cal.list_events(date(2026, 11, 10), date(2026, 11, 20))

    list_kwargs = service._events_resource.list_kwargs
    assert list_kwargs is not None
    assert list_kwargs["calendarId"] == "primary"
    assert list_kwargs["timeMin"].startswith("2026-11-10")
    assert list_kwargs["timeMax"].startswith("2026-11-21")
    assert list_kwargs["singleEvents"] is True

    assert [e.title for e in events] == ["Existing meeting", "Public holiday"]
    assert events[0].start_time == time(9, 0)
    assert events[1].start_time is None
