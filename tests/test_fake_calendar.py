from datetime import date, time

from dobby.calendar.base import Event
from dobby.calendar.fake import InMemoryCalendar


def test_create_event_appends_and_returns():
    cal = InMemoryCalendar()
    event = cal.create_event(title="Sports Day", date=date(2026, 11, 14))
    assert event.title == "Sports Day"
    assert cal.list_events(date(2026, 11, 1), date(2026, 11, 30)) == [event]


def test_list_events_filters_by_date_range():
    seed = [
        Event(title="Early", date=date(2026, 10, 1)),
        Event(title="Mid", date=date(2026, 10, 15)),
        Event(title="Late", date=date(2026, 11, 1)),
    ]
    cal = InMemoryCalendar(seed=seed)
    result = cal.list_events(date(2026, 10, 10), date(2026, 10, 20))
    assert [e.title for e in result] == ["Mid"]


def test_create_event_with_all_fields():
    cal = InMemoryCalendar()
    event = cal.create_event(
        title="Camp",
        date=date(2026, 12, 1),
        start_time=time(9, 0),
        end_time=time(17, 0),
        end_date=date(2026, 12, 5),
        location="School field",
        notes="Bring sunscreen",
    )
    assert event.end_date == date(2026, 12, 5)
    assert event.location == "School field"
