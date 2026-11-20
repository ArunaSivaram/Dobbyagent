from __future__ import annotations

import uuid
from datetime import date, time

from dobby.calendar.base import Event


class InMemoryCalendar:
    """A calendar that lives in a list — no network, seedable for tests and evals."""

    def __init__(self, seed: list[Event] | None = None) -> None:
        self._events: list[Event] = list(seed) if seed else []

    def list_events(self, start_date: date, end_date: date) -> list[Event]:
        return [e for e in self._events if start_date <= e.date <= end_date]

    def create_event(
        self,
        title: str,
        date: date,
        start_time: time | None = None,
        end_time: time | None = None,
        end_date: date | None = None,
        location: str | None = None,
        notes: str | None = None,
    ) -> Event:
        event = Event(
            title=title,
            date=date,
            start_time=start_time,
            end_time=end_time,
            end_date=end_date,
            location=location,
            notes=notes,
            id=uuid.uuid4().hex,
        )
        self._events.append(event)
        return event

    def delete_event(self, event_id: str) -> None:
        self._events = [e for e in self._events if e.id != event_id]
