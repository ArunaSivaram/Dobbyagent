from __future__ import annotations

from dataclasses import dataclass
from datetime import date, time
from typing import Protocol


@dataclass(frozen=True)
class Event:
    title: str
    date: date
    start_time: time | None = None
    end_time: time | None = None
    end_date: date | None = None
    location: str | None = None
    notes: str | None = None


class CalendarPort(Protocol):
    def list_events(self, start_date: date, end_date: date) -> list[Event]: ...

    def create_event(
        self,
        title: str,
        date: date,
        start_time: time | None = None,
        end_time: time | None = None,
        end_date: date | None = None,
        location: str | None = None,
        notes: str | None = None,
    ) -> Event: ...
