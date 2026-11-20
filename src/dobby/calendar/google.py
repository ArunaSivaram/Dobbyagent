"""Google Calendar adapter. Implements CalendarPort, tags events with dobby=1."""
from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Any

from dobby.calendar.base import Event
from dobby.config import TIMEZONE

DOBBY_TAG_KEY = "dobby"
DOBBY_TAG_VALUE = "1"


class GoogleCalendar:
    def __init__(self, service: Any, calendar_id: str = "primary") -> None:
        self._service = service
        self._calendar_id = calendar_id

    @classmethod
    def from_env(cls) -> GoogleCalendar:
        from googleapiclient.discovery import build

        from dobby.calendar.google_auth import get_credentials
        from dobby.config import (
            GOOGLE_CALENDAR_ID,
            GOOGLE_CREDENTIALS_PATH,
            GOOGLE_TOKEN_PATH,
        )

        creds = get_credentials(GOOGLE_CREDENTIALS_PATH, GOOGLE_TOKEN_PATH)
        service = build("calendar", "v3", credentials=creds, cache_discovery=False)
        return cls(service=service, calendar_id=GOOGLE_CALENDAR_ID)

    def list_events(self, start_date: date, end_date: date) -> list[Event]:
        time_min = datetime.combine(start_date, time.min).isoformat() + "Z"
        time_max = datetime.combine(end_date + timedelta(days=1), time.min).isoformat() + "Z"
        response = (
            self._service.events()
            .list(
                calendarId=self._calendar_id,
                timeMin=time_min,
                timeMax=time_max,
                singleEvents=True,
                orderBy="startTime",
                maxResults=250,
            )
            .execute()
        )
        return [_from_google(item) for item in response.get("items", [])]

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
        )
        body = _to_google(event)
        self._service.events().insert(calendarId=self._calendar_id, body=body).execute()
        return event


def _to_google(event: Event) -> dict[str, Any]:
    body: dict[str, Any] = {
        "summary": event.title,
        "extendedProperties": {"private": {DOBBY_TAG_KEY: DOBBY_TAG_VALUE}},
    }
    if event.location:
        body["location"] = event.location
    if event.notes:
        body["description"] = event.notes

    if event.start_time is not None:
        start_dt = datetime.combine(event.date, event.start_time)
        end_t = event.end_time or (time(hour=(event.start_time.hour + 1) % 24, minute=event.start_time.minute))
        end_dt = datetime.combine(event.date, end_t)
        if end_dt <= start_dt:
            end_dt = start_dt + timedelta(hours=1)
        body["start"] = {"dateTime": start_dt.isoformat(), "timeZone": TIMEZONE}
        body["end"] = {"dateTime": end_dt.isoformat(), "timeZone": TIMEZONE}
    else:
        inclusive_end = event.end_date or event.date
        body["start"] = {"date": event.date.isoformat()}
        body["end"] = {"date": (inclusive_end + timedelta(days=1)).isoformat()}
    return body


def _from_google(item: dict[str, Any]) -> Event:
    title = item.get("summary", "")
    location = item.get("location")
    notes = item.get("description")

    start = item.get("start", {})
    end = item.get("end", {})

    if "dateTime" in start:
        start_dt = datetime.fromisoformat(start["dateTime"])
        end_dt = datetime.fromisoformat(end["dateTime"]) if "dateTime" in end else start_dt
        return Event(
            title=title,
            date=start_dt.date(),
            start_time=start_dt.time().replace(microsecond=0),
            end_time=end_dt.time().replace(microsecond=0),
            end_date=end_dt.date() if end_dt.date() != start_dt.date() else None,
            location=location,
            notes=notes,
        )

    start_date = date.fromisoformat(start["date"])
    exclusive_end = date.fromisoformat(end["date"]) if "date" in end else start_date + timedelta(days=1)
    inclusive_end = exclusive_end - timedelta(days=1)
    return Event(
        title=title,
        date=start_date,
        end_date=inclusive_end if inclusive_end != start_date else None,
        location=location,
        notes=notes,
    )
