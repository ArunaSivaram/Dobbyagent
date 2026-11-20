"""Pure handler functions — no Telegram library imports. Testable in isolation.

The bot module wraps these in async Telegram handlers; everything that touches
the calendar or the agent lives here.
"""
from __future__ import annotations

from datetime import date, time, timedelta

from dobby.agent import run_agent
from dobby.calendar.base import CalendarPort, Event
from dobby.llm.base import LLMClient

UPCOMING_WINDOW_DAYS = 7


def handle_text(
    text: str,
    calendar: CalendarPort,
    today: date,
    client: LLMClient,
) -> tuple[str, list[str]]:
    """Run the agent on a free-text message. Returns (reply, created_event_ids)."""
    result = run_agent(message=text, calendar=calendar, today=today, client=client)
    ids = [e.id for e in result.created_events if e.id]
    return result.reply, ids


def handle_upcoming(calendar: CalendarPort, today: date) -> str:
    """List events in the next UPCOMING_WINDOW_DAYS days."""
    events = calendar.list_events(today, today + timedelta(days=UPCOMING_WINDOW_DAYS))
    if not events:
        return f"No events in the next {UPCOMING_WINDOW_DAYS} days."
    lines = [f"Next {UPCOMING_WINDOW_DAYS} days:"]
    for e in sorted(events, key=_event_sort_key):
        lines.append(f"• {_format_event(e)}")
    return "\n".join(lines)


def handle_undo(calendar: CalendarPort, event_ids: list[str]) -> str:
    """Delete the given event IDs. Returns a human-readable summary."""
    if not event_ids:
        return "Nothing to undo."
    deleted = 0
    errors: list[str] = []
    for eid in event_ids:
        try:
            calendar.delete_event(eid)
            deleted += 1
        except Exception as e:  # noqa: BLE001 — surface the error to the user, don't crash the bot
            errors.append(f"{type(e).__name__}: {e}")
    if errors and deleted == 0:
        return "Undo failed: " + "; ".join(errors)
    if errors:
        return f"Undid {deleted} event(s). {len(errors)} failed: " + "; ".join(errors)
    return f"Undid {deleted} event(s) from your last run."


def _event_sort_key(e: Event) -> tuple[date, time]:
    return (e.date, e.start_time or time.min)


def _format_event(e: Event) -> str:
    parts = [e.date.strftime("%a %b %d")]
    if e.start_time:
        parts.append(e.start_time.strftime("%H:%M"))
    parts.append("—")
    parts.append(e.title)
    if e.location:
        parts.append(f"({e.location})")
    return " ".join(parts)
