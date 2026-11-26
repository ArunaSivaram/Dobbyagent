"""Pure handler functions — no Telegram library imports. Testable in isolation.

The bot module wraps these in async Telegram handlers; everything that touches
the calendar or the agent lives here.
"""
from __future__ import annotations

from datetime import date, time, timedelta
from io import BytesIO
from typing import Any

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from dobby.agent import run_agent
from dobby.calendar.base import CalendarPort, Event
from dobby.calendar.fake import InMemoryCalendar
from dobby.llm.base import LLMClient, Transcriber
from dobby.telegram.session import PendingEvent

UPCOMING_WINDOW_DAYS = 7

MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_PDF_BYTES = 10 * 1024 * 1024
MAX_PDF_PAGES = 5

SUPPORTED_IMAGE_MIME: tuple[str, ...] = (
    "image/jpeg", "image/png", "image/webp", "image/gif",
)
PDF_MIME = "application/pdf"


def plan_from_text(
    text: str,
    calendar: CalendarPort,
    today: date,
    client: LLMClient,
) -> tuple[list[PendingEvent], str]:
    """Run the agent against a staging buffer. Returns (drafts, agent_reply).

    Reads go to the real calendar (so duplicate-check still works), writes
    land in a throwaway InMemoryCalendar. Nothing is written to the real
    calendar until the dialogue layer confirms each event.

    The agent's final reply is returned alongside the drafts so the bot can
    surface it to the user — critical for duplicate detection, where the
    agent correctly produces zero drafts but explains why in the reply.
    """
    staging = _StagingCalendar(real=calendar)
    agent_result = run_agent(message=text, calendar=staging, today=today, client=client)
    drafts = [PendingEvent(draft=e) for e in agent_result.created_events]
    return drafts, agent_result.reply


def plan_from_file(
    data: bytes,
    mime_type: str,
    calendar: CalendarPort,
    today: date,
    client: LLMClient,
    transcriber: Transcriber,
) -> tuple[list[PendingEvent] | None, str, str | None]:
    """Transcribe, then plan.

    Returns (drafts, agent_reply, error):
      - On validation/unreadable failure: (None, "", error)
      - On success: (drafts, agent_reply, None)
    """
    error = _validate_file(data, mime_type)
    if error:
        return None, "", error

    result = transcriber.transcribe(data, mime_type)
    if not result.is_readable or not result.transcript.strip():
        reason = result.reason or "no event information found"
        msg = (
            f"I couldn't read that file: {reason}. "
            "Try a clearer photo or paste the details as text."
        )
        return None, "", msg

    pending, agent_reply = plan_from_text(result.transcript, calendar, today, client)
    return pending, agent_reply, None


def commit_pending(pending: PendingEvent, calendar: CalendarPort) -> str:
    """Write a user-confirmed pending event to the real calendar."""
    start, end = pending.resolved_times()
    created = calendar.create_event(
        title=pending.final_title(),
        date=pending.draft.date,
        start_time=start,
        end_time=end,
        end_date=pending.draft.end_date,
        location=pending.draft.location,
        notes=pending.draft.notes,
    )
    if created.id is None:
        raise RuntimeError("calendar returned event without an id")
    return created.id


class _StagingCalendar:
    """Reads go to the real calendar; writes land in a fresh InMemoryCalendar.

    Lets the agent's existing duplicate-check logic (list_events on the user's
    calendar) still work, while the actual writes are quarantined until the
    dialogue layer commits them.
    """

    def __init__(self, real: CalendarPort) -> None:
        self._real = real
        self._staging = InMemoryCalendar()

    def list_events(self, start_date: date, end_date: date) -> list[Event]:
        return self._real.list_events(start_date, end_date)

    def create_event(self, **kwargs: Any) -> Event:
        return self._staging.create_event(**kwargs)

    def delete_event(self, event_id: str) -> None:
        raise NotImplementedError("agent must not delete during planning")


def _validate_file(data: bytes, mime_type: str) -> str | None:
    """Return an error message if the file should be rejected, else None."""
    if mime_type in SUPPORTED_IMAGE_MIME:
        if len(data) > MAX_IMAGE_BYTES:
            mb = len(data) / (1024 * 1024)
            return (
                f"That image is {mb:.1f} MB. Please send one under "
                f"{MAX_IMAGE_BYTES // (1024 * 1024)} MB."
            )
        return None
    if mime_type == PDF_MIME:
        if len(data) > MAX_PDF_BYTES:
            mb = len(data) / (1024 * 1024)
            return (
                f"That PDF is {mb:.1f} MB. Please send one under "
                f"{MAX_PDF_BYTES // (1024 * 1024)} MB."
            )
        pages = _pdf_page_count(data)
        if pages is None:
            return "I couldn't open that PDF — it may be corrupted."
        if pages > MAX_PDF_PAGES:
            return (
                f"That PDF has {pages} pages. Please send one with "
                f"{MAX_PDF_PAGES} or fewer."
            )
        return None
    return f"I can only read images and PDFs, not {mime_type!r}."


def _pdf_page_count(data: bytes) -> int | None:
    try:
        return len(PdfReader(BytesIO(data)).pages)
    except (PdfReadError, ValueError, OSError):
        return None


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
