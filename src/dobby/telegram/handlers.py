"""Pure handler functions — no Telegram library imports. Testable in isolation.

The bot module wraps these in async Telegram handlers; everything that touches
the calendar or the agent lives here.
"""
from __future__ import annotations

from datetime import date, time, timedelta
from io import BytesIO

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from dobby.agent import run_agent
from dobby.calendar.base import CalendarPort, Event
from dobby.llm.base import LLMClient, Transcriber

UPCOMING_WINDOW_DAYS = 7

MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_PDF_BYTES = 10 * 1024 * 1024
MAX_PDF_PAGES = 5

SUPPORTED_IMAGE_MIME: tuple[str, ...] = (
    "image/jpeg", "image/png", "image/webp", "image/gif",
)
PDF_MIME = "application/pdf"


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


def handle_file(
    data: bytes,
    mime_type: str,
    calendar: CalendarPort,
    today: date,
    client: LLMClient,
    transcriber: Transcriber,
) -> tuple[str, list[str]]:
    """Transcribe an image or PDF, then run the agent on the transcript.

    Returns (reply, created_event_ids). If the file is rejected or unreadable,
    created_event_ids is empty and the reply explains why.
    """
    error = _validate_file(data, mime_type)
    if error:
        return error, []

    result = transcriber.transcribe(data, mime_type)
    if not result.is_readable or not result.transcript.strip():
        reason = result.reason or "no event information found"
        msg = (
            f"I couldn't read that file: {reason}. "
            "Try a clearer photo or paste the details as text."
        )
        return msg, []

    agent_result = run_agent(
        message=result.transcript, calendar=calendar, today=today, client=client,
    )
    ids = [e.id for e in agent_result.created_events if e.id]
    return agent_result.reply, ids


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
