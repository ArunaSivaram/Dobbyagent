"""Per-chat pending-event state for the Module 5 dialogue flow.

Pure state and transitions. Nothing here imports Telegram.

The bot uses this to queue events the agent has drafted but not yet
written to the real calendar. Each PendingEvent walks through a short
dialogue (category, type, duration, confirm) before being committed.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, time, timedelta
from typing import Literal

from dobby.calendar.base import Event

Category = Literal["Family", "Professional"]
EventType = Literal["Event", "Meeting"]

DEFAULT_CATEGORY: Category = "Family"
DEFAULT_EVENT_TYPE: EventType = "Event"
DEFAULT_DURATION_HOURS = 1
MIN_DURATION_HOURS = 1
MAX_DURATION_HOURS = 24

# Per Module 5 spec: "If no time is mentioned default it to Event, 9 AM, 1h."
DEFAULT_START_TIME = time(hour=9, minute=0)


@dataclass(frozen=True)
class PendingEvent:
    """One event waiting on user confirmation.

    The `draft` is what the agent extracted (title, date, times, etc.).
    The three structured fields (category, event_type, duration_hours)
    layer on top and can be edited via buttons before the event is
    written to the real calendar.
    """

    draft: Event
    category: Category = DEFAULT_CATEGORY
    event_type: EventType = DEFAULT_EVENT_TYPE
    duration_hours: int = DEFAULT_DURATION_HOURS

    def with_category(self, category: Category) -> PendingEvent:
        return replace(self, category=category)

    def with_event_type(self, event_type: EventType) -> PendingEvent:
        return replace(self, event_type=event_type)

    def with_duration_delta(self, delta: int) -> PendingEvent:
        new_duration = max(
            MIN_DURATION_HOURS,
            min(MAX_DURATION_HOURS, self.duration_hours + delta),
        )
        return replace(self, duration_hours=new_duration)

    def final_title(self) -> str:
        """`"<Category> : <title>"` per Module 5 spec."""
        return f"{self.category} : {self.draft.title}"

    def is_timed(self) -> bool:
        """True when this event will have start/end times (not all-day).

        - Draft has a start_time → always timed.
        - Draft has an end_date spanning multiple days → stays all-day.
        - Draft has neither (no time mentioned, single-day) → treated as
          timed with a 9 AM default per the Module 5 spec.
        """
        if self.draft.start_time is not None:
            return True
        multi_day = (
            self.draft.end_date is not None
            and self.draft.end_date != self.draft.date
        )
        return not multi_day

    def effective_start_time(self) -> time | None:
        """The start time used when committing. None for multi-day all-day."""
        if not self.is_timed():
            return None
        return self.draft.start_time or DEFAULT_START_TIME

    def resolved_times(self) -> tuple[time | None, time | None]:
        """What start/end time to use when writing to the real calendar.

        - Multi-day all-day events return (None, None); duration is ignored.
        - Otherwise end_time = start + duration_hours. Any end_time the
          draft had is overridden, since the user explicitly chose a
          duration via buttons.
        """
        start = self.effective_start_time()
        if start is None:
            return None, None
        start_dt = datetime.combine(self.draft.date, start)
        end_dt = start_dt + timedelta(hours=self.duration_hours)
        return start, end_dt.time().replace(microsecond=0)


@dataclass
class ChatSession:
    """Everything a chat needs to walk through its current event queue."""

    chat_id: int
    queue: list[PendingEvent] = field(default_factory=list)
    current_idx: int = 0
    committed_ids: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.queue)

    @property
    def is_done(self) -> bool:
        return self.current_idx >= len(self.queue)

    def current(self) -> PendingEvent:
        if self.is_done:
            raise IndexError("no current event: queue is exhausted")
        return self.queue[self.current_idx]

    def replace_current(self, updated: PendingEvent) -> None:
        if self.is_done:
            raise IndexError("no current event to replace")
        self.queue[self.current_idx] = updated

    def advance(self) -> None:
        """Move past the current event. Called after a Confirm commits."""
        self.current_idx += 1

    def record_commit(self, event_id: str) -> None:
        self.committed_ids.append(event_id)
