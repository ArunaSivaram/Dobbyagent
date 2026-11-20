"""CLI: paste a message, Dobby runs the agent and reports what it did.

Usage:
    python -m dobby "Soccer practice Saturday 10am at field 3"
    echo "message here" | python -m dobby

The calendar backend is picked from CALENDAR_BACKEND (fake | google).
"""
from __future__ import annotations

import sys
from datetime import date

from dobby.agent import run_agent
from dobby.calendar.base import CalendarPort
from dobby.config import CALENDAR_BACKEND
from dobby.llm import get_client


def _build_calendar() -> CalendarPort:
    backend = CALENDAR_BACKEND.lower()
    if backend == "fake":
        from dobby.calendar.fake import InMemoryCalendar

        return InMemoryCalendar()
    if backend == "google":
        from dobby.calendar.google import GoogleCalendar

        return GoogleCalendar.from_env()
    raise ValueError(f"unknown CALENDAR_BACKEND: {backend!r} (expected 'fake' or 'google')")


def _read_message(argv: list[str]) -> str:
    if len(argv) > 1:
        return " ".join(argv[1:])
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()
    print("Paste a message, then Ctrl-D:", file=sys.stderr)
    return sys.stdin.read().strip()


def main() -> None:
    message = _read_message(sys.argv)
    if not message:
        print("No message provided.", file=sys.stderr)
        sys.exit(1)

    calendar = _build_calendar()
    client = get_client()
    result = run_agent(message=message, calendar=calendar, today=date.today(), client=client)

    print(result.reply)
    if result.created_events:
        print()
        print(f"Created {len(result.created_events)} event(s):")
        for e in result.created_events:
            when = e.date.isoformat()
            if e.start_time:
                when += f" {e.start_time.isoformat(timespec='minutes')}"
            print(f"  - {e.title} ({when})")
    print()
    print(f"Cost: ${result.total_cost_usd:.6f} across {len(result.usages)} LLM call(s).")


if __name__ == "__main__":
    main()
