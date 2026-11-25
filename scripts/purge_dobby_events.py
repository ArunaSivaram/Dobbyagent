"""Delete every event Dobby ever wrote to the configured Google Calendar.

Safe by design:
  - Only deletes events tagged with extendedProperties.private.dobby=1
  - Dry-run by default; prints what it would delete and exits
  - Requires --apply AND a typed "yes" confirmation to actually delete

Usage:
    python scripts/purge_dobby_events.py           # dry-run (default)
    python scripts/purge_dobby_events.py --apply   # delete after confirmation
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, time, timedelta

from googleapiclient.errors import HttpError

from dobby.calendar.google import DOBBY_TAG_KEY, DOBBY_TAG_VALUE, GoogleCalendar

DAYS_BACK = 60
DAYS_AHEAD = 365 * 2


def _window() -> tuple[str, str]:
    today = date.today()
    start = today - timedelta(days=DAYS_BACK)
    end = today + timedelta(days=DAYS_AHEAD)
    time_min = datetime.combine(start, time.min).isoformat() + "Z"
    time_max = datetime.combine(end, time.min).isoformat() + "Z"
    return time_min, time_max


def _list_dobby_events(cal: GoogleCalendar) -> list[dict]:
    time_min, time_max = _window()
    tag = f"{DOBBY_TAG_KEY}={DOBBY_TAG_VALUE}"
    all_items: list[dict] = []
    page_token: str | None = None
    while True:
        resp = (
            cal._service.events()
            .list(
                calendarId=cal._calendar_id,
                privateExtendedProperty=tag,
                timeMin=time_min,
                timeMax=time_max,
                singleEvents=True,
                showDeleted=False,
                maxResults=250,
                pageToken=page_token,
            )
            .execute()
        )
        all_items.extend(resp.get("items", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return all_items


def _format(item: dict) -> str:
    title = item.get("summary", "(no title)")
    start = item.get("start", {})
    when = start.get("dateTime") or start.get("date") or "?"
    return f"  • {when}   {title}   [id={item.get('id')}]"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply", action="store_true",
        help="Actually delete (default is dry-run).",
    )
    args = parser.parse_args()

    cal = GoogleCalendar.from_env()
    events = _list_dobby_events(cal)

    if not events:
        print("No Dobby-tagged events found in the window "
              f"({DAYS_BACK}d back → {DAYS_AHEAD}d ahead). Nothing to do.")
        return 0

    print(f"Found {len(events)} Dobby-tagged event(s) on calendar "
          f"'{cal._calendar_id}':")
    for item in events:
        print(_format(item))

    if not args.apply:
        print("\nDry-run. Re-run with --apply to delete.")
        return 0

    print(f"\nAbout to DELETE {len(events)} event(s) from '{cal._calendar_id}'.")
    answer = input("Type 'yes' to confirm: ").strip().lower()
    if answer != "yes":
        print("Aborted. No changes made.")
        return 1

    deleted = 0
    for item in events:
        event_id = item.get("id")
        if not event_id:
            continue
        try:
            cal.delete_event(event_id)
            deleted += 1
        except HttpError as exc:
            print(f"  ! failed to delete {event_id}: {exc}")

    print(f"\nDeleted {deleted}/{len(events)} event(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
