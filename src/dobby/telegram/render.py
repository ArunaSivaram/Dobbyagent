"""Render a PendingEvent as text + a 2D button grid.

Returns neutral `list[list[tuple[label, callback_data]]]` so these
helpers stay pure and testable. `bot.py` materialises the grid into
telegram.InlineKeyboardMarkup at the edge.
"""
from __future__ import annotations

from dobby.telegram.session import PendingEvent

Button = tuple[str, str]  # (label, callback_data)
Keyboard = list[list[Button]]


def render_proposal(pending: PendingEvent, position: int, total: int) -> str:
    """The message text shown with the inline keyboard.

    `position` is 1-based (event 1 of 3, event 2 of 3, …).
    """
    draft = pending.draft
    lines = [f"Proposed event {position} of {total}:", ""]
    lines.append(f"  {pending.category} · {pending.event_type}")
    lines.append(f"  {draft.title}")

    if draft.end_date and draft.end_date != draft.date:
        lines.append(f"  📅 {_fmt_date(draft.date)} – {_fmt_date(draft.end_date)}")
    else:
        lines.append(f"  📅 {_fmt_date(draft.date)}")

    start, end = pending.resolved_times()
    if start is None:
        lines.append("  📆 All day")
    else:
        assert end is not None
        lines.append(
            f"  ⏰ {start.strftime('%H:%M')} – {end.strftime('%H:%M')}  "
            f"({pending.duration_hours}h)"
        )

    lines.append("")
    lines.append("Tap to adjust, then Confirm.")
    return "\n".join(lines)


def render_keyboard(pending: PendingEvent) -> Keyboard:
    """Inline keyboard matching the proposal.

    Active values get a ✓ in the label so the user sees the current state
    without re-reading the body text.
    """
    rows: Keyboard = [
        [
            _toggle("Family", pending.category == "Family", "cat:Family"),
            _toggle("Professional", pending.category == "Professional", "cat:Professional"),
        ],
        [
            _toggle("Event", pending.event_type == "Event", "type:Event"),
            _toggle("Meeting", pending.event_type == "Meeting", "type:Meeting"),
        ],
    ]
    # Duration row whenever the event will be timed (single-day gets a 9 AM
    # default per Module 5 spec). Only multi-day all-day events hide this.
    if pending.is_timed():
        rows.append(
            [
                ("– 1h", "dur:-1"),
                (f"{pending.duration_hours}h", "dur:0"),
                ("+ 1h", "dur:+1"),
            ]
        )
    rows.append([("✅ Confirm", "confirm"), ("❌ Cancel", "cancel")])
    return rows


def render_done(committed_count: int, total: int) -> str:
    if committed_count == total == 0:
        return "Nothing to confirm."
    if committed_count == total:
        return f"Added {committed_count} event(s) to your calendar."
    return (
        f"Added {committed_count} of {total} event(s). "
        "The rest were cancelled."
    )


def _fmt_date(d) -> str:
    return d.strftime("%a %d %b %Y")


def _toggle(label: str, active: bool, callback_data: str) -> Button:
    return (f"✓ {label}" if active else label, callback_data)
