"""Telegram bot entrypoint. Polls Telegram, dispatches to handlers.py.

Run with: make bot
"""
from __future__ import annotations

import logging
from datetime import date

from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
)
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from dobby.calendar.base import CalendarPort
from dobby.calendar.google import GoogleCalendar
from dobby.config import TELEGRAM_ALLOWED_USER_IDS, TELEGRAM_BOT_TOKEN
from dobby.llm import get_client
from dobby.llm.base import Transcriber
from dobby.telegram.handlers import (
    PDF_MIME,
    commit_pending,
    handle_undo,
    handle_upcoming,
    plan_from_file,
    plan_from_text,
)
from dobby.telegram.render import (
    Keyboard,
    render_done,
    render_keyboard,
    render_proposal,
)
from dobby.telegram.session import (
    MAX_DURATION_HOURS,
    MIN_DURATION_HOURS,
    ChatSession,
    PendingEvent,
)

logger = logging.getLogger("dobby.telegram")

# Per-chat state.
_sessions: dict[int, ChatSession] = {}
# Last-run event IDs kept for /undo compatibility across dialogue sessions.
_last_run_ids: dict[int, list[str]] = {}


def _is_allowed(update: Update) -> bool:
    if update.effective_user is None:
        return False
    return update.effective_user.id in TELEGRAM_ALLOWED_USER_IDS


# ---------- inbound messages (start a planning run) ----------


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_allowed(update) or update.message is None or update.message.text is None:
        return
    chat_id = update.effective_chat.id
    calendar: CalendarPort = context.bot_data["calendar"]
    client = context.bot_data["llm"]
    try:
        pending, agent_reply = plan_from_text(
            text=update.message.text,
            calendar=calendar,
            today=date.today(),
            client=client,
        )
    except Exception as e:
        logger.exception("plan_from_text failed")
        await update.message.reply_text(f"Sorry, something broke: {type(e).__name__}: {e}")
        return
    await _start_dialogue(update, context, chat_id, pending, agent_reply)


async def on_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_allowed(update) or update.message is None:
        return
    msg = update.message
    if msg.photo:
        tg_file = await msg.photo[-1].get_file()
        mime = "image/jpeg"
    elif msg.document and (msg.document.mime_type or "").startswith("image/"):
        tg_file = await msg.document.get_file()
        mime = msg.document.mime_type or "image/jpeg"
    else:
        return
    data = bytes(await tg_file.download_as_bytearray())
    await _dispatch_file(update, context, data=data, mime_type=mime)


async def on_pdf(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_allowed(update) or update.message is None or update.message.document is None:
        return
    doc = update.message.document
    tg_file = await doc.get_file()
    data = bytes(await tg_file.download_as_bytearray())
    mime = doc.mime_type or PDF_MIME
    await _dispatch_file(update, context, data=data, mime_type=mime)


async def _dispatch_file(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    *,
    data: bytes,
    mime_type: str,
) -> None:
    assert update.message is not None
    chat_id = update.effective_chat.id
    calendar: CalendarPort = context.bot_data["calendar"]
    client = context.bot_data["llm"]
    transcriber: Transcriber = context.bot_data["transcriber"]

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    try:
        pending, agent_reply, error = plan_from_file(
            data=data, mime_type=mime_type,
            calendar=calendar, today=date.today(),
            client=client, transcriber=transcriber,
        )
    except Exception as e:
        logger.exception("plan_from_file failed")
        await update.message.reply_text(f"Sorry, something broke: {type(e).__name__}: {e}")
        return
    if error is not None:
        await update.message.reply_text(error)
        return
    assert pending is not None
    await _start_dialogue(update, context, chat_id, pending, agent_reply)


# ---------- dialogue state machine ----------


async def _start_dialogue(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    pending: list[PendingEvent],
    agent_reply: str,
) -> None:
    """Show the agent's summary (covers duplicate detection), then walk
    through any new drafts. If there are no drafts, the summary is the
    only message the user sees."""
    assert update.message is not None
    reply = agent_reply.strip() or "Nothing to schedule from that message."
    await update.message.reply_text(reply)
    if not pending:
        return
    # New run starts fresh — drop any half-finished session for this chat.
    _sessions[chat_id] = ChatSession(chat_id=chat_id, queue=list(pending))
    await _send_current(update.message.chat_id, context, chat_id)


async def _send_current(
    tg_chat_id: int,
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
) -> None:
    """Post the proposal + keyboard for the current pending event."""
    session = _sessions[chat_id]
    text = render_proposal(session.current(), session.current_idx + 1, session.total)
    markup = _to_markup(render_keyboard(session.current()))
    await context.bot.send_message(chat_id=tg_chat_id, text=text, reply_markup=markup)


async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle every button tap on a proposal message."""
    query = update.callback_query
    if query is None or not _is_allowed(update):
        return
    await query.answer()  # stop Telegram's loading spinner
    chat_id = update.effective_chat.id
    session = _sessions.get(chat_id)
    if session is None or session.is_done:
        await query.edit_message_text("This dialogue is no longer active.")
        return

    data = query.data or ""
    if data == "confirm":
        await _confirm(update, context, chat_id, session)
        return
    if data == "cancel":
        await _cancel(update, context, chat_id, session)
        return

    # Field edits re-render the current message in place.
    updated = _apply_edit(session.current(), data)
    if updated is None:
        return  # unknown callback, ignore
    session.replace_current(updated)
    await query.edit_message_text(
        text=render_proposal(updated, session.current_idx + 1, session.total),
        reply_markup=_to_markup(render_keyboard(updated)),
    )


def _apply_edit(pending: PendingEvent, data: str) -> PendingEvent | None:
    if data.startswith("cat:"):
        value = data[4:]
        if value in ("Family", "Professional"):
            return pending.with_category(value)  # type: ignore[arg-type]
    if data.startswith("type:"):
        value = data[5:]
        if value in ("Event", "Meeting"):
            return pending.with_event_type(value)  # type: ignore[arg-type]
    if data == "dur:+1" and pending.duration_hours < MAX_DURATION_HOURS:
        return pending.with_duration_delta(+1)
    if data == "dur:-1" and pending.duration_hours > MIN_DURATION_HOURS:
        return pending.with_duration_delta(-1)
    if data == "dur:0":
        return pending  # middle "label" button, no-op
    return None


async def _confirm(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    session: ChatSession,
) -> None:
    query = update.callback_query
    assert query is not None
    calendar: CalendarPort = context.bot_data["calendar"]
    pending = session.current()
    try:
        event_id = commit_pending(pending, calendar)
    except Exception as e:
        logger.exception("commit_pending failed")
        await query.edit_message_text(f"Could not save: {type(e).__name__}: {e}")
        return
    session.record_commit(event_id)
    await query.edit_message_text(
        f"✅ Saved: {pending.final_title()}"
    )
    session.advance()
    await _advance_or_finish(update, context, chat_id, session)


async def _cancel(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    session: ChatSession,
) -> None:
    query = update.callback_query
    assert query is not None
    skipped = session.total - session.current_idx
    await query.edit_message_text(
        f"❌ Skipped {skipped} remaining event(s)."
    )
    session.current_idx = session.total  # mark done without further commits
    await _advance_or_finish(update, context, chat_id, session)


async def _advance_or_finish(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    session: ChatSession,
) -> None:
    if not session.is_done:
        await _send_current(update.effective_chat.id, context, chat_id)
        return
    committed = session.committed_ids
    _last_run_ids[chat_id] = list(committed)
    _sessions.pop(chat_id, None)
    if committed:
        summary = render_done(len(committed), session.total)
        await context.bot.send_message(chat_id=update.effective_chat.id, text=summary)


# ---------- commands ----------


async def on_upcoming(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_allowed(update) or update.message is None:
        return
    calendar: CalendarPort = context.bot_data["calendar"]
    reply = handle_upcoming(calendar=calendar, today=date.today())
    await update.message.reply_text(reply)


async def on_undo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_allowed(update) or update.message is None:
        return
    chat_id = update.effective_chat.id
    calendar: CalendarPort = context.bot_data["calendar"]
    ids = _last_run_ids.pop(chat_id, [])
    reply = handle_undo(calendar=calendar, event_ids=ids)
    await update.message.reply_text(reply)


# ---------- plumbing ----------


def _to_markup(keyboard: Keyboard) -> InlineKeyboardMarkup:
    """Convert the neutral button grid into Telegram's native type."""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(label, callback_data=data) for label, data in row]
        for row in keyboard
    ])


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        level=logging.INFO,
    )
    if not TELEGRAM_BOT_TOKEN:
        raise SystemExit("TELEGRAM_BOT_TOKEN is not set. Add it to .env.")
    if not TELEGRAM_ALLOWED_USER_IDS:
        raise SystemExit(
            "TELEGRAM_ALLOWED_USER_IDS is empty. Add your numeric user ID to .env."
        )

    app = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .connect_timeout(30.0)
        .read_timeout(30.0)
        .write_timeout(30.0)
        .get_updates_connect_timeout(30.0)
        .get_updates_read_timeout(40.0)
        .build()
    )
    llm = get_client()
    app.bot_data["calendar"] = GoogleCalendar.from_env()
    app.bot_data["llm"] = llm
    app.bot_data["transcriber"] = llm  # AnthropicClient satisfies both protocols

    app.add_handler(CommandHandler("upcoming", on_upcoming))
    app.add_handler(CommandHandler("undo", on_undo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, on_photo))
    app.add_handler(MessageHandler(filters.Document.PDF, on_pdf))
    app.add_handler(CallbackQueryHandler(on_callback))

    logger.info("Dobby bot starting (allowlist: %s)", TELEGRAM_ALLOWED_USER_IDS)
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
