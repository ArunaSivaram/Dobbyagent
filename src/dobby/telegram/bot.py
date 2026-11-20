"""Telegram bot entrypoint. Polls Telegram, dispatches to handlers.py.

Run with: make bot
"""
from __future__ import annotations

import logging
from datetime import date

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from dobby.calendar.base import CalendarPort
from dobby.calendar.google import GoogleCalendar
from dobby.config import TELEGRAM_ALLOWED_USER_IDS, TELEGRAM_BOT_TOKEN
from dobby.llm import get_client
from dobby.telegram.handlers import handle_text, handle_undo, handle_upcoming

logger = logging.getLogger("dobby.telegram")

# Per-chat state: the IDs of events created on the most recent text run.
_last_run_ids: dict[int, list[str]] = {}


def _is_allowed(update: Update) -> bool:
    if update.effective_user is None:
        return False
    return update.effective_user.id in TELEGRAM_ALLOWED_USER_IDS


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_allowed(update) or update.message is None or update.message.text is None:
        return
    chat_id = update.effective_chat.id
    calendar: CalendarPort = context.bot_data["calendar"]
    client = context.bot_data["llm"]
    try:
        reply, created_ids = handle_text(
            text=update.message.text,
            calendar=calendar,
            today=date.today(),
            client=client,
        )
    except Exception as e:
        logger.exception("agent run failed")
        await update.message.reply_text(f"Sorry, something broke: {type(e).__name__}: {e}")
        return
    _last_run_ids[chat_id] = created_ids
    await update.message.reply_text(reply)


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
    app.bot_data["calendar"] = GoogleCalendar.from_env()
    app.bot_data["llm"] = get_client()

    app.add_handler(CommandHandler("upcoming", on_upcoming))
    app.add_handler(CommandHandler("undo", on_undo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))

    logger.info("Dobby bot starting (allowlist: %s)", TELEGRAM_ALLOWED_USER_IDS)
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
