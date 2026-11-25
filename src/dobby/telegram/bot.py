"""Telegram bot entrypoint. Polls Telegram, dispatches to handlers.py.

Run with: make bot
"""
from __future__ import annotations

import logging
from datetime import date

from telegram import Update
from telegram.constants import ChatAction
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
from dobby.llm.base import Transcriber
from dobby.telegram.handlers import (
    PDF_MIME,
    handle_file,
    handle_text,
    handle_undo,
    handle_upcoming,
)

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
        reply, created_ids = handle_file(
            data=data,
            mime_type=mime_type,
            calendar=calendar,
            today=date.today(),
            client=client,
            transcriber=transcriber,
        )
    except Exception as e:
        logger.exception("file handler failed")
        await update.message.reply_text(f"Sorry, something broke: {type(e).__name__}: {e}")
        return
    if created_ids:
        _last_run_ids[chat_id] = created_ids
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
    llm = get_client()
    app.bot_data["calendar"] = GoogleCalendar.from_env()
    app.bot_data["llm"] = llm
    app.bot_data["transcriber"] = llm  # AnthropicClient satisfies both protocols

    app.add_handler(CommandHandler("upcoming", on_upcoming))
    app.add_handler(CommandHandler("undo", on_undo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, on_photo))
    app.add_handler(MessageHandler(filters.Document.PDF, on_pdf))

    logger.info("Dobby bot starting (allowlist: %s)", TELEGRAM_ALLOWED_USER_IDS)
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
