"""Telegram bot: send a voice message, get the speaker's gender back.

Requires the TELEGRAM_BOT_TOKEN environment variable and ffmpeg on PATH.
Only one polling instance may run per token at a time.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from genderid import load_default
from genderid.inference import NO_SPEECH, TOO_SHORT, UNCERTAIN, Prediction

log = logging.getLogger(__name__)

CLASSIFIER = load_default()
MAX_FILE_MB = 20  # Telegram Bot API download limit

WELCOME = (
    "Salom! Men ovozdan jinsni aniqlayman.\n\n"
    "Menga *ovozli xabar* yuboring yoki audio fayl tashlang.\n\n"
    "Model o'zbek nutqida o'qitilgan, test aniqligi 98.1%. "
    "Ishonchim past bo'lsa, buni yashirmayman."
)


def format_result(result: Prediction) -> str:
    """Render a prediction as a Telegram message."""
    if result.label == NO_SPEECH:
        return (
            "🔇 *Nutq topilmadi*\n\n"
            "Balandroq gapiring yoki tinchroq joyda qayta yozing."
        )
    if result.label == TOO_SHORT:
        return "⏱ *Juda qisqa*\n\nKamida 1 sekundlik yozuv yuboring."

    if result.label == UNCERTAIN:
        body = (
            "🤔 *Aniq emas*\n\n"
            "Ovoz chegaraviy zonada. Sabablari: ovoz juda past yoki baland, "
            "yoki yozuvda bir nechta kishi gapiryapti."
        )
    else:
        icon = "👩" if result.label == CLASSIFIER.labels[0] else "👨"
        body = f"{icon} *{result.label}*\n\nIshonch: *{result.confidence * 100:.0f}%*"

    return (
        f"{body}\n\n"
        f"_{result.duration:.1f} s · {result.n_windows} oynadan "
        f"{result.n_speech} tasida nutq topildi_"
    )


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_markdown(WELCOME)


async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Download the attachment, run inference, reply."""
    message = update.message
    media = message.voice or message.audio or message.video_note or message.document
    if media is None:
        return

    if media.file_size and media.file_size > MAX_FILE_MB * 1024 * 1024:
        await message.reply_text(
            f"Fayl juda katta (maksimum {MAX_FILE_MB} MB). Qisqaroq yozuv yuboring."
        )
        return

    await context.bot.send_chat_action(message.chat_id, ChatAction.TYPING)
    try:
        telegram_file = await media.get_file()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input"
            await telegram_file.download_to_drive(path)
            result = CLASSIFIER.predict_file(path)
        await message.reply_markdown(format_result(result))
    except Exception:
        log.exception("Failed to process audio from chat %s", message.chat_id)
        await message.reply_text(
            "Audioni o'qib bo'lmadi. Boshqa formatda yoki qaytadan yozib yuboring."
        )


async def handle_other(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Men faqat ovoz bilan ishlayman. Ovozli xabar yuboring yoki audio fayl tashlang."
    )


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    log.exception("Unhandled error", exc_info=context.error)


def build_application(token: str) -> Application:
    """Configure the bot.

    Timeouts are raised well above the library defaults (5 s): container
    cold starts plus cloud egress regularly exceed that and the bot would die
    during initialisation.
    """
    app = (
        Application.builder()
        .token(token)
        .connect_timeout(30.0)
        .read_timeout(30.0)
        .write_timeout(30.0)
        .pool_timeout(30.0)
        .get_updates_connect_timeout(30.0)
        .get_updates_read_timeout(40.0)  # must exceed the long-poll timeout
        .build()
    )
    app.add_handler(CommandHandler(["start", "help"], cmd_start))
    app.add_handler(
        MessageHandler(
            filters.VOICE | filters.AUDIO | filters.VIDEO_NOTE | filters.Document.AUDIO,
            handle_audio,
        )
    )
    app.add_handler(MessageHandler(~filters.COMMAND, handle_other))
    app.add_error_handler(on_error)
    return app


def main() -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise SystemExit(
            "TELEGRAM_BOT_TOKEN is not set.\n"
            "  PowerShell: $env:TELEGRAM_BOT_TOKEN=\"...\"\n"
            "  Railway:    add it under Variables"
        )

    log.info("Connecting to Telegram...")
    # bootstrap_retries=-1: retry the initial handshake forever instead of
    # letting one transient network failure kill the container.
    build_application(token).run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True,
        bootstrap_retries=-1,
    )


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=logging.INFO
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    main()
