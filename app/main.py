"""Container entry point: Gradio web UI and Telegram bot in one process.

Both share a single loaded model. Gradio serves on $PORT in a background
thread; the bot polls on the main thread.

Environment:
    TELEGRAM_BOT_TOKEN  if set, the bot runs
    RUN_WEB=0           disables the web UI (bot only)
    PORT                HTTP port for the web UI (default 7860)
"""

from __future__ import annotations

import inspect
import logging
import os
import threading

log = logging.getLogger(__name__)


def start_web() -> None:
    """Launch Gradio without blocking the calling thread.

    ``launch()`` keyword arguments differ between Gradio majors (``show_api``
    was removed in v6), so supported names are discovered from the signature
    instead of being hard-coded.
    """
    import gradio

    from app import gradio_app

    port = int(os.environ.get("PORT", 7860))
    log.info("Gradio %s starting on port %s", gradio.__version__, port)

    supported = inspect.signature(gradio_app.demo.launch).parameters
    kwargs: dict[str, object] = {"server_name": "0.0.0.0", "server_port": port}
    for name, value in {"prevent_thread_lock": True, "quiet": True}.items():
        if name in supported:
            kwargs[name] = value

    if kwargs.get("prevent_thread_lock"):
        gradio_app.demo.launch(**kwargs)
    else:
        # This Gradio version blocks in launch(); move it off the main thread
        # so the bot can still start.
        threading.Thread(
            target=gradio_app.demo.launch, kwargs=kwargs, daemon=True
        ).start()
    log.info("Web UI listening on port %s", port)


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=logging.INFO
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    run_web = os.environ.get("RUN_WEB", "1") != "0"
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()

    if not run_web and not token:
        raise SystemExit("Nothing to run: RUN_WEB=0 and no TELEGRAM_BOT_TOKEN.")

    if run_web:
        start_web()

    if token:
        from app import telegram_bot

        telegram_bot.main()  # blocks
    else:
        log.warning("No TELEGRAM_BOT_TOKEN — running the web UI only")
        threading.Event().wait()


if __name__ == "__main__":
    main()
