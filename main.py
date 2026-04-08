"""Mini Coding Agent — entry point."""

import logging
import uvicorn

from config.settings import get_settings


def main():
    # ── Bootstrap logging ──
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    settings = get_settings()
    logger.info(
        "Starting Mini Coding Agent — model=%s  backend=%s",
        settings.model.name,
        settings.model.backend,
    )

    uvicorn.run(
        "interface.api:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=False,  # disable reload in production; use --reload for dev
        log_level="info",
    )


if __name__ == "__main__":
    main()