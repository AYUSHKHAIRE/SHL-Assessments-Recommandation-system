import logging
import colorlog

logger = colorlog.getLogger("docinium")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    handler = colorlog.StreamHandler()

    formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s - %(levelname)s - %(message)s",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        },
    )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

# VERY IMPORTANT (prevents double logging via root logger)
logger.propagate = False

# Silence noisy libs
# logging.getLogger("requests").setLevel(logging.WARNING)