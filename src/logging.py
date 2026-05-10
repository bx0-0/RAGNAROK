import sys
import os
from loguru import logger

def setup_logging(debug=None):
    if debug is None:
        debug = os.environ.get("DEBUG_MODE", "False").lower() in ("true", "1", "yes")
    level = "DEBUG" if debug else "WARNING"

    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
        level=level,
    )
    return logger

logger = setup_logging()
