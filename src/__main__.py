import os

from src.logging import setup_logging, logger
from src.server import run_server

if __name__ == "__main__":
    debug = os.environ.get("DEBUG_MODE", "False").lower() in ("true", "1", "yes")
    setup_logging(debug)
    logger.info("Starting server via __main__...")
    run_server()
