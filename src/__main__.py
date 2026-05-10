import sys
from src.server import run_server
from src.logging import setup_logging, logger

if __name__ == "__main__":
    setup_logging()
    logger.info("Starting via python -m src")
    run_server()
