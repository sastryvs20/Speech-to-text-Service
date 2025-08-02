# logger.py
import logging
import os
from logging.handlers import RotatingFileHandler

LOG_DIR = os.getenv("LOG_DIR", "/home/gcp-admin/Whisper-Service/Whisper-Service/helpers")
LOG_FILE = os.path.join(LOG_DIR, "transcription_service.log")

os.makedirs(LOG_DIR, exist_ok=True)

def get_logger(name: str = "global_logger", level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # prevent adding multiple handlers in case of repeated imports

    logger.setLevel(level)

    # Format: timestamp - level - module - message
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Rotating file handler (5MB max, 3 backups)
    handler = RotatingFileHandler(LOG_FILE, maxBytes=5*1024*1024, backupCount=3)
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.propagate = False  # avoid duplicate logs in console

    return logger
