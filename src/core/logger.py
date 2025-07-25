# src/core/logger.py

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
DATE_FMT = "%Y-%m-%d %H:%M:%S"

def setup_logger(name: str = "horizon_forecast", level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.hasHandlers():
        return logger

    formatter = logging.Formatter(fmt=FORMAT, datefmt=DATE_FMT)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Файловый хендлер (ротация каждые 5 МБ, до 5 файлов)
    file_handler = RotatingFileHandler(
        LOG_DIR / "app.log",
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger