# src/config.py
import logging
import os


class Settings:
    def __init__(self):
        self.LOGGER_LEVEL = logging.DEBUG if os.getenv("DEBUG") == "true" else logging.INFO
        self.PUBLIC_OR_LOCAL = os.getenv("PUBLIC_OR_LOCAL", "LOCAL")


settings = Settings()
