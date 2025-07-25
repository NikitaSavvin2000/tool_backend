# src/config.py
import os
import logging


class Settings:
    def __init__(self):
        self.LOGGER_LEVEL = logging.DEBUG if os.getenv("DEBUG") == "true" else logging.INFO
        self.PUBLIC_OR_LOCAL = os.getenv("PUBLIC_OR_LOCAL", "LOCAL")


settings = Settings()
