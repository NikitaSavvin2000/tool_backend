# src/core/config.py
import logging

from environs import Env


class Settings:
    def __init__(self):
        env = Env()
        env.read_env()

        self.LOGGER_LEVEL = logging.DEBUG if env.bool("DEBUG", False) else logging.INFO
        self.PUBLIC_OR_LOCAL = env.str("PUBLIC_OR_LOCAL", "LOCAL")

        self.HOST = env.str("HOST", '0.0.0.0')
        self.PORT = env.int('PORT', 7071)

        self.TOKENS_LIST = env.str('TOKENS_LIST')
        self.VERIFY_TOKEN = env.bool('VERIFY_TOKEN', True)

    def get_origins_urls(self):
        if self.PUBLIC_OR_LOCAL == 'PUBLIC':
            return 'http://11.11.11.11'


        return 'http://localhost'





settings = Settings()