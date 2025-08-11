import logging
from typing import List, Optional

import yaml
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.core.configuration.config import settings

logger = logging.getLogger(__name__)

class TokenValidator:
    def __init__(self):
        self.security = HTTPBearer()
        self.valid_tokens: Optional[List[str]] = None

    def load_tokens(self) -> List[str]:
        try:
            tokens_link = settings.TOKENS_LIST
            if not tokens_link:
                raise ValueError("Environment variable TOKEN_LIST is not set or empty.")

            logger.info(f"Loading tokens from: {tokens_link}")
            with open(tokens_link, 'r', encoding='utf-8') as file:
                tokens_data = yaml.safe_load(file)

            if not tokens_data or "tokens" not in tokens_data:
                raise ValueError("Tokens file is empty or does not contain the 'tokens' key.")

            valid_tokens = [
                token["token"] for token in tokens_data["tokens"]
                if token.get("source") == "tool_backend"
            ]

            if not valid_tokens:
                raise ValueError("No valid tokens found for 'tool_backend' source.")

            logger.info("Tokens loaded successfully.")
            return valid_tokens
        except Exception as e:
            logger.error(f"Failed to load tokens: {e}")
            raise HTTPException(
                status_code=500,
                detail="Internal server error - token validation failed."
            )

    async def __call__(self, credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
        if self.valid_tokens is None:
            self.valid_tokens = self.load_tokens()

        token = credentials.credentials
        if token not in self.valid_tokens:
            logger.warning(f"Unauthorized access attempt with token: {token[:5]}...")
            raise HTTPException(
                status_code=401,
                detail="Unauthorized.",
            )
        logger.info(f"Token verified successfully: {token[:5]}...")
        return token
    

token_validator = TokenValidator()