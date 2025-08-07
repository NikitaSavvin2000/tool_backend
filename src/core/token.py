import logging

import yaml
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.core.configuration.config import settings

logger = logging.getLogger(__name__)

security = HTTPBearer()


def load_tokens():
    try:
        tokens_link = settings.TOKENS_LIST
        if not tokens_link:
            raise ValueError("Environment variable TOKEN_LIST is not set or empty.")

        logger.info(f"Loading tokens from: {tokens_link}")
        with open(tokens_link, 'r', encoding='utf-8') as file:
            tokens_data = yaml.safe_load(file)

        if not tokens_data or "tokens" not in tokens_data:
            raise ValueError("Tokens file is empty or does not contain the 'tokens' key.")

        VALID_TOKENS = [
            token["token"] for token in tokens_data["tokens"]
            if token.get("source") == "tool_backend"
        ]

        if not VALID_TOKENS:
            raise ValueError("No valid tokens found for 'tool_backend' source.")

        logger.info("Tokens loaded successfully.")
        return VALID_TOKENS
    except Exception as e:
        logger.error(f"Failed to load tokens: {e}")

    return None


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    VALID_TOKENS = load_tokens()
    if VALID_TOKENS is None or token not in VALID_TOKENS:
        logger.warning(f"Unauthorized access attempt with token: {token[:5]}...")
        raise HTTPException(
            status_code=401,
            detail="Unauthorized.",
        )
    logger.info(f"Token verified successfully: {token[:5]}...")
    return token