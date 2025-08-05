# src/server.py
import multiprocessing
import os
import sys
import yaml

import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.core.configuration.config import settings
from src.core.logger import logger
from src.routers.api_router import api_router
from src.routers.forecast_router import router as forecast_router


# Загрузка переменных окружения
load_dotenv()

logger.info("Starting microservice main forecast")

# Настройка CORS
origins = ["http://localhost", "http://77.37.136.11"] if settings.PUBLIC_OR_LOCAL == "LOCAL" else ["http://77.37.136.11"]

# Определение количества воркеров
workers = multiprocessing.cpu_count()
logger.info(f"[WORKERS] Count workers = {workers}")

# Инициализация системы безопасности
security = HTTPBearer()

# Проверка и загрузка токенов
try:
    tokens_link = os.getenv("TOKEN_LIST")
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
except Exception as e:
    logger.error(f"Failed to load tokens: {e}")
    sys.exit(1)

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    if token not in VALID_TOKENS:
        logger.warning(f"Unauthorized access attempt with token: {token[:5]}...")
        raise HTTPException(
            status_code=401,
            detail="Unauthorized.",
        )
    logger.info(f"Token verified successfully: {token[:5]}...")
    return token

# Создание FastAPI приложения
docs_url = "/docs"
app = FastAPI(
    docs_url=docs_url,
    dependencies=[Depends(verify_token)],
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(f"Validation error: {exc.errors()} on {request.url}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "detail": exc.errors(),
            "body": exc.body,
            "message": "Ошибка валидации входных данных. Проверьте формат запроса."
        },
    )

# Добавление middleware для CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключение роутеров API
app.include_router(api_router, prefix="/api/v1")

for route in app.routes:
    print(route.path)


@app.get("/")
def read_root():
    logger.info("Root endpoint accessed.")
    return {"message": "Welcome to the Horizon System API"}

if __name__ == "__main__":
    try:
        port = 7070
        logger.info(f"Starting server on http://0.0.0.0:{port}{docs_url}")
        uvicorn.run(
            "server:app",
            host="0.0.0.0",
            port=port,
            workers=4,
            log_level="debug",
        )
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)