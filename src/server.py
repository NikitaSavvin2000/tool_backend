# src/server.py
import multiprocessing

import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.core.configuration.config import settings
from src.core.logger import logger
from src.core.token import verify_token
from src.routers.api_router import api_router

# Загрузка переменных окружения
load_dotenv()

logger.info("Starting microservice main forecast")

# Определение количества воркеров
workers = multiprocessing.cpu_count()
logger.info(f"[WORKERS] Count workers = {workers}")

# Создание FastAPI приложения
app = FastAPI(
    docs_url="/docs",
    dependencies=[Depends(verify_token)] if settings.VERIFY_TOKEN else [],
)

# Добавление middleware для CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_origins_urls(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключение роутеров API
app.include_router(api_router, prefix="/api/v1")
for route in app.routes:
    print(route.path)


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


@app.get("/")
def read_root():
    logger.info("Root endpoint accessed.")
    return {"message": "Welcome to the Horizon System API"}


if __name__ == "__main__":
    try:
        logger.info(f"Starting server on http://{settings.HOST}:{settings.PORT}")
        uvicorn.run(
            "server:app",
            host=settings.HOST,
            port=settings.PORT,
            workers=4,
            log_level="debug",
        )
    except Exception as e:
        logger.error(f"Failed to start server: {e}")