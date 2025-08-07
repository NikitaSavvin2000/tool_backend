# src/routers/col_update_router.py
"""
Роутер для обновления колонок, используемых для обучения.
Подключает эндпоинты из api.v1.col_update.
"""
from fastapi import APIRouter

from src.api.v1.col_update import router as col_update_api_router

# Создаём роутер и подключаем к нему API-роуты
router = APIRouter(tags=["Col Update"])
router.include_router(col_update_api_router)